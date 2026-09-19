"""Same-GPU baseline comparison. Never reports unmeasured GPU throughput.

python benchmark.py --seconds 2 --repeats 5
python benchmark.py --pipeline-seconds 30   # Also measure the complete new pipeline.
"""
import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import cupy as cp
import tron_vanity_gpu as app
from cpu_worker import gen_startpoints_batch

BASELINE = '59dfbc1d8d971898c360af8912a9f22c6fd1de7e'
OLD_PATTERN = np.dtype([('prefix_len', np.int32), ('suffix_len', np.int32),
                        ('repeat_n', np.int32), ('prefix', np.uint8, 40),
                        ('suffix', np.uint8, 40)], align=True)


def measure(kernel, params, m, blocks, seconds, repeats):
    threads = blocks*64
    xb, yb, _ = gen_startpoints_batch(threads*m)
    sx = np.frombuffer(xb, dtype='>u8').reshape(-1,4)[:,::-1].astype(np.uint64).ravel()
    sy = np.frombuffer(yb, dtype='>u8').reshape(-1,4)[:,::-1].astype(np.uint64).ravel()
    x, y = cp.asarray(sx), cp.asarray(sy)
    params = cp.asarray(params)
    out, count = cp.empty(16384, dtype=app.MATCH_DTYPE), cp.zeros(1, dtype=cp.uint32)
    offset = 0

    def launch(steps):
        nonlocal offset
        count.fill(0)
        cp.cuda.Stream.null.synchronize()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        kernel((blocks,), (64,), (x, y, np.int32(steps), np.uint64(offset), params,
                                 out, count, np.uint32(len(out))))
        end.record()
        end.synchronize()
        offset += steps
        return cp.cuda.get_elapsed_time(begin, end)/1000, int(count.get()[0])

    warm, _ = launch(8)
    steps = max(1, min(2048, int(.25*8/max(warm, .000001))))
    rates, durations, candidates = [], [], 0
    hardware = []
    stop = threading.Event()

    def sample():
        while not stop.wait(.2):
            gpu, cpu = app._gpu_stats(), app._cpu_percent()
            hardware.append((gpu[0] if gpu else None, cpu))

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        for _ in range(repeats):
            elapsed, addresses = 0., 0
            while elapsed < seconds:
                duration, hits = launch(steps)
                elapsed += duration
                addresses += threads*m*steps
                candidates += hits
            durations.append(elapsed)
            rates.append(addresses/elapsed)
    finally:
        stop.set()
        monitor.join()
    def mean_column(index):
        vals = [x[index] for x in hardware if x[index] is not None]
        return statistics.mean(vals) if vals else None
    return dict(m=m, addresses_per_second=statistics.median(rates), trial_rates=rates,
                candidates_per_second=candidates/sum(durations),
                gpu_utilization_percent=mean_column(0), cpu_system_percent=mean_column(1),
                scope='GPU kernel only; candidate counting enabled, no CPU classification/output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--pipeline-seconds', type=float, default=0)
    parser.add_argument('--output', default='benchmark_results.json')
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats < 1 or args.pipeline_seconds < 0:
        parser.error('times must be positive and repeats >= 1')
    report = dict(baseline_commit=BASELINE, measured=False, cases=[], limitations=[])
    exit_code = 0
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA GPU')
        props = cp.cuda.runtime.getDeviceProperties(0)
        report['gpu'] = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        report['driver_version'] = cp.cuda.runtime.driverGetVersion()
        arch = 'sm_{}{}'.format(props['major'], props['minor'])
        blocks = props['multiProcessorCount']*4
        from verify_kernel import main as verify
        if verify() != 0:
            raise RuntimeError('GPU verification failed; refusing benchmark')
        original = subprocess.check_output(['git','show',BASELINE+':kernels.cu'], cwd=Path(__file__).parent).decode('utf-8')
        suffix = dict(mode='exact', suffix='888888', prefix='')
        wide = dict(mode='wide', min_len=8, max_len=34)
        cases = [('original_suffix', None), ('new_suffix', suffix), ('new_wide', wide)]
        for name, cfg in cases:
            for m in ((8,16) if cfg is None else (8,16,24,32)):
                try:
                    if cfg is None:
                        module = cp.RawModule(code=original, options=('-std=c++14','--use_fast_math',
                                              '-arch='+arch, '-DPOINTS_PER_THREAD='+str(m)))
                        kernel = module.get_function('vanity_kernel')
                        params = np.zeros(1, dtype=OLD_PATTERN)
                        params['suffix_len'] = 6
                        params['suffix'][0,:6] = np.frombuffer(b'888888', dtype=np.uint8)
                    else:
                        kernel = app.load_kernel(arch, m, cfg['mode'])
                        params = app.make_pattern_params(cfg)
                    result = measure(kernel, params, m, blocks, args.seconds, args.repeats)
                    result['case'] = name
                    report['cases'].append(result)
                    print(name, 'M='+str(m), round(result['addresses_per_second']), 'addresses/s', flush=True)
                except Exception as exc:
                    report['cases'].append(dict(case=name, m=m, error=str(exc)))
        winners = {}
        for name, _ in cases:
            measured = [r for r in report['cases'] if r['case']==name and 'addresses_per_second' in r]
            if not measured:
                raise RuntimeError('No successful benchmark for '+name)
            winners[name] = max(measured, key=lambda x:x['addresses_per_second'])
        report['best'] = winners
        regression = 1-winners['new_suffix']['addresses_per_second']/winners['original_suffix']['addresses_per_second']
        report['suffix_regression_percent'] = 100*regression
        report['tail_performance_gate'] = 'FAIL' if regression > .15 else 'PASS'
        report['measured'] = True
        if regression > .15:
            exit_code = 1
            print('FAIL: normal suffix throughput regression exceeds 15%; optimize before accepting.')
        if args.pipeline_seconds:
            folder = Path('命中地址')
            folder.mkdir(exist_ok=True)
            report['pipeline'] = {}
            for name, cfg in cases[1:]:
                path = folder/('benchmark_'+name+'_'+str(time.time_ns())+'.jsonl')
                report['pipeline'][name] = app.run_search(cfg, str(path), args.pipeline_seconds/60)
        else:
            report['limitations'].append('No full-pipeline measurement; use --pipeline-seconds 30.')
    except Exception as exc:
        exit_code = 1
        report['error'] = str(exc)
        print('GPU benchmark unavailable/failed:', exc)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
