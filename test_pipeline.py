"""Exercise the real pipeline using a deterministic in-memory CUDA transport.

These are scheduling/overflow tests, not GPU execution or throughput tests.
"""
import ctypes
import json
import signal
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tron_vanity_gpu as app
from cpu_worker import _priv_to_address


class Array:
    def __init__(self, array):
        self.array = array
        self.data = SimpleNamespace(ptr=array.ctypes.data)
    def fill(self, value):
        self.array.fill(value)
    def __len__(self):
        return len(self.array)


class Stream:
    ptr = 0
    def __init__(self, **_): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def synchronize(self): pass


Stream.null = Stream()


def pinned(size):
    buf = ctypes.create_string_buffer(size)
    buf.ptr = ctypes.addressof(buf)
    return buf


fake_cp = SimpleNamespace(
    cuda=SimpleNamespace(Stream=Stream, alloc_pinned_memory=pinned,
                         runtime=SimpleNamespace(memcpyDeviceToHost=2,
                            memcpyAsync=lambda dst, src, size, *_: ctypes.memmove(dst, src, size))),
    uint32=np.uint32, empty=lambda size, dtype: Array(np.empty(size, dtype=dtype)),
    zeros=lambda size, dtype: Array(np.zeros(size, dtype=dtype)),
    empty_like=lambda a: Array(np.empty_like(a.array)),
    copyto=lambda dst, src: np.copyto(dst.array, src.array))


def executor(max_workers, **_):
    return ThreadPoolExecutor(max_workers=max_workers)


class PipelineTests(unittest.TestCase):
    def run_case(self, hits=True, interrupt=False, fail_worker=False):
        points, steps = 24, 7
        states = [dict(cur_x=Array(np.array([0], dtype=np.uint64)),
                       cur_y=Array(np.array([0], dtype=np.uint64)),
                       base_keys=list(range(1000+stream*10000, 1000+stream*10000+points*100, 100)),
                       step_offset=0) for stream in range(2)]
        launches = [0]
        def kernel(grid, block, args, **_):
            x, y, count_steps, offset, params, out, count, capacity = args
            count_steps, offset = int(count_steps), int(offset)
            self.assertEqual(int(x.array[0]), offset)
            keys = states[0 if x is states[0]['cur_x'] else 1]['base_keys']
            count.array[0] = points*count_steps if hits else 0
            if hits:
                for i in range(min(int(capacity), points*count_steps)):
                    step, chain = divmod(i, points)
                    record = out.array[i]
                    record['thread_id'] = chain
                    record['step'] = offset+step
                    address = _priv_to_address((keys[chain]+offset+step).to_bytes(32, 'big'))
                    record['address'] = np.frombuffer(address.encode(), dtype=np.uint8)
            x.array[0] += count_steps
            y.array[0] += count_steps
            launches[0] += 1
            if interrupt and launches[0] == 2:
                signal.raise_signal(signal.SIGINT)
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder)/'matches.jsonl')
            with patch.object(app, 'cp', fake_cp), patch.object(app, 'ProcessPoolExecutor', executor), \
                 patch.object(app, '_gpu_stats', return_value=None), patch.object(app, '_cpu_percent', return_value=None):
                if fail_worker:
                    with patch.object(app, 'classify_batch', side_effect=RuntimeError('injected worker failure')):
                        with self.assertRaises(RuntimeError):
                            app.run_pipeline(kernel, None, dict(mode='exact', prefix='T'), output,
                                             0.00000001, states, 1, 1, points, steps, 3)
                    recovered = [json.loads(line) for line in Path(output+'.pending.jsonl').read_text().splitlines()]
                    self.assertEqual(len(recovered), 2*points*steps)
                    self.assertEqual(len({r['address'] for r in recovered}), len(recovered))
                    return
                metrics = app.run_pipeline(kernel, None, dict(mode='exact', prefix='T'), output,
                                           0 if interrupt else 0.00000001,
                                           states, 1, 1, points, steps, 3)
            records = [json.loads(line) for line in Path(output).read_text(encoding='utf-8').splitlines()]
            expected = points*steps*2
            self.assertEqual(metrics['addresses'], expected)
            self.assertEqual(metrics['candidates'], expected if hits else 0)
            self.assertEqual(metrics['classified'], expected if hits else 0)
            self.assertEqual(len(records), expected if hits else 0)
            self.assertEqual(len({x['address'] for x in records}), len(records))
            self.assertEqual(metrics['dropped_candidates'], 0)
            if hits: self.assertGreater(metrics['overflow_replays'], 0)
            self.assertEqual([s['step_offset'] for s in states], [steps, steps])

    def test_overflow_replay_and_timed_drain(self):
        self.run_case()
    def test_no_hits_still_exits(self):
        self.run_case(hits=False)
    def test_ctrl_c_drains_both_streams(self):
        self.run_case(interrupt=True)
    def test_worker_failure_retains_candidates(self):
        self.run_case(fail_worker=True)


if __name__ == '__main__':
    unittest.main()
