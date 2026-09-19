"""Exercise the real pipeline using a deterministic in-memory CUDA transport.

These are scheduling/overflow tests, not GPU execution or throughput tests.
"""
import ctypes
import json
import signal
import threading
import time
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, mock_open

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
    def run_case(self, hits=True, interrupt=False, fail_worker=False,
                 fail_save=False, capacity=3, rounds=1, on_launch=None, real_pool=False):
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
            if on_launch:
                on_launch(launches[0])
            if interrupt and launches[0] == 2*rounds:
                signal.raise_signal(signal.SIGINT)
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder)/'matches.jsonl')
            pool_factory = app.ProcessPoolExecutor if real_pool else executor
            with patch.object(app, 'cp', fake_cp), patch.object(app, 'ProcessPoolExecutor', pool_factory), \
                 patch.object(app, '_gpu_stats', return_value=None), patch.object(app, '_cpu_percent', return_value=None):
                if fail_worker or fail_save:
                    failure = (patch.object(app, 'classify_batch', side_effect=RuntimeError('injected worker failure'))
                               if fail_worker else patch.object(app.os, 'fsync', side_effect=OSError('injected disk failure')))
                    with failure:
                        with self.assertRaises((RuntimeError, OSError)):
                            app.run_pipeline(kernel, None, dict(mode='exact', prefix='T'), output,
                                             0.00000001, states, 1, 1, points, steps, capacity)
                    recovered = [json.loads(line) for line in Path(output+'.pending.jsonl').read_text().splitlines()]
                    self.assertEqual(len(recovered), 2*points*steps)
                    self.assertEqual(len({r['address'] for r in recovered}), len(recovered))
                    return
                metrics = app.run_pipeline(kernel, None, dict(mode='exact', prefix='T'), output,
                                           0 if interrupt else 0.00000001,
                                           states, 1, 1, points, steps, capacity)
            records = [json.loads(line) for line in Path(output).read_text(encoding='utf-8').splitlines()]
            expected = points*steps*2*rounds
            self.assertEqual(metrics['addresses'], expected)
            self.assertEqual(metrics['candidates'], expected if hits else 0)
            self.assertEqual(metrics['classified'], expected if hits else 0)
            self.assertEqual(len(records), expected if hits else 0)
            self.assertEqual(len({x['address'] for x in records}), len(records))
            self.assertEqual(metrics['dropped_candidates'], 0)
            if hits and capacity < points*steps:
                self.assertGreater(metrics['overflow_replays'], 0)
            self.assertEqual([s['step_offset'] for s in states], [steps*rounds]*2)
            return metrics

    def test_overflow_replay_and_timed_drain(self):
        self.run_case()
    def test_rounded_pinned_buffers_and_overflow_reallocation(self):
        allocations = []
        def rounded_pinned(size):
            # CUDA's memory pool can expose more bytes than requested, even
            # for the 4-byte counter. Record buffers need not divide evenly.
            allocated = ((size + 511) // 512) * 512
            allocations.append((size, allocated))
            return pinned(allocated)
        with patch.object(fake_cp.cuda, 'alloc_pinned_memory', rounded_pinned):
            self.run_case()
        self.assertIn((4, 512), allocations)
        self.assertTrue(any(size > 3*app.MATCH_DTYPE.itemsize
                            for size, _ in allocations))
    def test_no_hits_still_exits(self):
        self.run_case(hits=False)
    def test_ctrl_c_drains_both_streams(self):
        self.run_case(interrupt=True)
    def test_worker_failure_retains_candidates(self):
        self.run_case(fail_worker=True)
    def test_disk_failure_retains_candidates(self):
        self.run_case(fail_save=True)

    def test_real_spawn_pool_from_background_coordinator(self):
        with patch.object(app, 'classifier_workers', return_value=2):
            self.run_case(capacity=10000, real_pool=True)

    def test_pool_submission_failure_retains_candidates(self):
        class BrokenPool(ThreadPoolExecutor):
            def submit(self, *args, **kwargs):
                raise RuntimeError('injected submission failure')
        with patch(__name__+'.executor', lambda max_workers, **_: BrokenPool(max_workers)):
            self.run_case(fail_worker=True)

    def test_ready_batch_saved_before_slow_first_batch(self):
        release = threading.Event()
        lock = threading.Lock()
        calls = [0]
        real_classify, real_sync = app.classify_batch, app.os.fsync
        def slow_first(*args):
            with lock:
                calls[0] += 1
                first = calls[0] == 1
            if first and not release.wait(5):
                raise RuntimeError('ready batch blocked behind first batch')
            return real_classify(*args)
        def sync_then_release(fd):
            real_sync(fd)
            release.set()
        with patch.object(app, 'classify_batch', slow_first), \
             patch.object(app.os, 'fsync', sync_then_release), \
             patch.object(app, 'classifier_workers', return_value=2):
            try:
                self.run_case(capacity=10000)
            finally:
                release.set()

    def test_gpu_launches_while_classifier_waits(self):
        entered, release = threading.Event(), threading.Event()
        real_classify = app.classify_batch
        def delayed_classify(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('GPU scheduling waited for classification')
            return real_classify(*args)
        def on_launch(count):
            if count == 3:
                self.assertTrue(entered.wait(5))
            if count == 4:
                release.set()
        with patch.object(app, 'classify_batch', delayed_classify):
            try:
                self.run_case(interrupt=True, capacity=10000, rounds=2, on_launch=on_launch)
            finally:
                release.set()

    def test_gpu_launches_while_disk_waits(self):
        entered, release = threading.Event(), threading.Event()
        def delayed_sync(_):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('GPU scheduling waited for disk')
        def on_launch(count):
            if count == 3:
                self.assertTrue(entered.wait(5))
            if count == 4:
                release.set()
        with patch.object(app.os, 'fsync', delayed_sync):
            try:
                self.run_case(interrupt=True, capacity=10000, rounds=2, on_launch=on_launch)
            finally:
                release.set()

    def test_full_handoff_queue_drains_without_loss(self):
        real_classify = app.classify_batch
        def delayed_classify(*args):
            time.sleep(0.02)
            return real_classify(*args)
        with patch.object(app, 'classifier_workers', return_value=1), \
             patch.object(app, 'classify_batch', delayed_classify):
            metrics = self.run_case()
        self.assertGreater(metrics['backpressure_seconds'], 0)

    def test_worker_count_respects_affinity_and_cpu_quota(self):
        with patch.object(app.os, 'process_cpu_count', return_value=32, create=True), \
             patch.object(app.os, 'sched_getaffinity', return_value=set(range(16)), create=True):
            for quota, expected in [('max 100000', 14), ('800000 100000', 6), ('100000 100000', 1)]:
                with patch('builtins.open', mock_open(read_data=quota)):
                    self.assertEqual(app.classifier_workers(), expected)


if __name__ == '__main__':
    unittest.main()
