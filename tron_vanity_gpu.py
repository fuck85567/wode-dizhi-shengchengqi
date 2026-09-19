import os
import sys
import time
import hashlib
import threading
import multiprocessing as mp
import signal
import atexit
from datetime import datetime
import json
import math
import statistics
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from cpu_worker import (classify_batch, gen_startpoints_batch, init_classifier,
                        validate_config, split_targets, rule_mask)
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    for _rel in ("", os.path.join("nvidia", "cuda_runtime", "bin"), os.path.join("nvidia", "cuda_nvrtc", "bin")):
        _dll_dir = os.path.join(_base, _rel)
        if os.path.isdir(_dll_dir):
            try:
                os.add_dll_directory(_dll_dir)
            except Exception:
                pass
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetConsoleOutputCP(65001)
        k32.SetConsoleCP(65001)
    except Exception:
        pass
try:
    import warnings
    warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
    import cupy as cp
    import numpy as np
except ImportError as e:
    sys.stderr.write(
        "缺少依赖: {}\n请运行: pip install cupy-cuda12x numpy coincurve pycryptodome base58\n".format(e)
    )
    sys.exit(1)
try:
    import coincurve
    from Crypto.Hash import keccak as keccak_lib
    import base58
except ImportError as e:
    sys.stderr.write("缺少依赖: {}\n请运行: pip install coincurve pycryptodome base58\n".format(e))
    sys.exit(1)
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
    atexit.register(lambda: pynvml.nvmlShutdown())
except Exception:
    _NVML_HANDLE = None
    _HAS_NVML = False
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False
def _gpu_stats():
    if not _HAS_NVML or _NVML_HANDLE is None:
        return None
    try:
        u = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu
        p = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
        t = pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
        m = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).used / (1024 * 1024)
        return (u, p, t, m)
    except Exception:
        return None
def _cpu_percent():
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BASE58_SET = set(BASE58_ALPHABET)
ADDRESS_LEN = 34
def cpu_priv_to_address(priv_int: int) -> str:
    pub = coincurve.PublicKey.from_valid_secret(priv_int.to_bytes(32, "big")).format(compressed=False)
    k = keccak_lib.new(digest_bits=256)
    k.update(pub[1:])
    payload = b"\x41" + k.digest()[-20:]
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode()
def validate_pattern(prefix, suffix, combine_or=False):
    try:
        validate_config(dict(mode="exact", prefix=prefix, suffix=suffix, combine_or=combine_or))
        return []
    except ValueError as exc:
        return [str(exc)]


def fmt_time(seconds):
    if seconds is None or seconds != seconds or seconds == float("inf") or seconds < 0:
        return "?"
    if seconds < 1:    return "{:.0f}毫秒".format(seconds * 1000)
    if seconds < 60:   return "{:.1f}秒".format(seconds)
    if seconds < 3600: return "{:.1f}分".format(seconds / 60)
    if seconds < 86400:return "{:.1f}小时".format(seconds / 3600)
    if seconds < 86400 * 365: return "{:.1f}天".format(seconds / 86400)
    return "{:.1f}年".format(seconds / 86400 / 365)
def fmt_num(n):
    if n < 1e3:
        return "{:.0f}".format(n)
    if n < 1e4:
        return "{:.0f}".format(n)
    if n < 1e8:
        v = n / 1e4
        if v < 10:    return "{:.2f}万".format(v)
        if v < 100:   return "{:.1f}万".format(v)
        return "{:.0f}万".format(v)
    if n < 1e12:
        v = n / 1e8
        if v < 10:    return "{:.2f}亿".format(v)
        if v < 100:   return "{:.1f}亿".format(v)
        return "{:.0f}亿".format(v)
    v = n / 1e12
    if v < 10:    return "{:.2f}万亿".format(v)
    if v < 100:   return "{:.1f}万亿".format(v)
    return "{:.0f}万亿".format(v)
APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
RESOURCE_BASE_DIR = getattr(sys, "_MEIPASS", APP_BASE_DIR)
KERNEL_PATH = os.path.join(RESOURCE_BASE_DIR, "kernels.cu")
PATTERN_DTYPE = np.dtype([
    ("mode", np.int32), ("combine_or", np.int32),
    ("prefix_count", np.int32), ("suffix_count", np.int32),
    ("min_len", np.int32), ("max_len", np.int32), ("rules", np.int32),
    ("prefix_len", np.int32, 8), ("suffix_len", np.int32, 8),
    ("prefixes", np.uint8, (8, 40)), ("suffixes", np.uint8, (8, 40)),
], align=True)
MATCH_DTYPE = np.dtype([
    ("thread_id", np.uint32),
    ("_pad",      np.uint32),
    ("step",      np.uint64),
    ("address",   np.uint8, 34),
    ("_pad2",     np.uint8, 6),
], align=True)
def load_kernel(arch=None, points_per_thread=8, mode=None):
    with open(KERNEL_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    options = ["-std=c++14", "--use_fast_math",
               "-DPOINTS_PER_THREAD={}".format(points_per_thread),
               "-DSEARCH_MODE={}".format(-1 if mode is None else int(mode == "wide"))]
    if arch:
        options.append("-arch=" + arch)
    module = cp.RawModule(
        code=src,
        options=tuple(options),
        backend="nvrtc",
        name_expressions=("vanity_kernel",),
    )
    return module.get_function("vanity_kernel")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def make_pattern_params(cfg):
    validate_config(cfg)
    pp_local = np.zeros(1, dtype=PATTERN_DTYPE)
    pp_local["mode"] = 1 if cfg.get("mode") == "wide" else 0
    pp_local["combine_or"] = int(bool(cfg.get("combine_or", False)))
    ps = split_targets(cfg.get("prefix", ""))
    ss = split_targets(cfg.get("suffix", ""))
    pp_local["prefix_count"] = len(ps)
    pp_local["suffix_count"] = len(ss)
    pp_local["min_len"] = int(cfg.get("min_len", 8))
    pp_local["max_len"] = int(cfg.get("max_len", 34))
    pp_local["rules"] = rule_mask(cfg)
    for i, value in enumerate(ps):
        pp_local["prefix_len"][0, i] = len(value)
        pp_local["prefixes"][0, i, :len(value)] = np.frombuffer(value.encode(), dtype=np.uint8)
    for i, value in enumerate(ss):
        pp_local["suffix_len"][0, i] = len(value)
        pp_local["suffixes"][0, i, :len(value)] = np.frombuffer(value.encode(), dtype=np.uint8)
    return pp_local

def run_search(config: dict, output_path: str, duration_minutes: float = 0):
    validate_config(config)
    if not math.isfinite(duration_minutes) or duration_minutes < 0:
        raise ValueError("运行时间必须为有限的非负数")
    mode = config["mode"]
    prefix = config.get("prefix", "")
    suffix = config.get("suffix", "")
    combine_or = bool(config.get("combine_or", False))
    try:
        n_dev = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        sys.stderr.write(
            "✗ CUDA 初始化失败: {}\n"
            "  请检查:\n"
            "    1) NVIDIA 显卡驱动是否已安装 (终端跑 nvidia-smi 应能看到设备)\n"
            "    2) 显卡是否支持 CUDA (compute capability ≥ 7.0, 即 RTX 20 系及以上)\n"
            "    3) Python 是否能找到 CUDA 库 (重启电脑或重装 cupy-cuda12x)\n".format(e))
        sys.exit(1)
    if n_dev == 0:
        sys.stderr.write("✗ 没检测到 NVIDIA GPU. 请确认驱动已装, nvidia-smi 能列出设备.\n")
        sys.exit(1)
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
    except Exception as e:
        sys.stderr.write("✗ 读取 GPU 0 信息失败: {}\n".format(e))
        sys.exit(1)
    gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    n_sms = props["multiProcessorCount"]
    cc_major = props["major"]
    cc_minor = props["minor"]
    if cc_major < 7:
        sys.stderr.write(
            "✗ GPU 计算能力 {}.{} 太低 (需要 ≥ 7.0, 即 RTX 20 系或更新).\n"
            "  当前 GPU: {}\n".format(cc_major, cc_minor, gpu_name))
        sys.exit(1)
    M_CANDIDATES = [32, 24, 16, 8]
    POINTS_PER_THREAD = None
    THREADS_PER_BLOCK = 64
    BLOCKS_PER_SM = 4
    n_blocks = n_sms * BLOCKS_PER_SM
    n_threads = n_blocks * THREADS_PER_BLOCK
    STEPS_PER_LAUNCH = 128
    MAX_MATCHES = 16384
    N_STREAMS = 2
    cpu_count = mp.cpu_count()
    parts = []
    if mode == "wide":
        pattern_desc = "全地址靓号宽筛 {}-{} 位".format(config.get("min_len", 8), config.get("max_len", 34))
    else:
        if prefix: parts.append("前缀={}".format(prefix))
        if suffix: parts.append("后缀={}".format(suffix))
        pattern_desc = (" OR " if combine_or else " AND ").join(parts)
    prob = 0.0  # Overlapping rules do not have a reliable simple ETA.
    print()
    print("=" * 70)
    print("  TRON 靓号生成器 — GPU + CPU 全速版")
    print("=" * 70)
    print("  GPU 设备     : {}".format(gpu_name))
    print("  计算能力     : {}.{}   SM 数: {}".format(cc_major, cc_minor, n_sms))
    print("  GPU 并发线程 : {} (= {} block × {})".format(n_threads, n_blocks, THREADS_PER_BLOCK))
    print("  CPU 分类      : GPU 命中后二次校验和分类 (CPU 暴力搜索关闭, 本机 {} 核)".format(cpu_count))
    print("  搜索模式     : {}".format(pattern_desc))
    if prob > 0:
        print("  理论概率     : 平均 {} 个地址出 1 个".format(fmt_num(1 / prob)))
    print("  输出文件     : {}".format(output_path))
    print("  按 Ctrl+C 退出")
    print("=" * 70)
    print()
    print()
    print("按当前模式实测 M (8 / 16 / 24 / 32)...")
    arch = "sm_{}{}".format(cc_major, cc_minor)
    def _bench_m(m_val):
        try:
            k = load_kernel(arch=arch, points_per_thread=m_val, mode=mode)
            # Use the production grid and independent random starts. A small,
            # identical-point microbenchmark biases both occupancy and hits.
            xb, yb, _ = gen_startpoints_batch(n_threads*m_val)
            sx = np.frombuffer(xb, dtype=">u8").reshape(-1, 4)[:, ::-1].astype(np.uint64).ravel()
            sy = np.frombuffer(yb, dtype=">u8").reshape(-1, 4)[:, ::-1].astype(np.uint64).ravel()
            cur_x, cur_y = cp.asarray(sx), cp.asarray(sy)
            params = cp.asarray(make_pattern_params(config))
            hits, count = cp.zeros(16384, dtype=MATCH_DTYPE), cp.zeros(1, dtype=cp.uint32)
            samples = []
            for trial in range(4):
                count.fill(0)
                cp.cuda.Stream.null.synchronize()
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record()
                k((n_blocks,), (THREADS_PER_BLOCK,),
                  (cur_x, cur_y, np.int32(16), np.uint64(trial*16),
                   params, hits, count, np.uint32(len(hits))))
                end.record()
                end.synchronize()
                if trial:
                    samples.append(n_threads*m_val*16 / (cp.cuda.get_elapsed_time(begin, end)/1000))
            return (statistics.median(samples), k), None
        except Exception as exc:
            return None, "编译或运行失败: {}".format(exc)
    best_rate, best_M, best_kernel = 0.0, None, None
    for M_try in M_CANDIDATES:
        result, err = _bench_m(M_try)
        if err:
            print("  M={}: {}".format(M_try, err))
            continue
        rate, k_obj = result
        print("  M={}: {:.1f}M/秒".format(M_try, rate / 1e6))
        if rate > best_rate:
            best_rate, best_M, best_kernel = rate, M_try, k_obj
    if best_kernel is None:
        sys.stderr.write(
            "\n✗ CUDA 内核所有 M 值都编译失败, GPU 不兼容.\n"
            "  可能原因:\n"
            "    1) NVRTC 找不到 CUDA 头文件 (重装 nvidia-cuda-nvrtc-cu12 和 nvidia-cuda-runtime-cu12)\n"
            "    2) GPU 驱动版本太老, 升级 NVIDIA 驱动到 535+\n"
            "    3) GPU 计算能力 < 7.0 (RTX 20 系以下不支持)\n")
        sys.exit(1)
    POINTS_PER_THREAD = best_M
    kernel = best_kernel
    BATCH = n_threads * STEPS_PER_LAUNCH * POINTS_PER_THREAD
    print("→ 选用 M={} (摊销 {:.0f} mul/点, 当前模式实测最快)".format(
        POINTS_PER_THREAD, 256 / POINTS_PER_THREAD + 3))
    print()
    print("GPU 自检: 单线程执行 1 步 (M={}个点)...".format(POINTS_PER_THREAD))
    self_t0 = time.time()
    try:
        k_tests = list(range(2, POINTS_PER_THREAD + 2))
        sx_test = np.zeros(POINTS_PER_THREAD * 4, dtype=np.uint64)
        sy_test = np.zeros(POINTS_PER_THREAD * 4, dtype=np.uint64)
        for p, k_test in enumerate(k_tests):
            pub_test = coincurve.PublicKey.from_valid_secret(k_test.to_bytes(32, "big")).format(compressed=False)
            x_test = int.from_bytes(pub_test[1:33], "big")
            y_test = int.from_bytes(pub_test[33:65], "big")
            for j in range(4):
                sx_test[p * 4 + j] = (x_test >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
                sy_test[p * 4 + j] = (y_test >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
        cur_x_test = cp.asarray(sx_test)
        cur_y_test = cp.asarray(sy_test)
        pp_test = np.zeros(1, dtype=PATTERN_DTYPE)
        pp_test["mode"] = 0
        pp_test["prefix_count"] = 1
        pp_test["prefix_len"][0, 0] = 1
        pp_test["prefixes"][0, 0, 0] = ord("T")
        pp_test_dev = cp.asarray(pp_test)
        m_test = cp.zeros(POINTS_PER_THREAD * 2, dtype=MATCH_DTYPE)
        c_test = cp.zeros(1, dtype=cp.uint32)
        check_kernel = load_kernel(arch=arch, points_per_thread=POINTS_PER_THREAD)
        check_kernel(
            (1,), (1,),
            (cur_x_test, cur_y_test, np.int32(1), np.uint64(0),
             pp_test_dev, m_test, c_test, np.uint32(POINTS_PER_THREAD * 2))
        )
        cp.cuda.Stream.null.synchronize()
        n = int(c_test.get()[0])
        if n != POINTS_PER_THREAD:
            raise RuntimeError("GPU 内核未执行 (期望 {} 命中, 实际 {}). GPU 可能没有响应.".format(POINTS_PER_THREAD, n))
        matches_back = m_test[:n].get()
        for m in matches_back:
            packed = int(m["thread_id"])
            p_idx = packed % POINTS_PER_THREAD
            addr_gpu = bytes(m["address"]).decode("ascii")
            addr_cpu = cpu_priv_to_address(k_tests[p_idx])
            if addr_gpu != addr_cpu:
                raise RuntimeError("GPU 自检不匹配 (point={}): \n  GPU: {}\n  CPU: {}".format(p_idx, addr_gpu, addr_cpu))
        print("GPU 自检通过, 用时 {:.2f}s. GPU 确实在执行内核 ✓".format(time.time() - self_t0))
    except Exception as e:
        print()
        print("✗ GPU 自检失败: {}".format(e))
        print("  请检查: nvidia-smi 是否能看到 GPU; 显存是否充足; 驱动是否最新")
        sys.exit(1)
    print()
    pp = make_pattern_params(config)
    pp_dev = cp.asarray(pp)
    pts_per_stream = n_threads * POINTS_PER_THREAD
    total_pts = pts_per_stream * N_STREAMS
    print("生成 {} 个 GPU 起始点 ({} 套 × {} 线程 × {} 点, 用 {} 个 CPU 核并行)...".format(
        total_pts, N_STREAMS, n_threads, POINTS_PER_THREAD, cpu_count))
    t1 = time.time()
    n_chunks = cpu_count * 4
    chunk = (total_pts + n_chunks - 1) // n_chunks
    tasks = []
    remain = total_pts
    while remain > 0:
        c = min(chunk, remain)
        tasks.append(c)
        remain -= c
    ctx_pool = mp.get_context("spawn")
    try:
        with ctx_pool.Pool(processes=cpu_count) as pool:
            results = pool.map(gen_startpoints_batch, tasks)
    except Exception as e:
        sys.stderr.write(
            "✗ 起点生成失败: {}\n"
            "  可能原因: 内存不足 / coincurve 安装损坏 / 子进程被杀\n".format(e))
        sys.exit(1)
    base_keys_all = []
    sx_all = np.zeros(total_pts * 4, dtype=np.uint64)
    sy_all = np.zeros(total_pts * 4, dtype=np.uint64)
    idx = 0
    for xb, yb, privs in results:
        n_in_chunk = len(privs)
        for i in range(n_in_chunk):
            x_be = xb[i*32:(i+1)*32]
            y_be = yb[i*32:(i+1)*32]
            x = int.from_bytes(x_be, "big")
            y = int.from_bytes(y_be, "big")
            for j in range(4):
                sx_all[idx*4 + j] = (x >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
                sy_all[idx*4 + j] = (y >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
            base_keys_all.append(privs[i])
            idx += 1
    stream_state = []
    for s in range(N_STREAMS):
        beg = s * pts_per_stream
        end = (s + 1) * pts_per_stream
        sx_slice = sx_all[beg*4:end*4]
        sy_slice = sy_all[beg*4:end*4]
        base_keys = base_keys_all[beg:end]
        cur_x_dev = cp.asarray(sx_slice)
        cur_y_dev = cp.asarray(sy_slice)
        stream_state.append({
            "cur_x": cur_x_dev,
            "cur_y": cur_y_dev,
            "base_keys": base_keys,
            "step_offset": 0,
        })
    print("起点准备完成, 用时 {:.1f}s ({:.0f} pts/s)".format(
        time.time() - t1, total_pts / max(time.time() - t1, 0.001)))
    print()
    print("自适应调参: 测量单步耗时...")
    pp_warmup_dev = cp.asarray(pp)
    m_warmup = cp.zeros(MAX_MATCHES, dtype=MATCH_DTYPE)
    c_warmup = cp.zeros(1, dtype=cp.uint32)
    warmup_state = {"cur_x": stream_state[0]["cur_x"].copy(),
                    "cur_y": stream_state[0]["cur_y"].copy()}
    cp.cuda.Stream.null.synchronize()
    t_warm = time.time()
    WARMUP_STEPS = 64
    kernel(
        (n_blocks,), (THREADS_PER_BLOCK,),
        (warmup_state["cur_x"], warmup_state["cur_y"],
         np.int32(WARMUP_STEPS), np.uint64(0),
         pp_warmup_dev, m_warmup, c_warmup, np.uint32(MAX_MATCHES))
    )
    cp.cuda.Stream.null.synchronize()
    elapsed_warm = time.time() - t_warm
    warmup_hits = int(c_warmup.get()[0])
    target_sec = 1.5
    per_step = elapsed_warm / WARMUP_STEPS
    STEPS_PER_LAUNCH = max(1, int(target_sec / max(per_step, 1e-6)))
    STEPS_PER_LAUNCH = max(1, min(STEPS_PER_LAUNCH, 2048))
    if warmup_hits > 0:
        # Keep the GPU->CPU candidate batch bounded.  The wide screen is
        # intentionally recall-first, so shorten launches rather than drop
        # candidates when a GPU produces a dense structural region.
        estimated_steps = int((MAX_MATCHES * 0.5) / max(warmup_hits / float(WARMUP_STEPS), 1e-9))
        STEPS_PER_LAUNCH = max(1, min(STEPS_PER_LAUNCH, estimated_steps))
    BATCH = n_threads * STEPS_PER_LAUNCH * POINTS_PER_THREAD
    print("  warmup: {} 步用 {:.2f}s ({:.2f}ms/步)".format(WARMUP_STEPS, elapsed_warm, per_step * 1000))
    if warmup_hits > MAX_MATCHES:
        print("  警告: 当前条件命中率过高, warmup 已超过 GPU 命中缓冲上限 {}".format(MAX_MATCHES))
    print("  自适应 STEPS_PER_LAUNCH = {} (预期 ~{:.1f}s/launch, ~{:.1f}M 地址/launch)".format(
        STEPS_PER_LAUNCH, STEPS_PER_LAUNCH * per_step, BATCH / 1e6))
    print()
    return run_pipeline(kernel, pp_dev, config, output_path, duration_minutes,
                        stream_state, n_blocks, THREADS_PER_BLOCK, POINTS_PER_THREAD,
                        STEPS_PER_LAUNCH, MAX_MATCHES)


def run_pipeline(kernel, params, config, output_path, duration_minutes, states,
                 n_blocks, block_size, points, launch_steps, capacity):
    """Double-stream generation with bounded asynchronous CPU classification.

    Each launch checkpoints its curve state. Overflow replays that exact range
    in smaller launches; no candidate is discarded to enforce a rate limit.
    """
    import sqlite3
    streams = [cp.cuda.Stream(non_blocking=True) for _ in states]
    n_threads = n_blocks * block_size
    # The device counter is uint32, including for very broad user filters.
    launch_steps = min(launch_steps, (2**32-1)//(n_threads*points))
    if launch_steps < 1:
        raise ValueError("GPU 网格超过命中计数器容量")
    buffers = []

    def allocate(size):
        pinned = cp.cuda.alloc_pinned_memory(MATCH_DTYPE.itemsize * size)
        count_host = cp.cuda.alloc_pinned_memory(4)
        return {"device": cp.empty(size, dtype=MATCH_DTYPE),
                "count": cp.zeros(1, dtype=cp.uint32), "pinned": pinned,
                "host": np.frombuffer(pinned, dtype=MATCH_DTYPE),
                "count_host": count_host,
                "count_view": np.frombuffer(count_host, dtype=np.uint32), "capacity": size}

    for state in states:
        state["backup_x"] = cp.empty_like(state["cur_x"])
        state["backup_y"] = cp.empty_like(state["cur_y"])
        buffers.append(allocate(capacity))
    cp.cuda.Stream.null.synchronize()
    workers = max(1, min(8, mp.cpu_count()-1))
    pending, order = deque(), deque()
    unsubmitted = []
    batches = iter(())
    metrics = {"addresses": 0, "candidates": 0, "classified": 0, "saved": 0,
               "overflow_replays": 0, "dropped_candidates": 0, "backpressure_seconds": 0.0}
    started = time.monotonic()
    deadline = started + duration_minutes*60 if duration_minutes else float("inf")
    interrupted = threading.Event()
    display_stop = threading.Event()
    old_handlers = {}
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", signal.SIGINT)):
        if sig not in old_handlers:
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: interrupted.set())
    initial_steps = launch_steps
    last_warning = [0.0]
    hardware_totals, hardware_counts = [0.0, 0.0], [0, 0]

    def warn(message):
        now = time.monotonic()
        if now-last_warning[0] >= 10:
            print("\n[警告] " + message)
            last_warning[0] = now

    def launch(idx, steps):
        state, buf, stream = states[idx], buffers[idx], streams[idx]
        state["batch_start"], state["batch_steps"] = state["step_offset"], steps
        with stream:
            cp.copyto(state["backup_x"], state["cur_x"])
            cp.copyto(state["backup_y"], state["cur_y"])
            buf["count"].fill(0)
            kernel((n_blocks,), (block_size,),
                   (state["cur_x"], state["cur_y"], np.int32(steps),
                    np.uint64(state["batch_start"]), params, buf["device"],
                    buf["count"], np.uint32(buf["capacity"])), stream=stream)
            cp.cuda.runtime.memcpyAsync(buf["count_host"].ptr, buf["count"].data.ptr, 4,
                                       cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        state["step_offset"] += steps

    def collect(idx):
        nonlocal launch_steps
        state, stream, buf = states[idx], streams[idx], buffers[idx]
        stream.synchronize()
        count, steps = int(buf["count_view"][0]), state["batch_steps"]
        if count > buf["capacity"]:
            metrics["overflow_replays"] += 1
            warn("GPU 缓冲溢出：重放原区间并缩短批次，不丢弃候选。")
            with stream:
                cp.copyto(state["cur_x"], state["backup_x"])
                cp.copyto(state["cur_y"], state["backup_y"])
            stream.synchronize()
            state["step_offset"] = state["batch_start"]
            if steps == 1:
                # One candidate per point, bounded by n_threads*points.
                buffers[idx] = allocate(count)
                cp.cuda.Stream.null.synchronize()
                launch(idx, 1)
                yield from collect(idx)
            else:
                launch_steps = max(1, min(launch_steps, steps//2))
                remaining = steps
                while remaining:
                    part = min(remaining, launch_steps)
                    launch(idx, part)
                    yield from collect(idx)
                    remaining -= part
            return
        if count:
            cp.cuda.runtime.memcpyAsync(buf["pinned"].ptr, buf["device"].data.ptr,
                                       count*MATCH_DTYPE.itemsize,
                                       cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
            stream.synchronize()
        # Copy before reusing pinned memory on the next stream launch.
        result = buf["host"][:count].copy()
        if count > buf["capacity"]//2:
            launch_steps = max(1, min(launch_steps, int(steps*buf["capacity"]/(2*count))))
        yield result, steps

    def status():
        while not display_stop.wait(1):
            elapsed = max(time.monotonic()-started, 0.001)
            gpu, cpu = _gpu_stats(), _cpu_percent()
            for column, value in enumerate((gpu[0] if gpu else None, cpu)):
                if value is not None:
                    hardware_totals[column] += value
                    hardware_counts[column] += 1
            print("\rGPU {}/秒 | 候选 {}/秒 | 已分类 {} | 已保存 {} | 待处理 {}{}{}   ".format(
                fmt_num(metrics["addresses"]/elapsed), fmt_num(metrics["candidates"]/elapsed),
                metrics["classified"], metrics["saved"], len(pending),
                " | GPU {:.0f}%".format(gpu[0]) if gpu else "",
                " | CPU {:.0f}%".format(cpu) if cpu is not None else ""), end="", flush=True)
            if metrics["candidates"]/elapsed > 10000:
                warn("候选持续超过每秒一万个；保留规则，必要时等待 CPU 清空队列。")

    def candidates_from(arr, idx):
        keys = states[idx]["base_keys"]
        result = []
        for item in arr:
            chain, step = int(item["thread_id"]), int(item["step"])
            if chain >= len(keys):
                raise RuntimeError("GPU 命中索引越界")
            private_key = (keys[chain]+step) % SECP256K1_N
            result.append((private_key.to_bytes(32, "big").hex(),
                           bytes(item["address"]).decode("ascii")))
        return result

    # Disk-backed dedup avoids unbounded RAM growth in an unlimited search.
    index = sqlite3.connect(output_path + ".index.sqlite3")
    index.execute("CREATE TABLE IF NOT EXISTS addresses (address TEXT PRIMARY KEY)")
    output = open(output_path, "a", encoding="utf-8")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=init_classifier)
    display = threading.Thread(target=status, daemon=True)
    display.start()
    error = None

    def finish_one():
        future, raw = pending[0]
        records = future.result()  # propagate verification/worker errors
        saved = 0
        try:
            for record in records:
                if index.execute("INSERT OR IGNORE INTO addresses VALUES (?)", (record["address"],)).rowcount:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    saved += 1
            if saved:
                output.flush()
                os.fsync(output.fileno())
            index.commit()
        except Exception:
            index.rollback()
            raise
        metrics["saved"] += saved
        metrics["classified"] += len(raw)
        pending.popleft()

    def enqueue(arr, idx, steps):
        nonlocal unsubmitted
        metrics["addresses"] += n_threads*points*steps
        metrics["candidates"] += len(arr)
        raw = candidates_from(arr, idx)
        unsubmitted = raw
        # Bounded batches and queue: normal classification overlaps both GPU streams.
        for start in range(0, len(raw), 128):
            while len(pending) >= workers*2:
                if not pending[0][0].done():
                    warn("CPU 分类队列已满，暂缓提交以保留全部候选。")
                before = time.monotonic()
                finish_one()
                metrics["backpressure_seconds"] += time.monotonic()-before
            batch = raw[start:start+128]
            pending.append((pool.submit(classify_batch, batch, config), batch))
            unsubmitted = raw[start+128:]

    print("开始搜索；Ctrl+C 或到时后停止提交，排空在途候选再退出。")
    try:
        for idx in range(len(streams)):
            launch(idx, launch_steps)
            order.append(idx)
        while order:
            idx = order.popleft()
            # Normal case yields one batch. Replay batches are submitted as
            # they finish, bounding memory even under a dense filter.
            batches = collect(idx)
            for arr, steps in batches:
                enqueue(arr, idx, steps)
            if not interrupted.is_set() and time.monotonic() < deadline:
                launch(idx, launch_steps)
                order.append(idx)
            while pending and pending[0][0].done():
                finish_one()
        while pending:
            finish_one()
    except Exception as exc:
        error = exc
        # Retain unclassified private-key/address pairs for recovery, never print keys.
        recovery = output_path + ".pending.jsonl"
        with open(recovery, "a", encoding="utf-8") as f:
            for _, raw in pending:
                for key, address in raw:
                    f.write(json.dumps({"private_key": key, "address": address, "config": config})+"\n")
            for key, address in unsubmitted:
                f.write(json.dumps({"private_key": key, "address": address, "config": config})+"\n")
            for arr, _ in batches:
                for key, address in candidates_from(arr, idx):
                    f.write(json.dumps({"private_key": key, "address": address, "config": config})+"\n")
            for idx in order:
                for arr, _ in collect(idx):
                    for key, address in candidates_from(arr, idx):
                        f.write(json.dumps({"private_key": key, "address": address, "config": config})+"\n")
        print("\n处理失败，待验证候选已保存到 {}".format(recovery))
    finally:
        pool.shutdown(wait=True)
        display_stop.set()
        display.join()
        for stream in streams:
            stream.synchronize()
        output.close()
        index.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        elapsed = time.monotonic()-started
        for column, label in ((0, "gpu_utilization_percent"), (1, "cpu_system_percent")):
            metrics[label] = hardware_totals[column]/hardware_counts[column] if hardware_counts[column] else None
        metrics.update(elapsed_seconds=elapsed, addresses_per_second=metrics["addresses"]/max(elapsed, .001),
                       candidates_per_second=metrics["candidates"]/max(elapsed, .001),
                       points_per_thread=points, initial_steps=initial_steps, final_steps=launch_steps,
                       complete=error is None, config=config)
        with open(output_path + ".stats.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("\n生成 {}，候选 {}，已分类 {}，保存 {}，溢出重放 {} 次。".format(
            metrics["addresses"], metrics["candidates"], metrics["classified"],
            metrics["saved"], metrics["overflow_replays"]))
    if error:
        raise error
    return metrics


def prompt_mode_1():
    while True:
        prefix = input("请输入前缀列表 (逗号分隔, 留空表示无, 必须以 T 开头): ").strip()
        suffix = input("请输入后缀列表 (逗号分隔, 留空表示无): ").strip()
        if not prefix and not suffix:
            print("错误: 前缀和后缀至少填写一项。\n")
            continue
        combine = input("前缀/后缀组合 [AND/OR，默认 AND]: ").strip().upper() or "AND"
        if combine not in ("AND", "OR"):
            print("请输入 AND 或 OR。")
            continue
        errs = validate_pattern(prefix, suffix, combine == "OR")
        if errs:
            print("输入有误: " + "; ".join(errs))
            continue
        return {"mode": "exact", "prefix": prefix, "suffix": suffix,
                "combine_or": combine == "OR"}

def prompt_mode_2():
    while True:
        s = input("靓号长度范围 [默认 8-34；单值 8 表示 8-8]: ").strip() or "8-34"
        try:
            if "-" in s:
                a, b = [int(x.strip()) for x in s.split("-", 1)]
            else:
                a = b = int(s)
        except ValueError:
            print("错误: 请输入整数或 min-max。\n"); continue
        if a < 2 or b < a or b > 34:
            print("错误: 长度必须满足 2 <= 最小值 <= 最大值 <= 34。\n"); continue
        rules = {}
        for key, label in (("same", "连续相同字符"), ("folded", "同字母忽略大小写"),
                           ("groups", "连续分组（含顺序分组）"),
                           ("straight", "数字顺子"), ("periodic", "周期重复"),
                           ("palindrome", "回文/对称")):
            ans = input("开启{}? [Y/n]: ".format(label)).strip().lower()
            rules[key] = ans not in ("n", "no", "否", "0")
        if not any(rules.values()):
            print("至少开启一种规则。")
            continue
        if a < 8:
            print("提示: 小于 8 位可能产生大量候选；过载将报警并减速，不会自动提高长度。")
        return {"mode": "wide", "min_len": a, "max_len": b, "rules": rules}
def main():
    print("=" * 70)
    print("  TRON 靓号地址生成器 (CUDA + CPU 全速版)")
    print("=" * 70)
    print()
    print("  模式 1: 前缀/后缀模式 — 支持多个目标和 AND/OR")
    print("  模式 2: 靓号模式     — 全地址宽筛 + CPU 分类")
    print()
    while True:
        mode = input("请选择模式 [1/2]: ").strip()
        if mode in ("1", "2"): break
        print("无效输入。\n")
    if mode == "1":
        config = prompt_mode_1()
    else:
        config = prompt_mode_2()
    while True:
        raw_minutes = input("运行时间（分钟，0=一直运行）: ").strip() or "0"
        try:
            duration_minutes = float(raw_minutes)
            if math.isfinite(duration_minutes) and duration_minutes >= 0: break
        except ValueError:
            pass
        print("请输入不小于 0 的数字。")
    out_dir = os.path.join(APP_BASE_DIR, "命中地址")

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        sys.stderr.write(
            "✗ 无法创建输出目录 {}: {}\n"
            "  请检查脚本所在目录是否有写权限\n".format(out_dir, e))
        sys.exit(1)
    output_path = os.path.join(
        out_dir, "matches_{}.jsonl".format(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    run_search(config, output_path, duration_minutes)
if __name__ == "__main__":
    mp.freeze_support()
    main()
