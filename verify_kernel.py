import os
import sys
import secrets
import time
import warnings
warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
import numpy as np
import cupy as cp
import coincurve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tron_vanity_gpu import (
    PATTERN_DTYPE, MATCH_DTYPE, load_kernel,
    cpu_priv_to_address, SECP256K1_N, make_pattern_params, KERNEL_PATH
)
def verify_with_m(M):
    props = cp.cuda.runtime.getDeviceProperties(0)
    cc_major = props["major"]
    cc_minor = props["minor"]
    print("\n=== 测试 M={} ===".format(M))
    print("加载并编译 CUDA 内核 (arch=sm_{}{})...".format(cc_major, cc_minor))
    t0 = time.time()
    kernel = load_kernel(arch="sm_{}{}".format(cc_major, cc_minor),
                         points_per_thread=M)
    print("编译完成: {:.1f}s".format(time.time() - t0))
    THREADS = 4
    STEPS = 64
    n_chains = THREADS * M
    base_keys = []
    sx = np.zeros(n_chains * 4, dtype=np.uint64)
    sy = np.zeros(n_chains * 4, dtype=np.uint64)
    for i in range(n_chains):
        k = secrets.randbelow(SECP256K1_N - 1) + 1
        base_keys.append(k)
        pub = coincurve.PublicKey.from_valid_secret(k.to_bytes(32, "big")).format(compressed=False)
        x = int.from_bytes(pub[1:33], "big")
        y = int.from_bytes(pub[33:65], "big")
        for j in range(4):
            sx[i*4 + j] = (x >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
            sy[i*4 + j] = (y >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
    start_x = cp.asarray(sx)
    start_y = cp.asarray(sy)
    pp = np.zeros(1, dtype=PATTERN_DTYPE)
    pp["mode"] = 0
    pp["prefix_count"] = 1
    pp["prefix_len"][0, 0] = 1
    pp["prefixes"][0, 0, 0] = ord("T")
    pp_dev = cp.asarray(pp)
    MAX_MATCHES = THREADS * STEPS * M * 4 + 10
    matches_dev = cp.zeros(MAX_MATCHES, dtype=MATCH_DTYPE)
    count_dev = cp.zeros(1, dtype=cp.uint32)
    print("启动内核 (THREADS={}, M={}, STEPS={} x 2 次)...".format(THREADS, M, STEPS))
    for launch_idx in range(2):
        offset = launch_idx * STEPS
        kernel(
            (1,), (THREADS,),
            (start_x, start_y,
             np.int32(STEPS),
             np.uint64(offset),
             pp_dev,
             matches_dev, count_dev, np.uint32(MAX_MATCHES))
        )
    cp.cuda.Stream.null.synchronize()
    n_hits = int(count_dev.get()[0])
    expected = THREADS * STEPS * M * 2
    print("内核报告命中: {} (期望 {})".format(n_hits, expected))
    if n_hits != expected:
        raise AssertionError("命中数不匹配")
    n_check = min(n_hits, MAX_MATCHES)
    matches = matches_dev[:n_check].get()
    mismatches = 0
    assert len({(int(m["thread_id"]), int(m["step"])) for m in matches}) == expected
    for m in matches:
        packed = int(m["thread_id"])
        real_tid, p_idx = divmod(packed, M)
        step = int(m["step"])
        gpu_addr = bytes(m["address"]).decode("ascii", errors="replace")
        if real_tid >= THREADS or p_idx >= M:
            print("✗ 越界 tid={} p={}".format(real_tid, p_idx))
            mismatches += 1
            continue
        chain_idx = real_tid * M + p_idx
        priv_int = (base_keys[chain_idx] + step) % SECP256K1_N
        cpu_addr = cpu_priv_to_address(priv_int)
        if gpu_addr != cpu_addr:
            mismatches += 1
            if mismatches <= 5:
                print("✗ 不匹配 tid={} p={} step={}".format(real_tid, p_idx, step))
                print("    GPU: {}".format(gpu_addr))
                print("    CPU: {}".format(cpu_addr))
    if mismatches == 0:
        print("✓ M={}: 全部 {} 个 GPU 地址 (2 次启动, 状态延续) 与 CPU 完全一致".format(M, n_check))
        verify_filtered_search(M, THREADS, STEPS*2, sx, sy, matches)
        return True
    else:
        print("✗ M={}: 发现 {} 个不匹配".format(M, mismatches))
        return False
def verify_filtered_search(m, threads, steps, sx, sy, all_matches):
    from cpu_worker import classify_vanity
    known = {(int(r["thread_id"]), int(r["step"])): bytes(r["address"]).decode("ascii")
             for r in all_matches}
    targets = list(known.values())
    first, last = targets[0], targets[-1]
    cases = [dict(mode="exact", suffix=first[-6:]+","+last[-5:]),
             dict(mode="exact", prefix=first[:5]+","+last[:6]),
             dict(mode="exact", prefix=first[:5], suffix=first[-6:]),
             dict(mode="exact", prefix=first[:5], suffix=last[-6:], combine_or=True),
             dict(mode="exact", suffix=first[-6:], combine_or=True),
             dict(mode="exact", prefix=first[:5], combine_or=True),
             dict(mode="wide", min_len=2, max_len=8),
             dict(mode="wide", rule_minima={"same": 3, "folded": 4, "groups": 4, "digits": 2}),
             dict(mode="wide", rule_minima={"cyclic": 2, "digit_periodic": 4, "mixed_periodic": 4,
                                          "digit_palindrome": 3, "mixed_palindrome": 3})]
    kernels = {mode: load_kernel(points_per_thread=m, mode=mode) for mode in ("exact", "wide")}
    for cfg in cases:
        x, y = cp.asarray(sx), cp.asarray(sy)
        params = cp.asarray(make_pattern_params(cfg))
        out = cp.empty(len(known), dtype=MATCH_DTYPE)
        count = cp.zeros(1, dtype=cp.uint32)
        kernels[cfg["mode"]]((1,), (threads,),
                            (x, y, np.int32(steps), np.uint64(0), params, out, count, np.uint32(len(out))))
        n = int(count.get()[0])
        assert n <= len(out)
        actual = {}
        for r in out[:n].get():
            key = (int(r["thread_id"]), int(r["step"]))
            address = bytes(r["address"]).decode("ascii")
            assert known[key] == address
            assert key not in actual
            actual[key] = address
        expected = {key for key, address in known.items() if classify_vanity(address, cfg)}
        assert expected <= actual.keys(), (m, cfg, "missed candidates")
        if cfg["mode"] == "exact":
            assert expected == actual.keys(), (m, cfg, "false exact matches")
    print("✓ M={} 单独编译的前后缀/全地址路径与 CPU 结果一致".format(m))
def verify_screen():
    from test_vanity import fixtures
    from cpu_worker import classify_vanity, RULE_BITS
    addresses = list(fixtures())
    module = cp.RawModule(code=open(KERNEL_PATH, encoding="utf-8").read(),
                          options=("-std=c++14", "-DPOINTS_PER_THREAD=8"))
    kernel = module.get_function("screen_addresses")
    encoded = cp.asarray(np.frombuffer("".join(addresses).encode(), dtype=np.uint8))
    output = cp.zeros(len(addresses), dtype=cp.uint8)
    comparisons = 0
    for lo, hi in ((2, 5), (5, 8), (8, 8), (8, 12), (8, 34), (9, 34), (12, 34), (34, 34)):
        for only in [None]+list(RULE_BITS):
            cfg = dict(mode="wide", min_len=lo, max_len=hi)
            if only:
                cfg["rules"] = {key: key == only for key in RULE_BITS}
            params = cp.asarray(make_pattern_params(cfg))
            kernel(((len(addresses)+127)//128,), (128,),
                   (encoded, np.int32(len(addresses)), params, output))
            hits = output.get()
            for address, hit in zip(addresses, hits):
                assert not classify_vanity(address, cfg) or hit, (cfg, address)
                comparisons += 1
    print("✓ 实际 CUDA 粗筛/CPU 分类对照: {} 项无漏筛".format(comparisons))
    from test_independent_rules import independent_cases
    addresses, configs = independent_cases()
    encoded = cp.asarray(np.frombuffer("".join(addresses).encode(), dtype=np.uint8))
    output = cp.zeros(len(addresses), dtype=cp.uint8)
    comparisons = 0
    for cfg in configs:
        params = cp.asarray(make_pattern_params(cfg))
        kernel(((len(addresses)+127)//128,), (128,),
               (encoded, np.int32(len(addresses)), params, output))
        for address, hit in zip(addresses, output.get()):
            assert not classify_vanity(address, cfg) or hit, (cfg, address)
            comparisons += 1
    print("✓ 独立长度 CUDA 粗筛/CPU 分类对照: {} 项无漏筛".format(comparisons))


def verify_host_screen():
    """Execute the unmodified CUDA predicate as native C++ (not a GPU test)."""
    import ctypes
    import subprocess
    import tempfile
    from pathlib import Path
    from test_vanity import fixtures
    from cpu_worker import classify_vanity, RULE_BITS, rule_mask
    source = Path(KERNEL_PATH).read_text(encoding="utf-8")
    source = 'typedef unsigned int u32; typedef unsigned long long u64;\n' + source[source.index("struct PatternParams"):source.index("// Test entry point")]
    source = source.replace("__device__", "").replace("__forceinline__", "inline")
    source += '\nextern "C" __declspec(dllexport) int host_screen(const char *a, int lo, int hi, int mask) { return wide_candidate(a,lo,hi,mask); }\n'
    source += '\nextern "C" __declspec(dllexport) int host_independent(const char *a, const int *lo) { return independent_candidate(a,lo); }\n'
    source += '\nextern "C" __declspec(dllexport) int host_edge(const char *a, int n, int rule, const char *targets, int count) { return edge_candidate(a,n,rule,(const char (*)[40])targets,count); }\n'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"screen.cpp"
        path.write_text(source, encoding="utf-8")
        library = Path(directory)/"screen.dll"
        compiled = subprocess.run([sys.executable, "-m", "ziglang", "c++", "-shared", "-O2",
                                   "-nostdlib++", "-std=c++14", str(path), "-o", str(library)],
                                  capture_output=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr.decode(errors="replace"))
        lib = ctypes.CDLL(str(library))
        fn = lib.host_screen
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int
        comparisons = 0
        addresses = list(fixtures())
        for lo, hi in ((2,5),(5,8),(8,8),(8,12),(8,34),(9,34),(12,34),(34,34)):
            for only in [None]+list(RULE_BITS):
                config = dict(mode="wide", min_len=lo, max_len=hi)
                if only:
                    config["rules"] = {key: key == only for key in RULE_BITS}
                for address in addresses:
                    hit = fn(address.encode(), lo, hi, rule_mask(config))
                    assert not classify_vanity(address, config) or hit, (config, address)
                    comparisons += 1
        print("Host execution of actual coarse predicate: {} comparisons, no false negatives (not GPU execution).".format(comparisons))
        from test_independent_rules import independent_cases
        from cpu_worker import RULE_SPECS
        addresses, configs = independent_cases()
        fn = lib.host_independent
        fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        comparisons = 0
        for config in configs:
            thresholds = (ctypes.c_int*10)(*[config['rule_minima'].get(s[0], 0) for s in RULE_SPECS])
            for address in addresses:
                hit = fn(address.encode(), thresholds)
                assert not classify_vanity(address, config) or hit, (config, address)
                comparisons += 1
        print("Independent thresholds: {} comparisons, no false negatives (host execution).".format(comparisons))
        from cpu_worker import edge_matches, EDGE_RULES
        edge = lib.host_edge
        edge.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        edge.restype = ctypes.c_int
        checked = 0
        for address in addresses:
            for length in (1, 2, 8, 9, 16, 33):
                for text in (address[1:1+length], address[-length:]):
                    for rule, code in EDGE_RULES.items():
                        spec = dict(rule=rule, targets=text)
                        targets = text.encode().ljust(40,b'\0')+b'\0'*280
                        assert bool(edge(text.encode(),length,code,targets,1)) == edge_matches(text,spec)
                        checked += 1
        print("Edge CPU/CUDA predicate agreement: {} comparisons (host execution).".format(checked))
        # Windows holds loaded DLLs open until FreeLibrary.
        import _ctypes
        _ctypes.FreeLibrary(lib._handle)
    return 0


def compile_offline():
    """NVRTC -> PTX; compilation only, does not imply GPU execution passed."""
    import ctypes as c
    from pathlib import Path
    import site
    locations = [path for root in site.getsitepackages()
                 for path in Path(root).glob("nvidia/cuda_nvrtc/bin/nvrtc64_*.dll")
                 if ".alt." not in path.name]
    if not locations:
        raise RuntimeError("未找到 Windows NVRTC DLL；安装 nvidia-cuda-nvrtc-cu12")
    with os.add_dll_directory(str(locations[0].parent)):
        lib = c.CDLL(str(locations[0]))
        lib.nvrtcCreateProgram.argtypes = [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p,
                                          c.c_int, c.c_void_p, c.c_void_p]
        lib.nvrtcCompileProgram.argtypes = [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)]
        lib.nvrtcGetProgramLogSize.argtypes = [c.c_void_p, c.POINTER(c.c_size_t)]
        lib.nvrtcGetProgramLog.argtypes = [c.c_void_p, c.c_void_p]
        lib.nvrtcGetPTXSize.argtypes = [c.c_void_p, c.POINTER(c.c_size_t)]
        lib.nvrtcDestroyProgram.argtypes = [c.POINTER(c.c_void_p)]
        source = Path(KERNEL_PATH).read_bytes()
        for mode in (0, 1, -1):
            for points in (8, 16, 24, 32):
                program = c.c_void_p()
                assert lib.nvrtcCreateProgram(c.byref(program), source, b"kernels.cu", 0, None, None) == 0
                options = [b"--std=c++14", b"--gpu-architecture=compute_75", b"--use_fast_math",
                           ("-DPOINTS_PER_THREAD="+str(points)).encode(), ("-DSEARCH_MODE="+str(mode)).encode()]
                result = lib.nvrtcCompileProgram(program, len(options), (c.c_char_p*len(options))(*options))
                size = c.c_size_t()
                lib.nvrtcGetProgramLogSize(program, c.byref(size))
                log = c.create_string_buffer(size.value)
                lib.nvrtcGetProgramLog(program, log)
                if result:
                    raise RuntimeError(log.value.decode())
                assert lib.nvrtcGetPTXSize(program, c.byref(size)) == 0
                lib.nvrtcDestroyProgram(c.byref(program))
                print("NVRTC OK: mode={} M={} PTX={} bytes (未执行 GPU)".format(mode, points, size.value), flush=True)
    return 0


def main():
    if "--host-screen" in sys.argv:
        return verify_host_screen()
    if "--compile-only" in sys.argv:
        return compile_offline()
    try:
        cp.cuda.Device(0).use()
        n_dev = cp.cuda.runtime.getDeviceCount()
        if n_dev == 0:
            print("✗ 没检测到 NVIDIA GPU. 请确认驱动已安装, nvidia-smi 能看到设备.")
            return 1
    except Exception as e:
        print("✗ CUDA 初始化失败: {}".format(e))
        print("  请确认 NVIDIA 驱动已安装, GPU 可用 (跑 nvidia-smi 测试).")
        return 1
    verify_screen()
    all_ok = True
    passed = []
    for M in (8, 16, 24, 32):
        try:
            ok = verify_with_m(M)
        except AssertionError:
            raise
        except Exception as exc:
            if M in (24, 32):
                print("M={} 不可用，保留 8/16 回退: {}".format(M, exc))
                continue
            raise
        if not ok:
            all_ok = False
        else:
            passed.append(M)
    print()
    if all_ok:
        print("✓ M={} 配置通过验证".format(passed))
        print("  Montgomery 批量求逆 + GPU 持久化状态 + 全部加密原语 OK")
        return 0
    else:
        return 1
if __name__ == "__main__":
    sys.exit(main())
