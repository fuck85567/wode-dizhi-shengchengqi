"""GPU-hit verification and classification. No independent CPU brute force."""
import hashlib
import secrets
import signal
from datetime import datetime, timezone

import base58
import coincurve
from Crypto.Hash import keccak as keccak_lib

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
RULE_BITS = {"same": 1, "folded": 2, "groups": 4, "straight": 8,
             "periodic": 16, "palindrome": 32}
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def _priv_to_address(priv_bytes):
    pub = coincurve.PublicKey.from_valid_secret(priv_bytes).format(compressed=False)
    k = keccak_lib.new(digest_bits=256)
    k.update(pub[1:])
    payload = b"\x41" + k.digest()[-20:]
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode()


def split_targets(value):
    return list(dict.fromkeys(x.strip() for x in value.replace("，", ",").split(",") if x.strip()))


def rule_mask(config):
    rules = config.get("rules", {})
    return sum(bit for key, bit in RULE_BITS.items() if rules.get(key, True))


def validate_config(config):
    if config.get("mode") not in ("wide", "exact"):
        raise ValueError("模式必须为 wide 或 exact")
    if config["mode"] == "wide":
        lo, hi = config.get("min_len", 8), config.get("max_len", 34)
        if not isinstance(lo, int) or not isinstance(hi, int) or not 2 <= lo <= hi <= 34:
            raise ValueError("长度范围必须满足 2 <= 最小值 <= 最大值 <= 34（默认 8-34）")
        if not rule_mask(config):
            raise ValueError("至少开启一种规则")
    else:
        ps, ss = split_targets(config.get("prefix", "")), split_targets(config.get("suffix", ""))
        if not ps and not ss:
            raise ValueError("至少输入一个前缀或后缀")
        if max(len(ps), len(ss)) > 8:
            raise ValueError("前缀和后缀分别最多 8 个；不会截断多余目标")
        for value in ps + ss:
            if len(value) > 34 or any(c not in BASE58_ALPHABET for c in value):
                raise ValueError("目标必须为最多 34 位的 Base58 字符（不含 0/O/I/l）")
        if any(not p.startswith("T") for p in ps):
            raise ValueError("前缀必须以 T 开头")
        if ps and ss and not config.get("combine_or", False):
            # Overlap is valid if any prefix/suffix pair agrees in its overlap.
            if not any(len(p) + len(s) <= 34 or p[34-len(s):] == s[:len(p)+len(s)-34]
                       for p in ps for s in ss):
                raise ValueError("所有前后缀组合在重叠位置均冲突")
    return config


def _run_groups(text):
    groups = []
    for ch in text:
        if groups and groups[-1][0] == ch:
            groups[-1][1] += 1
        else:
            groups.append([ch, 1])
    return groups


def _group_direction(chars):
    if len(chars) < 2:
        return 0
    # Do not bridge digit/letter boundaries or wrap missing Base58 symbols.
    if not (all(c in "123456789" for c in chars) or all("a" <= c <= "z" for c in chars)):
        return 0
    delta = ord(chars[1]) - ord(chars[0])
    return delta if delta in (-1, 1) and all(ord(b)-ord(a) == delta for a, b in zip(chars, chars[1:])) else 0


def _summary(matches):
    if not matches:
        return None
    priority = {"连续相同字符": 0, "同字母忽略大小写": 1, "连续分组": 2,
                "数字顺子": 3, "周期重复": 4, "回文": 5, "前缀": 6, "后缀": 7}
    matches.sort(key=lambda m: (-m["length"], priority[m["type"]], m["start"]))
    # Keep independent locations and types, suppress shorter contained copies.
    kept = []
    for m in matches:
        if not any(m["type"] == k["type"] and k["start"] <= m["start"] and
                   m["start"] + m["length"] <= k["start"] + k["length"] for k in kept):
            kept.append(m)
    result = dict(kept[0])
    result["tags"] = sorted(set(result["tags"]) | {m["type"] for m in kept[1:]})
    result["matches"] = kept
    result["position_base"] = 0
    return result


def _match(address, start, length, kind, groups=None, tags=None):
    content = address[start:start+length]
    return {"type": kind, "start": start, "length": length, "content": content,
            "normalized": content.lower(), "groups": len(groups or []),
            "group_lengths": [g[1] for g in groups or []], "tags": tags or []}


def classify_vanity(address, config):
    """Longest match plus independent secondary matches, one record/address.

    Groups: adjacent runs of >=2, OR consecutive ascending/descending run
    characters (singletons allowed). Case folding never splits AAAaaa.
    Periods: 2..4, >=2 repeats, partial final repeat allowed. Palindromes: folded.
    """
    if len(address) != 34 or not address.startswith("T") or any(c not in BASE58_ALPHABET for c in address):
        return None
    if config.get("mode") == "exact":
        prefixes, suffixes = split_targets(config.get("prefix", "")), split_targets(config.get("suffix", ""))
        ps = [p for p in prefixes if address.startswith(p)]
        ss = [s for s in suffixes if address.endswith(s)]
        if config.get("combine_or", False):
            ok = bool(ps or ss)
        else:
            ok = (not prefixes or bool(ps)) and (not suffixes or bool(ss))
        if not ok:
            return None
        return _summary([_match(address, 0, len(p), "前缀") for p in ps] +
                        [_match(address, 34-len(s), len(s), "后缀") for s in ss])
    lo, hi = config.get("min_len", 8), config.get("max_len", 34)
    mask = rule_mask(config)
    norm = address.lower()
    matches = []
    # For a fixed start/type, every shorter hit is contained in its longest
    # hit and would be removed by _summary. Scan boundaries instead of all
    # 378 windows for the default 8..34 range; keep every independent start.
    folded_end = list(range(1, 35))
    raw_end = folded_end.copy()
    for i in range(32, -1, -1):
        if norm[i] == norm[i+1]:
            folded_end[i] = folded_end[i+1]
        if address[i] == address[i+1]:
            raw_end[i] = raw_end[i+1]

    if mask & 3:
        for start in range(35-lo):
            length = min(hi, raw_end[start]-start)
            if mask & 1 and length >= lo:
                matches.append(_match(address, start, length, "连续相同字符", [[norm[start], length]]))
            length = min(hi, folded_end[start]-start)
            if mask & 2 and length >= lo and norm[start].isalpha():
                original = address[start:start+length]
                tags = ["混合大小写"] if original != original.upper() and original != original.lower() else []
                matches.append(_match(address, start, length, "同字母忽略大小写", [[norm[start], length]], tags))

    if mask & 4:
        up, down = list(range(1, 35)), list(range(1, 35))
        for i in range(32, -1, -1):
            a, b = norm[i], norm[i+1]
            family = (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())
            delta = ord(b)-ord(a)
            if family and delta in (0, 1):
                up[i] = up[i+1]
            if family and delta in (0, -1):
                down[i] = down[i+1]
        # End of each uninterrupted sequence of runs of length >= 2.
        paired_end = [0]*35
        for i in range(32, -1, -1):
            end = folded_end[i]
            if end-i >= 2:
                paired_end[i] = max(end, paired_end[end])
        for start in range(35-lo):
            paired = min(start+hi, paired_end[start])
            # The upper length bound must not leave a singleton final group.
            if paired > start+1 and norm[paired-1] != norm[paired-2]:
                paired -= 1
            end = max(paired, min(start+hi, max(up[start], down[start])))
            if end-start >= lo and end > folded_end[start]:
                direction = 1 if end <= up[start] else -1 if end <= down[start] else 0
                tags = ["分组递增" if direction > 0 else "分组递减"] if direction else []
                matches.append(_match(address, start, end-start, "连续分组",
                                      _run_groups(norm[start:end]), tags))

    if mask & 8 and lo <= 9:
        for start in range(35-lo):
            if norm[start] not in "123456789":
                continue
            for delta in (1, -1):
                end = start+1
                while end < min(34, start+hi, start+9) and norm[end] in "123456789" and ord(norm[end])-ord(norm[end-1]) == delta:
                    end += 1
                if end-start >= lo:
                    matches.append(_match(address, start, end-start, "数字顺子"))

    if mask & 16:
        best = [0]*34
        periods = [0]*34
        for period in (2, 3, 4):
            repeated = 0
            for start in range(33-period, -1, -1):
                repeated = repeated+1 if norm[start] == norm[start+period] else 0
                length = min(hi, repeated+period)
                if length >= max(lo, 2*period) and start+length > folded_end[start] and length > best[start]:
                    best[start], periods[start] = length, period
        for start, length in enumerate(best):
            if length:
                matches.append(_match(address, start, length, "周期重复", tags=["周期{}位".format(periods[start])]))

    if mask & 32:
        # Each center contributes only its longest allowed palindrome.
        for center in range(67):
            left, right = center//2, (center+1)//2
            while left >= 0 and right < 34 and right-left+1 <= hi and norm[left] == norm[right]:
                left -= 1
                right += 1
            length = right-left-1
            if length >= lo:
                matches.append(_match(address, left+1, length, "回文"))
    return _summary(matches)


def classify_and_verify(priv_hex, gpu_address, config):
    address = _priv_to_address(bytes.fromhex(priv_hex))
    if address != gpu_address:
        raise ValueError("GPU/CPU 地址校验不一致，停止本次搜索")
    result = classify_vanity(address, config)
    if result is not None:
        result.update(private_key=priv_hex, address=address, source="GPU",
                      verified_by="CPU", classified_by="CPU",
                      time=datetime.now(timezone.utc).isoformat(), schema_version=1)
    return result


def init_classifier():
    # Parent handles Ctrl+C and drains already submitted candidate batches.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)


def classify_batch(candidates, config):
    results = []
    for private_key, address in candidates:
        result = classify_and_verify(private_key, address, config)
        if result:
            results.append(result)
    return results


def gen_startpoints_batch(n):
    xs, ys = bytearray(n*32), bytearray(n*32)
    privs = []
    for i in range(n):
        key = secrets.randbelow(SECP256K1_N-1)+1
        privs.append(key)
        pub = coincurve.PublicKey.from_valid_secret(key.to_bytes(32, "big")).format(compressed=False)
        xs[i*32:(i+1)*32], ys[i*32:(i+1)*32] = pub[1:33], pub[33:65]
    return bytes(xs), bytes(ys), privs
