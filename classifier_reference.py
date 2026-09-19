"""Frozen pre-optimization classifier, used only for equivalence tests."""
from cpu_worker import BASE58_ALPHABET, split_targets, rule_mask, _match, _summary, _run_groups


def classify_reference(address, config):
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
    # Prefix counts let random windows fail grouping in O(1), instead of
    # constructing hundreds of run lists for every coarse candidate.
    singletons = [0]
    for i in range(34):
        singletons.append(singletons[-1] + int(0 < i < 33 and norm[i] != norm[i-1] and norm[i] != norm[i+1]))
    bad_up, bad_down = [0], [0]
    for a, b in zip(norm, norm[1:]):
        family = (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())
        delta = ord(b)-ord(a)
        bad_up.append(bad_up[-1] + int(not family or delta not in (0, 1)))
        bad_down.append(bad_down[-1] + int(not family or delta not in (0, -1)))
    # Descending lengths permit containment suppression without losing locations.
    for length in range(hi, lo-1, -1):
        for start in range(35-length):
            text = norm[start:start+length]
            original = address[start:start+length]
            same = text.count(text[0]) == length
            if mask & 1 and original.count(original[0]) == length:
                matches.append(_match(address, start, length, "连续相同字符", [[text[0], length]]))
            if mask & 2 and same and text[0].isalpha():
                tags = ["混合大小写"] if original != original.upper() and original != original.lower() else []
                matches.append(_match(address, start, length, "同字母忽略大小写", [[text[0], length]], tags))
            if mask & 4 and not same:
                end = start+length
                paired = (text[0] == text[1] and text[-1] == text[-2] and
                          singletons[end-1] == singletons[start+1])
                direction = (1 if bad_up[end-1] == bad_up[start] else
                             -1 if bad_down[end-1] == bad_down[start] else 0)
                if paired or direction:
                    groups = _run_groups(text)
                    tags = ["分组递增" if direction > 0 else "分组递减"] if direction else []
                    matches.append(_match(address, start, length, "连续分组", groups, tags))
            if mask & 8 and length <= 9 and (text in "123456789" or text in "987654321"):
                matches.append(_match(address, start, length, "数字顺子"))
            if mask & 16 and not same:
                for period in range(2, min(4, length//2)+1):
                    if text[period:] == text[:-period]:
                        matches.append(_match(address, start, length, "周期重复", tags=["周期{}位".format(period)]))
                        break
            if mask & 32 and text == text[::-1]:
                matches.append(_match(address, start, length, "回文"))
    return _summary(matches)

