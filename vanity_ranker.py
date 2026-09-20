"""Standalone TRON ranking: longest structure first, then closest to either edge."""
import argparse
import codecs
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ADDRESS_RE = re.compile(r"(?<![A-Za-z0-9])T[" + ALPHABET + r"]{33}(?![A-Za-z0-9])")
KEY_RE = re.compile(r"(?<![A-Za-z0-9])[0-9a-fA-F]{64}(?![A-Za-z0-9])")
ADDRESS_FIELDS = ("address", "地址", "钱包地址", "TRON地址")
KEY_FIELDS = ("private_key", "privkey", "私钥")
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SERIES = ("纯豹子", "同字母混合大小写", "成对分组", "等长多连分组",
          "阶梯分组", "对称分组", "变长分组", "顺序分组", "数字顺子", "数字循环顺子",
          "数字周期重复", "含字母周期重复", "纯数字回文", "含字母回文", "连续纯数字")
DEFAULT_CONFIG = {"排序规则": "长度优先，其次靠近首尾"}
SCORING_VERSION = 3


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if supplied != config:
            raise ValueError("新版固定按长度、首尾距离排序，请使用新版 ranker_config.example.json")
    return config


def valid_address(address):
    if not isinstance(address, str) or not re.fullmatch(r"T["+ALPHABET+r"]{33}", address):
        return False
    value = 0
    for char in address:
        value = value*58 + ALPHABET.index(char)
    try:
        raw = value.to_bytes(25, "big")
    except OverflowError:
        return False
    return raw[0] == 0x41 and hashlib.sha256(hashlib.sha256(raw[:21]).digest()).digest()[:4] == raw[21:]


def runs(text):
    result = []
    for char in text:
        if result and result[-1][0] == char:
            result[-1][1] += 1
        else:
            result.append([char, 1])
    return result


def group_style(groups, direction):
    lengths = [g[1] for g in groups]
    tags = ["字符递增" if direction > 0 else "字符递减"] if direction else []
    if all(n == 2 for n in lengths):
        return "成对分组", min(1, .94+.06*bool(direction)), tags+["每组2位"]
    if len(set(lengths)) == 1 and lengths[0] >= 3:
        return "等长多连分组", min(1, .94+.06*bool(direction)), tags+["组长相等"]
    deltas = [b-a for a, b in zip(lengths, lengths[1:])]
    if len(lengths) >= 3 and (all(d == 1 for d in deltas) or all(d == -1 for d in deltas)):
        return "阶梯分组", min(1, .95+.05*bool(direction)), tags+["组长递增" if deltas[0] > 0 else "组长递减"]
    if len(lengths) >= 3 and lengths == lengths[::-1] and len(set(lengths)) > 1:
        return "对称分组", min(1, .92+.05*bool(direction)), tags+["组长对称"]
    if all(n == 1 for n in lengths):
        return "顺序分组", 1.0, tags
    uniformity = max(Counter(lengths).values())/len(lengths)
    monotonic = all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas)
    if monotonic:
        tags.append("组长单调变化")
    return "变长分组", min(.92, .55+.25*uniformity+.1*monotonic+.1*bool(direction)), tags


def find_matches(address, minimum=8, maximum=34):
    """Inspect eligible windows, preserving the distinct structure families.

    A subwindow can belong to a different series than its containing window.
    Each series chooses its longest match, then the closest one to either edge.
    """
    norm = address.lower()
    singleton, bad_up, bad_down = [0], [0], [0]
    for i in range(34):
        singleton.append(singleton[-1]+int(0 < i < 33 and norm[i] != norm[i-1] and norm[i] != norm[i+1]))
    for a, b in zip(norm, norm[1:]):
        family = (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())
        delta = ord(b)-ord(a)
        bad_up.append(bad_up[-1]+int(not family or delta not in (0, 1)))
        bad_down.append(bad_down[-1]+int(not family or delta not in (0, -1)))
    for length in range(maximum, minimum-1, -1):
        for start in range(35-length):
            end = start+length
            folded, original = norm[start:end], address[start:end]
            same = folded.count(folded[0]) == length
            kinds = []
            if original.count(original[0]) == length:
                kinds.append(("纯豹子", 1.0, [[folded[0], length]], []))
            elif same and folded[0].isalpha():
                kinds.append(("同字母混合大小写", 1.0, [[folded[0], length]], ["混合大小写"]))
            if not same:
                paired = (folded[0] == folded[1] and folded[-1] == folded[-2] and
                          singleton[end-1] == singleton[start+1])
                direction = (1 if bad_up[end-1] == bad_up[start] else
                             -1 if bad_down[end-1] == bad_down[start] else 0)
                if paired or direction:
                    groups = runs(folded)
                    series, quality, tags = group_style(groups, direction)
                    kinds.append((series, quality, groups, tags))
                for period in range(2, min(4, length//2)+1):
                    if folded[period:] == folded[:-period]:
                        complete = length % period == 0
                        kinds.append(("数字周期重复" if folded.isdigit() else "含字母周期重复", .8+.12*complete+.08*(2/period), [],
                                      ["周期{}位".format(period), "完整周期" if complete else "末尾不足一周期"]))
                        break
            if length <= 9 and (folded in "123456789" or folded in "987654321"):
                kinds.append(("数字顺子", 1.0, [], ["递增" if folded[0] < folded[-1] else "递减"]))
            if folded.isdigit():
                # Digits alone are meaningful length, not a perfect structure.
                kinds.append(("连续纯数字", .3, [], ["纯数字；不要求排列规律"]))
                deltas = [(int(b)-int(a)) % 9 for a, b in zip(folded, folded[1:])]
                if all(d == 1 for d in deltas) or all(d == 8 for d in deltas):
                    kinds.append(("数字循环顺子", 1.0, [], ["循环递增" if deltas[0] == 1 else "循环递减"]))
            if folded == folded[::-1]:
                kinds.append(("纯数字回文" if folded.isdigit() else "含字母回文", 1.0, [], ["偶数对称" if length % 2 == 0 else "奇数对称"]))
            for series, quality, groups, tags in kinds:
                yield dict(series=series, start=start, length=length, content=original,
                           group_lengths=[g[1] for g in groups], tags=tags, structure_quality=quality)


def score_match(match, config):
    result = dict(match)
    length, start = match["length"], match["start"]
    # Fixed leading T is not an extra gap: start 0 or 1 both touch the head.
    head_gap = max(0, start-1)
    tail_gap = 34-start-length
    distance = min(head_gap, tail_gap)
    position = "首尾" if head_gap == tail_gap == 0 else "开头" if head_gap == 0 else "尾部" if tail_gap == 0 else "靠近开头" if head_gap <= tail_gap else "靠近尾部"
    # One character of length always outweighs every possible edge bonus.
    # Score is kept for machine-readable exports/filtering, not display.
    score = round(100*(length*100+34-distance)/3434, 6)
    result.update(score=score, edge_distance=distance, head_distance=head_gap, tail_distance=tail_gap,
                  position=position, groups=len(match["group_lengths"]), position_base=0)
    result.pop("structure_quality", None)
    return result


def match_order(match):
    return (-match["length"], match["edge_distance"], SERIES.index(match["series"]), match["start"])


def rank_address(address, config, minimum=8, maximum=34):
    if not 2 <= minimum <= maximum <= 34:
        raise ValueError("长度范围需为 2～34")
    if len(address) != 34 or not address.startswith("T") or any(c not in ALPHABET for c in address):
        raise ValueError("地址格式错误")
    best = {}
    for match in find_matches(address, minimum, maximum):
        scored = score_match(match, config)
        series = scored["series"]
        if series not in best or match_order(scored) < match_order(best[series]):
            best[series] = scored
    matches = sorted(best.values(), key=match_order)
    return dict(address=address, record_id=hashlib.sha256(address.encode()).hexdigest()[:20],
                best=matches[0] if matches else None, series_matches=matches,
                scoring_version=SCORING_VERSION, address_checksum_valid=valid_address(address))


def text_encoding(path):
    with path.open("rb") as f:
        sample = f.read(65536)
    if sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    try:
        codecs.getincrementaldecoder("utf-8")(errors="strict").decode(sample, final=False)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "gb18030"


def read_records(path, issue):
    """Yield source line and record; parser errors never expose raw key text."""
    with path.open(encoding=text_encoding(path), newline="") as f:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(f)
            for raw in reader:
                yield reader.line_num, raw
            return
        first = next((line for line in f if line.strip()), "")
        f.seek(0)
        is_json = path.suffix.lower() == ".jsonl" or first.lstrip().startswith("{")
        pending = None
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            if not is_json and line.split() == ["地址", "位数"]:
                continue
            if is_json:
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError()
                except (ValueError, json.JSONDecodeError):
                    issue(path, number, "不是有效的单行 JSON 对象")
                    continue
                yield number, raw
                continue
            addresses, keys = ADDRESS_RE.findall(line), KEY_RE.findall(line)
            if addresses:
                if pending:
                    yield pending
                    pending = None
                if len(addresses) != 1 or len(keys) > 1:
                    issue(path, number, "同一行包含多个地址或私钥，无法确定对应关系")
                    continue
                pending = (number, {"address": addresses[0]})
                if keys:
                    pending[1]["private_key"] = keys[0]
            elif keys:
                if len(keys) == 1 and pending and not pending[1].get("private_key"):
                    pending[1]["private_key"] = keys[0]
                else:
                    issue(path, number, "私钥缺少唯一对应的前置地址")
            elif line.strip().startswith(("T", "地址", "address", "私钥", "private_key")):
                issue(path, number, "地址或私钥格式无法识别")
        if pending:
            yield pending


def normalize_record(raw):
    addresses = {str(raw[k]).strip() for k in ADDRESS_FIELDS if k in raw and raw[k] not in (None, "")}
    if len(addresses) != 1:
        raise ValueError("缺少唯一地址字段")
    address = addresses.pop()
    if not valid_address(address):
        raise ValueError("TRON 地址格式或 Base58Check 校验失败")
    keys = {str(raw[k]).strip().lower() for k in KEY_FIELDS if k in raw and raw[k] not in (None, "")}
    if len(keys) > 1:
        raise ValueError("同一记录含冲突的私钥字段")
    key = next(iter(keys), "")
    if key and (not re.fullmatch(r"[0-9a-f]{64}", key) or not 0 < int(key, 16) < SECP256K1_N):
        raise ValueError("私钥不是有效的 64 位十六进制标量")
    metadata = {k: v for k, v in raw.items() if k not in ADDRESS_FIELDS+KEY_FIELDS}
    return address, key, metadata


def expand_inputs(paths):
    found = []
    for value in paths:
        path = Path(str(value).strip().strip('"')).expanduser().resolve()
        if path.is_dir():
            found.extend(p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in (".jsonl", ".txt", ".csv"))
        elif path.is_file():
            found.append(path)
        else:
            raise ValueError("找不到输入文件：{}".format(path))
    found = list(dict.fromkeys(found))
    if not found:
        raise ValueError("没有找到 JSONL、TXT 或 CSV 文件")
    return found


CSV_COLUMNS = ["地址", "位数"]


def export_csv(db, path, rows):
    # The TXT is the primary human-readable view; CSV has the same two columns.
    with path.open("w", encoding="utf-8-sig", newline="") as f, path.with_suffix(".txt").open("w", encoding="utf-8-sig") as text:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        text.write("地址                                  位数\n")
        for address, record_id, payload in rows:
            match = json.loads(payload) if payload else None
            length = match["length"] if match else 0
            writer.writerow([address, length])
            text.write("{}    {}\n".format(address, length))


def run_ranking(inputs, output, config, minimum=8, maximum=34, top=0, min_score=0):
    if not 2 <= minimum <= maximum <= 34 or top < 0 or not math.isfinite(min_score) or not 0 <= min_score <= 100:
        raise ValueError("长度范围需为 2～34，数量上限需非负，最低分需为 0～100")
    files = expand_inputs(inputs)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，请指定新目录，避免覆盖已有结果")
    output.mkdir(parents=True, exist_ok=True)
    details = output/"详细数据"
    categories = output/"分类"
    details.mkdir()
    categories.mkdir()
    (details/"评分配置.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    db = sqlite3.connect(str(details/"评分索引.sqlite3"))
    db.executescript("""
        CREATE TABLE records(address TEXT PRIMARY KEY, record_id TEXT, payload TEXT, best TEXT, score REAL, length INTEGER);
        CREATE TABLE original_records(address TEXT, source TEXT, line INTEGER, metadata TEXT);
        CREATE INDEX source_address ON original_records(address);
        CREATE TABLE private_keys(address TEXT, key TEXT, PRIMARY KEY(address,key));
        CREATE TABLE series(address TEXT, name TEXT, payload TEXT, score REAL, length INTEGER, PRIMARY KEY(address,name));
    """)
    started, last_report = time.monotonic(), time.monotonic()
    stats = dict(input_records=0, unique_addresses=0, duplicates=0, matched_addresses=0,
                 unmatched_addresses=0, import_errors=0, conflicting_addresses=0)
    try:
        with (details/"导入问题.csv").open("w", encoding="utf-8-sig", newline="") as errors:
            writer = csv.writer(errors)
            writer.writerow(["来源文件", "行号", "问题"])
            def issue(path, line, reason):
                stats["import_errors"] += 1
                writer.writerow([str(path), line, reason])
            for path in files:
                print("读取：{}".format(path), flush=True)
                try:
                    recognized, errors_before = 0, stats["import_errors"]
                    for number, raw in read_records(path, issue):
                        recognized += 1
                        stats["input_records"] += 1
                        try:
                            address, key, metadata = normalize_record(raw)
                        except ValueError as exc:
                            issue(path, number, str(exc))
                            continue
                        exists = db.execute("SELECT 1 FROM records WHERE address=?", (address,)).fetchone()
                        if exists:
                            stats["duplicates"] += 1
                        else:
                            ranked = rank_address(address, config, minimum, maximum)
                            best = ranked["best"]
                            db.execute("INSERT INTO records VALUES (?,?,?,?,?,?)", (
                                address, ranked["record_id"], json.dumps(ranked, ensure_ascii=False),
                                json.dumps(best, ensure_ascii=False) if best else None,
                                best["score"] if best else 0, best["length"] if best else 0))
                            for match in ranked["series_matches"]:
                                db.execute("INSERT INTO series VALUES (?,?,?,?,?)", (
                                    address, match["series"], json.dumps(match, ensure_ascii=False), match["score"], match["length"]))
                            stats["unique_addresses"] += 1
                            stats["matched_addresses" if best else "unmatched_addresses"] += 1
                        db.execute("INSERT INTO original_records VALUES (?,?,?,?)", (
                            address, str(path), number, json.dumps(metadata, ensure_ascii=False)))
                        if key:
                            db.execute("INSERT OR IGNORE INTO private_keys VALUES (?,?)", (address, key))
                        if stats["input_records"] % 1000 == 0:
                            db.commit()
                        if time.monotonic()-last_report >= 2:
                            print("  已读取 {} 条，去重后 {} 个，有结构 {} 个".format(
                                stats["input_records"], stats["unique_addresses"], stats["matched_addresses"]), flush=True)
                            last_report = time.monotonic()
                    if not recognized and stats["import_errors"] == errors_before and path.stat().st_size:
                        issue(path, 0, "文件中未识别到任何地址记录")
                except (OSError, UnicodeError, csv.Error) as exc:
                    issue(path, 0, "文件读取失败：{}".format(type(exc).__name__))
            db.commit()
        db.executescript("CREATE INDEX ranking ON records(score DESC,length DESC,address);"
                         "CREATE INDEX series_ranking ON series(name,score DESC,length DESC,address);")
        stats["conflicting_addresses"] = db.execute(
            "SELECT COUNT(*) FROM (SELECT address FROM private_keys GROUP BY address HAVING COUNT(*)>1)").fetchone()[0]
        print("正在导出两列排行榜（地址、位数）……", flush=True)
        limit = top if top else -1
        export_csv(db, output/"总排行榜.csv", db.execute(
            "SELECT address,record_id,best FROM records WHERE best IS NOT NULL AND score>=? "
            "ORDER BY score DESC,length DESC,address LIMIT ?", (min_score, limit)))
        for series in SERIES:
            export_csv(db, categories/(series+".csv"), db.execute(
                "SELECT s.address,r.record_id,s.payload FROM series s JOIN records r ON r.address=s.address "
                "WHERE s.name=? AND s.score>=? ORDER BY s.score DESC,s.length DESC,s.address LIMIT ?",
                (series, min_score, limit)))
        export_csv(db, output/"未匹配.csv", db.execute(
            "SELECT address,record_id,best FROM records WHERE best IS NULL ORDER BY address"))
        with (details/"完整评分结果.jsonl").open("w", encoding="utf-8") as f:
            for address, payload in db.execute("SELECT address,payload FROM records ORDER BY score DESC,length DESC,address"):
                record = json.loads(payload)
                keys = [row[0] for row in db.execute("SELECT key FROM private_keys WHERE address=? ORDER BY key", (address,))]
                record["private_key"] = keys[0] if len(keys) == 1 else None
                record["private_key_conflict"] = len(keys) > 1
                if len(keys) > 1:
                    record["conflicting_private_keys"] = keys
                record["sources"] = [dict(file=p, line=n, metadata=json.loads(m)) for p, n, m in db.execute(
                    "SELECT source,line,metadata FROM original_records WHERE address=? ORDER BY rowid", (address,))]
                f.write(json.dumps(record, ensure_ascii=False)+"\n")
        stats.update(scoring_version=SCORING_VERSION, min_length=minimum, max_length=maximum,
                     csv_top_per_list=top, csv_min_score=min_score, inputs=[str(p) for p in files],
                     elapsed_seconds=round(time.monotonic()-started, 2), series_counts=dict(db.execute(
                         "SELECT name,COUNT(*) FROM series GROUP BY name")), complete=True)
        (details/"统计.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print("完成：{} 个地址，{} 个有结构，{} 个未匹配，{} 条导入问题。\n打开查看：{}\n分类结果：{}".format(
            stats["unique_addresses"], stats["matched_addresses"], stats["unmatched_addresses"], stats["import_errors"],
            output/"总排行榜.txt", categories))
        return stats
    finally:
        db.close()


def main(argv=None):
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="TRON 靓号精品筛选器：长度优先，同长度越靠近首尾越靠前；仅显示地址、位数")
    parser.add_argument("inputs", nargs="*", help="JSONL/TXT/CSV 文件或包含这些文件的目录")
    parser.add_argument("-o", "--output", help="新的输出目录，默认 精选结果/当前时间")
    parser.add_argument("--config", help="评分配置 JSON 文件")
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=34)
    parser.add_argument("--top", type=int, default=0, help="每张排行榜最多几条；0 为全部")
    parser.add_argument("--min-score", type=float, default=0, help="CSV 排行榜最低分；完整 JSONL 始终保留全部有效地址")
    args = parser.parse_args(argv)
    if not args.inputs:
        print("TRON 靓号精品筛选器｜长度优先，其次靠近首尾｜仅显示地址、位数")
        value = input("请输入结果文件或目录路径（可拖入文件）：").strip().strip('"')
        if not value:
            parser.error("需要一个输入文件或目录")
        args.inputs = [value]
    output = args.output or str(Path("精选结果")/datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    try:
        stats = run_ranking(args.inputs, output, load_config(args.config), args.min_length, args.max_length,
                            args.top, args.min_score)
        return 2 if stats["import_errors"] or stats["conflicting_addresses"] else 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print("错误：{}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中止；输入文件未修改，本次输出可能不完整。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
