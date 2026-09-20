"""Offline ranker tests. Uses only the standard library, no GPU dependencies."""
import contextlib
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import vanity_ranker as ranker


def embed(fragment, start=6):
    address = list('TZk7Q9mX4sN6pR2uV8wY5hJ3eF9qC7dS2gK'[:34])
    address[start:start+len(fragment)] = fragment
    return ''.join(address)


def checked_address(fragment, start=6):
    """Synthetic valid Base58Check address; no claim about a corresponding key."""
    value = 0
    address = list('TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC')
    address[start:start+len(fragment)] = fragment
    for char in address:
        value = value*58+ranker.ALPHABET.index(char)
    payload = value.to_bytes(25, 'big')[:21]
    assert payload[0] == 0x41
    value = int.from_bytes(payload+hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4], 'big')
    result = ''
    while value:
        value, remainder = divmod(value, 58)
        result = ranker.ALPHABET[remainder]+result
    assert result[start:start+len(fragment)] == fragment
    return result


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class RankerTests(unittest.TestCase):
    def setUp(self):
        self.config = ranker.load_config()

    def match(self, fragment, series, start=6):
        result = ranker.rank_address(embed(fragment, start), self.config)
        return next(m for m in result['series_matches'] if m['series'] == series)

    def test_user_examples_classified(self):
        examples = {
            '88888888': '纯豹子', 'aAaAAAaaAa': '同字母混合大小写',
            'aabbccdd': '成对分组', 'AAccBBDd': '成对分组',
            '112233445566': '成对分组', 'aaabbbccc': '等长多连分组',
            'abbcccdddd': '阶梯分组', 'aaBBBccccDDDee': '对称分组',
            'aaabbbbcccc': '变长分组', '11122233': '变长分组',
            'abbcccddddeeee': '变长分组', '1122233344455566': '对称分组',
            'abbccddeeff': '变长分组', '12345678': '数字顺子',
            '987654321': '数字顺子', 'ABABABAB': '含字母周期重复', 'ABCABCABC': '含字母周期重复',
            '12344321': '纯数字回文', '123454321': '纯数字回文'}
        for text, series in examples.items():
            with self.subTest(text=text):
                self.assertEqual(self.match(text, series)['content'], text)

    def test_scoring_weights_and_dimensions(self):
        self.assertEqual(self.config, {'排序规则': '长度优先，其次靠近首尾'})
        for text in ('aabbccdd', 'aAaAAAaaAa', '12344321', 'ABABABAB'):
            for match in ranker.rank_address(embed(text), self.config)['series_matches']:
                self.assertGreaterEqual(match['score'], 0)
                self.assertLessEqual(match['score'], 100)
                self.assertNotIn('scores', match)

    def test_visual_structure_and_length_order(self):
        neat = self.match('aabbccdd', '成对分组')
        mixed = self.match('AAccBBDd', '成对分组')
        self.assertEqual(neat['score'], mixed['score'])
        alternating = self.match('AaAaAaAaAa', '同字母混合大小写')
        messy = self.match('aAaAAAaaAa', '同字母混合大小写')
        self.assertEqual(alternating['score'], messy['score'])
        self.assertGreater(self.match('aaaaaaaaaa', '纯豹子', 12)['score'],
                           self.match('aaaaaaaa', '纯豹子', 26)['score'])
        full = ranker.score_match(next(m for m in ranker.find_matches(embed('ABCABCABCABC'))
                                      if m['series'] == '含字母周期重复' and m['content'] == 'ABCABCABCABC'), self.config)
        partial = ranker.score_match(next(m for m in ranker.find_matches(embed('ABCABCABCAB'))
                                         if m['series'] == '含字母周期重复' and m['content'] == 'ABCABCABCAB'), self.config)
        self.assertGreater(full['score'], partial['score'])

    def test_positions_and_same_fragment_scoring(self):
        tail = self.match('aabbccdd', '成对分组', 26)
        front = self.match('aabbccdd', '成对分组', 1)
        middle = self.match('aabbccdd', '成对分组', 6)
        self.assertEqual([m['edge_distance'] for m in (tail, front, middle)], [0, 0, 5])
        self.assertEqual(tail['score'], front['score'])
        self.assertGreater(front['score'], middle['score'])
        near_head = self.match('aabbccdd', '成对分组', 2)
        near_tail = self.match('aabbccdd', '成对分组', 25)
        self.assertEqual(near_head['score'], near_tail['score'])
        self.assertGreater(front['score'], near_head['score'])
        self.assertGreater(near_head['score'], middle['score'])
        address = embed('aabbccdd', 1)
        address = address[:26]+'AAbbCCdd'
        ranked = ranker.rank_address(address, self.config)
        for match in ranked['series_matches']:
            self.assertEqual(match['content'], address[match['start']:match['start']+match['length']])
            expected = ranker.score_match(next(m for m in ranker.find_matches(address)
                if (m['series'], m['start'], m['length']) == (match['series'], match['start'], match['length'])), self.config)
            self.assertEqual(match, expected)

    def test_no_padding_no_duplicate_family_and_shorter_pretty_window(self):
        result = ranker.rank_address(embed('aabbccdd'), self.config)
        self.assertEqual(result['best']['length'], 8)
        self.assertEqual(len(result['series_matches']), len({m['series'] for m in result['series_matches']}))
        self.assertIsNone(ranker.rank_address(embed('aaQbbRcc'), self.config)['best'])
        result = ranker.rank_address(embed('Aaaaaaaaa'), self.config)
        # The pure lower-case sub-run must remain available as its own series.
        self.assertTrue(any(m['series'] == '纯豹子' and m['content'] == 'aaaaaaaa' for m in result['series_matches']))

    def test_config_preferences_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps(self.config), encoding='utf-8')
            config = ranker.load_config(path)
            self.assertEqual(config, self.config)
            for invalid in ({'权重': {'有意义长度': 99}}, {'位置系数': {'尾部': 2}}, {'偏好字符': '0'},
                            {'偏好系列': ['不存在']}, {'未知参数': 1}):
                path.write_text(json.dumps(invalid), encoding='utf-8')
                with self.assertRaises(ValueError):
                    ranker.load_config(path)

    def test_checksum_and_key_import_validation(self):
        address = checked_address('aabbccdd')
        self.assertTrue(ranker.valid_address(address))
        self.assertTrue(ranker.valid_address('TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC'))
        damaged = address[:-1]+('1' if address[-1] != '1' else '2')
        self.assertFalse(ranker.valid_address(damaged))
        for raw in ({'address': damaged}, {'address': address, 'private_key': '0'*64},
                    {'address': address, 'private_key': 'bad'}, {'address': address, '地址': damaged}):
            with self.assertRaises(ValueError):
                ranker.normalize_record(raw)

    def test_txt_csv_json_bom_and_encodings(self):
        address, key = checked_address('aabbccdd'), '0'*63+'1'
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
                path = folder/(encoding+'.txt')
                path.write_text('地址：'+address+'\n私钥：'+key+'\n类型：分组\n', encoding=encoding)
                records = list(ranker.read_records(path, lambda *args: self.fail(str(args))))
                self.assertEqual(ranker.normalize_record(records[0][1])[:2], (address, key))
            path = folder/'json.txt'
            path.write_text(json.dumps(dict(address=address, private_key=key)), encoding='utf-8-sig')
            self.assertEqual(len(list(ranker.read_records(path, lambda *args: self.fail(str(args))))), 1)
            path = folder/'addresses.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['地址', '私钥', '备注'])
                writer.writerow([address, key, '原始\n备注'])
            raw = next(ranker.read_records(path, lambda *args: self.fail(str(args))))[1]
            self.assertEqual(ranker.normalize_record(raw), (address, key, {'备注': '原始\n备注'}))

    def test_end_to_end_dedup_sort_filters_metadata_and_key_conflicts(self):
        addresses = [checked_address('aabbccdd'), checked_address('aaaabbbbcccc'),
                     'TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC']
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'input.jsonl'
            rows = [dict(address=addresses[0], private_key='0'*63+'1', time='original-time', matches=[{'old': True}]),
                    dict(address=addresses[0], private_key='0'*63+'2'),
                    dict(address=addresses[1]), dict(address=addresses[2]), dict(address='not-an-address')]
            source.write_text('\n'.join(json.dumps(r) for r in rows)+'\nBROKEN JSON\n', encoding='utf-8')
            before = source.read_bytes()
            output = Path(folder)/'ranked'
            with contextlib.redirect_stdout(io.StringIO()):
                stats = ranker.run_ranking([source], output, self.config, top=1)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual((stats['unique_addresses'], stats['duplicates'], stats['import_errors'], stats['conflicting_addresses']),
                             (3, 1, 2, 1))
            self.assertEqual(len(read_csv(output/'总排行榜.csv')), 1)
            self.assertEqual(len(read_csv(output/'未匹配.csv')), 1)
            records = [json.loads(line) for line in (output/'详细数据'/'完整评分结果.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 3)
            record = next(r for r in records if r['address'] == addresses[0])
            self.assertIsNone(record['private_key'])
            self.assertTrue(record['private_key_conflict'])
            self.assertEqual(len(record['conflicting_private_keys']), 2)
            self.assertEqual(record['sources'][0]['metadata']['time'], 'original-time')
            self.assertEqual(record['sources'][0]['metadata']['matches'], [{'old': True}])
            self.assertEqual(list(read_csv(output/'分类'/'成对分组.csv')[0]), ['地址', '位数'])
            for series in ranker.SERIES:
                data = read_csv(output/'分类'/(series+'.csv'))
                self.assertEqual(len({r['地址'] for r in data}), len(data))
                self.assertEqual([int(r['位数']) for r in data], sorted([int(r['位数']) for r in data], reverse=True))
            with self.assertRaises(ValueError):
                ranker.run_ranking([source], output, self.config)
            with contextlib.redirect_stdout(io.StringIO()):
                ranker.run_ranking([source], Path(folder)/'filtered', self.config, min_score=100)
            self.assertEqual(read_csv(Path(folder)/'filtered'/'总排行榜.csv'), [])
            self.assertEqual(len((Path(folder)/'filtered'/'详细数据'/'完整评分结果.jsonl').read_text(encoding='utf-8').splitlines()), 3)

    def test_cli_standalone_and_series_sorting(self):
        addresses = [checked_address(s) for s in ('AAccBBDd', 'aabbccdd', 'aabbccddeeff', 'aaccbbdd')]
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'input.txt', Path(folder)/'output'
            source.write_text('\n'.join(addresses), encoding='utf-8')
            process = subprocess.run([sys.executable, '-S', str(Path(ranker.__file__)), str(source), '-o', str(output)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr.decode('utf-8', errors='replace'))
            data = read_csv(output/'分类'/'成对分组.csv')
            self.assertEqual(len(data), len(addresses))
            # The following RR is also part of the longest valid grouping.
            self.assertEqual(data[0], {'地址': addresses[2], '位数': '14'})
            self.assertEqual([int(r['位数']) for r in data], sorted([int(r['位数']) for r in data], reverse=True))
            txt_rows = (output/'分类'/'成对分组.txt').read_text(encoding='utf-8-sig').splitlines()[1:]
            self.assertEqual([row.split() for row in txt_rows], [[r['地址'], r['位数']] for r in data])
            self.assertEqual(len(read_csv(output/'总排行榜.csv')), 4)
            self.assertTrue((output/'总排行榜.csv').read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_length_and_edge_priority_for_all_windows(self):
        by_length = {}
        for length in range(2, 35):
            matches = [ranker.score_match(dict(series='纯豹子', content='A'*length,
                       length=length, start=start, group_lengths=[length], tags=[]), self.config)
                       for start in range(35-length)]
            by_length[length] = matches
            for a in matches:
                for b in matches:
                    if a['edge_distance'] < b['edge_distance']:
                        self.assertGreater(a['score'], b['score'])
            if length > 2:
                self.assertGreater(min(m['score'] for m in matches), max(m['score'] for m in by_length[length-1]))

    def test_export_equal_lengths_sorted_by_edge_distance(self):
        addresses = [checked_address('aabbccdd', start) for start in (12, 4, 20)]
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'input.txt', Path(folder)/'out'
            source.write_text('\n'.join(addresses), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                ranker.run_ranking([source], output, self.config, minimum=8, maximum=8)
            rows = read_csv(output/'分类'/'成对分组.csv')
            self.assertEqual([r['地址'] for r in rows], [addresses[1], addresses[2], addresses[0]])
            self.assertTrue(all(set(r) == {'地址', '位数'} for r in rows))

    def test_unrecognized_input_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'unknown.txt'
            source.write_text('这不是结果文件', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                stats = ranker.run_ranking([source], Path(folder)/'out', self.config)
            self.assertEqual(stats['import_errors'], 1)
            self.assertEqual(stats['unique_addresses'], 0)


if __name__ == '__main__':
    unittest.main()
