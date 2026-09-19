"""CPU regression tests; no CUDA device required."""
import random
import unittest
from cpu_worker import (BASE58_ALPHABET, RULE_BITS, classify_vanity, classify_and_verify,
                        validate_config, _priv_to_address)

EXAMPLES = ['aaabbbbcccc', 'abbccddeeff', 'abbcccddddeeee', '1122233344455566',
            '11122233', 'aabbccdd', 'AAccBBDd', '112233445566', '12345678',
            '98765432', 'aAaAAAaaAa', '12312312', 'ABCABCABC', 'ABABABAB',
            '12344321', '123454321', 'aaaaaaaaabbbbbbbbbb']


def embed(fragment, start=6):
    address = list('TZk7Q9mX4sN6pR2uV8wY5hJ3eF9qC7dS2gK'[:34])
    assert len(address) == 34
    address[start:start+len(fragment)] = fragment
    return ''.join(address)


def fixtures():
    for s in EXAMPLES:
        for start in (1, 6, 34-len(s)):
            yield embed(s, start)
    rng = random.Random(458)
    for _ in range(200):
        yield 'T'+''.join(rng.choices(BASE58_ALPHABET, k=33))
    for _ in range(100):
        s = ''.join(c*rng.randint(2, 5) for c in rng.sample('aBcDeF123456789', rng.randint(2, 6)))[:30]
        yield embed(s, rng.randint(1, 34-len(s)))


class ClassifierTests(unittest.TestCase):
    def test_optimized_classifier_matches_reference(self):
        from classifier_reference import classify_reference
        rng = random.Random(20260920)
        addresses = list(fixtures())
        addresses += ['T'+'a'*33, 'T'+'AaB'*11, 'T'+('aabbCCdd'*5)[:33],
                      'T'+('123454321'*4)[:33], 'T'+('ABCD'*9)[:33]]
        addresses += ['T'+''.join(rng.choices('aAbBcC123', k=33)) for _ in range(150)]
        for address in addresses:
            configs = [dict(mode='wide'), dict(mode='wide', min_len=8, max_len=8)]
            for _ in range(5):
                lo = rng.randint(2, 34)
                configs.append(dict(mode='wide', min_len=lo, max_len=rng.randint(lo, 34),
                                    rules={key: bool(rng.getrandbits(1)) for key in RULE_BITS}))
            for cfg in configs:
                with self.subTest(address=address, config=cfg):
                    self.assertEqual(classify_vanity(address, cfg), classify_reference(address, cfg))

    def test_all_user_examples_all_positions(self):
        for fragment in EXAMPLES:
            for start in range(1, 35-len(fragment)):
                with self.subTest(fragment=fragment, start=start):
                    result = classify_vanity(embed(fragment, start), {'mode': 'wide'})
                    self.assertIsNotNone(result)
                    self.assertTrue(any(m['start'] <= start and m['start']+m['length'] >= start+len(fragment)
                                        for m in result['matches']))

    def test_groups_and_positions(self):
        expected = {'aaabbbbcccc': [3, 4, 4], 'abbccddeeff': [1, 2, 2, 2, 2, 2],
                    'abbcccddddeeee': [1, 2, 3, 4, 4], '11122233': [3, 3, 2],
                    '1122233344455566': [2, 3, 3, 3, 3, 2], 'AAccBBDd': [2, 2, 2, 2]}
        for text, lengths in expected.items():
            result = classify_vanity(embed(text), {'mode': 'wide'})
            self.assertEqual(result['content'], text)
            self.assertEqual(result['start'], 6)
            self.assertEqual(result['group_lengths'], lengths)

    def test_no_arbitrary_padding(self):
        result = classify_vanity(embed('aabbccdd'), {'mode': 'wide'})
        self.assertEqual(result['length'], 8)
        self.assertIsNone(classify_vanity(embed('aaQbbRcc'), {'mode': 'wide'}))

    def test_casefold_not_two_groups(self):
        result = classify_vanity(embed('AAAAaaaa'), {'mode': 'wide'})
        self.assertEqual(result['type'], '同字母忽略大小写')
        self.assertFalse(any(m['type'] == '连续分组' for m in result['matches']))

    def test_ranges_and_longest(self):
        address = embed('aAaAaAaAaAaA')
        result = classify_vanity(address, {'mode': 'wide', 'min_len': 8, 'max_len': 12})
        self.assertEqual(result['length'], 12)
        same = [m for m in result['matches'] if m['type'] == '同字母忽略大小写']
        self.assertEqual(len(same), 1)
        self.assertEqual(classify_vanity(address, {'mode': 'wide', 'min_len': 8, 'max_len': 8})['length'], 8)
        self.assertIsNone(classify_vanity(address, {'mode': 'wide', 'min_len': 13, 'max_len': 34}))

    def test_secondary_matches(self):
        result = classify_vanity(embed('aabbccddZ12344321'), {'mode': 'wide'})
        self.assertTrue(any(m['content'] == 'aabbccdd' for m in result['matches']))
        self.assertTrue(any(m['content'] == '12344321' for m in result['matches']))

    def test_disabled_rules(self):
        rules = {key: False for key in RULE_BITS}
        self.assertIsNone(classify_vanity(embed('AaAaAaAa'), {'mode': 'wide', 'rules': rules}))
        rules['same'] = True
        self.assertIsNone(classify_vanity(embed('AaAaAaAa'), {'mode': 'wide', 'rules': rules}))
        rules['folded'] = True
        self.assertIsNotNone(classify_vanity(embed('AaAaAaAa'), {'mode': 'wide', 'rules': rules}))

    def test_exact_and_or_empty_sides(self):
        addr = embed('aabbccdd')
        configs = [dict(mode='exact', prefix='TZZ,'+addr[:5], suffix=addr[-6:]),
                   dict(mode='exact', prefix='', suffix=addr[-6:]+',888888', combine_or=True),
                   dict(mode='exact', prefix=addr[:5], suffix='', combine_or=True)]
        for cfg in configs:
            self.assertIsNotNone(classify_vanity(addr, cfg))
        cfg = dict(mode='exact', prefix='TXXXX', suffix=addr[-6:])
        self.assertIsNone(classify_vanity(addr, cfg))
        cfg['combine_or'] = True
        self.assertIsNotNone(classify_vanity(addr, cfg))

    def test_config_overlap_and_limits(self):
        address = embed('aabbccdd')
        validate_config(dict(mode='exact', prefix=address[:22], suffix=address[-22:]))
        with self.assertRaises(ValueError):
            validate_config(dict(mode='exact', prefix='T'+'a'*25, suffix='b'*20))
        validate_config(dict(mode='exact', prefix='T'+'a'*25, suffix='b'*20, combine_or=True))
        with self.assertRaises(ValueError):
            validate_config(dict(mode='exact', suffix=','.join(str(x)*8 for x in range(1,10))))
        for lo, hi in [(2, 4), (5, 8), (8, 34), (34, 34)]:
            validate_config(dict(mode='wide', min_len=lo, max_len=hi))

    def test_crypto_and_validation(self):
        # Public test vector, never a production key.
        key = (1).to_bytes(32, 'big').hex()
        address = _priv_to_address(bytes.fromhex(key))
        self.assertEqual(address, 'TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC')
        result = classify_and_verify(key, address, dict(mode='exact', suffix=address[-6:]))
        self.assertEqual(result['content'], address[-6:])
        self.assertEqual(result['start'], 28)
        with self.assertRaises(ValueError):
            classify_and_verify(key, address[:-1]+'1', dict(mode='wide'))

    def test_windows_spawn_classifier(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from cpu_worker import classify_batch, init_classifier
        key = (1).to_bytes(32, 'big').hex()
        address = _priv_to_address(bytes.fromhex(key))
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
                                 initializer=init_classifier) as pool:
            result = pool.submit(classify_batch, [(key, address)], dict(mode='exact', prefix='T')).result(timeout=30)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['address'], address)


if __name__ == '__main__':
    unittest.main()
