"""Independent thresholds, UI, parameter layout and CPU reference coverage."""
import contextlib
import io
import random
import unittest
from unittest.mock import patch

from cpu_worker import RULE_SPECS, classify_vanity, validate_config
from test_vanity import embed, fixtures
from vanity_ranker import find_matches


EXAMPLES = {
    'same': 'AAAAAAAA', 'folded': 'aAaAAAaaAa', 'groups': 'aabbccdd',
    'straight': '123456789', 'cyclic': '123456789123',
    'digit_periodic': '123123123', 'mixed_periodic': 'ABCDABCDABCD',
    'digit_palindrome': '1234554321', 'mixed_palindrome': 'ABCDEFGGFEDCBA',
    'digits': '5837291648357291',
}


def independent_cases():
    addresses = list(fixtures())
    for example in EXAMPLES.values():
        for start in range(1, 35-len(example)):
            addresses.append(embed(example, start))
    addresses += [embed('AA12344321aa'), embed('AB123321ba'), embed('Z12344321z'),
                  embed('12344321ABCCBA'), 'T'+'1'*33, 'T'+'A'*33,
                  'T'+('123456789'*4)[:33], 'T'+('aB1'*11)]
    rng = random.Random(34581)
    configs = [dict(mode='wide', rule_minima={s[0]: s[5] for s in RULE_SPECS})]
    for key, _, maximum, *_ in RULE_SPECS:
        for lo in (2, 8, min(9, maximum), maximum):
            configs.append(dict(mode='wide', rule_minima={key: lo}))
    for _ in range(10):
        configs.append(dict(mode='wide', rule_minima={s[0]: rng.choice([0, rng.randint(2, s[2])]) for s in RULE_SPECS}))
    return addresses, configs


class IndependentTests(unittest.TestCase):
    def test_examples_every_position(self):
        for key, fragment in EXAMPLES.items():
            for start in range(1, 35-len(fragment)):
                result = classify_vanity(embed(fragment, start), dict(mode='wide', rule_minima={key: len(fragment)}))
                self.assertIsNotNone(result, (key, start))
                self.assertTrue(any(m['start'] <= start and m['start']+m['length'] >= start+len(fragment)
                                    and m['rule'] == key for m in result['matches']))

    def test_different_thresholds_disabled_rules_and_digit_split(self):
        address = embed('88888888ZaAaAaAaAaA')
        result = classify_vanity(address, dict(mode='wide', rule_minima={'same': 8, 'folded': 10}))
        self.assertEqual({m['rule'] for m in result['matches']}, {'same', 'folded'})
        result = classify_vanity(address, dict(mode='wide', rule_minima={'same': 9, 'folded': 10}))
        self.assertEqual({m['rule'] for m in result['matches']}, {'folded'})
        for rule, text in [('digit_palindrome', 'ABCDEFGGFEDCBA'), ('mixed_palindrome', '1234554321'),
                           ('digit_periodic', 'ABCABCABC'), ('mixed_periodic', '123123123')]:
            self.assertIsNone(classify_vanity(embed(text), dict(mode='wide', rule_minima={rule: 8})))
        self.assertIsNotNone(classify_vanity(embed('AA12344321aa'), dict(mode='wide', rule_minima={'digit_palindrome': 8})))
        self.assertIsNone(classify_vanity(embed('AAAAaaaa'), dict(mode='wide', rule_minima={'groups': 8})))

    def test_bounds_and_config(self):
        for cfg in ({}, {'same': 0}, {'same': True}, {'same': 35}, {'straight': 10}, {'cyclic': -1}, {'unknown': 8}):
            with self.assertRaises(ValueError):
                validate_config(dict(mode='wide', rule_minima=cfg))
        validate_config(dict(mode='wide', rule_minima={'cyclic': 34}))
        self.assertIsNotNone(classify_vanity('T'+'1'*33, dict(mode='wide', rule_minima={'digits': 33})))
        self.assertIsNone(classify_vanity('T'+'1'*33, dict(mode='wide', rule_minima={'digits': 34})))

    def test_prompt_blank_disables_and_invalid_input_retries(self):
        from tron_vanity_gpu import prompt_mode_2
        # First entire pass blank; second invalid range, then valid minimum.
        answers = ['']*10+['10-34', '8', '', '', '10', '8']+['']*6
        with patch('builtins.input', side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
            config = prompt_mode_2()
        self.assertEqual({k:v for k,v in config['rule_minima'].items() if v}, {'same': 8, 'straight': 8})

    def test_numpy_cuda_parameter_layout(self):
        from tron_vanity_gpu import make_pattern_params, PATTERN_DTYPE
        values = {s[0]: s[5] for s in RULE_SPECS}
        params = make_pattern_params(dict(mode='wide', rule_minima=values))
        self.assertEqual(PATTERN_DTYPE.itemsize, 792)
        self.assertEqual(PATTERN_DTYPE.fields['independent_rules'][1], 732)
        self.assertEqual(PATTERN_DTYPE.fields['rule_minima'][1], 736)
        self.assertEqual(params['rule_minima'][0].tolist(), list(values.values()))
        self.assertEqual(params['independent_rules'][0], 1)
        self.assertEqual(make_pattern_params(dict(mode='exact', suffix='8888'))['independent_rules'][0], 0)

    def test_ranker_exhaustive_windows_agree_with_classifier(self):
        mapping = {'same': {'纯豹子'}, 'folded': {'纯豹子', '同字母混合大小写'},
                   'groups': {'成对分组', '等长多连分组', '阶梯分组', '对称分组', '变长分组', '顺序分组'},
                   'straight': {'数字顺子'}, 'cyclic': {'数字循环顺子'}, 'digit_periodic': {'数字周期重复'},
                   'mixed_periodic': {'含字母周期重复'}, 'digit_palindrome': {'纯数字回文'},
                   'mixed_palindrome': {'含字母回文'}, 'digits': {'连续纯数字'}}
        addresses, configs = independent_cases()
        for address in addresses:
            windows = list(find_matches(address, 2, 34))
            for cfg in configs:
                expected = any(m['series'] in mapping[key] and m['length'] >= lo and
                               (key != 'folded' or m['content'][0].isalpha())
                               for key, lo in cfg['rule_minima'].items() if lo for m in windows)
                self.assertEqual(bool(classify_vanity(address, cfg)), expected, (address, cfg))

    def test_full_match_metadata_and_private_key_validation(self):
        from cpu_worker import classify_and_verify, _priv_to_address
        key = (1).to_bytes(32, 'big').hex()
        address = _priv_to_address(bytes.fromhex(key))
        # Find a permissive real match without changing cryptographic validation.
        cfg = dict(mode='wide', rule_minima={'same': 2, 'groups': 2, 'mixed_palindrome': 2, 'digits': 2})
        result = classify_and_verify(key, address, cfg)
        self.assertIsNotNone(result)
        self.assertEqual(result['schema_version'], 2)
        self.assertEqual(result['rule_minima'], cfg['rule_minima'])
        self.assertEqual(result['verified_by'], 'CPU')
        for match in result['matches']:
            self.assertEqual(match['content'], address[match['start']:match['start']+match['length']])
            self.assertEqual(match['minimum_length'], cfg['rule_minima'][match['rule']])
        with self.assertRaises(ValueError):
            classify_and_verify(key, 'T'+'1'*33, cfg)


if __name__ == '__main__':
    unittest.main()
