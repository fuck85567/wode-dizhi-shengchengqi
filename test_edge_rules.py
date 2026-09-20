import contextlib
import io
import unittest
from unittest.mock import patch

from cpu_worker import EDGE_RULES, classify_vanity, edge_matches, validate_config
from test_vanity import embed
from tron_vanity_gpu import make_pattern_params, prompt_mode_1


class EdgeTests(unittest.TestCase):
    def test_rules_and_positions(self):
        for rule, fragment in [('same', '88888888'), ('folded', 'AaAAaaAa'),
                               ('straight', '23456789'), ('cyclic', '78912345'), ('literal', 'Ab123XYZ')]:
            for side, start in [('prefix', 1), ('suffix', 26)]:
                spec = dict(rule=rule, length=8, targets=fragment)
                cfg = dict(mode='exact', edges={side: spec})
                validate_config(cfg)
                result = classify_vanity(embed(fragment, start), cfg)
                self.assertIsNotNone(result)
                self.assertEqual(result['start'], start)
                self.assertEqual(result['content'], fragment)
                self.assertIsNone(classify_vanity(embed(fragment, 10), cfg))
                params = make_pattern_params(cfg)
                index = int(side == 'suffix')
                self.assertEqual(params['edge_rules'][0,index,0], EDGE_RULES[rule])
                self.assertEqual(params['edge_lengths'][0,index,0], 8)
        self.assertFalse(edge_matches('AaAAAAAA', dict(rule='same')))
        self.assertFalse(edge_matches('11111111', dict(rule='folded')))
        self.assertFalse(edge_matches('78912345', dict(rule='straight')))
        self.assertTrue(edge_matches('32198765', dict(rule='cyclic')))

    def test_and_or_and_disabled_side(self):
        address = embed('88888888', 26)
        edges = dict(prefix=dict(rule='literal', length=3, targets='XYZ'), suffix=dict(rule='same', length=8))
        self.assertIsNone(classify_vanity(address, dict(mode='exact', edges=edges)))
        self.assertIsNotNone(classify_vanity(address, dict(mode='exact', edges=edges, combine_or=True)))
        self.assertIsNone(classify_vanity(address, dict(mode='exact', edges={'prefix': edges['prefix']}, combine_or=True)))

    def test_validation_and_multiple_literals(self):
        for spec in [dict(rule='straight', length=10), dict(rule='same', length=35),
                     dict(rule='literal', length=8, targets='888'), dict(rule='literal', length=1, targets='0')]:
            with self.assertRaises(ValueError):
                validate_config(dict(mode='exact', edges={'suffix':spec}))
        spec = dict(rule='literal', length=8, targets='88888888,66666666')
        cfg = dict(mode='exact', edges={'suffix':spec})
        self.assertIsNotNone(classify_vanity(embed('66666666',26),cfg))
        self.assertEqual(make_pattern_params(cfg)['suffix_count'][0],2)

    def test_prompt_flow(self):
        with patch('builtins.input', side_effect=['n','y','8','2']), contextlib.redirect_stdout(io.StringIO()):
            cfg = prompt_mode_1()
        self.assertEqual(cfg, dict(mode='exact',edges={'suffix':[dict(rule='folded',length=8)]},combine_or=False))
        with patch('builtins.input', side_effect=['y','3','15','ABC','y','8','14','OR']), contextlib.redirect_stdout(io.StringIO()):
            cfg = prompt_mode_1()
        self.assertTrue(cfg['combine_or'])
        self.assertEqual([x['rule'] for x in cfg['edges']['prefix']], ['same','literal'])
        self.assertEqual(cfg['edges']['prefix'][1]['targets'],'ABC')

    def test_multiple_rules_same_side(self):
        cfg = dict(mode='exact', edges={'suffix': [
            dict(rule='same', length=8),
            dict(rule='straight', length=8),
            dict(rule='cyclic', length=8),
        ]})
        self.assertIsNotNone(classify_vanity(embed('88888888', 26), cfg))
        self.assertIsNotNone(classify_vanity(embed('23456789', 26), cfg))
        self.assertIsNotNone(classify_vanity(embed('78912345', 26), cfg))


if __name__ == '__main__':
    unittest.main()
