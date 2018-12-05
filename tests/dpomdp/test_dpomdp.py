import unittest

from rl_parsers.dpomdp import parse


class DPOMDP_Test(unittest.TestCase):
    def parse_file(self, fname):
        with open(fname) as f:
            return parse(f.read())

    def test_parser(self):
        dpomdps = [
            'dectiger',
            'dectiger_skewed',
            'prisoners',
            'recycling',
            '2generals',
        ]

        for dpomdp in dpomdps:
            with self.subTest(dpomdp):
                self.parse_file(f'tests/dpomdp/{dpomdp}.dpomdp')


if __name__ == '__main__':
    unittest.main(module='test_dpomdp')
