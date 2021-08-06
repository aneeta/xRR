import unittest
import pandas as pd
import numpy as np

from xrr.xrr_class import xRR

class TestXRR(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.X = pd.DataFrame(
            {
                'item_id': {
                    0: 1,
                    1: 2,
                    2: 3,
                    3: 4,
                    4: 5,
                    5: 6,
                    6: 1,
                    7: 2,
                    8: 3,
                    9: 4,
                    10: 5,
                    11: 6,
                    12: 1,
                    13: 2,
                    14: 3,
                    15: 4,
                    16: 5,
                    17: 6
                    },
                'rater': {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 2,
                    7: 2,
                    8: 2,
                    9: 2,
                    10: 2,
                    11: 2,
                    12: 3,
                    13: 3,
                    14: 3,
                    15: 3,
                    16: 3,
                    17: 3
                    },
                'rating': {
                    0: 1.0,
                    1: 1.0,
                    2: 0.0,
                    3: 0.0,
                    4: 0.0,
                    5: 0.0,
                    6: 1.0,
                    7: 1.0,
                    8: 0.0,
                    9: 0.0,
                    10: 1.0,
                    11: 0.0,
                    12: np.nan,
                    13: 1.0,
                    14: np.nan,
                    15: 0.0,
                    16: np.nan,
                    17: 0.0
                    }
                }
        )

        self.Y = pd.DataFrame(
            {
                'item_id': {
                    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5
                    },
                'rater': {
                    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2
                    },
                'rating': {
                    0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 0
                    }
                }
        )

        self.xRR_obj = xRR(self.X, self.Y, "rating", "rater", "item_id")

        self.expected_do = 0.4090909
        self.expected_de = 0.5
        self.expected_xRR = 0.1818182
    
    @classmethod
    def tearDownClass(self):
        pass

    def test_d_e(self):
        self.assertEqual(self.xRR_obj.d_e(), self.expected_de)

    def test_d_o(self):
        self.assertEqual(round(self.xRR_obj.d_o(), 7), self.expected_do)
    
    def test_kappa_x(self):
        self.assertEqual(round(self.xRR_obj.kappa_x(), 7), self.expected_xRR)
        self.assertEqual(round(self.xRR_obj.kappa, 7), self.expected_xRR)
    
    def test_normalized_kappa_x(self):
        with self.assertRaises(ValueError):
            self.xRR_obj.normalized_kappa_x("kappa")

    def test_IRR(self):
        self.assertEqual(round(self.xRR_obj._IRR(self.xRR_obj.Y_, "kappa"), 3), -0.154)


if __name__ == '__main__':
    unittest.main()