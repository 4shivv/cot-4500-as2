import unittest
import numpy as np
from src.main.assignment_2 import neville_method, newton_forward_interpolation

class TestAssignment2(unittest.TestCase):
    def test_neville_method(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        self.assertAlmostEqual(neville_method(x_vals, y_vals, 3.7), 1.5549999999999999, places=4)

    def test_newton_forward_interpolation(self):
        x_vals = [7.2, 7.4, 7.5, 7.6]
        y_vals = [23.5492, 25.3913, 26.8224, 27.4589]
        self.assertAlmostEqual(newton_forward_interpolation(x_vals, y_vals, 7.3), 24.47718457889519, places=4)

if __name__ == '__main__':
    unittest.main()
