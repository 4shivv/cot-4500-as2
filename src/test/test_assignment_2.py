import unittest
import numpy as np
from src.main.assignment_2 import (
    neville_interpolation, newton_forward_interpolation,
    newton_forward_eval, hermite_interpolation, cubic_spline_interpolation
)

class TestInterpolationMethods(unittest.TestCase):
    def test_neville_interpolation(self):
        x_points = np.array([3.6, 3.8, 3.9])
        y_points = np.array([1.675, 1.436, 1.318])
        result = neville_interpolation(x_points, y_points, 3.7)
        self.assertAlmostEqual(result, 1.5549999999999995, places=5)

    def test_newton_forward(self):
        x_points = np.array([7.2, 7.4, 7.5, 7.6])
        y_points = np.array([23.5492, 25.3913, 26.8224, 27.4589])
        diff_table = newton_forward_interpolation(x_points, y_points)
        result = newton_forward_eval(diff_table, x_points, 7.3)
        self.assertAlmostEqual(result, 24.47718457889519, places=5)

    def test_cubic_spline(self):
        x_points = np.array([2, 5, 8, 10])
        y_points = np.array([3, 5, 7, 9])
        A, b, x = cubic_spline_interpolation(x_points, y_points)
        self.assertEqual(A.shape, (4, 4))

if __name__ == "__main__":
    unittest.main()
