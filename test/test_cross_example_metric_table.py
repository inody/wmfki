import math
import unittest

import numpy as np

from scripts.build_cross_example_metric_table import (
    format_relative,
    rmse_with_standard_error,
)


class CrossExampleMetricTableTest(unittest.TestCase):
    def test_delta_method_standard_error(self):
        estimate, standard_error = rmse_with_standard_error([3.0, 4.0])
        expected_estimate = math.sqrt(12.5)
        expected_standard_error = np.std([9.0, 16.0], ddof=1) / math.sqrt(2.0)
        expected_standard_error /= 2.0 * expected_estimate
        self.assertAlmostEqual(estimate, expected_estimate)
        self.assertAlmostEqual(standard_error, expected_standard_error)

    def test_nonfinite_values_are_reported_as_missing(self):
        self.assertEqual(rmse_with_standard_error([1.0, np.inf]), (None, None))

    def test_table_entry_formats_dimensionless_point_estimate(self):
        self.assertEqual(format_relative(2.2585), "2.26")
        self.assertEqual(format_relative(0.5362), "0.536")
        self.assertEqual(format_relative(None), "div.")


if __name__ == "__main__":
    unittest.main()
