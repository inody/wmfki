import unittest

import numpy as np

from kernels import make_kernel_basis, make_radial_basis
from simulate import make_true_kernel
from scripts.run_localized10d_pilot import (
    DEFAULT_CONFIG,
    _augment_record,
    effective_protocol,
    read_json,
    validate_protocol,
)
from scripts.run_morse10d_sweeps import _radial_drift_fast, physical_conditions, repetition_plan
from utils import make_nested_pair_sampler


class TestLocalized10DPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(DEFAULT_CONFIG)

    def test_protocol_is_matched_sparse_and_pilot_only(self):
        validate_protocol(self.spec)
        truth = self.spec["model"]["interaction"]
        dictionary = self.spec["dictionary"]
        self.assertTrue(self.spec["pilot_only"])
        self.assertEqual(self.spec["evaluation"]["scientific_repetitions"], 3)
        self.assertEqual(truth["centers"], dictionary["centers"])
        self.assertEqual(truth["h"], dictionary["h"])
        self.assertEqual(np.count_nonzero(truth["coefficients"]), 2)
        self.assertFalse(self.spec["estimator"]["truth_used_for_selection"])
        self.assertFalse(self.spec["estimator"]["test_error_used_for_selection"])

    def test_scientific_protocol_is_frozen_after_the_pilot(self):
        scientific = read_json(
            DEFAULT_CONFIG.with_name("localized10d_scientific.json")
        )
        validate_protocol(scientific)
        self.assertFalse(scientific["pilot_only"])
        self.assertEqual(scientific["evaluation"]["scientific_repetitions"], 20)
        for key in ("model", "dictionary", "weak_tests", "pair_sampling", "sweeps", "estimator", "seeds"):
            self.assertEqual(scientific[key], self.spec[key])

    def test_same_population_evaluation_is_fixed_without_refitting(self):
        scientific = read_json(
            DEFAULT_CONFIG.with_name("localized10d_scientific.json")
        )
        evaluation = read_json(
            DEFAULT_CONFIG.with_name("localized10d_sameN_evaluation.json")
        )
        self.assertEqual(evaluation["protocol_version"], 1)
        self.assertEqual(evaluation["rollout_particles"], scientific["model"]["particles"])
        self.assertEqual(evaluation["display_repetition"], 0)
        self.assertIn("do not refit or retune", evaluation["policy"])

    def test_dictionary_coefficients_exactly_reproduce_truth(self):
        radius = np.linspace(0.0, 4.5, 101)
        radial_basis, _ = make_radial_basis(self.spec["dictionary"])
        _, _, _, true_profile = make_true_kernel(self.spec["model"]["interaction"])
        theta = np.asarray(self.spec["model"]["interaction"]["coefficients"])
        np.testing.assert_allclose(radial_basis(radius) @ theta, true_profile(radius))
        z = np.random.default_rng(9).normal(size=(31, 10))
        kernel_basis, _ = make_kernel_basis(self.spec["dictionary"])
        true_kernel, _, _, _ = make_true_kernel(self.spec["model"]["interaction"])
        np.testing.assert_allclose(
            np.einsum("nmd,m->nd", kernel_basis(z), theta), true_kernel(z)
        )

    def test_fast_radial_drift_matches_direct_shell_evaluation(self):
        rng = np.random.default_rng(43)
        x = rng.normal(size=(19, 10))
        interaction = self.spec["model"]["interaction"]
        kernel, _, _, _ = make_true_kernel(interaction)
        direct = kernel(x[:, None, :] - x[None, :, :]).mean(axis=1)
        np.testing.assert_allclose(
            _radial_drift_fast(x, interaction), direct, rtol=2e-13, atol=2e-13
        )

    def test_smoke_is_separate_and_keeps_scientific_protocol_unchanged(self):
        smoke = effective_protocol(self.spec, smoke=True)
        self.assertEqual(smoke["evaluation"]["scientific_repetitions"], 1)
        self.assertLess(smoke["model"]["particles"], self.spec["model"]["particles"])
        self.assertEqual(self.spec["evaluation"]["scientific_repetitions"], 3)
        self.assertEqual(self.spec["model"]["particles"], 500)

    def test_conditions_and_split_seeds_are_paired(self):
        conditions = physical_conditions(self.spec)
        self.assertEqual(conditions.count((500, 0.05)), 1)
        self.assertIn((500, 0.0), conditions)
        self.assertIn((100, 0.05), conditions)
        plan = repetition_plan(self.spec)[0]
        self.assertNotEqual(plan["training_seed"], plan["validation_seed"])
        self.assertNotEqual(plan["training_seed"], plan["test_seed"])

    def test_nested_pair_budgets_share_timewise_prefixes(self):
        small = make_nested_pair_sampler(17, 991)
        large = make_nested_pair_sampler(41, 991)
        for _ in range(4):
            i_small, j_small = small(30)
            i_large, j_large = large(30)
            np.testing.assert_array_equal(i_small, i_large[:17])
            np.testing.assert_array_equal(j_small, j_large[:17])
            self.assertTrue(np.all(i_small != j_small))

    def test_coefficient_error_uses_explicit_matched_target(self):
        target = np.asarray(self.spec["model"]["interaction"]["coefficients"])
        exact = _augment_record({"coefficients": target.tolist()}, target)
        perturbed = _augment_record({"coefficients": (target + 0.1).tolist()}, target)
        self.assertEqual(exact["coefficient_error_relative"], 0.0)
        self.assertGreater(perturbed["coefficient_error_relative"], 0.0)


if __name__ == "__main__":
    unittest.main()
