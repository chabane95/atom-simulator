"""Distribution tests that detect missing spherical-coordinate Jacobians."""

import json
import unittest

import numpy as np
from scipy.special import gammainc, gammaincc
from scipy.stats import kstest

from atom_simulator.physics import QuantumNumbers
from atom_simulator.sampling import RADIAL_TAIL_PROBABILITY, sample_orbital


def one_s_radial_cdf(radius: np.ndarray) -> np.ndarray:
    return gammainc(3.0, 2.0 * radius)


def two_p_radial_cdf(radius: np.ndarray) -> np.ndarray:
    return gammainc(5.0, radius)


def two_p_zero_polar_cdf(coordinate: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + coordinate**3)


class SamplingTests(unittest.TestCase):
    def test_seeded_reproducibility_and_output_contract(self) -> None:
        first = sample_orbital(QuantumNumbers(), 1000, seed=7)
        repeated = sample_orbital(QuantumNumbers(), 1000, seed=7)
        changed = sample_orbital(QuantumNumbers(), 1000, seed=8)
        np.testing.assert_array_equal(first.positions, repeated.positions)
        self.assertFalse(np.array_equal(first.positions, changed.positions))
        self.assertEqual(first.positions.shape, (1000, 3))
        self.assertEqual(first.positions.dtype, np.dtype(np.float32))
        self.assertTrue(np.all(np.isfinite(first.positions)))
        self.assertTrue(np.all(first.radii <= first.bound_radius * (1.0 + 1e-6)))
        self.assertEqual(first.bound_radius, changed.bound_radius)
        self.assertEqual(json.loads(json.dumps(first.metadata))["position_unit"], "a_mu")

    def test_identical_sampling_for_opposite_m(self) -> None:
        positive = sample_orbital(QuantumNumbers(5, 3, 2), 3000, seed=81)
        negative = sample_orbital(QuantumNumbers(5, 3, -2), 3000, seed=81)
        np.testing.assert_array_equal(positive.positions, negative.positions)

    def test_one_s_radial_and_spherical_distribution(self) -> None:
        cloud = sample_orbital(QuantumNumbers(1, 0, 0), 100000, seed=31)
        radius = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)
        cos_theta = cloud.positions[:, 2] / radius
        phi = np.arctan2(cloud.positions[:, 1], cloud.positions[:, 0])
        self.assertAlmostEqual(float(np.mean(radius)), 1.5, delta=0.012)
        self.assertAlmostEqual(float(np.mean(radius**2)), 3.0, delta=0.05)
        self.assertAlmostEqual(float(np.mean(cos_theta)), 0.0, delta=0.008)
        self.assertAlmostEqual(float(np.mean(cos_theta**2)), 1.0 / 3.0, delta=0.004)
        self.assertAlmostEqual(float(np.mean(np.cos(phi))), 0.0, delta=0.008)
        self.assertAlmostEqual(float(np.mean(np.sin(phi))), 0.0, delta=0.008)
        self.assertLess(float(kstest(radius, one_s_radial_cdf).statistic), 0.008)

    def test_two_p_zero_radial_and_polar_cdfs(self) -> None:
        cloud = sample_orbital(QuantumNumbers(2, 1, 0), 100000, seed=41)
        radius = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)
        cos_theta = cloud.positions[:, 2] / radius
        self.assertAlmostEqual(float(np.mean(radius)), 5.0, delta=0.035)
        self.assertAlmostEqual(float(np.mean(cos_theta**2)), 3.0 / 5.0, delta=0.005)
        self.assertLess(float(kstest(radius, two_p_radial_cdf).statistic), 0.008)
        self.assertLess(float(kstest(cos_theta, two_p_zero_polar_cdf).statistic), 0.008)

    def test_two_p_one_polar_moment(self) -> None:
        cloud = sample_orbital(QuantumNumbers(2, 1, 1), 80000, seed=51)
        radius = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)
        cos_theta = cloud.positions[:, 2] / radius
        self.assertAlmostEqual(float(np.mean(cos_theta**2)), 1.0 / 5.0, delta=0.004)

    def test_default_excited_state_mean_radius(self) -> None:
        cloud = sample_orbital(QuantumNumbers(), 80000, seed=61)
        radius = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)
        self.assertAlmostEqual(float(np.mean(radius)), 51.0, delta=0.30)
        self.assertEqual(cloud.metadata["radial_node_count"], 3)

    def test_radial_cutoff_is_true_high_probability_quantile(self) -> None:
        cloud = sample_orbital(QuantumNumbers(1, 0, 0), 1000, seed=71)
        true_tail = float(gammaincc(3.0, 2.0 * cloud.bound_radius))
        self.assertAlmostEqual(true_tail, RADIAL_TAIL_PROBABILITY, delta=2e-12)
        self.assertGreaterEqual(cloud.metadata["radial_retained_probability"], 0.999999)
        second = sample_orbital(QuantumNumbers(1, 0, 0), 17, seed=999)
        self.assertEqual(cloud.bound_radius, second.bound_radius)

    def test_highest_supported_state_grid_normalization(self) -> None:
        states = [QuantumNumbers(12, 0, 0), QuantumNumbers(12, 11, 0), QuantumNumbers(12, 11, 11)]
        for state in states:
            with self.subTest(state=state):
                cloud = sample_orbital(state, 512, seed=91)
                retained = cloud.metadata["radial_retained_probability"]
                self.assertGreaterEqual(retained, 0.999999)
                self.assertAlmostEqual(cloud.metadata["radial_grid_integral"], retained, delta=2e-7)
                self.assertAlmostEqual(cloud.metadata["polar_grid_integral"], 1.0, delta=2e-7)
                self.assertTrue(np.all(np.isfinite(cloud.positions)))

    def test_every_supported_radial_and_polar_distribution(self) -> None:
        lower, upper = np.triu_indices(12)
        for index in range(lower.size):
            state = QuantumNumbers(n=int(upper[index]) + 1, l=int(lower[index]), m=0)
            with self.subTest(radial_state=state):
                cloud = sample_orbital(state, 3, seed=101)
                self.assertGreaterEqual(cloud.metadata["radial_retained_probability"], 0.999999)
                self.assertTrue(np.all(np.isfinite(cloud.positions)))
        for index in range(lower.size):
            degree = int(upper[index])
            state = QuantumNumbers(n=degree + 1, l=degree, m=int(lower[index]))
            with self.subTest(polar_state=state):
                cloud = sample_orbital(state, 3, seed=102)
                self.assertAlmostEqual(cloud.metadata["polar_grid_integral"], 1.0, delta=2e-7)
                self.assertTrue(np.all(np.isfinite(cloud.positions)))

    def test_invalid_sampling_inputs(self) -> None:
        state = QuantumNumbers()
        with self.assertRaises(ValueError):
            sample_orbital(state, 0)
        with self.assertRaises(ValueError):
            sample_orbital(state, -4)
        with self.assertRaises(ValueError):
            sample_orbital(state, 10, seed=-1)
        with self.assertRaises(TypeError):
            sample_orbital(state, 10.0)
        with self.assertRaises(TypeError):
            sample_orbital(state, True)
        with self.assertRaises(TypeError):
            sample_orbital(state, 10, seed=False)
        with self.assertRaises(TypeError):
            sample_orbital((1, 0, 0), 10)


if __name__ == "__main__":
    unittest.main()
