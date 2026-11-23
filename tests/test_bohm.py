"""Spinless Bohmian flow: analytical motion and an independent current check."""

import math
import unittest

import numpy as np

from atom_simulator.physics import (
    A_MU_METERS,
    HBAR_JOULE_SECONDS,
    REDUCED_MASS_KG,
    T_MU_SECONDS,
    QuantumNumbers,
    bohm_advance,
    bohm_angular_velocity,
    bohm_velocity,
    density,
    sample_orbital,
    wavefunction,
)


class BohmVelocityTests(unittest.TestCase):
    def test_direction_and_magnitude_for_positive_and_negative_m(self) -> None:
        points = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, -1.0]])
        expected = np.array([[0.0, 0.5, 0.0], [-0.25, 0.0, 0.0]])
        positive = bohm_velocity(QuantumNumbers(2, 1, 1), points)
        negative = bohm_velocity(QuantumNumbers(2, 1, -1), points)
        np.testing.assert_allclose(positive, expected, atol=1e-15)
        np.testing.assert_allclose(negative, -expected, atol=1e-15)
        np.testing.assert_allclose(
            bohm_angular_velocity(QuantumNumbers(2, 1, 1), points), [0.25, 0.0625], atol=1e-15
        )

    def test_m_zero_has_zero_velocity_and_an_exact_copy(self) -> None:
        state = QuantumNumbers(6, 2, 0)
        points = sample_orbital(state, 2000, seed=12).positions
        advanced = bohm_advance(state, points, 1.0e9)
        np.testing.assert_array_equal(advanced, points)
        self.assertFalse(np.shares_memory(advanced, points))
        np.testing.assert_array_equal(bohm_velocity(state, points), np.zeros(points.shape))

    def test_velocity_equals_current_over_density_from_cartesian_finite_difference(self) -> None:
        state = QuantumNumbers(6, 2, 1)
        points = np.array([[2.0, 1.5, 0.7], [-4.0, 2.0, -1.2], [8.0, -3.0, 2.5]])
        step = 1.0e-5
        gradient = np.empty(points.shape, dtype=np.complex128)
        for axis in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[:, axis] += step
            minus[:, axis] -= step
            gradient[:, axis] = (wavefunction(state, plus) - wavefunction(state, minus)) / (
                2.0 * step
            )
        values = wavefunction(state, points)
        current_over_density = np.imag(gradient / values[:, None])
        np.testing.assert_allclose(
            bohm_velocity(state, points), current_over_density, rtol=2e-8, atol=2e-9
        )

    def test_no_artificial_cap_close_to_the_axis(self) -> None:
        points = np.array([[1.0e-9, 0.0, 1.0]])
        velocity = bohm_velocity(QuantumNumbers(2, 1, 1), points)
        self.assertAlmostEqual(float(velocity[0, 1]) / 1.0e9, 1.0, delta=2e-15)
        self.assertAlmostEqual(
            float(bohm_angular_velocity(QuantumNumbers(2, 1, 1), points)[0]) / 1.0e18,
            1.0,
            delta=2e-15,
        )

    def test_axis_and_resolved_radial_or_polar_nodes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bohm_velocity(QuantumNumbers(2, 1, 1), [[0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            bohm_advance(QuantumNumbers(3, 1, 1), [[6.0, 0.0, 0.0]], 1.0)
        with self.assertRaises(ValueError):
            bohm_velocity(QuantumNumbers(3, 2, 1), [[2.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            bohm_velocity(QuantumNumbers(2, 0, 0), [[2.0, 0.0, 0.0]])

    def test_natural_units_consistent_with_codata_reduced_mass(self) -> None:
        self.assertGreater(A_MU_METERS, 5.291e-11)
        self.assertLess(A_MU_METERS, 5.295e-11)
        self.assertAlmostEqual(
            T_MU_SECONDS, REDUCED_MASS_KG * A_MU_METERS**2 / HBAR_JOULE_SECONDS, delta=1e-31
        )
        self.assertGreater(T_MU_SECONDS, 2.418e-17)
        self.assertLess(T_MU_SECONDS, 2.422e-17)


class BohmAdvanceTests(unittest.TestCase):
    def test_known_quarter_turn(self) -> None:
        points = np.array([[2.0, 0.0, 1.0]], dtype=np.float32)
        advanced = bohm_advance(QuantumNumbers(2, 1, 1), points, 2.0 * math.pi)
        np.testing.assert_allclose(advanced, [[0.0, 2.0, 1.0]], atol=3e-7)

    def test_radius_cylindrical_radius_and_height_are_preserved(self) -> None:
        state = QuantumNumbers(6, 2, 1)
        points = sample_orbital(state, 10000, seed=16).positions
        advanced = bohm_advance(state, points, 1.0e6)
        np.testing.assert_allclose(
            np.linalg.norm(advanced, axis=1), np.linalg.norm(points, axis=1), rtol=2e-7, atol=1e-5
        )
        np.testing.assert_allclose(
            np.hypot(advanced[:, 0], advanced[:, 1]),
            np.hypot(points[:, 0], points[:, 1]),
            rtol=2e-7,
            atol=5e-6,
        )
        np.testing.assert_array_equal(advanced[:, 2], points[:, 2])
        self.assertEqual(advanced.dtype, np.dtype(np.float32))

    def test_semigroup_up_to_documented_float32_rounding(self) -> None:
        state = QuantumNumbers(3, 2, 2)
        points = np.array([[2.0, 1.0, 0.4], [-3.0, 2.0, -0.6]], dtype=np.float32)
        first_step = bohm_advance(state, points, 3.0)
        two_steps = bohm_advance(state, first_step, 4.0)
        one_step = bohm_advance(state, points, 7.0)
        np.testing.assert_allclose(two_steps, one_step, rtol=3e-7, atol=5e-7)

    def test_negative_m_reverses_time_direction(self) -> None:
        points = np.array([[2.0, 1.0, 0.4], [-3.0, 2.0, -0.6]], dtype=np.float32)
        backward = bohm_advance(QuantumNumbers(3, 2, 2), points, -12.0)
        reversed_m = bohm_advance(QuantumNumbers(3, 2, -2), points, 12.0)
        np.testing.assert_array_equal(backward, reversed_m)

    def test_stationary_density_is_preserved_along_trajectories(self) -> None:
        state = QuantumNumbers(2, 1, 1)
        points = sample_orbital(state, 5000, seed=26).positions
        advanced = bohm_advance(state, points, 345.0)
        np.testing.assert_allclose(
            density(state, advanced), density(state, points), rtol=3e-6, atol=2e-10
        )

    def test_different_cylindrical_radii_have_different_angular_speeds(self) -> None:
        points = np.array([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
        state = QuantumNumbers(2, 1, 1)
        omega = bohm_angular_velocity(state, points)
        self.assertAlmostEqual(float(omega[0] / omega[1]), 4.0)

    def test_empty_arrays_and_zero_time_are_supported(self) -> None:
        empty = np.empty((0, 3), dtype=np.float32)
        state = QuantumNumbers(2, 1, 1)
        self.assertEqual(bohm_velocity(state, empty).shape, (0, 3))
        self.assertEqual(bohm_angular_velocity(state, empty).shape, (0,))
        self.assertEqual(bohm_advance(state, empty, 1.0).shape, (0, 3))
        points = np.array([[2.0, 1.0, 0.4]], dtype=np.float32)
        np.testing.assert_array_equal(bohm_advance(state, points, 0.0), points)

    def test_invalid_inputs(self) -> None:
        state = QuantumNumbers(2, 1, 1)
        with self.assertRaises(ValueError):
            bohm_velocity(state, np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            bohm_advance(state, [[1.0, np.nan, 0.0]], 2.0)
        with self.assertRaises(ValueError):
            bohm_advance(state, [[1.0, 0.0, 0.0]], np.inf)
        with self.assertRaises(TypeError):
            bohm_advance(state, [[1.0, 0.0, 0.0]], True)


if __name__ == "__main__":
    unittest.main()
