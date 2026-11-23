"""Scientific checks independent of the renderer and CLI."""

import math
import unittest

import numpy as np

from atom_simulator.physics import (
    QuantumNumbers,
    angular_wavefunction,
    associated_legendre,
    density,
    generalized_laguerre,
    polar_probability_density,
    radial_nodes,
    radial_probability_density,
    radial_wavefunction,
    wavefunction,
)


class QuantumNumberTests(unittest.TestCase):
    def test_default_state_and_node_count(self) -> None:
        state = QuantumNumbers()
        self.assertEqual(state.label, "n=6, l=2, m=0")
        self.assertEqual(state.radial_node_count, 3)

    def test_invalid_quantum_numbers(self) -> None:
        invalid = [(0, 0, 0), (13, 0, 0), (2, -1, 0), (2, 2, 0), (2, 1, 2), (2, 1, -2)]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    QuantumNumbers(*values)

    def test_non_integer_quantum_numbers(self) -> None:
        invalid = [(True, 0, 0), (2.0, 1, 0), (2, 1.0, 0), (2, 1, False)]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    QuantumNumbers(*values)


class WavefunctionTests(unittest.TestCase):
    def test_laguerre_recurrence_against_explicit_cubic(self) -> None:
        coordinate = np.array([0.0, 0.5, 2.0, 5.0, 12.0, 30.0])
        expected = (-(coordinate**3) + 24.0 * coordinate**2 - 168.0 * coordinate + 336.0) / 6.0
        np.testing.assert_allclose(generalized_laguerre(3, 5.0, coordinate), expected, rtol=2e-14)

    def test_associated_legendre_recurrence_and_condon_shortley_phase(self) -> None:
        coordinate = np.linspace(-1.0, 1.0, 101)
        expected_p21 = -3.0 * coordinate * np.sqrt(1.0 - coordinate**2)
        expected_p30 = 0.5 * (5.0 * coordinate**3 - 3.0 * coordinate)
        np.testing.assert_allclose(associated_legendre(2, 1, coordinate), expected_p21, atol=2e-15)
        np.testing.assert_allclose(associated_legendre(3, 0, coordinate), expected_p30, atol=2e-15)

    def test_analytic_1s_radial_function(self) -> None:
        radius = np.array([0.0, 0.1, 0.7, 1.0, 3.0, 10.0])
        expected = 2.0 * np.exp(-radius)
        np.testing.assert_allclose(
            radial_wavefunction(QuantumNumbers(1, 0, 0), radius), expected, rtol=2e-14, atol=1e-15
        )

    def test_analytic_full_1s(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-0.2, 0.3, 0.5]])
        expected = np.exp(-np.linalg.norm(positions, axis=1)) / np.sqrt(np.pi)
        np.testing.assert_allclose(
            wavefunction(QuantumNumbers(1, 0, 0), positions), expected, rtol=2e-14, atol=1e-15
        )

    def test_analytic_full_2p_zero(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-0.2, 0.3, 0.5], [2.0, 1.0, 0.0]])
        radius = np.linalg.norm(positions, axis=1)
        expected = positions[:, 2] * np.exp(-radius / 2.0) / (4.0 * np.sqrt(2.0 * np.pi))
        np.testing.assert_allclose(
            wavefunction(QuantumNumbers(2, 1, 0), positions), expected, rtol=2e-14, atol=1e-15
        )

    def test_analytic_full_complex_2p_plus_one(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-0.2, 0.3, 0.5], [2.0, 1.0, 0.0]])
        radius = np.linalg.norm(positions, axis=1)
        expected = (
            -(positions[:, 0] + 1j * positions[:, 1])
            * np.exp(-radius / 2.0)
            / (8.0 * np.sqrt(np.pi))
        )
        np.testing.assert_allclose(
            wavefunction(QuantumNumbers(2, 1, 1), positions), expected, rtol=2e-14, atol=1e-15
        )

    def test_negative_m_conjugation_and_equal_density(self) -> None:
        positions = np.random.default_rng(21).normal(size=(100, 3)) * 10.0
        positive = wavefunction(QuantumNumbers(5, 3, 3), positions)
        negative = wavefunction(QuantumNumbers(5, 3, -3), positions)
        np.testing.assert_allclose(negative, -positive.conjugate(), rtol=2e-13, atol=1e-15)
        np.testing.assert_allclose(
            density(QuantumNumbers(5, 3, 3), positions),
            density(QuantumNumbers(5, 3, -3), positions),
            atol=1e-17,
        )

    def test_radial_normalization_and_first_two_moments(self) -> None:
        states = [
            QuantumNumbers(1, 0, 0),
            QuantumNumbers(2, 1, 0),
            QuantumNumbers(6, 2, 0),
            QuantumNumbers(12, 0, 0),
            QuantumNumbers(12, 11, 11),
        ]
        for state in states:
            with self.subTest(state=state):
                integration_limit = 4.0 * state.n**2 + 20.0 * state.n
                radius = np.linspace(0.0, integration_limit, 262145)
                probability = radial_probability_density(state, radius)
                normalization = np.trapezoid(probability, radius)
                first = np.trapezoid(radius * probability, radius)
                second = np.trapezoid(radius**2 * probability, radius)
                expected_first = (3 * state.n**2 - state.l * (state.l + 1)) / 2.0
                expected_second = (
                    state.n**2 * (5 * state.n**2 + 1 - 3 * state.l * (state.l + 1)) / 2.0
                )
                self.assertAlmostEqual(normalization, 1.0, delta=2e-10)
                self.assertAlmostEqual(first, expected_first, delta=2e-7)
                self.assertAlmostEqual(second, expected_second, delta=2e-5)

    def test_angular_normalization_and_independent_legendre_density(self) -> None:
        coordinate, weights = np.polynomial.legendre.leggauss(64)
        states = [
            QuantumNumbers(1, 0, 0),
            QuantumNumbers(6, 2, 0),
            QuantumNumbers(6, 3, -2),
            QuantumNumbers(12, 11, 0),
            QuantumNumbers(12, 11, 11),
        ]
        for state in states:
            with self.subTest(state=state):
                angular_values = angular_wavefunction(
                    state, np.arccos(coordinate), np.zeros_like(coordinate)
                )
                independent_density = 2.0 * np.pi * np.abs(angular_values) ** 2
                marginal = polar_probability_density(state, coordinate)
                self.assertAlmostEqual(
                    float(np.dot(weights, independent_density)), 1.0, delta=2e-12
                )
                np.testing.assert_allclose(marginal, independent_density, rtol=2e-11, atol=2e-13)

    def test_known_2s_and_3p_radial_nodes(self) -> None:
        np.testing.assert_allclose(radial_nodes(QuantumNumbers(2, 0, 0)), [2.0], atol=1e-14)
        np.testing.assert_allclose(radial_nodes(QuantumNumbers(3, 1, 0)), [6.0], atol=1e-14)

    def test_default_radial_nodes_independent_polynomial(self) -> None:
        expected = np.array([10.429186528440166, 21.960738680260906, 39.61007479129889])
        np.testing.assert_allclose(radial_nodes(QuantumNumbers()), expected, rtol=2e-13)

    def test_radial_node_counts_zeros_and_sign_changes(self) -> None:
        states = [
            QuantumNumbers(2, 0, 0),
            QuantumNumbers(6, 2, 0),
            QuantumNumbers(12, 0, 0),
            QuantumNumbers(12, 11, 0),
        ]
        for state in states:
            with self.subTest(state=state):
                nodes = radial_nodes(state)
                self.assertEqual(nodes.size, state.n - state.l - 1)
                self.assertTrue(np.all(nodes > 0.0))
                offset = 1e-5 * np.maximum(nodes, 1.0)
                left = radial_wavefunction(state, nodes - offset)
                right = radial_wavefunction(state, nodes + offset)
                self.assertTrue(np.all(left * right < 0.0))
                np.testing.assert_allclose(radial_wavefunction(state, nodes), 0.0, atol=2e-15)

    def test_degree_eleven_roots_obey_independent_viete_identities(self) -> None:
        state = QuantumNumbers(12, 0, 0)
        scaled_roots = radial_nodes(state) * (2.0 / state.n)
        degree = state.radial_node_count
        alpha = 2 * state.l + 1
        expected_sum = degree * (degree + alpha)
        expected_product = math.factorial(degree + alpha) / math.factorial(alpha)
        self.assertAlmostEqual(float(np.sum(scaled_roots)), expected_sum, delta=2e-11)
        self.assertAlmostEqual(float(np.prod(scaled_roots)) / expected_product, 1.0, delta=2e-12)

    def test_angular_node_of_default_d_state(self) -> None:
        theta = np.arccos(1.0 / np.sqrt(3.0))
        positions = np.array(
            [
                [10.0 * np.sin(theta), 0.0, 10.0 * np.cos(theta)],
                [10.0 * np.sin(theta), 0.0, -10.0 * np.cos(theta)],
            ]
        )
        np.testing.assert_allclose(wavefunction(QuantumNumbers(), positions), 0.0, atol=2e-17)

    def test_complex_m_density_is_azimuth_independent(self) -> None:
        phi = np.linspace(0.0, 2.0 * np.pi, 100)
        theta = np.full(phi.shape, 0.8)
        values = angular_wavefunction(QuantumNumbers(6, 3, 2), theta, phi)
        np.testing.assert_allclose(np.abs(values) ** 2, np.abs(values[0]) ** 2, rtol=2e-14)

    def test_invalid_positions_radii_and_angles(self) -> None:
        state = QuantumNumbers()
        with self.assertRaises(ValueError):
            wavefunction(state, np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            wavefunction(state, [[np.nan, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            radial_wavefunction(state, [-1.0, 0.0])
        with self.assertRaises(ValueError):
            radial_wavefunction(state, [np.inf])
        with self.assertRaises(ValueError):
            angular_wavefunction(state, -0.1, 0.0)
        with self.assertRaises(ValueError):
            polar_probability_density(state, 1.01)


if __name__ == "__main__":
    unittest.main()
