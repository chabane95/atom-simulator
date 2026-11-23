"""Independent inverse-CDF sampling of normalized hydrogen orbital densities.

No Cartesian volume mesh or rejection cloud is used.  The factorized measure
is p(r) dr * p(u) du * dphi/(2*pi), where u=cos(theta).  The infinite radial
tail is truncated at a deterministic 1-1e-8 probability quantile.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq
from scipy.special import roots_jacobi

from .physics import (
    QuantumNumbers,
    polar_probability_density,
    radial_nodes,
    radial_probability_density,
)

RADIAL_TAIL_PROBABILITY: float = 1.0e-8
RADIAL_GRID_SIZE: int = 65537
POLAR_GRID_SIZE: int = 131073


@dataclass(frozen=True)
class PointCloud:
    """Samples of one orbital, with positions expressed in units of a_mu."""

    positions: NDArray[np.float32]
    bound_radius: float
    metadata: dict[str, Any]

    @property
    def radii(self) -> NDArray[np.float32]:
        return np.asarray(np.linalg.norm(self.positions, axis=1), dtype=np.float32)


@dataclass(frozen=True)
class _TabulatedDistribution:
    coordinate: NDArray[np.float64]
    probability: NDArray[np.float64]
    integral: float
    original_grid_points: int
    bound: float
    tail_probability: float


def _radial_integrand(radius: float, state: QuantumNumbers) -> float:
    return float(radial_probability_density(state, radius))


def _radial_tail(radius: float, state: QuantumNumbers) -> float:
    probability, error_estimate = quad(
        _radial_integrand, radius, np.inf, args=(state,), epsabs=2.0e-14, epsrel=2.0e-10, limit=200
    )
    if not np.isfinite(probability) or error_estimate > 1.0e-9:
        raise ArithmeticError("Radial tail integration did not reach the required accuracy.")
    return float(max(0.0, probability))


def _tail_difference(radius: float, state: QuantumNumbers) -> float:
    return _radial_tail(radius, state) - RADIAL_TAIL_PROBABILITY


def _make_cdf(
    coordinate: NDArray[np.float64],
    probability_density: NDArray[np.float64],
    bound: float,
    tail_probability: float,
) -> _TabulatedDistribution:
    cumulative: NDArray[np.float64] = cumulative_trapezoid(
        probability_density, coordinate, initial=0.0
    )
    integral: float = float(cumulative[-1])
    expected_integral: float = 1.0 - tail_probability
    if not np.isfinite(integral) or abs(integral - expected_integral) > 2.0e-7:
        raise ArithmeticError("The probability grid did not reproduce the required normalization.")
    cumulative = cumulative / integral
    cumulative[0] = 0.0
    cumulative[-1] = 1.0

    # A trapezoidal CDF is nondecreasing.  Flat floating-point steps can occur
    # at nodes or very small tails; inverse interpolation needs distinct values.
    keep: NDArray[np.bool_] = np.concatenate((np.array([True]), np.diff(cumulative) > 0.0))
    inverse_coordinate: NDArray[np.float64] = np.asarray(coordinate[keep], dtype=np.float64)
    inverse_probability: NDArray[np.float64] = np.asarray(cumulative[keep], dtype=np.float64)
    if inverse_probability.size < 2 or inverse_probability[-1] != 1.0:
        raise ArithmeticError("The inverse probability table has invalid endpoints.")
    inverse_coordinate.setflags(write=False)
    inverse_probability.setflags(write=False)
    return _TabulatedDistribution(
        coordinate=inverse_coordinate,
        probability=inverse_probability,
        integral=integral,
        original_grid_points=int(coordinate.size),
        bound=float(bound),
        tail_probability=float(tail_probability),
    )


@lru_cache(maxsize=128)
def _radial_distribution(n: int, l: int) -> _TabulatedDistribution:
    state: QuantumNumbers = QuantumNumbers(n=n, l=l, m=0)
    upper_bracket: float = float(4 * n * n + 20 * n)
    if _tail_difference(upper_bracket, state) >= 0.0:
        raise ArithmeticError("The radial quantile was not bracketed.")
    bound: float = float(
        brentq(_tail_difference, 0.0, upper_bracket, args=(state,), xtol=1.0e-10, rtol=1.0e-12)
    )
    tail: float = _radial_tail(bound, state)
    parameter: NDArray[np.float64] = np.linspace(0.0, 1.0, RADIAL_GRID_SIZE)
    regular_grid: NDArray[np.float64] = bound * parameter * parameter
    nodes: NDArray[np.float64] = radial_nodes(state)
    coordinate: NDArray[np.float64] = np.unique(np.concatenate((regular_grid, nodes)))
    probability_density: NDArray[np.float64] = radial_probability_density(state, coordinate)
    return _make_cdf(coordinate, probability_density, bound, tail)


@lru_cache(maxsize=128)
def _polar_distribution(l: int, absolute_m: int) -> _TabulatedDistribution:
    state: QuantumNumbers = QuantumNumbers(n=l + 1, l=l, m=absolute_m)
    regular_grid: NDArray[np.float64] = np.linspace(-1.0, 1.0, POLAR_GRID_SIZE)
    node_count: int = l - absolute_m
    nodes: NDArray[np.float64] = np.empty(0, dtype=np.float64)
    if node_count > 0:
        nodes, unused_weights = roots_jacobi(node_count, absolute_m, absolute_m)
        del unused_weights
    coordinate: NDArray[np.float64] = np.unique(np.concatenate((regular_grid, nodes)))
    probability_density: NDArray[np.float64] = polar_probability_density(state, coordinate)
    return _make_cdf(coordinate, probability_density, 1.0, 0.0)


def sample_orbital(state: QuantumNumbers, count: int, seed: int = 7) -> PointCloud:
    """Return reproducible samples of |psi_nlm|**2, conditioned on the radial cutoff.

    count must be positive.  The cutoff depends on n and l, never on the seed
    or the number of requested samples.  Positive/negative m have equal density.
    """
    if not isinstance(state, QuantumNumbers):
        raise TypeError("state must be a QuantumNumbers instance.")
    if type(count) is not int or type(seed) is not int:
        raise TypeError("count and seed must be integers, not booleans or floats.")
    if count < 1:
        raise ValueError("count must be positive.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")

    radial_table: _TabulatedDistribution = _radial_distribution(state.n, state.l)
    polar_table: _TabulatedDistribution = _polar_distribution(state.l, abs(state.m))
    generator: np.random.Generator = np.random.default_rng(seed)
    uniform_values: NDArray[np.float64] = generator.random((count, 3))
    radius: NDArray[np.float64] = np.interp(
        uniform_values[:, 0], radial_table.probability, radial_table.coordinate
    )
    cos_theta: NDArray[np.float64] = np.interp(
        uniform_values[:, 1], polar_table.probability, polar_table.coordinate
    )
    sin_theta: NDArray[np.float64] = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    phi: NDArray[np.float64] = 2.0 * np.pi * uniform_values[:, 2]

    positions: NDArray[np.float32] = np.empty((count, 3), dtype=np.float32)
    positions[:, 0] = radius * sin_theta * np.cos(phi)
    positions[:, 1] = radius * sin_theta * np.sin(phi)
    positions[:, 2] = radius * cos_theta
    metadata: dict[str, Any] = {
        "state": {"n": state.n, "l": state.l, "m": state.m, "label": state.label},
        "count": count,
        "seed": seed,
        "position_unit": "a_mu",
        "angular_basis": "normalized complex spherical harmonics",
        "radial_node_count": state.radial_node_count,
        "radial_nodes_a_mu": radial_nodes(state).tolist(),
        "sampler": "independent inverse CDF: r, cos(theta), uniform phi",
        "radial_bound_a_mu": radial_table.bound,
        "radial_tail_probability": radial_table.tail_probability,
        "radial_retained_probability": 1.0 - radial_table.tail_probability,
        "radial_grid_integral": radial_table.integral,
        "radial_grid_points": radial_table.original_grid_points,
        "radial_inverse_cdf_points": int(radial_table.probability.size),
        "polar_grid_integral": polar_table.integral,
        "polar_grid_points": polar_table.original_grid_points,
        "polar_inverse_cdf_points": int(polar_table.probability.size),
        "probability_interpretation": "|psi|^2 conditioned on the stated radial cutoff",
    }
    return PointCloud(positions=positions, bound_radius=radial_table.bound, metadata=metadata)
