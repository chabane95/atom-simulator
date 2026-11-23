"""Central spinless hydrogen physics: orbitals, sampling and Bohmian flow.

The physical wavefunction is a_mu**(-3/2) times the dimensionless value here.
Spherical harmonics are complex Y_lm, including the Condon--Shortley phase.
theta is measured from +z; phi is the azimuth in the xy plane.

References:
https://www.damtp.cam.ac.uk/user/tong/qm/qmhtml/S4.html
https://dlmf.nist.gov/18.9
https://dlmf.nist.gov/14.10
https://physics.nist.gov/cuu/pdf/wall_2022.pdf

Only NumPy and Python's standard library are required.  Radial tail estimates
come from a normalized, finite numerical grid; they are not exact tail bounds.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAX_PRINCIPAL_QUANTUM_NUMBER: int = 12
RADIAL_TAIL_PROBABILITY: float = 1.0e-8
RADIAL_GRID_SIZE: int = 65537
POLAR_GRID_SIZE: int = 131073
GRID_NORMALIZATION_TOLERANCE: float = 2.0e-7

# CODATA 2022 central values.  h is exact in SI; the other measurements have
# uncertainties not propagated here.  Coordinates use a_mu and time uses t_mu.
PLANCK_CONSTANT_JOULE_SECONDS: float = 6.62607015e-34
HBAR_JOULE_SECONDS: float = PLANCK_CONSTANT_JOULE_SECONDS / (2.0 * math.pi)
ELECTRON_MASS_KG: float = 9.1093837139e-31
PROTON_ELECTRON_MASS_RATIO: float = 1836.152673426
BOHR_RADIUS_METERS: float = 5.29177210544e-11
REDUCED_MASS_KG: float = ELECTRON_MASS_KG / (1.0 + 1.0 / PROTON_ELECTRON_MASS_RATIO)
A_MU_METERS: float = BOHR_RADIUS_METERS * (1.0 + 1.0 / PROTON_ELECTRON_MASS_RATIO)
T_MU_SECONDS: float = REDUCED_MASS_KG * A_MU_METERS * A_MU_METERS / HBAR_JOULE_SECONDS


@dataclass(frozen=True)
class QuantumNumbers:
    """One hydrogen energy/angular-momentum eigenstate, without spin."""

    n: int = 6
    l: int = 2
    m: int = 0

    def __post_init__(self) -> None:
        if type(self.n) is not int or type(self.l) is not int or type(self.m) is not int:
            raise TypeError("n, l and m must be integers, not booleans or floats.")
        if self.n < 1 or self.n > MAX_PRINCIPAL_QUANTUM_NUMBER:
            raise ValueError("n must be between 1 and 12.")
        if self.l < 0 or self.l >= self.n:
            raise ValueError("l must satisfy 0 <= l < n.")
        if abs(self.m) > self.l:
            raise ValueError("m must satisfy abs(m) <= l.")

    @property
    def label(self) -> str:
        return f"n={self.n}, l={self.l}, m={self.m}"

    @property
    def radial_node_count(self) -> int:
        return self.n - self.l - 1


@dataclass(frozen=True)
class PointCloud:
    """Samples of one orbital; positions are in a_mu units."""

    positions: NDArray[np.float32]
    bound_radius: float
    metadata: dict[str, Any]

    @property
    def radii(self) -> NDArray[np.float32]:
        return np.asarray(np.linalg.norm(self.positions, axis=1), dtype=np.float32)


def generalized_laguerre(degree: int, alpha: float, x: ArrayLike) -> NDArray[np.float64]:
    """Evaluate L_degree^alpha with its three-term recurrence."""
    if type(degree) is not int or degree < 0:
        raise ValueError("Laguerre degree must be a nonnegative integer.")
    if not math.isfinite(alpha) or alpha <= -1.0:
        raise ValueError("Laguerre alpha must be finite and greater than -1.")
    coordinate: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("Laguerre arguments must be finite.")
    previous: NDArray[np.float64] = np.ones_like(coordinate)
    if degree == 0:
        return previous
    current: NDArray[np.float64] = 1.0 + alpha - coordinate
    for index in range(1, degree):
        first_term: NDArray[np.float64] = (2.0 * index + alpha + 1.0 - coordinate) * current
        second_term: NDArray[np.float64] = (index + alpha) * previous
        next_value: NDArray[np.float64] = (first_term - second_term) / (index + 1.0)
        previous = current
        current = next_value
    return np.asarray(current, dtype=np.float64)


def _legendre_core(degree: int, order: int, coordinate: NDArray[np.float64]) -> NDArray[np.float64]:
    """P_l^m(x) with (1-x*x)^(m/2) factored out, including its sign.

    Factoring this term also allows node checks arbitrarily close to the axis
    without rounding a very small nonzero sin(theta) to a false angular node.
    """
    diagonal_factor: float = ((-1.0) ** order) * math.factorial(2 * order)
    diagonal_factor /= (2**order) * math.factorial(order)
    previous: NDArray[np.float64] = np.full_like(coordinate, diagonal_factor)
    if degree == order:
        return previous
    current: NDArray[np.float64] = (2.0 * order + 1.0) * coordinate * previous
    for current_degree in range(order + 2, degree + 1):
        first_term: NDArray[np.float64] = (2.0 * current_degree - 1.0) * coordinate * current
        second_term: NDArray[np.float64] = (current_degree + order - 1.0) * previous
        next_value: NDArray[np.float64] = (first_term - second_term) / (current_degree - order)
        previous = current
        current = next_value
    return np.asarray(current, dtype=np.float64)


def associated_legendre(degree: int, order: int, x: ArrayLike) -> NDArray[np.float64]:
    """Evaluate P_l^m(x) for integer 0 <= m <= l with Condon--Shortley phase."""
    if type(degree) is not int or type(order) is not int or order < 0 or degree < order:
        raise ValueError("Associated Legendre indices must satisfy 0 <= order <= degree.")
    coordinate: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(coordinate)) or np.any(np.abs(coordinate) > 1.0):
        raise ValueError("Associated Legendre arguments must lie in [-1, 1].")
    sine: NDArray[np.float64] = np.sqrt(np.maximum(0.0, 1.0 - coordinate * coordinate))
    core: NDArray[np.float64] = _legendre_core(degree, order, coordinate)
    return np.asarray(np.power(sine, order) * core, dtype=np.float64)


def radial_wavefunction(state: QuantumNumbers, r: ArrayLike) -> NDArray[np.float64]:
    """Return R_nl(r), with integral r**2 * |R_nl|**2 dr equal to one.

    r is nonnegative and dimensionless: its physical unit is a_mu.  An array
    keeps its shape; scalar input returns a zero-dimensional NumPy array.
    """
    radius: NDArray[np.float64] = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("Radius must be finite and nonnegative.")

    # Beyond rho=10000 the answer is below floating-point underflow for every
    # supported state.  Clipping there avoids the indeterminate product 0*inf.
    safe_radius: NDArray[np.float64] = np.minimum(radius, 5000.0 * state.n)
    rho: NDArray[np.float64] = safe_radius * (2.0 / state.n)
    factorial_ratio: float = math.factorial(state.n - state.l - 1)
    factorial_ratio /= 2 * state.n * math.factorial(state.n + state.l)
    normalization: float = (2.0 / state.n) ** 1.5 * math.sqrt(factorial_ratio)
    polynomial: NDArray[np.float64] = generalized_laguerre(
        state.radial_node_count, 2 * state.l + 1, rho
    )
    exponential: NDArray[np.float64] = np.exp(-0.5 * rho)
    radial_power: NDArray[np.float64] = np.power(rho, state.l)
    result: NDArray[np.float64] = normalization * exponential * radial_power * polynomial
    return np.asarray(result, dtype=np.float64)


def radial_probability_density(state: QuantumNumbers, r: ArrayLike) -> NDArray[np.float64]:
    """Probability density with respect to dr, including the r**2 Jacobian."""
    radius: NDArray[np.float64] = np.asarray(r, dtype=np.float64)
    radial_value: NDArray[np.float64] = radial_wavefunction(state, radius)
    radial_amplitude: NDArray[np.float64] = radius * radial_value
    return np.square(radial_amplitude)


def _laguerre_coefficients(degree: int, alpha: int) -> NDArray[np.float64]:
    """Ascending coefficients: (-1)^k * binomial(degree+alpha,degree-k)/k!."""
    coefficients: NDArray[np.float64] = np.empty(degree + 1, dtype=np.float64)
    for power in range(degree + 1):
        binomial: int = math.comb(degree + alpha, degree - power)
        coefficients[power] = ((-1.0) ** power) * binomial / math.factorial(power)
    return coefficients


@lru_cache(maxsize=128)
def _radial_nodes_cached(n: int, l: int) -> NDArray[np.float64]:
    degree: int = n - l - 1
    alpha: int = 2 * l + 1
    if degree == 0:
        return np.empty(0, dtype=np.float64)
    coefficients: NDArray[np.float64] = _laguerre_coefficients(degree, alpha)
    roots = np.polynomial.polynomial.polyroots(coefficients)
    if np.any(np.abs(roots.imag) > 1.0e-9) or np.any(roots.real <= 0.0):
        raise ArithmeticError("The Laguerre polynomial did not produce positive real roots.")
    refined: NDArray[np.float64] = np.sort(np.asarray(roots.real, dtype=np.float64))
    # Degree <= 11 is small, but power-basis roots can lose precision.  Correct
    # all roots together using the stable recurrence and dL_q^a/dx=-L_(q-1)^(a+1).
    for _iteration in range(4):
        values: NDArray[np.float64] = generalized_laguerre(degree, alpha, refined)
        derivative: NDArray[np.float64] = -generalized_laguerre(degree - 1, alpha + 1, refined)
        refined = refined - values / derivative
    residual_step: NDArray[np.float64] = generalized_laguerre(degree, alpha, refined)
    residual_step /= -generalized_laguerre(degree - 1, alpha + 1, refined)
    if np.any(np.abs(residual_step) > 1.0e-11 * np.maximum(1.0, refined)):
        raise ArithmeticError("Laguerre node refinement did not converge.")
    result: NDArray[np.float64] = 0.5 * n * refined
    result.setflags(write=False)
    return result


def radial_nodes(state: QuantumNumbers) -> NDArray[np.float64]:
    """Positive radial nodes in a_mu units; the origin is not counted."""
    return _radial_nodes_cached(state.n, state.l).copy()


def angular_wavefunction(
    state: QuantumNumbers,
    theta: ArrayLike,
    phi: ArrayLike,
) -> NDArray[np.complex128]:
    """Return normalized complex Y_lm(theta, phi), not a real-orbital mixture."""
    polar_angle: NDArray[np.float64] = np.asarray(theta, dtype=np.float64)
    azimuth: NDArray[np.float64] = np.asarray(phi, dtype=np.float64)
    if not np.all(np.isfinite(polar_angle)) or not np.all(np.isfinite(azimuth)):
        raise ValueError("Angles must be finite.")
    if np.any(polar_angle < 0.0) or np.any(polar_angle > np.pi):
        raise ValueError("theta must lie between 0 and pi.")
    wrapped_azimuth: NDArray[np.float64] = np.mod(azimuth, 2.0 * np.pi)
    order: int = abs(state.m)
    factorial_ratio: float = math.factorial(state.l - order) / math.factorial(state.l + order)
    normalization: float = math.sqrt((2.0 * state.l + 1.0) * factorial_ratio / (4.0 * math.pi))
    sine: NDArray[np.float64] = np.sin(polar_angle)
    sine = np.where((polar_angle == 0.0) | (polar_angle == np.pi), 0.0, sine)
    core: NDArray[np.float64] = _legendre_core(state.l, order, np.cos(polar_angle))
    amplitude: NDArray[np.float64] = normalization * np.power(sine, order) * core
    if state.m < 0:
        amplitude = ((-1.0) ** order) * amplitude
    phase: NDArray[np.complex128] = np.exp(1j * state.m * wrapped_azimuth)
    result: NDArray[np.complex128] = amplitude * phase
    return np.asarray(result, dtype=np.complex128)


def polar_probability_density(
    state: QuantumNumbers,
    cos_theta: ArrayLike,
) -> NDArray[np.float64]:
    """Return p(u)=2*pi*|Y_lm(arccos(u),0)|**2 for u=cos(theta).

    The spherical sin(theta) Jacobian is already included by using u, so no
    extra sin(theta) factor belongs in this one-dimensional density.
    """
    coordinate: NDArray[np.float64] = np.asarray(cos_theta, dtype=np.float64)
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("cos(theta) must be finite.")
    if np.any(coordinate < -1.0) or np.any(coordinate > 1.0):
        raise ValueError("cos(theta) must lie between -1 and 1.")
    order: int = abs(state.m)
    normalization: float = (2.0 * state.l + 1.0) / 2.0
    normalization *= math.factorial(state.l - order) / math.factorial(state.l + order)
    polynomial: NDArray[np.float64] = associated_legendre(state.l, order, coordinate)
    result: NDArray[np.float64] = normalization * np.square(polynomial)
    return np.asarray(result, dtype=np.float64)


def _positions_array(positions: ArrayLike) -> NDArray[np.float64]:
    points: NDArray[np.float64] = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must contain only finite values.")
    return points


def wavefunction(state: QuantumNumbers, positions: ArrayLike) -> NDArray[np.complex128]:
    """Evaluate the full spatial orbital at an N-by-3 array in a_mu units."""
    points: NDArray[np.float64] = _positions_array(positions)
    cylindrical_radius: NDArray[np.float64] = np.hypot(points[:, 0], points[:, 1])
    radius: NDArray[np.float64] = np.hypot(cylindrical_radius, points[:, 2])
    theta: NDArray[np.float64] = np.arctan2(cylindrical_radius, points[:, 2])
    phi: NDArray[np.float64] = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    radial_value: NDArray[np.float64] = radial_wavefunction(state, radius)
    angular_value: NDArray[np.complex128] = angular_wavefunction(state, theta, phi)
    return np.asarray(radial_value * angular_value, dtype=np.complex128)


def density(state: QuantumNumbers, positions: ArrayLike) -> NDArray[np.float64]:
    """Return |psi|**2, a density per dimensionless Cartesian volume."""
    values: NDArray[np.complex128] = wavefunction(state, positions)
    return np.asarray(values.real * values.real + values.imag * values.imag, dtype=np.float64)


@dataclass(frozen=True)
class _ProbabilityTable:
    coordinate: NDArray[np.float64]
    cumulative_probability: NDArray[np.float64]
    primitive_integral: float
    integration_limit: float
    sampling_probability: float
    bound_radius: float
    grid_points: int


def _integrated_probability(
    coordinate: NDArray[np.float64],
    probability_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cumulative trapezoidal integration written as widths, areas and a sum."""
    widths: NDArray[np.float64] = coordinate[1:] - coordinate[:-1]
    average_heights: NDArray[np.float64] = 0.5 * (
        probability_density[1:] + probability_density[:-1]
    )
    interval_areas: NDArray[np.float64] = widths * average_heights
    cumulative: NDArray[np.float64] = np.empty_like(coordinate)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(interval_areas, dtype=np.float64)
    return cumulative


def _probability_table(
    coordinate: NDArray[np.float64],
    probability_density: NDArray[np.float64],
    sampling_probability: float,
) -> _ProbabilityTable:
    cumulative: NDArray[np.float64] = _integrated_probability(coordinate, probability_density)
    primitive_integral: float = float(cumulative[-1])
    if not math.isfinite(primitive_integral):
        raise ArithmeticError("Probability integration produced a nonfinite result.")
    if abs(primitive_integral - 1.0) > GRID_NORMALIZATION_TOLERANCE:
        raise ArithmeticError("The finite probability grid failed its normalization check.")
    cumulative = cumulative / primitive_integral
    cumulative[0] = 0.0
    cumulative[-1] = 1.0

    # Inverse interpolation requires strictly increasing CDF values.  Tiny
    # probabilities can round to repeated values; discard only these flat steps.
    keep: NDArray[np.bool_] = np.empty(cumulative.size, dtype=np.bool_)
    keep[0] = True
    keep[1:] = cumulative[1:] > cumulative[:-1]
    inverse_coordinate: NDArray[np.float64] = coordinate[keep]
    inverse_probability: NDArray[np.float64] = cumulative[keep]
    if inverse_probability.size < 2 or inverse_probability[-1] != 1.0:
        raise ArithmeticError("The cumulative distribution has invalid endpoints.")
    bound_radius: float = float(
        np.interp(sampling_probability, inverse_probability, inverse_coordinate)
    )
    inverse_coordinate.setflags(write=False)
    inverse_probability.setflags(write=False)
    return _ProbabilityTable(
        coordinate=inverse_coordinate,
        cumulative_probability=inverse_probability,
        primitive_integral=primitive_integral,
        integration_limit=float(coordinate[-1]),
        sampling_probability=sampling_probability,
        bound_radius=bound_radius,
        grid_points=int(coordinate.size),
    )


@lru_cache(maxsize=128)
def _radial_probability_table(n: int, l: int) -> _ProbabilityTable:
    state: QuantumNumbers = QuantumNumbers(n=n, l=l, m=0)
    integration_limit: float = float(4 * n * n + 20 * n)
    parameter: NDArray[np.float64] = np.linspace(0.0, 1.0, RADIAL_GRID_SIZE)
    coordinate: NDArray[np.float64] = integration_limit * parameter * parameter
    probability_density: NDArray[np.float64] = radial_probability_density(state, coordinate)
    sampling_probability: float = 1.0 - RADIAL_TAIL_PROBABILITY
    return _probability_table(coordinate, probability_density, sampling_probability)


@lru_cache(maxsize=128)
def _polar_probability_table(l: int, absolute_m: int) -> _ProbabilityTable:
    state: QuantumNumbers = QuantumNumbers(n=l + 1, l=l, m=absolute_m)
    coordinate: NDArray[np.float64] = np.linspace(-1.0, 1.0, POLAR_GRID_SIZE)
    probability_density: NDArray[np.float64] = polar_probability_density(state, coordinate)
    return _probability_table(coordinate, probability_density, 1.0)


def sample_orbital(state: QuantumNumbers, count: int, seed: int = 7) -> PointCloud:
    """Sample |psi|**2 with independent radial, cos(theta) and azimuth variables.

    The deterministic radial bound is the approximate 1-1e-8 quantile of a
    normalized finite-grid CDF.  It is not an analytically certified tail bound.
    Grid normalization is checked before interpolation.  count must be positive.
    """
    if not isinstance(state, QuantumNumbers):
        raise TypeError("state must be a QuantumNumbers instance.")
    if type(count) is not int or type(seed) is not int:
        raise TypeError("count and seed must be integers, not booleans or floats.")
    if count < 1:
        raise ValueError("count must be positive.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")

    radial_table: _ProbabilityTable = _radial_probability_table(state.n, state.l)
    polar_table: _ProbabilityTable = _polar_probability_table(state.l, abs(state.m))
    generator: np.random.Generator = np.random.default_rng(seed)
    uniform_values: NDArray[np.float64] = generator.random((count, 3))
    radial_probabilities: NDArray[np.float64] = (
        uniform_values[:, 0] * radial_table.sampling_probability
    )
    radius: NDArray[np.float64] = np.interp(
        radial_probabilities, radial_table.cumulative_probability, radial_table.coordinate
    )
    cos_theta: NDArray[np.float64] = np.interp(
        uniform_values[:, 1], polar_table.cumulative_probability, polar_table.coordinate
    )
    sin_theta: NDArray[np.float64] = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    phi: NDArray[np.float64] = 2.0 * math.pi * uniform_values[:, 2]
    positions: NDArray[np.float32] = np.empty((count, 3), dtype=np.float32)
    positions[:, 0] = radius * sin_theta * np.cos(phi)
    positions[:, 1] = radius * sin_theta * np.sin(phi)
    positions[:, 2] = radius * cos_theta

    metadata: dict[str, Any] = {
        "state": {"n": state.n, "l": state.l, "m": state.m, "label": state.label},
        "count": count,
        "seed": seed,
        "position_unit": "a_mu",
        "a_mu_meters": A_MU_METERS,
        "t_mu_seconds": T_MU_SECONDS,
        "angular_basis": "normalized complex spherical harmonics",
        "radial_node_count": state.radial_node_count,
        "radial_nodes_a_mu": radial_nodes(state).tolist(),
        "sampler": "independent finite-grid inverse CDF: r, cos(theta), uniform phi",
        "radial_bound_a_mu": radial_table.bound_radius,
        "radial_tail_probability": RADIAL_TAIL_PROBABILITY,
        "radial_tail_probability_is_estimate": True,
        "radial_tail_probability_method": (
            "target of normalized finite-grid CDF; not an exact tail bound"
        ),
        "radial_retained_probability": radial_table.sampling_probability,
        "radial_grid_integral": radial_table.primitive_integral * radial_table.sampling_probability,
        "radial_full_grid_integral": radial_table.primitive_integral,
        "radial_integration_limit_a_mu": radial_table.integration_limit,
        "radial_grid_points": radial_table.grid_points,
        "radial_inverse_cdf_points": int(radial_table.cumulative_probability.size),
        "polar_grid_integral": polar_table.primitive_integral,
        "polar_grid_points": polar_table.grid_points,
        "polar_inverse_cdf_points": int(polar_table.cumulative_probability.size),
        "probability_interpretation": (
            "numerical approximation to |psi|^2 conditioned on the radial cutoff"
        ),
    }
    return PointCloud(
        positions=positions, bound_radius=radial_table.bound_radius, metadata=metadata
    )


def _bohm_angular_velocity_for_points(
    state: QuantumNumbers,
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate the domain once, then evaluate the azimuthal angular speed."""
    if not isinstance(state, QuantumNumbers):
        raise TypeError("state must be a QuantumNumbers instance.")
    cylindrical_radius: NDArray[np.float64] = np.hypot(points[:, 0], points[:, 1])
    if state.m != 0 and np.any(cylindrical_radius == 0.0):
        raise ValueError("Bohmian velocity is undefined on the axis for nonzero m.")
    radius: NDArray[np.float64] = np.hypot(cylindrical_radius, points[:, 2])
    if state.l > 0 and np.any(radius == 0.0):
        raise ValueError(
            "The origin is a node of this orbital; Bohmian velocity is undefined there."
        )
    scaled_radius: NDArray[np.float64] = radius * (2.0 / state.n)
    radial_factor: NDArray[np.float64] = generalized_laguerre(
        state.radial_node_count, 2 * state.l + 1, scaled_radius
    )
    cos_theta: NDArray[np.float64] = np.ones_like(radius)
    np.divide(points[:, 2], radius, out=cos_theta, where=radius > 0.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angular_factor: NDArray[np.float64] = _legendre_core(state.l, abs(state.m), cos_theta)
    if np.any(radial_factor == 0.0) or np.any(angular_factor == 0.0):
        raise ValueError("Bohmian velocity is undefined at an orbital node.")
    if state.m == 0:
        return np.zeros(points.shape[0], dtype=np.float64)

    denominator: NDArray[np.float64] = cylindrical_radius * cylindrical_radius
    if np.any(denominator == 0.0) or not np.all(np.isfinite(denominator)):
        raise ArithmeticError("The cylindrical radius is outside floating-point range.")
    angular_velocity: NDArray[np.float64] = state.m / denominator
    if not np.all(np.isfinite(angular_velocity)):
        raise ArithmeticError("Bohmian angular velocity exceeds floating-point range.")
    return angular_velocity


def bohm_angular_velocity(state: QuantumNumbers, positions: ArrayLike) -> NDArray[np.float64]:
    """Return omega=m/(x*x+y*y) in inverse natural time units.

    Defined only where psi is nonzero.  Axis points for m!=0 and nodes resolved
    exactly by the recurrences are rejected.  No epsilon cutoff or speed cap is
    imposed near nodes.  Floating-point arithmetic cannot classify every exact
    mathematical node.  Empty arrays of shape (0,3) are accepted.
    """
    points: NDArray[np.float64] = _positions_array(positions)
    return _bohm_angular_velocity_for_points(state, points)


def bohm_velocity(state: QuantumNumbers, positions: ArrayLike) -> NDArray[np.float64]:
    """Return m*(-y,x,0)/(x*x+y*y) in a_mu/t_mu units, without spin current.

    The domain and node handling are the same as bohm_angular_velocity.
    """
    points: NDArray[np.float64] = _positions_array(positions)
    angular_velocity: NDArray[np.float64] = _bohm_angular_velocity_for_points(state, points)
    result: NDArray[np.float64] = np.zeros_like(points)
    result[:, 0] = -angular_velocity * points[:, 1]
    result[:, 1] = angular_velocity * points[:, 0]
    return result


def bohm_advance(
    state: QuantumNumbers,
    initial_positions: ArrayLike,
    elapsed_time_units: float,
) -> NDArray[np.float32]:
    """Analytically advance by tau=t/t_mu, returning renderer-ready float32 points.

    Each point rotates by m*tau/(x*x+y*y), with its radius and z unchanged.
    Always pass the original positions and total elapsed time for animation;
    repeatedly feeding rounded float32 results back would accumulate rounding.
    This is exact orbital flow up to floating-point rounding, not Euler stepping.
    For m=0 (or zero elapsed time), valid positions are copied without rotation.
    """
    if isinstance(elapsed_time_units, (bool, np.bool_)) or not np.isscalar(elapsed_time_units):
        raise TypeError("elapsed_time_units must be a real scalar.")
    elapsed: float = float(elapsed_time_units)
    if not math.isfinite(elapsed):
        raise ValueError("elapsed_time_units must be finite.")
    points: NDArray[np.float64] = _positions_array(initial_positions)
    if np.any(np.abs(points) > np.finfo(np.float32).max):
        raise ValueError("Positions exceed the float32 output range.")
    angular_velocity: NDArray[np.float64] = _bohm_angular_velocity_for_points(state, points)
    if state.m == 0 or elapsed == 0.0:
        return np.array(points, dtype=np.float32, copy=True)
    phase: NDArray[np.float64] = angular_velocity * elapsed
    if not np.all(np.isfinite(phase)):
        raise ArithmeticError("The elapsed phase exceeds floating-point range.")
    phase = np.remainder(phase, 2.0 * math.pi)
    cosine: NDArray[np.float64] = np.cos(phase)
    sine: NDArray[np.float64] = np.sin(phase)
    result: NDArray[np.float32] = np.empty(points.shape, dtype=np.float32)
    result[:, 0] = points[:, 0] * cosine - points[:, 1] * sine
    result[:, 1] = points[:, 0] * sine + points[:, 1] * cosine
    result[:, 2] = points[:, 2]
    return result
