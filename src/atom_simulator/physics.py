"""Normalized, nonrelativistic hydrogen eigenfunctions in units of a_mu.

The physical wavefunction is a_mu**(-3/2) times the dimensionless value here.
Spherical harmonics are complex Y_lm, including the Condon--Shortley phase.
theta is measured from +z; phi is the azimuth in the xy plane.

References:
https://www.damtp.cam.ac.uk/user/tong/qm/qmhtml/S4.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y.html
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_genlaguerre, gammaln, lpmv, roots_genlaguerre, sph_harm_y

MAX_PRINCIPAL_QUANTUM_NUMBER: int = 12


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
    log_factor: float = 1.5 * np.log(2.0 / state.n)
    log_factor += 0.5 * (
        gammaln(state.n - state.l) - np.log(2.0 * state.n) - gammaln(state.n + state.l + 1)
    )
    normalization: float = float(np.exp(log_factor))
    polynomial: NDArray[np.float64] = eval_genlaguerre(
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


def radial_nodes(state: QuantumNumbers) -> NDArray[np.float64]:
    """Positive radial nodes in a_mu units; the origin is not counted."""
    if state.radial_node_count == 0:
        return np.empty(0, dtype=np.float64)
    scaled_roots, unused_weights = roots_genlaguerre(state.radial_node_count, 2 * state.l + 1)
    del unused_weights
    return np.asarray(0.5 * state.n * scaled_roots, dtype=np.float64)


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
    result: NDArray[np.complex128] = sph_harm_y(state.l, state.m, polar_angle, wrapped_azimuth)
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
    log_normalization: float = np.log((2.0 * state.l + 1.0) / 2.0)
    log_normalization += gammaln(state.l - order + 1) - gammaln(state.l + order + 1)
    polynomial: NDArray[np.float64] = lpmv(order, state.l, coordinate)
    result: NDArray[np.float64] = np.exp(log_normalization) * np.square(polynomial)
    return np.asarray(result, dtype=np.float64)


def wavefunction(state: QuantumNumbers, positions: ArrayLike) -> NDArray[np.complex128]:
    """Evaluate the full spatial orbital at an N-by-3 array in a_mu units."""
    points: NDArray[np.float64] = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must contain only finite values.")
    cylindrical_radius: NDArray[np.float64] = np.hypot(points[:, 0], points[:, 1])
    radius: NDArray[np.float64] = np.hypot(cylindrical_radius, points[:, 2])
    cos_theta: NDArray[np.float64] = np.ones(radius.shape, dtype=np.float64)
    np.divide(points[:, 2], radius, out=cos_theta, where=radius > 0.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta: NDArray[np.float64] = np.arccos(cos_theta)
    phi: NDArray[np.float64] = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    radial_value: NDArray[np.float64] = radial_wavefunction(state, radius)
    angular_value: NDArray[np.complex128] = angular_wavefunction(state, theta, phi)
    return np.asarray(radial_value * angular_value, dtype=np.complex128)


def density(state: QuantumNumbers, positions: ArrayLike) -> NDArray[np.float64]:
    """Return |psi|**2, a density per dimensionless Cartesian volume."""
    values: NDArray[np.complex128] = wavefunction(state, positions)
    return np.asarray(values.real * values.real + values.imag * values.imag, dtype=np.float64)
