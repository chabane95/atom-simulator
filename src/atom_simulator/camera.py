"""Small, explicit camera calculations; no scene-graph dependency."""

import math

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]


def view_matrix(azimuth: float, elevation: float, distance: float) -> FloatMatrix:
    """Look at the origin with z pointing up; angles are in degrees."""
    azimuth_radians = math.radians(azimuth)
    elevation_radians = math.radians(elevation)

    eye_x = distance * math.cos(elevation_radians) * math.cos(azimuth_radians)
    eye_y = distance * math.cos(elevation_radians) * math.sin(azimuth_radians)
    eye_z = distance * math.sin(elevation_radians)
    eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)

    forward = -eye / np.linalg.norm(eye)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def projection_matrix(aspect: float, field_of_view: float = 32.0) -> FloatMatrix:
    """Perspective projection for the dimensionless display volume."""
    near = 0.1
    far = 30.0
    focal_length = 1.0 / math.tan(math.radians(field_of_view) * 0.5)

    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = focal_length / aspect
    matrix[1, 1] = focal_length
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -(2.0 * far * near) / (far - near)
    matrix[3, 2] = -1.0
    return matrix


def smoothstep(value: float) -> float:
    """Cubic easing with zero velocity at both ends."""
    bounded = min(1.0, max(0.0, value))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def preview_pose(seconds: float, duration: float) -> tuple[float, float, float]:
    """A quiet camera arc, a progressive cutaway, then a still final view."""
    progress = seconds / duration
    orbit = smoothstep((progress - 0.08) / 0.80)
    cutaway = smoothstep((progress - 0.24) / 0.38)
    azimuth = 24.0 + 38.0 * orbit
    elevation = 13.0 + 12.0 * math.sin(math.pi * orbit)
    cut_angle = 110.0 * cutaway
    return azimuth, elevation, cut_angle
