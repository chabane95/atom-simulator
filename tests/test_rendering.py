"""Independent camera, input, control, and export-failure checks without a GPU."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import glfw
import numpy as np
from PIL import Image

from atom_simulator.camera import preview_pose, projection_matrix, view_matrix
from atom_simulator.cli import build_parser, validate_arguments
from atom_simulator.export import render_still, render_video
from atom_simulator.physics import QuantumNumbers
from atom_simulator.renderer import RenderSettings
from atom_simulator.viewer import OrbitalViewer


class CameraTests(unittest.TestCase):
    def test_look_at_known_direction(self) -> None:
        """At +x, screen right is +y, screen up is +z, and the origin is in front."""
        transform = view_matrix(0.0, 0.0, 4.0)
        point = np.array([1.0, 2.0, 3.0, 1.0])
        np.testing.assert_allclose(transform @ point, [2.0, 3.0, -3.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(transform @ [4.0, 0.0, 0.0, 1.0], [0, 0, 0, 1])

    def test_rotation_preserves_geometry_at_extreme_viewer_elevations(self) -> None:
        poses = [(0.0, -85.0), (47.0, 0.0), (269.0, 85.0)]
        for azimuth, elevation in poses:
            transform = view_matrix(azimuth, elevation, 4.3)
            rotation = transform[:3, :3]
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=3e-7)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)
            np.testing.assert_allclose(transform @ [0, 0, 0, 1], [0, 0, -4.3, 1], atol=5e-7)

    def test_projection_does_not_stretch_equal_world_lengths(self) -> None:
        """One horizontal and one vertical unit must occupy the same pixel length."""
        width, height = 1920, 1080
        transform = projection_matrix(width / height)
        points = np.array([[1, 0, -4, 1], [0, 1, -4, 1]], dtype=float).T
        clip = transform @ points
        normalized = clip[:2] / clip[3]
        horizontal_pixels = normalized[0, 0] * width / 2
        vertical_pixels = normalized[1, 1] * height / 2
        self.assertAlmostEqual(horizontal_pixels, vertical_pixels, places=4)

    def test_preview_opens_cut_monotonically_and_has_a_final_hold(self) -> None:
        duration = 20.0
        poses = np.array(
            [preview_pose(float(time), duration) for time in np.linspace(0, duration, 101)]
        )
        self.assertTrue(np.all(np.isfinite(poses)))
        self.assertTrue(np.all(np.diff(poses[:, 0]) >= -1e-10))
        self.assertTrue(np.all(np.diff(poses[:, 2]) >= -1e-10))
        self.assertEqual(float(poses[0, 2]), 0.0)
        np.testing.assert_allclose(
            poses[90:], np.broadcast_to(poses[-1], poses[90:].shape), atol=1e-10
        )


class CommandLineTests(unittest.TestCase):
    def test_rejects_invalid_inputs_without_sampling_or_gpu(self) -> None:
        cases = [
            ["render", "--output", "collision.json"],
            ["render", "--output", "wrong.png"],
            ["image", "--output", "wrong.mp4"],
            ["render", "--duration", "nan"],
            ["render", "--duration", "inf"],
            ["render", "--duration", "0"],
            ["render", "--duration", "601"],
            ["render", "--fps", "0"],
            ["render", "--fps", "121"],
            ["render", "--width", "1921"],
            ["render", "--height", "1081"],
            ["image", "--points", "999"],
            ["image", "--points", "2000001"],
            ["image", "--seed", "-1"],
            ["view", "--test-frames", "-1"],
        ]
        parser = build_parser()
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_arguments(parser.parse_args(arguments))

    def test_accepts_valid_boundaries_and_static_odd_dimensions(self) -> None:
        cases = [
            ["render", "--duration", "1", "--fps", "1", "--points", "1000"],
            ["render", "--duration", "600", "--fps", "120", "--points", "2000000"],
            ["image", "--width", "1921", "--height", "1081", "--seed", "0"],
            ["view", "--test-frames", "0"],
        ]
        parser = build_parser()
        for arguments in cases:
            validate_arguments(parser.parse_args(arguments))


class ViewerControlTests(unittest.TestCase):
    def setUp(self) -> None:
        cloud = Mock()
        cloud.positions = np.zeros((1000, 3), dtype=np.float32)
        self.viewer = OrbitalViewer(QuantumNumbers(1, 0, 0), cloud, RenderSettings(), seed=7)

    def test_cut_and_background_do_not_resample_the_state(self) -> None:
        cloud = self.viewer.cloud
        with patch("atom_simulator.viewer.sample_orbital") as sampler:
            self.viewer.on_key(None, glfw.KEY_C, 0, glfw.PRESS, 0)
            self.assertEqual(self.viewer.cut_angle, 0.0)
            self.viewer.on_key(None, glfw.KEY_C, 0, glfw.PRESS, 0)
            self.assertGreater(self.viewer.cut_angle, 0.0)
            self.viewer.on_key(None, glfw.KEY_B, 0, glfw.PRESS, 0)
            self.assertEqual(self.viewer.settings.background, "black")
            self.assertTrue(self.viewer.needs_rebuild)
            self.assertIs(self.viewer.cloud, cloud)
            sampler.assert_not_called()

    def test_mouse_and_zoom_remain_in_valid_camera_domain(self) -> None:
        self.viewer.dragging = True
        self.viewer.on_cursor(None, 1000.0, 10000.0)
        self.assertLessEqual(self.viewer.elevation, 85.0)
        self.viewer.on_cursor(None, 1000.0, -10000.0)
        self.assertGreaterEqual(self.viewer.elevation, -85.0)
        self.viewer.on_scroll(None, 0.0, 100.0)
        self.assertGreaterEqual(self.viewer.distance, 2.5)
        self.viewer.on_scroll(None, 0.0, -100.0)
        self.assertLessEqual(self.viewer.distance, 10.0)


class ExportFailureTests(unittest.TestCase):
    def test_encoder_failure_closes_resources_and_preserves_existing_output(self) -> None:
        context = Mock()
        context.info = {"GL_RENDERER": "test renderer"}
        renderer = Mock()
        renderer.read_image.return_value = Image.new("RGB", (320, 180))
        container = Mock()
        settings = RenderSettings(width=320, height=180)
        state = QuantumNumbers(1, 0, 0)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("atom_simulator.export.moderngl.create_standalone_context", return_value=context),
            patch("atom_simulator.export.OrbitalRenderer", return_value=renderer),
            patch("atom_simulator.export.av.open", return_value=container),
            patch("atom_simulator.export.encode_frame", side_effect=RuntimeError("encoder failed")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            output = Path(directory) / "previous.mp4"
            output.write_bytes(b"previous successful export")
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                render_video(state, Mock(), settings, output, duration=1.0, fps=1)
            self.assertEqual(output.read_bytes(), b"previous successful export")
        container.close.assert_called_once()
        renderer.close.assert_called_once()
        context.release.assert_called_once()

    def test_renderer_initialization_failure_releases_standalone_context(self) -> None:
        context = Mock()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("atom_simulator.export.moderngl.create_standalone_context", return_value=context),
            patch(
                "atom_simulator.export.OrbitalRenderer", side_effect=RuntimeError("shader failed")
            ),
            self.assertRaisesRegex(RuntimeError, "shader failed"),
        ):
            render_still(
                QuantumNumbers(1, 0, 0), Mock(), RenderSettings(), Path(directory) / "still.png"
            )
        context.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
