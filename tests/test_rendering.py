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

from atom_simulator.camera import flow_pose, preview_pose, projection_matrix, view_matrix
from atom_simulator.cli import build_parser, main, validate_arguments
from atom_simulator.export import render_still, render_video
from atom_simulator.physics import PointCloud, QuantumNumbers, bohm_advance
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

    def test_flow_camera_holds_while_particle_time_can_continue(self) -> None:
        times = [0.0, 2.5, 5.0, 8.0, 10.5, 13.0, 16.0, 19.0, 22.0]
        poses = np.array([flow_pose(time, 22.0) for time in times]).reshape(3, 3, 2)
        self.assertTrue(np.all(np.isfinite(poses)))
        np.testing.assert_allclose(np.diff(poses, axis=1), 0.0, atol=1e-12)
        distances_between_views = np.linalg.norm(np.diff(poses[:, 0], axis=0), axis=1)
        self.assertTrue(np.all(distances_between_views > 1.0))
        slower = np.array([flow_pose(time * 2.0, 44.0) for time in times])
        np.testing.assert_allclose(slower, poses.reshape(9, 2), atol=1e-12)


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
            ["view", "--time-rate", "nan"],
            ["view", "--time-rate", "inf"],
            ["view", "--time-rate", "-1"],
            ["view", "--time-rate", "0"],
            ["render", "--time-rate", "nan"],
            ["render", "--time-rate", "inf"],
            ["render", "--time-rate", "-1"],
            ["render", "--time-rate", "0"],
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

    def test_invalid_flow_rate_does_not_start_sampling(self) -> None:
        with (
            patch("sys.argv", ["atom-simulator", "render", "--time-rate", "nan"]),
            patch("atom_simulator.cli.sample_orbital") as sampler,
            patch("atom_simulator.export.moderngl.create_standalone_context") as context_factory,
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(main(), 1)
        sampler.assert_not_called()
        context_factory.assert_not_called()
        self.assertIn("Time rate", errors.getvalue())


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

    def test_reversing_flow_preserves_positions_and_retraces_the_path(self) -> None:
        state = QuantumNumbers(2, 1, 1)
        positions = np.array(
            [[1.5, 0.0, 0.4], [0.0, 2.0, 0.6], [-3.5, 1.0, -0.3]], dtype=np.float32
        )
        cloud = PointCloud(positions, 5.0, {"seed": 7, "count": 3})
        viewer = OrbitalViewer(state, cloud, RenderSettings(), seed=7)
        viewer.physics_time = 2.0
        before = bohm_advance(state, positions, viewer.physics_time)
        earlier = bohm_advance(state, positions, viewer.physics_time - 0.125)
        with patch("atom_simulator.viewer.sample_orbital") as sampler:
            viewer.on_key(None, glfw.KEY_F, 0, glfw.PRESS, 0)
        sampler.assert_not_called()
        self.assertEqual(viewer.state.m, -1)
        after = bohm_advance(viewer.state, viewer.cloud.positions, viewer.physics_time)
        retraced = bohm_advance(viewer.state, viewer.cloud.positions, viewer.physics_time + 0.125)
        np.testing.assert_allclose(after, before, atol=2e-6)
        np.testing.assert_allclose(retraced, earlier, atol=2e-6)
        self.assertTrue(viewer.needs_rebuild)

    def test_paused_viewer_keeps_physical_time_during_frame_rendering(self) -> None:
        viewer = self.viewer
        viewer.flow_running = False
        viewer.physics_time = 5.0
        viewer.renderer = Mock()
        viewer.context = Mock()
        with (
            patch("atom_simulator.viewer.time.perf_counter", side_effect=[10.0, 10.1]),
            patch("atom_simulator.viewer.glfw.window_should_close", side_effect=[False, True]),
            patch("atom_simulator.viewer.glfw.poll_events"),
            patch("atom_simulator.viewer.glfw.swap_buffers"),
        ):
            viewer.draw_loop()
        self.assertEqual(viewer.physics_time, 5.0)
        viewer.renderer.update_flow.assert_called_once_with(5.0)


class ExportFailureTests(unittest.TestCase):
    def test_invalid_camera_distances_are_rejected_before_gpu_creation(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "atom_simulator.export.moderngl.create_standalone_context",
                side_effect=AssertionError("Invalid distance reached GPU creation"),
            ) as factory,
        ):
            for distance in (0.0, -1.0, 1.499, 10.001, float("nan"), float("inf")):
                with self.subTest(distance=distance), self.assertRaises(ValueError):
                    render_video(
                        QuantumNumbers(2, 1, 1),
                        Mock(),
                        RenderSettings(),
                        Path(directory) / "invalid.mp4",
                        duration=1.0,
                        fps=30,
                        camera_distance=distance,
                    )
        factory.assert_not_called()

    def test_valid_camera_distance_boundaries_reach_context_creation(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "atom_simulator.export.moderngl.create_standalone_context",
                side_effect=RuntimeError("validation accepted camera distance"),
            ) as factory,
        ):
            for distance in (1.5, 10.0):
                with (
                    self.subTest(distance=distance),
                    self.assertRaisesRegex(RuntimeError, "validation accepted camera distance"),
                ):
                    render_video(
                        QuantumNumbers(2, 1, 1),
                        Mock(),
                        RenderSettings(),
                        Path(directory) / "valid.mp4",
                        duration=1.0,
                        fps=30,
                        camera_distance=distance,
                    )
        self.assertEqual(factory.call_count, 2)

    def test_invalid_flow_rates_are_rejected_before_gpu_creation(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "atom_simulator.export.moderngl.create_standalone_context",
                side_effect=AssertionError("Invalid input reached GPU creation"),
            ) as factory,
        ):
            for rate in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(rate=rate), self.assertRaises(ValueError):
                    render_video(
                        QuantumNumbers(2, 1, 1),
                        Mock(),
                        RenderSettings(),
                        Path(directory) / "invalid.mp4",
                        duration=1.0,
                        fps=30,
                        bohm_flow=True,
                        time_rate=rate,
                    )
        factory.assert_not_called()

    def test_still_export_works_without_optional_video_encoder(self) -> None:
        context = Mock()
        renderer = Mock()
        renderer.read_image.return_value = Image.new("RGB", (320, 180))
        settings = RenderSettings(width=320, height=180)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("atom_simulator.export.av", None),
            patch("atom_simulator.export.moderngl.create_standalone_context", return_value=context),
            patch("atom_simulator.export.OrbitalRenderer", return_value=renderer),
            patch("atom_simulator.export.write_metadata"),
        ):
            output = Path(directory) / "still.png"
            render_still(QuantumNumbers(1, 0, 0), Mock(), settings, output, supersample=1)
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (320, 180))
                self.assertEqual(saved.format, "PNG")
        renderer.close.assert_called_once()
        context.release.assert_called_once()

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
