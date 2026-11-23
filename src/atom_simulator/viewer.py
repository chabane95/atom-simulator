"""Interactive desktop viewer; GLFW callbacks only change explicit viewer state."""

import math
import time
from dataclasses import replace

import glfw
import moderngl

from atom_simulator.config import FLOW_CASES
from atom_simulator.physics import PointCloud, QuantumNumbers, bohm_advance, sample_orbital
from atom_simulator.renderer import OrbitalRenderer, RenderSettings


class OrbitalViewer:
    def __init__(
        self,
        state: QuantumNumbers,
        cloud: PointCloud,
        settings: RenderSettings,
        seed: int,
        test_frames: int = 0,
        bohm_flow: bool = True,
        time_rate: float = 4.0,
    ) -> None:
        if not math.isfinite(time_rate) or time_rate <= 0.0:
            raise ValueError("Time rate must be finite and positive.")
        self.state = state
        self.cloud = cloud
        self.settings = settings
        self.seed = seed
        self.test_frames = test_frames
        self.bohm_flow = bohm_flow
        self.flow_running = bohm_flow
        self.time_rate = time_rate
        self.physics_time = 0.0
        self.azimuth = 42.0
        self.elevation = 20.0
        self.distance = 3.0
        self.minimum_distance = 2.5
        self.cut_angle = 110.0
        self.rotating = False
        self.dragging = False
        self.last_cursor = (0.0, 0.0)
        self.needs_rebuild = False
        self.renderer: OrbitalRenderer | None = None
        self.context: moderngl.Context | None = None
        self.window = None

    def on_key(self, window: object, key: int, scancode: int, action: int, modifiers: int) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.flow_running = not self.flow_running
        elif key == glfw.KEY_A:
            self.rotating = not self.rotating
        elif key == glfw.KEY_F:
            current_positions = self.cloud.positions
            if self.bohm_flow:
                current_positions = bohm_advance(self.state, current_positions, self.physics_time)
            self.state = QuantumNumbers(self.state.n, self.state.l, -self.state.m)
            metadata = self.cloud.metadata.copy()
            metadata["state"] = {
                "n": self.state.n,
                "l": self.state.l,
                "m": self.state.m,
                "label": self.state.label,
            }
            self.cloud = PointCloud(current_positions, self.cloud.bound_radius, metadata)
            self.physics_time = 0.0
            self.needs_rebuild = True
        elif key == glfw.KEY_C:
            self.cut_angle = 0.0 if self.cut_angle > 0.0 else 110.0
        elif key == glfw.KEY_B:
            background = "black" if self.settings.background == "light" else "light"
            self.settings = replace(self.settings, background=background)
            self.needs_rebuild = True
        elif key == glfw.KEY_R:
            self.azimuth = 42.0
            self.elevation = 20.0
            self.distance = 3.0
            self.minimum_distance = 2.5
            self.physics_time = 0.0
        elif glfw.KEY_1 <= key <= glfw.KEY_4:
            case_index = key - glfw.KEY_1
            if case_index >= len(FLOW_CASES):
                return
            case = FLOW_CASES[case_index]
            self.state = case.state
            self.cloud = sample_orbital(self.state, len(self.cloud.positions), self.seed)
            self.time_rate = case.time_rate
            self.distance = case.camera_distance
            self.minimum_distance = min(2.5, case.camera_distance)
            self.physics_time = 0.0
            self.needs_rebuild = True

    def on_mouse_button(self, window: object, button: int, action: int, modifiers: int) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        self.dragging = action == glfw.PRESS
        self.last_cursor = glfw.get_cursor_pos(window)

    def on_cursor(self, window: object, x: float, y: float) -> None:
        if self.dragging:
            self.azimuth -= 0.25 * (x - self.last_cursor[0])
            new_elevation = self.elevation + 0.25 * (y - self.last_cursor[1])
            self.elevation = min(85.0, max(-85.0, new_elevation))
        self.last_cursor = (x, y)

    def on_scroll(self, window: object, x_offset: float, y_offset: float) -> None:
        new_distance = self.distance * math.exp(-0.08 * y_offset)
        self.distance = min(10.0, max(self.minimum_distance, new_distance))

    def rebuild_renderer(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        if self.context is None:
            raise RuntimeError("Cannot render without a current OpenGL context.")
        self.renderer = OrbitalRenderer(self.context, self.cloud, self.state, self.settings)
        if self.bohm_flow:
            self.renderer.enable_flow(self.time_rate)
        self.needs_rebuild = False

    def run(self) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize. A desktop and OpenGL 3.3 are required.")
        try:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
            if self.test_frames > 0:
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            title = "atom-simulator | Space: flow | A: camera | C: cut | F: reverse | 1-4: cases"
            self.window = glfw.create_window(
                self.settings.width, self.settings.height, title, None, None
            )
            if not self.window:
                raise RuntimeError(
                    "Could not create an OpenGL 3.3 window. Check the graphics driver."
                )
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.context = moderngl.create_context(require=330)
            glfw.set_key_callback(self.window, self.on_key)
            glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self.on_cursor)
            glfw.set_scroll_callback(self.window, self.on_scroll)
            self.rebuild_renderer()
            self.draw_loop()
        finally:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            if self.context is not None:
                self.context.release()
            if self.window is not None:
                glfw.destroy_window(self.window)
            glfw.terminate()

    def draw_loop(self) -> None:
        last_time = time.perf_counter()
        frames = 0
        while not glfw.window_should_close(self.window):
            now = time.perf_counter()
            elapsed = min(0.1, now - last_time)
            last_time = now
            glfw.poll_events()
            if self.needs_rebuild:
                self.rebuild_renderer()
            if self.rotating:
                self.azimuth += elapsed * 4.0
            if self.renderer is None or self.context is None:
                raise RuntimeError("Renderer lost its OpenGL context.")
            if self.flow_running:
                self.physics_time += elapsed * self.time_rate
            if self.bohm_flow:
                self.renderer.update_flow(self.physics_time)
            self.renderer.draw(self.azimuth, self.elevation, self.distance, self.cut_angle)
            self.context.copy_framebuffer(self.context.screen, self.renderer.output_buffer)
            glfw.swap_buffers(self.window)
            frames += 1
            if self.test_frames > 0 and frames >= self.test_frames:
                glfw.set_window_should_close(self.window, True)
