"""OpenGL renderer with explicit resource ownership and no per-particle Python loop."""

import math
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from atom_simulator.camera import projection_matrix, view_matrix
from atom_simulator.physics import QuantumNumbers
from atom_simulator.sampling import PointCloud
from atom_simulator.shaders import (
    OVERLAY_FRAGMENT,
    PARTICLE_FRAGMENT,
    PARTICLE_VERTEX,
    SCREEN_VERTEX,
    SHADE_FRAGMENT,
)


@dataclass(frozen=True)
class RenderSettings:
    width: int = 1920
    height: int = 1080
    bead_radius: float = 0.0035
    background: str = "light"
    labels: bool = True

    def __post_init__(self) -> None:
        if self.width < 320 or self.height < 180:
            raise ValueError("Resolution must be at least 320 x 180.")
        if not 0.001 <= self.bead_radius <= 0.03:
            raise ValueError("Bead radius must be between 0.001 and 0.03.")
        if self.background not in ("light", "black"):
            raise ValueError("Background must be light or black.")


def label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Prefer a restrained serif font; keep the viewer portable without a font download."""
    candidates = (
        Path("C:/Windows/Fonts/georgia.ttf"),
        Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def create_overlay(settings: RenderSettings, state: QuantumNumbers, cutaway: bool) -> Image.Image:
    image = Image.new("RGBA", (settings.width, settings.height))
    if not settings.labels:
        return image

    draw = ImageDraw.Draw(image)
    scale = settings.height / 1080.0
    margin = round(64 * scale)
    main_font = label_font(round(34 * scale))
    small_font = label_font(round(24 * scale))
    ink = (57, 61, 62, 255)
    secondary = (95, 102, 104, 255)
    if settings.background == "black":
        ink = (228, 222, 213, 255)
        secondary = (173, 180, 183, 255)

    draw.text((margin, margin), "H", font=main_font, fill=ink)
    quantum_label = f"n = {state.n}     l = {state.l}     m = {state.m}"
    draw.text(
        (settings.width - margin, margin), quantum_label, font=main_font, fill=ink, anchor="ra"
    )
    draw.text((margin, settings.height - margin), "|ψ|²", font=main_font, fill=ink, anchor="ld")
    if cutaway:
        draw.text(
            (settings.width - margin, settings.height - margin),
            "Schnittansicht",
            font=small_font,
            fill=secondary,
            anchor="rd",
        )
    return image


class OrbitalRenderer:
    """Own buffers/framebuffers; the caller owns and closes the OpenGL context."""

    def __init__(
        self,
        context: moderngl.Context,
        cloud: PointCloud,
        state: QuantumNumbers,
        settings: RenderSettings,
    ) -> None:
        self.context = context
        self.settings = settings
        self.state = state
        self.point_count = len(cloud.positions)
        self.display_radius = float(cloud.bound_radius)
        normalized_positions = cloud.positions / self.display_radius
        normalized_positions = np.ascontiguousarray(normalized_positions, dtype=np.float32)

        self.particle_program = context.program(
            vertex_shader=PARTICLE_VERTEX, fragment_shader=PARTICLE_FRAGMENT
        )
        self.shade_program = context.program(
            vertex_shader=SCREEN_VERTEX, fragment_shader=SHADE_FRAGMENT
        )
        self.overlay_program = context.program(
            vertex_shader=SCREEN_VERTEX, fragment_shader=OVERLAY_FRAGMENT
        )
        self.vertex_buffer = context.buffer(normalized_positions.tobytes())
        self.particle_array = context.vertex_array(
            self.particle_program, [(self.vertex_buffer, "3f", "in_position")]
        )
        self.screen_array = context.vertex_array(self.shade_program, [])
        self.overlay_array = context.vertex_array(self.overlay_program, [])

        size = (settings.width, settings.height)
        self.geometry_texture = context.texture(size, components=4, dtype="f4")
        self.geometry_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.geometry_texture.repeat_x = False
        self.geometry_texture.repeat_y = False
        self.depth_buffer = context.depth_renderbuffer(size)
        self.geometry_buffer = context.framebuffer(
            color_attachments=[self.geometry_texture], depth_attachment=self.depth_buffer
        )
        self.output_texture = context.texture(size, components=4, dtype="f1")
        self.output_buffer = context.framebuffer(color_attachments=[self.output_texture])
        self.overlay_texture = context.texture(size, components=4, dtype="f1")
        self.overlay_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.overlay_cutaway: bool | None = None

        self.shadow_depth = context.depth_texture((2048, 2048))
        self.shadow_depth.compare_func = ""
        self.shadow_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.shadow_depth.repeat_x = False
        self.shadow_depth.repeat_y = False
        self.shadow_buffer = context.framebuffer(depth_attachment=self.shadow_depth)

        projection = projection_matrix(settings.width / settings.height)
        self.projection = projection
        self.light_projection = projection_matrix(1.0, field_of_view=40.0)
        self.particle_program["projection"].write(projection.T.tobytes())
        self.particle_program["viewport_height"].value = float(settings.height)
        self.particle_program["bead_radius"].value = settings.bead_radius
        self.particle_program["cut_direction"].value = math.radians(45.0)
        self.shade_program["geometry_texture"].value = 0
        self.shade_program["shadow_texture"].value = 2
        self.shade_program["focal_lengths"].value = (projection[0, 0], projection[1, 1])
        self.shade_program["resolution"].value = (settings.width, settings.height)
        self.shade_program["dark_background"].value = settings.background == "black"
        self.overlay_program["overlay_texture"].value = 1

    def draw(
        self,
        azimuth: float = 40.0,
        elevation: float = 15.0,
        distance: float = 3.0,
        cut_angle: float = 0.0,
    ) -> None:
        """Draw a fixed sample; only the camera and geometric cutaway change."""
        context = self.context
        view = view_matrix(azimuth, elevation, distance)
        inverse_view = np.linalg.inv(view).astype(np.float32)
        eye_light = np.array([-0.50, 0.75, 1.30, 0.0], dtype=np.float32)
        world_light = inverse_view @ eye_light
        light_azimuth = math.degrees(math.atan2(world_light[1], world_light[0]))
        light_elevation = math.degrees(math.asin(world_light[2] / np.linalg.norm(world_light[:3])))
        light_view = view_matrix(light_azimuth, light_elevation, 4.0)
        light_view_projection = self.light_projection @ light_view
        self.shade_program["inverse_view"].write(inverse_view.T.tobytes())
        self.shade_program["light_view_projection"].write(light_view_projection.T.tobytes())
        self.particle_program["cut_angle"].value = math.radians(cut_angle)

        self.shadow_buffer.use()
        context.viewport = (0, 0, 2048, 2048)
        self.shadow_buffer.clear(depth=1.0)
        self.particle_program["view"].write(light_view.T.tobytes())
        self.particle_program["projection"].write(self.light_projection.T.tobytes())
        self.particle_program["viewport_height"].value = 2048.0
        context.enable_only(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self.particle_array.render(mode=moderngl.POINTS, vertices=self.point_count)

        self.geometry_buffer.use()
        context.viewport = (0, 0, self.settings.width, self.settings.height)
        self.geometry_buffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.particle_program["view"].write(view.T.tobytes())
        self.particle_program["projection"].write(self.projection.T.tobytes())
        self.particle_program["viewport_height"].value = float(self.settings.height)
        context.enable_only(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self.particle_array.render(mode=moderngl.POINTS, vertices=self.point_count)

        self.output_buffer.use()
        context.enable_only(moderngl.NOTHING)
        self.geometry_texture.use(location=0)
        self.shadow_depth.use(location=2)
        self.screen_array.render(mode=moderngl.TRIANGLES, vertices=3)

        cutaway = cut_angle > 0.1
        if self.overlay_cutaway != cutaway:
            overlay_image = create_overlay(self.settings, self.state, cutaway)
            self.overlay_texture.write(overlay_image.tobytes())
            self.overlay_cutaway = cutaway
        context.enable(moderngl.BLEND)
        context.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.overlay_texture.use(location=1)
        self.overlay_array.render(mode=moderngl.TRIANGLES, vertices=3)
        context.disable(moderngl.BLEND)

    def read_image(self) -> Image.Image:
        raw = self.output_buffer.read(components=3, alignment=1)
        image = Image.frombytes("RGB", (self.settings.width, self.settings.height), raw)
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def close(self) -> None:
        """Release GPU resources before the owning window/context is destroyed."""
        self.overlay_array.release()
        self.screen_array.release()
        self.particle_array.release()
        self.vertex_buffer.release()
        self.geometry_buffer.release()
        self.output_buffer.release()
        self.depth_buffer.release()
        self.shadow_buffer.release()
        self.shadow_depth.release()
        self.geometry_texture.release()
        self.output_texture.release()
        self.overlay_texture.release()
        self.particle_program.release()
        self.shade_program.release()
        self.overlay_program.release()
