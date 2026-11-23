"""Still and MP4 export through the same renderer used by the interactive viewer."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from PIL import Image

from atom_simulator.camera import flow_pose, preview_pose
from atom_simulator.physics import T_MU_SECONDS, PointCloud, QuantumNumbers, bohm_angular_velocity
from atom_simulator.renderer import OrbitalRenderer, RenderSettings

try:
    import av
except ImportError:
    av = None


def write_metadata(
    path: Path,
    state: QuantumNumbers,
    cloud: PointCloud,
    renderer: OrbitalRenderer,
    details: dict[str, Any],
) -> None:
    interpretation = "Fixed samples of |R_nl Y_lm|^2; camera motion."
    if details.get("bohm_flow", False):
        interpretation = "Spinless Bohmian ensemble in the relative coordinate; not many electrons."
    record = {
        "quantum_numbers": asdict(state),
        "sampling": cloud.metadata,
        "render_settings": asdict(renderer.settings),
        "renderer": renderer.context.info["GL_RENDERER"],
        "display_radius_in_a_mu": renderer.display_radius,
        "interpretation": interpretation,
        "cutaway": "A geometric azimuth wedge; its opening changes visibility, not the state.",
        "export": details,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_still(
    state: QuantumNumbers,
    cloud: PointCloud,
    settings: RenderSettings,
    output: Path,
    cut_angle: float = 110.0,
    supersample: int = 2,
) -> None:
    if output.suffix.lower() != ".png":
        raise ValueError("Image output must use the .png extension.")
    if supersample not in (1, 2, 3):
        raise ValueError("Supersampling must be 1, 2 or 3.")
    output.parent.mkdir(parents=True, exist_ok=True)
    context = moderngl.create_standalone_context(require=330)
    renderer: OrbitalRenderer | None = None
    try:
        render_settings = replace(
            settings, width=settings.width * supersample, height=settings.height * supersample
        )
        renderer = OrbitalRenderer(context, cloud, state, render_settings)
        renderer.draw(azimuth=42.0, elevation=20.0, cut_angle=cut_angle)
        image = renderer.read_image().resize(
            (settings.width, settings.height), Image.Resampling.LANCZOS
        )
        image.save(output)
        write_metadata(
            output.with_suffix(".json"),
            state,
            cloud,
            renderer,
            {
                "kind": "still",
                "cut_angle_degrees": cut_angle,
                "width": settings.width,
                "height": settings.height,
                "supersample": supersample,
            },
        )
    finally:
        if renderer is not None:
            renderer.close()
        context.release()


def encode_frame(
    container: av.container.OutputContainer,
    stream: av.video.VideoStream,
    image: Image.Image,
    index: int,
    fps: int,
) -> None:
    pixels = np.asarray(image)
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
    frame.pts = index
    frame.time_base = Fraction(1, fps)
    for packet in stream.encode(frame):
        container.mux(packet)


def flush_encoder(container: av.container.OutputContainer, stream: av.video.VideoStream) -> None:
    for packet in stream.encode():
        container.mux(packet)


def render_video(
    state: QuantumNumbers,
    cloud: PointCloud,
    settings: RenderSettings,
    output: Path,
    duration: float,
    fps: int,
    cutaway: bool = True,
    supersample: int = 2,
    bohm_flow: bool = False,
    time_rate: float = 4.0,
    camera_distance: float = 3.0,
) -> None:
    if av is None:
        raise RuntimeError('Video export needs the optional encoder: pip install -e ".[video]"')
    if not math.isfinite(time_rate) or time_rate <= 0.0:
        raise ValueError("Time rate must be finite and positive.")
    if not math.isfinite(camera_distance) or not 1.5 <= camera_distance <= 10.0:
        raise ValueError("Camera distance must be between 1.5 and 10.")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension.")
    if supersample not in (1, 2, 3):
        raise ValueError("Supersampling must be 1, 2 or 3.")
    if not 1.0 <= duration <= 600.0:
        raise ValueError("Duration must be between 1 and 600 seconds.")
    if not 1 <= fps <= 120:
        raise ValueError("Frame rate must be between 1 and 120.")
    if settings.width % 2 != 0 or settings.height % 2 != 0:
        raise ValueError("H.264 export requires even width and height.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    total_frames = round(duration * fps)
    context = moderngl.create_standalone_context(require=330)
    renderer: OrbitalRenderer | None = None
    container: av.container.OutputContainer | None = None
    try:
        render_settings = replace(
            settings, width=settings.width * supersample, height=settings.height * supersample
        )
        renderer = OrbitalRenderer(context, cloud, state, render_settings)
        flow_diagnostics = {}
        if bohm_flow:
            renderer.enable_flow(time_rate, fps)
            angular_rates = bohm_angular_velocity(state, cloud.positions)
            frame_angles = np.abs(angular_rates) * time_rate / fps
            flow_diagnostics = {
                "time_units_per_video_second": time_rate,
                "natural_time_seconds": T_MU_SECONDS,
                "physical_seconds_per_video_second": time_rate * T_MU_SECONDS,
                "max_angle_per_frame_radians": float(frame_angles.max()),
                "fraction_over_pi_per_frame": float(np.mean(frame_angles > np.pi)),
                "fraction_over_0_25_radians_per_frame": float(np.mean(frame_angles > 0.25)),
                "highlighted_samples": renderer.tracer_indices.tolist(),
                "highlight_max_angle_per_frame": float(
                    frame_angles[renderer.tracer_indices].max(initial=0.0)
                ),
                "trajectory_solver": "Exact azimuthal solution; one time scale; no speed caps.",
            }
        container = av.open(str(temporary), "w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx264", rate=fps)
        stream.width = settings.width
        stream.height = settings.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        print(f"GPU: {context.info['GL_RENDERER']}", flush=True)

        for frame_index in range(total_frames):
            seconds = frame_index / fps
            if bohm_flow:
                renderer.update_flow(seconds * time_rate)
                azimuth, elevation = flow_pose(seconds, duration)
                cut_angle = 110.0
            else:
                azimuth, elevation, cut_angle = preview_pose(seconds, duration)
            if not cutaway:
                cut_angle = 0.0
            renderer.draw(
                azimuth=azimuth, elevation=elevation, distance=camera_distance, cut_angle=cut_angle
            )
            image = renderer.read_image().resize(
                (settings.width, settings.height), Image.Resampling.LANCZOS
            )
            encode_frame(container, stream, image, frame_index, fps)
            if frame_index % (fps * 5) == 0:
                print(f"Rendering {seconds:.0f} / {duration:g} s", flush=True)

        flush_encoder(container, stream)
        container.close()
        container = None
        temporary.replace(output)
        write_metadata(
            output.with_suffix(".json"),
            state,
            cloud,
            renderer,
            {
                "kind": "video",
                "duration_seconds": total_frames / fps,
                "frames": total_frames,
                "fps": fps,
                "width": settings.width,
                "height": settings.height,
                "supersample": supersample,
                "cutaway_enabled": cutaway,
                "camera": "Three held views with two smooth moves." if bohm_flow else "Slow orbit.",
                "bohm_flow": bohm_flow,
                "camera_distance": camera_distance,
                "flow": flow_diagnostics,
            },
        )
    finally:
        if container is not None:
            container.close()
        if renderer is not None:
            renderer.close()
        context.release()


def append_clip(
    output_container: Any, output_stream: Any, path: Path, first_frame: int, fps: int
) -> int:
    """Decode one clip in order; keep the outer sequence loop separate and readable."""
    frame_index = first_frame
    with av.open(str(path)) as source:
        for frame in source.decode(video=0):
            encode_frame(output_container, output_stream, frame.to_image(), frame_index, fps)
            frame_index += 1
    return frame_index


def combine_clips(paths: list[Path], output: Path, width: int, height: int, fps: int) -> None:
    """A clean sequence of the individually usable clips, without sound or extra titles."""
    if av is None:
        raise RuntimeError("The optional video encoder is not installed.")
    temporary = output.with_name(output.stem + ".partial.mp4")
    with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        frame_index = 0
        chapters = []
        for path in paths:
            chapters.append({"file": path.name, "start_seconds": frame_index / fps})
            frame_index = append_clip(container, stream, path, frame_index, fps)
        flush_encoder(container, stream)
    temporary.replace(output)
    metadata = {
        "frames": frame_index,
        "fps": fps,
        "duration_seconds": frame_index / fps,
        "width": width,
        "height": height,
        "chapters": chapters,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def render_cases(output: Path) -> None:
    """The four explicit production cases from config.py, followed by their combined clip."""
    from atom_simulator.config import (
        BACKGROUND,
        BEAD_RADIUS,
        CLIP_SECONDS,
        FLOW_CASES,
        FPS,
        HEIGHT,
        PARTICLE_COUNT,
        RANDOM_SEED,
        WIDTH,
    )
    from atom_simulator.physics import sample_orbital

    output.mkdir(parents=True, exist_ok=True)
    settings = RenderSettings(
        width=WIDTH, height=HEIGHT, bead_radius=BEAD_RADIUS, background=BACKGROUND
    )
    paths = []
    for case in FLOW_CASES:
        target = output / (case.filename + ".mp4")
        cloud = sample_orbital(case.state, PARTICLE_COUNT, RANDOM_SEED)
        render_video(
            case.state,
            cloud,
            settings,
            target,
            CLIP_SECONDS,
            FPS,
            bohm_flow=True,
            time_rate=case.time_rate,
            camera_distance=case.camera_distance,
        )
        paths.append(target)
    combine_clips(paths, output / "Bohmscher_Fluss_Alle_Faelle.mp4", WIDTH, HEIGHT, FPS)
