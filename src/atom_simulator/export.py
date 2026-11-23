"""Still and MP4 export through the same renderer used by the interactive viewer."""

import json
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import moderngl
import numpy as np
from PIL import Image

from atom_simulator.camera import preview_pose
from atom_simulator.physics import QuantumNumbers
from atom_simulator.renderer import OrbitalRenderer, RenderSettings
from atom_simulator.sampling import PointCloud


def write_metadata(
    path: Path,
    state: QuantumNumbers,
    cloud: PointCloud,
    renderer: OrbitalRenderer,
    details: dict[str, Any],
) -> None:
    record = {
        "quantum_numbers": asdict(state),
        "sampling": cloud.metadata,
        "render_settings": asdict(renderer.settings),
        "renderer": renderer.context.info["GL_RENDERER"],
        "display_radius_in_a_mu": renderer.display_radius,
        "interpretation": (
            "Fixed samples of |R_nl Y_lm|^2; camera motion, no particle trajectories."
        ),
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
) -> None:
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
        container = av.open(str(temporary), "w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx264", rate=fps)
        stream.width = settings.width
        stream.height = settings.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        print(f"GPU: {context.info['GL_RENDERER']}", flush=True)

        for frame_index in range(total_frames):
            seconds = frame_index / fps
            azimuth, elevation, cut_angle = preview_pose(seconds, duration)
            if not cutaway:
                cut_angle = 0.0
            renderer.draw(azimuth=azimuth, elevation=elevation, cut_angle=cut_angle)
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
                "camera": "Slow orbit; fixed particle positions; final hold.",
            },
        )
    finally:
        if container is not None:
            container.close()
        if renderer is not None:
            renderer.close()
        context.release()
