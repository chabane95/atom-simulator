"""Typed, conventional command-line entry point with explicit configuration."""

import argparse
import math
import sys
from pathlib import Path

import moderngl

from atom_simulator import config
from atom_simulator.physics import QuantumNumbers, sample_orbital
from atom_simulator.renderer import RenderSettings


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--n", type=int, default=config.N, help="Principal quantum number (1 to 12)."
    )
    parser.add_argument("--l", type=int, default=config.L, help="Angular momentum quantum number.")
    parser.add_argument("--m", type=int, default=config.M, help="Magnetic quantum number.")
    parser.add_argument(
        "--points", type=int, default=config.PARTICLE_COUNT, help="Number of probability samples."
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Reproducible seed.")
    parser.add_argument("--width", type=int, default=config.WIDTH)
    parser.add_argument("--height", type=int, default=config.HEIGHT)
    parser.add_argument(
        "--bead-radius",
        type=float,
        default=config.BEAD_RADIUS,
        help="Visual point radius, relative to the cloud bounding radius.",
    )
    parser.add_argument("--background", choices=("light", "black"), default=config.BACKGROUND)
    parser.add_argument("--no-labels", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hydrogen orbitals as gold 3D probability clouds.")
    commands = parser.add_subparsers(dest="command", required=True)

    viewer = commands.add_parser("view", help="Open the interactive desktop viewer.")
    common_arguments(viewer)
    viewer.set_defaults(width=1280, height=720)
    viewer.add_argument("--test-frames", type=int, default=0, help=argparse.SUPPRESS)
    viewer.add_argument("--motion", choices=("bohm", "camera"), default="bohm")
    viewer.add_argument("--time-rate", type=float, default=None)

    image = commands.add_parser("image", help="Render one PNG without opening a window.")
    common_arguments(image)
    image.add_argument("--output", type=Path, default=Path("outputs/orbital.png"))
    image.add_argument("--no-cutaway", action="store_true")
    image.add_argument("--supersample", type=int, choices=(1, 2, 3), default=2)

    video = commands.add_parser("render", help="Export a calm camera animation as H.264 MP4.")
    common_arguments(video)
    video.add_argument("--output", type=Path, default=Path("outputs/bohm-preview.mp4"))
    video.add_argument("--duration", type=float, default=config.CLIP_SECONDS)
    video.add_argument("--fps", type=int, default=config.FPS)
    video.add_argument("--no-cutaway", action="store_true")
    video.add_argument("--supersample", type=int, choices=(1, 2, 3), default=2)
    video.add_argument("--motion", choices=("bohm", "camera"), default="bohm")
    video.add_argument("--time-rate", type=float, default=None)

    cases = commands.add_parser(
        "clips", help="Render the four fixed Bohm cases and one combined film."
    )
    cases.add_argument("--output", type=Path, default=Path("outputs/bohm_clips"))
    gallery = commands.add_parser(
        "gallery", help="Save full and cutaway views of all unique densities."
    )
    gallery.add_argument("--output", type=Path, default=Path("outputs/Orbitalgalerie"))
    gallery.add_argument("--max-n", type=int, choices=range(1, 13), default=12)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid export requests before sampling or opening an output file."""
    if not 1_000 <= args.points <= 2_000_000:
        raise ValueError("Point count must be between 1,000 and 2,000,000.")
    if args.seed < 0:
        raise ValueError("Seed must be non-negative.")
    time_rate = getattr(args, "time_rate", None)
    if time_rate is not None and (not math.isfinite(time_rate) or time_rate <= 0.0):
        raise ValueError("Time rate must be finite and positive.")
    if args.command == "view":
        if args.test_frames < 0:
            raise ValueError("Test frame count must be non-negative.")
        return
    if args.command == "image":
        if args.output.suffix.lower() != ".png":
            raise ValueError("Image output must use the .png extension.")
        return
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension.")
    if not math.isfinite(args.duration) or not 1.0 <= args.duration <= 600.0:
        raise ValueError("Duration must be between 1 and 600 seconds.")
    if not 1 <= args.fps <= 120:
        raise ValueError("Frame rate must be between 1 and 120.")
    if args.width % 2 != 0 or args.height % 2 != 0:
        raise ValueError("H.264 export requires even width and height.")


def main() -> int:
    parser = build_parser()
    arguments = sys.argv[1:]
    if not arguments:
        arguments = ["view"]
    args = parser.parse_args(arguments)
    try:
        if args.command == "clips":
            from atom_simulator.export import render_cases

            render_cases(args.output)
            return 0
        if args.command == "gallery":
            from atom_simulator.gallery import render_gallery

            render_gallery(
                args.output,
                args.max_n,
                points=config.PARTICLE_COUNT,
                width=config.GALLERY_WIDTH,
                height=config.GALLERY_HEIGHT,
            )
            return 0
        state = QuantumNumbers(n=args.n, l=args.l, m=args.m)
        validate_arguments(args)
        settings = RenderSettings(
            width=args.width,
            height=args.height,
            bead_radius=args.bead_radius,
            background=args.background,
            labels=not args.no_labels,
        )
        print(f"Sampling n={state.n}, l={state.l}, m={state.m}: {args.points:,} points", flush=True)
        cloud = sample_orbital(state, args.points, args.seed)

        if args.command == "view":
            from atom_simulator.viewer import OrbitalViewer

            time_rate = args.time_rate or config.default_time_rate(state)
            viewer = OrbitalViewer(
                state,
                cloud,
                settings,
                args.seed,
                args.test_frames,
                bohm_flow=args.motion == "bohm",
                time_rate=time_rate,
            )
            viewer.run()
        elif args.command == "image":
            from atom_simulator.export import render_still

            cut_angle = 0.0 if args.no_cutaway else 110.0
            render_still(state, cloud, settings, args.output, cut_angle, args.supersample)
            print(f"Saved: {args.output.resolve()}")
        elif args.command == "render":
            from atom_simulator.export import render_video

            render_video(
                state,
                cloud,
                settings,
                args.output,
                args.duration,
                args.fps,
                cutaway=not args.no_cutaway,
                supersample=args.supersample,
                bohm_flow=args.motion == "bohm",
                time_rate=args.time_rate or config.default_time_rate(state),
            )
            print(f"Saved: {args.output.resolve()}")
    except (ValueError, RuntimeError, OSError, ArithmeticError, moderngl.Error) as error:
        print(f"atom-simulator: {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        print(
            f'Missing optional component: {error}. Install with: pip install -e ".[all]"',
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0
