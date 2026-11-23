"""Typed, conventional command-line entry point with explicit configuration."""

import argparse
import math
import sys
from pathlib import Path

import moderngl

from atom_simulator.physics import QuantumNumbers
from atom_simulator.renderer import RenderSettings
from atom_simulator.sampling import sample_orbital


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=6, help="Principal quantum number (1 to 12).")
    parser.add_argument("--l", type=int, default=2, help="Angular momentum quantum number.")
    parser.add_argument("--m", type=int, default=0, help="Magnetic quantum number.")
    parser.add_argument(
        "--points", type=int, default=450_000, help="Number of probability samples."
    )
    parser.add_argument("--seed", type=int, default=7, help="Fixed seed for reproducible clouds.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--bead-radius",
        type=float,
        default=0.0035,
        help="Visual point radius, relative to the cloud bounding radius.",
    )
    parser.add_argument("--background", choices=("light", "black"), default="light")
    parser.add_argument("--no-labels", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hydrogen orbitals as gold 3D probability clouds.")
    commands = parser.add_subparsers(dest="command", required=True)

    viewer = commands.add_parser("view", help="Open the interactive desktop viewer.")
    common_arguments(viewer)
    viewer.set_defaults(width=1280, height=720)
    viewer.add_argument("--test-frames", type=int, default=0, help=argparse.SUPPRESS)

    image = commands.add_parser("image", help="Render one PNG without opening a window.")
    common_arguments(image)
    image.add_argument("--output", type=Path, default=Path("outputs/orbital.png"))
    image.add_argument("--no-cutaway", action="store_true")
    image.add_argument("--supersample", type=int, choices=(1, 2, 3), default=2)

    video = commands.add_parser("render", help="Export a calm camera animation as H.264 MP4.")
    common_arguments(video)
    video.add_argument("--output", type=Path, default=Path("outputs/orbital-preview.mp4"))
    video.add_argument("--duration", type=float, default=20.0)
    video.add_argument("--fps", type=int, default=30)
    video.add_argument("--no-cutaway", action="store_true")
    video.add_argument("--supersample", type=int, choices=(1, 2, 3), default=2)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid export requests before sampling or opening an output file."""
    if not 1_000 <= args.points <= 2_000_000:
        raise ValueError("Point count must be between 1,000 and 2,000,000.")
    if args.seed < 0:
        raise ValueError("Seed must be non-negative.")
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

            viewer = OrbitalViewer(state, cloud, settings, args.seed, args.test_frames)
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
            )
            print(f"Saved: {args.output.resolve()}")
    except (ValueError, RuntimeError, OSError, ArithmeticError, moderngl.Error) as error:
        print(f"atom-simulator: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0
