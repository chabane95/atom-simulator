"""Decode finished exports and check gallery completeness without invoking OpenGL.

Run from the repository root with the project Python.  All MP4 frames and all
gallery PNGs are decoded, but only three frames per clip and eight gallery
examples are retained.  A nonzero exit status means a required check failed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from html.parser import HTMLParser
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import av
from PIL import Image, ImageDraw, ImageFont, ImageStat


@dataclass
class Checks:
    errors: list[str] = field(default_factory=list)
    error_count: int = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.error_count += 1
            if len(self.errors) < 100:
                self.errors.append(message)


@dataclass
class Thumbnail:
    image: Image.Image
    label: str


class SourceVisitor(ast.NodeVisitor):
    """Find syntactically nested loops, including comprehensions, and imports."""

    def __init__(self) -> None:
        self.loop_depth = 0
        self.nested_lines: list[int] = []
        self.imports: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.update(alias.name.split(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self.imports.add(node.module.split(".")[0])

    def visit_loop(self, node: ast.AST) -> None:
        if self.loop_depth:
            self.nested_lines.append(node.lineno)
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_For = visit_loop
    visit_AsyncFor = visit_loop
    visit_While = visit_loop

    def visit_comprehension_expression(self, node: ast.AST) -> None:
        if self.loop_depth or len(node.generators) > 1:
            self.nested_lines.append(node.lineno)
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_ListComp = visit_comprehension_expression
    visit_SetComp = visit_comprehension_expression
    visit_DictComp = visit_comprehension_expression
    visit_GeneratorExp = visit_comprehension_expression


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def dependency_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise ValueError(f"Cannot parse dependency: {requirement}")
    return match.group().lower().replace("_", "-")


def dependency_names(requirements: list[str]) -> set[str]:
    return {dependency_name(item) for item in requirements}


def check_source(project: Path, checks: Checks) -> dict[str, Any]:
    configuration = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = dependency_names(configuration["project"]["dependencies"])
    extras = configuration["project"].get("optional-dependencies", {})
    optional_dependencies = {name: dependency_names(items) for name, items in extras.items()}
    dev_dependencies = optional_dependencies.get("dev", set())
    files = sorted((project / "src").rglob("*.py"))
    imports: set[str] = set()
    for path in files:
        visitor = SourceVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8-sig")))
        imports.update(visitor.imports)
        checks.require(
            not visitor.nested_lines,
            f"Nested source loops: {path.relative_to(project)}:{visitor.nested_lines}",
        )
        checks.require(
            "scipy" not in visitor.imports,
            f"Production SciPy import remains: {path.relative_to(project)}",
        )

    checks.require("scipy" not in dependencies, "SciPy remains a runtime dependency")
    import_to_package = {
        "PIL": "pillow",
        "numpy": "numpy",
        "av": "av",
        "glfw": "glfw",
        "moderngl": "moderngl",
    }
    external_imports = imports - sys.stdlib_module_names - {"atom_simulator", "__future__"}
    for imported in sorted(external_imports):
        package = import_to_package.get(imported, imported.lower().replace("_", "-"))
        feature = {"av": "video", "glfw": "viewer"}.get(imported)
        declared = dependencies | optional_dependencies.get(feature, set())
        checks.require(
            package in declared, f"Undeclared dependency for {imported}: {feature or 'runtime'}"
        )
        if feature is not None and "all" in optional_dependencies:
            checks.require(
                package in dependencies | optional_dependencies["all"],
                f"The all extra omits {package} from the {feature} feature",
            )

    test_uses_scipy = False
    for path in sorted((project / "tests").rglob("*.py")):
        visitor = SourceVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8-sig")))
        test_uses_scipy = test_uses_scipy or "scipy" in visitor.imports
    checks.require(
        not test_uses_scipy or "scipy" in dev_dependencies,
        "Tests use SciPy, but the dev extra does not declare it",
    )
    return {
        "files": len(files),
        "runtime_dependencies": sorted(dependencies),
        "dev_dependencies": sorted(dev_dependencies),
        "optional_dependencies": {
            name: sorted(items) for name, items in optional_dependencies.items()
        },
        "test_reference_uses_scipy": test_uses_scipy,
    }


def save_contact(thumbnails: list[Thumbnail], path: Path, columns: int = 3) -> None:
    if not thumbnails:
        return
    tile_width, tile_height, caption_height = 640, 360, 34
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + caption_height)), "#101214"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=19)
    for index, thumbnail in enumerate(thumbnails):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + caption_height)
        preview = thumbnail.image.copy()
        preview.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + (tile_width - preview.width) // 2, y + caption_height))
        draw.text((x + 10, y + 7), thumbnail.label, font=font, fill="#e4ded5")
    sheet.save(path, quality=92)


def capture_frame(frame: av.VideoFrame, output: Path, label: str) -> Thumbnail:
    image = frame.to_image()
    image.save(output)
    image.thumbnail((640, 360), Image.Resampling.LANCZOS)
    return Thumbnail(image=image, label=label)


def inspect_video(
    path: Path, args: argparse.Namespace, checks: Checks, captures: list[Thumbnail]
) -> dict[str, Any]:
    metadata = read_json(path.with_suffix(".json"))
    expected_count = round(args.duration * args.fps)
    selected = {round((expected_count - 1) * fraction) for fraction in (0.1, 0.5, 0.9)}
    prefix = path.name
    frames = 0
    missing_timestamps = 0
    nonmonotonic_timestamps = 0
    wrong_dimensions = 0
    previous_time: Fraction | None = None
    largest_timestamp_error = 0.0
    with av.open(str(path)) as container:
        checks.require(len(container.streams.video) == 1, f"{prefix}: expected one video stream")
        checks.require(not container.streams.audio, f"{prefix}: unexpected audio stream")
        if not container.streams.video:
            return {"file": path.name, "decoded_frames": 0}
        stream = container.streams.video[0]
        checks.require(stream.codec_context.name == "h264", f"{prefix}: codec is not H.264")
        checks.require(stream.average_rate == Fraction(args.fps), f"{prefix}: wrong nominal FPS")
        checks.require(stream.codec_context.pix_fmt == "yuv420p", f"{prefix}: expected yuv420p")
        for frame in container.decode(stream):
            wrong_dimensions += (frame.width, frame.height) != (args.clip_width, args.clip_height)
            if frame.pts is None or frame.time_base is None:
                missing_timestamps += 1
            else:
                actual_time = frame.pts * frame.time_base
                expected_time = Fraction(frames, args.fps)
                largest_timestamp_error = max(
                    largest_timestamp_error, float(abs(actual_time - expected_time))
                )
                if previous_time is not None and actual_time <= previous_time:
                    nonmonotonic_timestamps += 1
                previous_time = actual_time
            if frames in selected:
                label = f"{path.stem} | {frames / args.fps:.2f} s"
                capture_path = args.report / f"{path.stem}_{frames:04d}.png"
                thumbnail = capture_frame(frame, capture_path, label)
                checks.require(
                    max(ImageStat.Stat(thumbnail.image).stddev) > 3.0,
                    f"{prefix}: selected frame {frames} is nearly uniform",
                )
                captures.append(thumbnail)
            frames += 1
        container_duration = (
            None if container.duration is None else container.duration / av.time_base
        )
        stream_duration = (
            None if stream.duration is None else float(stream.duration * stream.time_base)
        )

    checks.require(
        frames == expected_count, f"{prefix}: decoded {frames}, expected {expected_count} frames"
    )
    checks.require(
        wrong_dimensions == 0, f"{prefix}: {wrong_dimensions} frames have wrong dimensions"
    )
    checks.require(missing_timestamps == 0, f"{prefix}: {missing_timestamps} missing timestamps")
    checks.require(nonmonotonic_timestamps == 0, f"{prefix}: nonmonotonic timestamps")
    checks.require(
        largest_timestamp_error <= 1e-6, f"{prefix}: timestamp error {largest_timestamp_error:g} s"
    )
    checks.require(
        stream_duration is not None and abs(stream_duration - args.duration) <= 1e-5,
        f"{prefix}: stream duration {stream_duration} s, expected {args.duration}",
    )
    checks.require(
        container_duration is not None and abs(container_duration - args.duration) <= 1e-3,
        f"{prefix}: container duration {container_duration} s",
    )
    export = metadata.get("export", metadata)
    checks.require(export.get("frames") == frames, f"{prefix}: metadata frame count mismatch")
    checks.require(export.get("fps") == args.fps, f"{prefix}: metadata FPS mismatch")
    checks.require(
        export.get("width") == args.clip_width and export.get("height") == args.clip_height,
        f"{prefix}: metadata resolution mismatch",
    )
    return {
        "file": path.name,
        "decoded_frames": frames,
        "fps": args.fps,
        "resolution": [args.clip_width, args.clip_height],
        "duration_seconds": stream_duration,
        "maximum_timestamp_error_seconds": largest_timestamp_error,
        "quantum_numbers": metadata.get("quantum_numbers"),
        "export": export,
    }


def check_clips(args: argparse.Namespace, checks: Checks) -> dict[str, Any]:
    explicit_files = getattr(args, "clip_files", None)
    if explicit_files:
        paths = [args.clips / filename for filename in explicit_files]
    else:
        paths = sorted(
            path
            for path in args.clips.glob("*.mp4")
            if path.name != "Bohmscher_Fluss_Alle_Faelle.mp4"
        )
    checks.require(
        len(paths) == args.clip_count, f"Expected {args.clip_count} clips, found {len(paths)}"
    )
    captures: list[Thumbnail] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        print(f"Decoding {path.name}", flush=True)
        try:
            records.append(inspect_video(path, args, checks, captures))
        except Exception as error:
            checks.require(False, f"{path.name}: {type(error).__name__}: {error}")
    save_contact(captures, args.report / "clips_contact.jpg")
    return {"count": len(paths), "clips": records, "retained_frames": len(captures)}


def gallery_key(metadata: dict[str, Any]) -> tuple[int, int, int, bool]:
    state = metadata["quantum_numbers"]
    export = metadata["export"]
    angle = float(export["cut_angle_degrees"])
    return int(state["n"]), int(state["l"]), int(state["m"]), angle > 0.1


class IndexLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.png_paths: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        link = attributes.get("href") if tag == "a" else attributes.get("src")
        if link and urlsplit(link).path.lower().endswith(".png"):
            self.png_paths.add(unquote(urlsplit(link).path))


def json_png_paths(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.lower().endswith(".png") else set()
    children = (
        value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    )
    paths: set[str] = set()
    for child in children:
        paths.update(json_png_paths(child))
    return paths


def resolve_index_paths(directory: Path, links: set[str]) -> set[Path]:
    return {(directory / unquote(link)).resolve() for link in links}


def check_gallery_index(directory: Path, images: set[Path], checks: Checks) -> dict[str, Any]:
    candidates = [
        directory / name for name in ("index.html", "index.json", "manifest.json", "gallery.json")
    ]
    indices = [path for path in candidates if path.is_file()]
    checks.require(
        bool(indices), "Gallery has no recognized index.html/index.json/manifest.json/gallery.json"
    )
    records: dict[str, int] = {}
    for path in indices:
        if path.suffix == ".html":
            parser = IndexLinks()
            parser.feed(path.read_text(encoding="utf-8-sig"))
            links = parser.png_paths
        else:
            links = json_png_paths(json.loads(path.read_text(encoding="utf-8-sig")))
        resolved = resolve_index_paths(path.parent, links)
        checks.require(
            resolved == images,
            f"{path.name}: index has {len(resolved)} PNG references; "
            f"missing {len(images - resolved)}, extra {len(resolved - images)}",
        )
        records[path.name] = len(resolved)
    return records


def check_gallery(args: argparse.Namespace, checks: Checks) -> dict[str, Any]:
    paths = sorted(args.gallery.rglob("*.png"))
    expected = {
        (n, angular, magnetic, cut)
        for n, angular, magnetic, cut in product(
            range(1, args.n_max + 1), range(args.n_max), range(args.n_max), (False, True)
        )
        if angular < n and magnetic <= angular
    }
    checks.require(
        len(paths) == len(expected), f"Gallery has {len(paths)} PNGs, expected {len(expected)}"
    )
    seen: set[tuple[int, int, int, bool]] = set()
    captures: list[Thumbnail] = []
    selected_states = {
        (1, 0, 0),
        (2, 1, 1),
        (6, 2, 0),
        (args.n_max, args.n_max - 1, args.n_max - 1),
    }
    for index, path in enumerate(paths):
        try:
            metadata = read_json(path.with_suffix(".json"))
            key = gallery_key(metadata)
            checks.require(key in expected, f"{path.name}: unexpected state/view {key}")
            checks.require(key not in seen, f"{path.name}: duplicate state/view {key}")
            seen.add(key)
            with Image.open(path) as image:
                image.load()
                checks.require(image.format == "PNG", f"{path.name}: not PNG data")
                checks.require(
                    image.size == (args.gallery_width, args.gallery_height),
                    f"{path.name}: wrong image dimensions {image.size}",
                )
                if key[:3] in selected_states:
                    thumbnail = image.convert("RGB")
                    thumbnail.thumbnail((640, 360), Image.Resampling.LANCZOS)
                    captures.append(Thumbnail(thumbnail, path.stem))
            export = metadata["export"]
            checks.require(
                export.get("width") == args.gallery_width
                and export.get("height") == args.gallery_height,
                f"{path.name}: metadata resolution mismatch",
            )
        except Exception as error:
            checks.require(False, f"{path.name}: {type(error).__name__}: {error}")
        if (index + 1) % 50 == 0:
            print(f"Decoded gallery PNGs: {index + 1}/{len(paths)}", flush=True)
    checks.require(seen == expected, f"Missing gallery state/views: {sorted(expected - seen)[:20]}")
    index_counts = check_gallery_index(args.gallery, {path.resolve() for path in paths}, checks)
    save_contact(captures, args.report / "gallery_contact.jpg", columns=2)
    return {
        "png_count": len(paths),
        "expected_png_count": len(expected),
        "unique_state_views": len(seen),
        "expected_unique_densities": len(expected) // 2,
        "resolution": [args.gallery_width, args.gallery_height],
        "indices": index_counts,
        "selected_contact_images": len(captures),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clips", type=Path, default=Path("outputs/bohm_clips"))
    parser.add_argument("--gallery", type=Path, default=Path("outputs/Orbitalgalerie"))
    parser.add_argument("--report", type=Path, default=Path("outputs/qa_media"))
    parser.add_argument("--clip-count", type=int, default=4)
    parser.add_argument(
        "--clip-files",
        nargs="+",
        metavar="NAME.mp4",
        help="Check only these files, including an explicitly named combined clip.",
    )
    parser.add_argument("--duration", type=float, default=22.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip-width", type=int, default=1920)
    parser.add_argument("--clip-height", type=int, default=1080)
    parser.add_argument("--gallery-width", type=int, default=3840)
    parser.add_argument("--gallery-height", type=int, default=2160)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--skip-clips", action="store_true")
    parser.add_argument("--skip-gallery", action="store_true")
    parser.add_argument("--skip-source", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    report: dict[str, Any] = {}
    if not args.skip_source:
        report["source"] = check_source(args.project, checks)
    if not args.skip_clips:
        report["videos"] = check_clips(args, checks)
    if not args.skip_gallery:
        report["gallery"] = check_gallery(args, checks)
    report["passed"] = checks.error_count == 0
    report["error_count"] = checks.error_count
    report["errors"] = checks.errors
    output = args.report / "verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "error_count": checks.error_count,
                "report": str(output.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
