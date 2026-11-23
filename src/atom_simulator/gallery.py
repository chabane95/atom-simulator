"""Render the distinct orbital densities with one shared OpenGL context.

Each state has a full view and a geometric cutaway.  Positive and negative m
have identical densities, so m >= 0 is sufficient for this static gallery.
"""

import json
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

import moderngl
from PIL import Image, ImageOps

from atom_simulator.config import BACKGROUND, BEAD_RADIUS, RANDOM_SEED
from atom_simulator.export import write_metadata
from atom_simulator.physics import PointCloud, QuantumNumbers, sample_orbital
from atom_simulator.renderer import OrbitalRenderer, RenderSettings

SOURCE_VERSION: str = "0.2"
GALLERY_SEED: int = RANDOM_SEED
THUMBNAIL_SIZE: tuple[int, int] = (512, 288)


def image_path(state: QuantumNumbers, cutaway: bool) -> Path:
    view_name = "schnitt" if cutaway else "voll"
    filename = f"n{state.n:02d}_l{state.l:02d}_m{state.m:02d}_{view_name}.png"
    return Path(f"n_{state.n:02d}") / filename


def thumbnail_path(relative_image: Path) -> Path:
    return Path("_thumbs") / relative_image.with_suffix(".jpg")


def index_entry(state: QuantumNumbers, cutaway: bool) -> dict[str, Any]:
    return {
        "file": image_path(state, cutaway).as_posix(),
        "n": state.n,
        "l": state.l,
        "m": state.m,
        "cutaway": cutaway,
    }


def export_details(settings: RenderSettings, cutaway: bool) -> dict[str, Any]:
    return {
        "kind": "still",
        "cut_angle_degrees": 110.0 if cutaway else 0.0,
        "width": settings.width,
        "height": settings.height,
        "supersample": 1,
        "source_version": SOURCE_VERSION,
    }


def complete_view(
    output: Path, state: QuantumNumbers, settings: RenderSettings, points: int, cutaway: bool
) -> bool:
    """Only reuse a complete, readable export made with this configuration."""
    relative = image_path(state, cutaway)
    image_file = output / relative
    metadata_file = image_file.with_suffix(".json")
    thumbnail_file = output / thumbnail_path(relative)
    if not image_file.is_file() or not metadata_file.is_file() or not thumbnail_file.is_file():
        return False
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return False
        sampling = metadata.get("sampling")
        details = metadata.get("export")
        if not isinstance(sampling, dict) or not isinstance(details, dict):
            return False
        if metadata.get("quantum_numbers") != asdict(state):
            return False
        if metadata.get("render_settings") != asdict(settings):
            return False
        if sampling.get("count") != points or sampling.get("seed") != GALLERY_SEED:
            return False
        expected = export_details(settings, cutaway)
        if not all(details.get(key) == value for key, value in expected.items()):
            return False
        with Image.open(image_file) as image:
            if image.format != "PNG" or image.size != (settings.width, settings.height):
                return False
            image.verify()
        with Image.open(thumbnail_file) as thumbnail:
            if thumbnail.format != "JPEG" or thumbnail.size != THUMBNAIL_SIZE:
                return False
            thumbnail.verify()
    except (OSError, ValueError, TypeError, KeyError, SyntaxError):
        return False
    return True


def save_view(
    output: Path, state: QuantumNumbers, cloud: PointCloud, renderer: OrbitalRenderer, cutaway: bool
) -> dict[str, Any]:
    """Publish each image, thumbnail and sidecar only after writing them fully."""
    relative = image_path(state, cutaway)
    image_file = output / relative
    thumbnail_file = output / thumbnail_path(relative)
    metadata_file = image_file.with_suffix(".json")
    image_file.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_file.parent.mkdir(parents=True, exist_ok=True)
    pending_image = image_file.with_name(image_file.name + ".pending")
    pending_thumbnail = thumbnail_file.with_name(thumbnail_file.name + ".pending")
    pending_metadata = metadata_file.with_name(metadata_file.name + ".pending")

    cut_angle = 110.0 if cutaway else 0.0
    renderer.draw(azimuth=42.0, elevation=20.0, distance=3.0, cut_angle=cut_angle)
    image = renderer.read_image()
    try:
        image.save(pending_image, format="PNG")
        thumbnail = ImageOps.pad(
            image, THUMBNAIL_SIZE, method=Image.Resampling.LANCZOS, color="#efede5"
        )
        thumbnail.save(pending_thumbnail, format="JPEG", quality=88, optimize=True)
        thumbnail.close()
        write_metadata(
            pending_metadata, state, cloud, renderer, export_details(renderer.settings, cutaway)
        )
        pending_image.replace(image_file)
        pending_thumbnail.replace(thumbnail_file)
        pending_metadata.replace(metadata_file)
    finally:
        image.close()
    return index_entry(state, cutaway)


def render_pair(
    context: moderngl.Context,
    output: Path,
    state: QuantumNumbers,
    settings: RenderSettings,
    points: int,
) -> list[dict[str, Any]]:
    """One sampled state and one renderer produce the two matching views."""
    cloud = sample_orbital(state, points, GALLERY_SEED)
    renderer: OrbitalRenderer | None = None
    try:
        renderer = OrbitalRenderer(context, cloud, state, settings)
        full_entry = save_view(output, state, cloud, renderer, cutaway=False)
        cut_entry = save_view(output, state, cloud, renderer, cutaway=True)
        return [full_entry, cut_entry]
    finally:
        if renderer is not None:
            renderer.close()


def html_card(entry: dict[str, Any]) -> str:
    filename = escape(entry["file"], quote=True)
    thumbnail = escape(thumbnail_path(Path(entry["file"])).as_posix(), quote=True)
    view = "cut" if entry["cutaway"] else "full"
    view_label = "Schnitt" if entry["cutaway"] else "Voll"
    label = f"n = {entry['n']} · ℓ = {entry['l']} · m = {entry['m']}"
    return (
        f'<article class="orbital" data-n="{entry["n"]}" data-view="{view}">'
        f'<a href="{filename}" target="_blank" rel="noopener">'
        f'<img src="{thumbnail}" width="512" height="288" loading="lazy" '
        f'decoding="async" alt="{label}, {view_label}"></a>'
        f'<div class="caption"><span>{label}</span><span>{view_label}</span></div></article>'
    )


def gallery_html(entries: list[dict[str, Any]], max_n: int, settings: RenderSettings) -> str:
    options = "".join(f'<option value="{n}">n = {n}</option>' for n in range(1, max_n + 1))
    cards = "\n".join(html_card(entry) for entry in entries)
    header = (
        f"{len(entries) // 2} Dichten · {len(entries)} Ansichten · "
        f"{settings.width} × {settings.height}"
    )
    return (
        """<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wasserstoff · Orbitalgalerie</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#333a3d;background:#f3f1eb}
*{box-sizing:border-box}body{margin:0}main{max-width:1600px;padding:48px 28px 64px;margin:auto}
header{display:flex;justify-content:space-between;align-items:end;gap:24px;flex-wrap:wrap}
h1{font-family:Georgia,serif;font-size:clamp(30px,4vw,48px);font-weight:normal;margin:0 0 14px}
.description{color:#626b70;line-height:1.65;margin:0}.controls{display:flex;gap:16px;align-items:end;flex-wrap:wrap}
label{display:grid;gap:7px;font-size:14px}
select{font:inherit;background:white;border:1px solid #c7ccca;
border-radius:7px;color:#333a3d;padding:10px 30px 10px 12px;min-width:145px}
#result{font-size:14px;color:#626b70;margin:30px 0 16px}.grid{display:grid;
grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:22px}.orbital{background:#fff;
border:1px solid #dedfd8;border-radius:8px;overflow:hidden}.orbital[hidden]{display:none}
.orbital a{display:block}.orbital a:focus-visible{outline:3px solid #71909e;outline-offset:-3px}
.orbital img{display:block;width:100%;height:auto;aspect-ratio:16/9;background:#eeeae2}
.caption{display:flex;justify-content:space-between;gap:12px;padding:14px 16px;font-size:14px}
.caption span:last-child{color:#747d80}
footer{margin-top:32px;color:#697175;font-size:13px;line-height:1.7}
</style></head><body><main><header><div><h1>Wasserstoff · Orbitalgalerie</h1>
<p class="description">"""
        + header
        + """</p></div><div class="controls">
<label>Hauptquantenzahl<select id="n"><option value="all">Alle n</option>"""
        + options
        + """</select></label>
<label>Ansicht<select id="view"><option value="all">Voll und Schnitt</option>
<option value="full">Voll</option><option value="cut">Schnitt</option>
</select></label></div></header>
<p id="result" aria-live="polite"></p><section class="grid">"""
        + cards
        + """</section>
<footer>Komplexe Kugelflächenfunktionen: +m und −m besitzen dieselbe Dichte; gezeigt wird m ≥ 0.<br>
Jeder Zustand ist automatisch eingepasst. Die Ansichten haben keine gemeinsame räumliche Skala.
Ein Klick öffnet die vollständige PNG-Datei.</footer></main>
<script>
const nSelect=document.querySelector('#n');
const viewSelect=document.querySelector('#view');
const cards=Array.from(document.querySelectorAll('.orbital'));
function filterGallery(){
  let visible=0;
  cards.forEach(card=>{
    const matchesN=nSelect.value==='all'||card.dataset.n===nSelect.value;
    const matchesView=viewSelect.value==='all'||card.dataset.view===viewSelect.value;
    card.hidden=!(matchesN&&matchesView);
    if(!card.hidden) visible++;
  });
  document.querySelector('#result').textContent=visible+' Ansichten';
}
nSelect.addEventListener('change',filterGallery);
viewSelect.addEventListener('change',filterGallery);
filterGallery();
</script></body></html>
"""
    )


def write_indices(
    output: Path, entries: list[dict[str, Any]], max_n: int, settings: RenderSettings
) -> None:
    manifest = {"count": len(entries), "states": len(entries) // 2, "images": entries}
    pending_json = output / "index.json.pending"
    pending_html = output / "index.html.pending"
    pending_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending_html.write_text(gallery_html(entries, max_n, settings), encoding="utf-8")
    pending_json.replace(output / "index.json")
    pending_html.replace(output / "index.html")


def render_gallery(
    output: Path, max_n: int = 12, points: int = 450_000, width: int = 3840, height: int = 2160
) -> None:
    """Build or resume all distinct densities through max_n, in two views each."""
    if type(max_n) is not int or type(points) is not int or points < 1:
        raise ValueError("max_n and points must be positive integers.")
    QuantumNumbers(max_n, 0, 0)
    if type(width) is not int or type(height) is not int:
        raise ValueError("Gallery width and height must be integers.")
    settings = RenderSettings(
        width=width, height=height, bead_radius=BEAD_RADIUS, background=BACKGROUND
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    total_states = max_n * (max_n + 1) * (max_n + 2) // 6
    entries: list[dict[str, Any]] = []
    context: moderngl.Context | None = None
    completed = 0
    resumed = 0
    n, angular, magnetic = 1, 0, 0
    try:
        while n <= max_n:
            state = QuantumNumbers(n, angular, magnetic)
            full_ready = complete_view(output, state, settings, points, cutaway=False)
            cut_ready = complete_view(output, state, settings, points, cutaway=True)
            if full_ready and cut_ready:
                entries.extend([index_entry(state, False), index_entry(state, True)])
                resumed += 1
            else:
                if context is None:
                    context = moderngl.create_standalone_context(require=330)
                    print(f"Gallery GPU: {context.info['GL_RENDERER']}", flush=True)
                entries.extend(render_pair(context, output, state, settings, points))
            completed += 1
            if completed % 10 == 0 or completed == total_states:
                print(f"Gallery states: {completed}/{total_states} ({resumed} resumed)", flush=True)
            magnetic += 1
            if magnetic > angular:
                magnetic = 0
                angular += 1
            if angular >= n:
                angular = 0
                n += 1
        write_indices(output, entries, max_n, settings)
    finally:
        if context is not None:
            context.release()
