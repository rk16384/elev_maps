"""
Add centered two-line mountain title text to the latest shadow render.

Usage:
    python add_mountain_titles.py <name> --title "Mt. Baker"

Behavior:
    - Reads latest shadow_*.png from /Volumes/sandisk1/data_cache/mountain_mesh_data/<name>/
    - Loads title defaults from mesh_settings.json -> title_settings
    - Uses system Montserrat font (by family name)
    - Saves output as title_<input_stem>.png in same folder
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path("/Volumes/sandisk1/data_cache/mountain_mesh_data")
WHITE_THRESHOLD = 250
POSTER_WIDTH_IN = 20.0
POSTER_HEIGHT_IN = 16.0
MARGIN_IN = 0.125


def find_latest_shadow_image(folder: Path) -> Path | None:
    images = list(folder.glob("shadow_*.png"))
    if not images:
        return None
    images.sort(key=lambda p: p.stat().st_mtime)
    return images[-1]


def parse_ratio(value: str) -> float:
    raw = value.strip()
    if ":" in raw:
        left, right = raw.split(":", maxsplit=1)
        a = float(left.strip())
        b = float(right.strip())
        if a <= 0 or b <= 0:
            raise ValueError("Aspect ratio components must be > 0.")
        return a / b

    ratio = float(raw)
    if ratio <= 0:
        raise ValueError("Aspect ratio must be > 0.")
    return ratio


def _font_search_dirs() -> list[Path]:
    return [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    ]


def _is_italic_font_file(path: Path) -> bool:
    name = path.name.lower()
    return ("italic" in name) or ("oblique" in name)


def resolve_system_font_path(family: str) -> Path:
    # Prefer regular variants first for predictable typography.
    preferred_names = [
        f"{family}-Regular.ttf",
        f"{family} Regular.ttf",
        f"{family}.ttf",
        f"{family}-Regular.otf",
        f"{family} Regular.otf",
        f"{family}.otf",
    ]

    for search_dir in _font_search_dirs():
        if not search_dir.exists():
            continue
        for filename in preferred_names:
            matches = list(search_dir.rglob(filename))
            if matches:
                return matches[0]

    # Broader fallback search.
    patterns = [f"*{family}*.ttf", f"*{family}*.otf", f"*{family}*.ttc"]
    non_italic_matches: list[Path] = []
    italic_matches: list[Path] = []
    for search_dir in _font_search_dirs():
        if not search_dir.exists():
            continue
        for pattern in patterns:
            matches = list(search_dir.rglob(pattern))
            if matches:
                for match in matches:
                    if _is_italic_font_file(match):
                        italic_matches.append(match)
                    else:
                        non_italic_matches.append(match)

    if non_italic_matches:
        non_italic_matches.sort()
        return non_italic_matches[0]
    if italic_matches:
        italic_matches.sort()
        return italic_matches[0]

    raise FileNotFoundError(
        f"Could not find system font family '{family}'. "
        "Install Montserrat in your OS fonts and retry."
    )


def load_font(family: str, size: int) -> ImageFont.FreeTypeFont:
    font_path = resolve_system_font_path(family)
    return ImageFont.truetype(str(font_path), size=size)


def load_settings(settings_path: Path) -> dict[str, Any]:
    if not settings_path.exists():
        raise FileNotFoundError(f"mesh_settings.json not found: {settings_path}")
    with settings_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("mesh_settings.json must contain a top-level JSON object.")
    return data


def as_rgb(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, list) and len(value) == 3:
        out = []
        for item in value:
            iv = int(item)
            if iv < 0 or iv > 255:
                raise ValueError("RGB values must be between 0 and 255.")
            out.append(iv)
        return (out[0], out[1], out[2])
    return default


def save_settings(settings_path: Path, data: dict[str, Any]) -> None:
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def non_white_bbox(image: Image.Image, threshold: int = WHITE_THRESHOLD) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size

    left = width
    top = height
    right = -1
    bottom = -1

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r < threshold or g < threshold or b < threshold:
                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

    if right < left or bottom < top:
        return None

    # PIL crop uses right/bottom as exclusive.
    return (left, top, right + 1, bottom + 1)


def add_bottom_space(image: Image.Image, extra_px: int) -> Image.Image:
    if extra_px <= 0:
        return image
    width, height = image.size
    out = Image.new("RGB", (width, height + extra_px), "white")
    out.paste(image, (0, 0))
    return out


def draw_centered_title(
    image: Image.Image,
    title: str,
    subtitle: str,
    *,
    font_family: str,
    title_font_size: int,
    subtitle_font_size: int,
    title_color: tuple[int, int, int],
    subtitle_color: tuple[int, int, int],
    title_weight_boost_px: int,
    subtitle_weight_boost_px: int,
    line_spacing_px: int,
    bottom_offset_px: int,
    content_gap_px: int,
) -> Image.Image:
    img = image.convert("RGB")
    title_font = load_font(font_family, title_font_size)
    subtitle_font = load_font(font_family, subtitle_font_size)
    draw = ImageDraw.Draw(img)

    tbox = draw.textbbox((0, 0), title, font=title_font)
    sbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    title_w = tbox[2] - tbox[0]
    title_h = tbox[3] - tbox[1]
    subtitle_w = sbox[2] - sbox[0]
    subtitle_h = sbox[3] - sbox[1]

    # PIL text bounding boxes can include non-zero offsets, so use relative
    # extents to position the full text block without clipping descenders.
    line2_rel_y = title_h + line_spacing_px
    min_rel_top = min(tbox[1], line2_rel_y + sbox[1])
    max_rel_bottom = max(tbox[3], line2_rel_y + sbox[3])
    block_h = max_rel_bottom - min_rel_top

    width, height = img.size
    content = non_white_bbox(img)
    if content is None:
        content_bottom = max(0, height // 2)
    else:
        content_bottom = content[3]

    # Center text block between content bottom and canvas bottom.
    available_top = content_bottom
    available_bottom = height
    available_height = available_bottom - available_top
    if available_height < block_h:
        # Expand bottom canvas if the block cannot fit.
        img = add_bottom_space(img, block_h - available_height)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        available_bottom = height

    block_top = available_top + ((available_bottom - available_top - block_h) // 2)
    title_y = block_top - min_rel_top

    # Offset draw positions by bbox origin so visual glyph bounds are centered.
    title_x = (width - title_w) // 2 - tbox[0]
    subtitle_y = title_y + line2_rel_y
    subtitle_x = (width - subtitle_w) // 2 - sbox[0]

    # Draw a tiny offset pass for a subtle faux-bold effect.
    if title_weight_boost_px > 0:
        draw.text(
            (title_x + title_weight_boost_px, title_y),
            title,
            fill=title_color,
            font=title_font,
        )
    if subtitle_weight_boost_px > 0:
        draw.text(
            (subtitle_x + subtitle_weight_boost_px, subtitle_y),
            subtitle,
            fill=subtitle_color,
            font=subtitle_font,
        )
    draw.text((title_x, title_y), title, fill=title_color, font=title_font)
    draw.text((subtitle_x, subtitle_y), subtitle, fill=subtitle_color, font=subtitle_font)
    return img


def _buffer_pixels(width: int, height: int) -> tuple[int, int]:
    """
    Convert physical 1/8" margins to pixels using a 20x16 poster reference.
    """
    buffer_x = max(1, int(round(width * (MARGIN_IN / POSTER_WIDTH_IN))))
    buffer_y = max(1, int(round(height * (MARGIN_IN / POSTER_HEIGHT_IN))))
    return buffer_x, buffer_y


def _trim_to_buffer_on_width(
    image: Image.Image,
    *,
    threshold: int = WHITE_THRESHOLD,
) -> Image.Image:
    img = image.convert("RGB")
    bounds = non_white_bbox(img, threshold=threshold)
    if bounds is None:
        return img

    width, _ = img.size
    left, _, right, _ = bounds
    buffer_x, _ = _buffer_pixels(*img.size)

    left_room = max(0, left - buffer_x)
    right_room = max(0, (width - right) - buffer_x)
    each_side_trim = min(left_room, right_room)
    if each_side_trim <= 0:
        return img

    return img.crop((each_side_trim, 0, width - each_side_trim, img.size[1]))


def _trim_to_buffer_on_height(
    image: Image.Image,
    *,
    threshold: int = WHITE_THRESHOLD,
) -> Image.Image:
    img = image.convert("RGB")
    bounds = non_white_bbox(img, threshold=threshold)
    if bounds is None:
        return img

    _, height = img.size
    _, top, _, bottom = bounds
    _, buffer_y = _buffer_pixels(*img.size)

    top_room = max(0, top - buffer_y)
    bottom_room = max(0, (height - bottom) - buffer_y)
    each_side_trim = min(top_room, bottom_room)
    if each_side_trim <= 0:
        return img

    return img.crop((0, each_side_trim, img.size[0], height - each_side_trim))


def _pad_to_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    img = image.convert("RGB")
    width, height = img.size
    current_ratio = width / height

    if abs(current_ratio - target_ratio) <= 1e-6:
        return img

    if current_ratio > target_ratio:
        # Too wide: add top/bottom white space.
        new_height = int(math.ceil(width / target_ratio))
        pad_total = max(0, new_height - height)
        top_pad = pad_total // 2
        bottom_pad = pad_total - top_pad
        out = Image.new("RGB", (width, new_height), "white")
        out.paste(img, (0, top_pad))
        return out

    # Too tall: add left/right white space.
    new_width = int(math.ceil(height * target_ratio))
    pad_total = max(0, new_width - width)
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad
    out = Image.new("RGB", (new_width, height), "white")
    out.paste(img, (left_pad, 0))
    return out


def enforce_aspect_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    img = image.convert("RGB")
    width, height = img.size
    current_ratio = width / height

    # Trim only on the dominant axis, symmetrically, down to 1/8" buffer.
    if current_ratio > target_ratio:
        trimmed = _trim_to_buffer_on_width(img, threshold=WHITE_THRESHOLD)
    elif current_ratio < target_ratio:
        trimmed = _trim_to_buffer_on_height(img, threshold=WHITE_THRESHOLD)
    else:
        trimmed = img

    # Then add whitespace on the opposite axis to hit exact ratio.
    return _pad_to_ratio(trimmed, target_ratio=target_ratio)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add centered mountain title text to latest shadow image.",
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="mt-baker",
        help="Folder name within /Volumes/sandisk1/data_cache/mountain_mesh_data/ (e.g., mt-baker)",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Main title text (line 1), e.g. 'Mt. Baker'",
    )
    parser.add_argument(
        "--elevation",
        help="Subtitle text override (line 2), e.g. 'Elevation - 10,781 ft'",
    )
    parser.add_argument(
        "--aspect-ratio",
        help="Optional output aspect ratio, e.g. 14:11",
    )
    parser.add_argument(
        "--title-weight-boost-px",
        "--title_weight_boost_px",
        dest="title_weight_boost_px",
        type=int,
        help="Optional faux-bold x-offset for title text in pixels.",
    )
    parser.add_argument(
        "--subtitle-weight-boost-px",
        "--subtitle_weight_boost_px",
        dest="subtitle_weight_boost_px",
        type=int,
        help="Optional faux-bold x-offset for subtitle text in pixels.",
    )
    parser.add_argument(
        "--overwrite-title-settings",
        "--overwrite_title_settings",
        action="store_true",
        help=(
            "Overwrite existing mesh_settings.title_settings using the effective "
            "values for this run."
        ),
    )
    args = parser.parse_args()

    folder = BASE_DIR / args.name
    if not folder.exists():
        print(f"Error: folder not found: {folder}")
        return 1

    input_path = find_latest_shadow_image(folder)
    if input_path is None:
        print(f"Error: no shadow_*.png found in {folder}")
        return 1

    settings_path = folder / "mesh_settings.json"
    try:
        settings = load_settings(settings_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    title_settings = settings.get("title_settings", {})
    if not isinstance(title_settings, dict):
        print("Error: mesh_settings.json 'title_settings' must be an object.")
        return 1

    font_family = str(title_settings.get("font_family", "Montserrat"))
    elevation_text = str(title_settings.get("elevation_text", "Elevation - 10,781 ft"))
    subtitle = args.elevation if args.elevation is not None else elevation_text
    if not subtitle:
        print(
            "Error: subtitle text is empty. "
            "Provide --elevation or set title_settings.elevation_text."
        )
        return 1

    title_font_size = int(title_settings.get("title_font_size", 120))
    subtitle_font_size = int(title_settings.get("subtitle_font_size", 80))
    title_weight_boost_px = (
        int(args.title_weight_boost_px)
        if args.title_weight_boost_px is not None
        else int(title_settings.get("title_weight_boost_px", 1))
    )
    subtitle_weight_boost_px = (
        int(args.subtitle_weight_boost_px)
        if args.subtitle_weight_boost_px is not None
        else int(title_settings.get("subtitle_weight_boost_px", 1))
    )
    line_spacing_px = int(title_settings.get("line_spacing_px", 50))
    bottom_offset_px = int(title_settings.get("bottom_offset_px", 100))
    content_gap_px = int(title_settings.get("content_gap_px", 5))

    title_color = as_rgb(title_settings.get("title_color"), default=(50, 50, 50))
    subtitle_color = as_rgb(title_settings.get("subtitle_color"), default=(120, 120, 120))

    ratio_str = args.aspect_ratio or title_settings.get("aspect_ratio_default", "4:3")
    target_ratio = None
    if ratio_str:
        try:
            target_ratio = parse_ratio(str(ratio_str))
        except Exception as exc:
            print(f"Error: invalid aspect ratio '{ratio_str}': {exc}")
            return 1

    effective_title_settings: dict[str, Any] = {
        "font_family": font_family,
        "elevation_text": subtitle,
        "title_font_size": title_font_size,
        "subtitle_font_size": subtitle_font_size,
        "title_color": [title_color[0], title_color[1], title_color[2]],
        "subtitle_color": [subtitle_color[0], subtitle_color[1], subtitle_color[2]],
        "title_weight_boost_px": title_weight_boost_px,
        "subtitle_weight_boost_px": subtitle_weight_boost_px,
        "line_spacing_px": line_spacing_px,
        "bottom_offset_px": bottom_offset_px,
        "content_gap_px": content_gap_px,
        "aspect_ratio_default": str(ratio_str) if ratio_str else "4:3",
    }

    title_settings_missing = "title_settings" not in settings
    should_overwrite_title_settings = args.overwrite_title_settings
    if title_settings_missing or should_overwrite_title_settings:
        settings["title_settings"] = effective_title_settings
        try:
            save_settings(settings_path, settings)
        except Exception as exc:
            print(f"Error: could not update mesh settings: {exc}")
            return 1
        if title_settings_missing:
            print("Saved default title_settings to mesh_settings.json")
        else:
            print("Overwrote title_settings in mesh_settings.json")

    try:
        base_image = Image.open(input_path).convert("RGB")
        if target_ratio is not None:
            base_image = enforce_aspect_ratio(base_image, target_ratio=target_ratio)
        result = draw_centered_title(
            base_image,
            title=args.title,
            subtitle=subtitle,
            font_family=font_family,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            title_color=title_color,
            subtitle_color=subtitle_color,
            title_weight_boost_px=title_weight_boost_px,
            subtitle_weight_boost_px=subtitle_weight_boost_px,
            line_spacing_px=line_spacing_px,
            bottom_offset_px=bottom_offset_px,
            content_gap_px=content_gap_px,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    out_path = input_path.parent / f"title_{input_path.stem}.png"
    result.save(out_path)
    print(f"Using: {input_path.name}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
