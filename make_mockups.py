"""
Create framed room mockups by replacing a magenta placeholder.

This module can be imported by other code or executed as a CLI script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage


def _validate_path(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _find_magenta_component(
    mockup_rgb: Image.Image,
    lower_hsv: Sequence[int],
    upper_hsv: Sequence[int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    hsv = np.asarray(mockup_rgb.convert("HSV"), dtype=np.uint8)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h_min, s_min, v_min = [int(x) for x in lower_hsv]
    h_max, s_max, v_max = [int(x) for x in upper_hsv]

    if h_min <= h_max:
        h_ok = (h >= h_min) & (h <= h_max)
    else:
        # Hue wraparound support (e.g. 250..20).
        h_ok = (h >= h_min) | (h <= h_max)

    raw_mask = h_ok & (s >= s_min) & (s <= s_max) & (v >= v_min) & (v <= v_max)

    structure = np.ones((3, 3), dtype=bool)
    cleaned = ndimage.binary_opening(raw_mask, structure=structure, iterations=1)
    cleaned = ndimage.binary_closing(cleaned, structure=structure, iterations=2)
    cleaned = ndimage.binary_fill_holes(cleaned)

    labels, count = ndimage.label(cleaned)
    if count == 0:
        raise ValueError(
            "Could not find a magenta placeholder mask. "
            "Try adjusting --lower-hsv/--upper-hsv."
        )

    areas = ndimage.sum(cleaned, labels, index=np.arange(1, count + 1))
    largest_idx = int(np.argmax(areas)) + 1
    largest = labels == largest_idx

    y_coords, x_coords = np.where(largest)
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    x_min, x_max = int(x_coords.min()), int(x_coords.max())

    bbox = (x_min, y_min, x_max + 1, y_max + 1)
    return largest, bbox


def _compute_channel_stats(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        pixels = rgb.reshape(-1, 3)
    else:
        pixels = rgb[mask]
    if pixels.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32), np.array(
            [40.0, 40.0, 40.0], dtype=np.float32
        )
    return pixels.mean(axis=0), pixels.std(axis=0)


def _adjust_for_scene_lighting(
    poster_rgb: Image.Image,
    mockup_rgb: Image.Image,
    placeholder_mask: np.ndarray,
    realism_strength: float,
) -> Image.Image:
    realism_strength = float(np.clip(realism_strength, 0.0, 1.0))
    if realism_strength <= 0:
        return poster_rgb

    ring_width = max(6, int(min(mockup_rgb.size) * 0.008))
    ring_mask = ndimage.binary_dilation(placeholder_mask, iterations=ring_width) & (~placeholder_mask)

    mockup_arr = np.asarray(mockup_rgb, dtype=np.float32)
    poster_arr = np.asarray(poster_rgb, dtype=np.float32)

    target_mean, target_std = _compute_channel_stats(mockup_arr, ring_mask)
    source_mean, source_std = _compute_channel_stats(poster_arr)

    contrast_scale = target_std / np.maximum(source_std, 1.0)
    contrast_scale = np.clip(contrast_scale, 0.75, 1.35)
    brightness_shift = target_mean - (source_mean * contrast_scale)
    brightness_shift = np.clip(brightness_shift, -30.0, 30.0)

    adjusted = poster_arr * contrast_scale + brightness_shift
    adjusted = np.clip(adjusted, 0.0, 255.0).astype(np.uint8)
    adjusted_img = Image.fromarray(adjusted, mode="RGB")

    # Match saturation conservatively so the inserted print feels less pasted-on.
    poster_hsv = np.asarray(poster_rgb.convert("HSV"), dtype=np.float32)
    ring_hsv = np.asarray(mockup_rgb.convert("HSV"), dtype=np.float32)
    source_sat = float(poster_hsv[..., 1].mean()) + 1e-6
    target_sat = float(ring_hsv[..., 1][ring_mask].mean()) if np.any(ring_mask) else source_sat
    sat_factor = float(np.clip(target_sat / source_sat, 0.85, 1.15))
    adjusted_img = ImageEnhance.Color(adjusted_img).enhance(sat_factor)

    return Image.blend(poster_rgb, adjusted_img, realism_strength)


def create_mockup(
    image_path: str | Path,
    mockup_path: str | Path,
    output_path: str | Path | None = None,
    *,
    lower_hsv: tuple[int, int, int] = (185, 70, 70),
    upper_hsv: tuple[int, int, int] = (240, 255, 255),
    realism: bool = True,
    realism_strength: float = 0.45,
    feather_px: float = 1.75,
    debug: bool = False,
) -> Image.Image:
    """
    Replace the magenta placeholder in a mockup with the provided image.

    The inserted image is intentionally stretched to fit the mask bounding box.
    """
    source_path = _validate_path(image_path, "Input image")
    frame_path = _validate_path(mockup_path, "Mockup image")

    mockup_rgb = Image.open(frame_path).convert("RGB")
    poster_rgb = Image.open(source_path).convert("RGB")

    placeholder_mask, bbox = _find_magenta_component(
        mockup_rgb=mockup_rgb,
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv,
    )

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if width <= 1 or height <= 1:
        raise ValueError(f"Detected mask is too small to use: bbox={bbox}")

    if realism:
        poster_rgb = _adjust_for_scene_lighting(
            poster_rgb=poster_rgb,
            mockup_rgb=mockup_rgb,
            placeholder_mask=placeholder_mask,
            realism_strength=realism_strength,
        )

    poster_resized = poster_rgb.resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")

    local_mask = placeholder_mask[top:bottom, left:right]
    mask_img = Image.fromarray((local_mask.astype(np.uint8) * 255), mode="L")
    if feather_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    poster_resized.putalpha(mask_img)

    output = mockup_rgb.convert("RGBA")
    output.alpha_composite(poster_resized, dest=(left, top))

    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        output.save(target)
        if debug:
            print(f"Saved mockup: {target}")
    elif debug:
        print("Generated mockup image in memory (not saved).")

    if debug:
        print(f"Detected placeholder bbox: {bbox}")

    return output


def create_mockups_batch(
    jobs: Sequence[dict[str, Any]],
    *,
    lower_hsv: tuple[int, int, int] = (185, 70, 70),
    upper_hsv: tuple[int, int, int] = (240, 255, 255),
    realism: bool = True,
    realism_strength: float = 0.45,
    feather_px: float = 1.75,
    continue_on_error: bool = True,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    Create multiple mockups in sequence.

    Each job dict must include: image, mockup, output.
    Optional per-job overrides:
      - lower_hsv (list/tuple of 3 ints)
      - upper_hsv (list/tuple of 3 ints)
      - realism (bool)
      - realism_strength (float)
      - feather_px (float)
    """
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            msg = f"Job {index} is not a dict."
            if continue_on_error:
                results.append({"index": index, "status": "error", "error": msg, "job": job})
                continue
            raise ValueError(msg)

        image = job.get("image")
        mockup = job.get("mockup")
        output = job.get("output")
        if not image or not mockup or not output:
            msg = f"Job {index} requires image, mockup, and output."
            if continue_on_error:
                results.append({"index": index, "status": "error", "error": msg, "job": job})
                continue
            raise ValueError(msg)

        try:
            create_mockup(
                image_path=image,
                mockup_path=mockup,
                output_path=output,
                lower_hsv=tuple(job.get("lower_hsv", lower_hsv)),
                upper_hsv=tuple(job.get("upper_hsv", upper_hsv)),
                realism=bool(job.get("realism", realism)),
                realism_strength=float(job.get("realism_strength", realism_strength)),
                feather_px=float(job.get("feather_px", feather_px)),
                debug=debug,
            )
            results.append(
                {
                    "index": index,
                    "status": "ok",
                    "output": str(Path(output).expanduser().resolve()),
                }
            )
        except Exception as exc:
            item = {"index": index, "status": "error", "error": str(exc), "job": job}
            results.append(item)
            if not continue_on_error:
                raise

    return results


def _load_batch_manifest(path_value: str | Path) -> list[dict[str, Any]]:
    manifest_path = _validate_path(path_value, "Batch manifest")
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Batch manifest must be a JSON list of jobs.")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Insert a poster image into a magenta mask placeholder mockup."
    )
    parser.add_argument("--image", help="Path to poster image (png/jpeg/etc).")
    parser.add_argument("--mockup", help="Path to mockup image with magenta placeholder.")
    parser.add_argument("--output", help="Path for output composited image.")
    parser.add_argument(
        "--batch-manifest",
        help="Path to JSON list of jobs. Each job needs image, mockup, output.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="In batch mode, stop immediately on first failed job.",
    )
    parser.add_argument(
        "--no-realism",
        action="store_true",
        help="Disable lighting/color realism matching for the inserted image.",
    )
    parser.add_argument(
        "--realism-strength",
        type=float,
        default=0.1,
        help="Blend amount for realism adjustments (0.0 to 1.0). Default: 0.45",
    )
    parser.add_argument(
        "--feather-px",
        type=float,
        default=1.75,
        help="Edge feather blur radius for mask in pixels. Default: 1.75",
    )
    parser.add_argument(
        "--lower-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(185, 70, 70),
        help="Lower HSV threshold for magenta mask detection (0-255 each).",
    )
    parser.add_argument(
        "--upper-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(240, 255, 255),
        help="Upper HSV threshold for magenta mask detection (0-255 each).",
    )
    parser.add_argument("--debug", action="store_true", help="Print extra diagnostic output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.batch_manifest:
            if args.image or args.mockup or args.output:
                raise ValueError(
                    "Use either --batch-manifest or --image/--mockup/--output, not both."
                )
            jobs = _load_batch_manifest(args.batch_manifest)
            results = create_mockups_batch(
                jobs=jobs,
                lower_hsv=tuple(args.lower_hsv),
                upper_hsv=tuple(args.upper_hsv),
                realism=not args.no_realism,
                realism_strength=args.realism_strength,
                feather_px=args.feather_px,
                continue_on_error=not args.stop_on_error,
                debug=args.debug,
            )

            ok_count = sum(1 for item in results if item["status"] == "ok")
            err_count = len(results) - ok_count
            print(f"Batch finished: {ok_count} succeeded, {err_count} failed.")
            if err_count > 0:
                for item in results:
                    if item["status"] == "error":
                        print(f"  - Job {item['index']} failed: {item['error']}")
                return 1
            return 0

        if not args.image or not args.mockup or not args.output:
            raise ValueError(
                "Single mode requires --image, --mockup, and --output. "
                "Or provide --batch-manifest for batch mode."
            )

        create_mockup(
            image_path=args.image,
            mockup_path=args.mockup,
            output_path=args.output,
            lower_hsv=tuple(args.lower_hsv),
            upper_hsv=tuple(args.upper_hsv),
            realism=not args.no_realism,
            realism_strength=args.realism_strength,
            feather_px=args.feather_px,
            debug=args.debug,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
