#!/usr/bin/env python3
"""
Composite one mountain title image into every frame mask for a ratio.

Example:
    python place_mountain.py mt-baker 5x4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

MOUNTAIN_ROOT = Path("/Volumes/sandisk1/data_cache/mountain_mesh_data")
MASK_ROOT = Path("/Users/ryankuhn/Desktop/frame_masks")
SUPPORTED_MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Place the latest title image for a mountain into all frame masks "
            "for the requested ratio."
        )
    )
    parser.add_argument("mountain", help="Mountain folder slug, e.g. mt-baker")
    parser.add_argument("ratio", help="Ratio folder name, e.g. 5x4")
    return parser


def _existing_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {resolved}")
    return resolved


def _latest_title_image(mountain_dir: Path) -> Path:
    candidates = sorted(
        mountain_dir.glob("title_*.png"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No title images found in {mountain_dir}. Expected files matching title_*.png."
        )
    return candidates[0]


def _mask_files(mask_dir: Path) -> list[Path]:
    files = sorted(
        [
            file_path
            for file_path in mask_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_MASK_EXTENSIONS
        ],
        key=lambda file_path: file_path.name.lower(),
    )
    if not files:
        raise FileNotFoundError(
            f"No supported mask images found in {mask_dir}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_MASK_EXTENSIONS))}"
        )
    return files


def _output_path(output_dir: Path, title_image: Path, mask_path: Path) -> Path:
    safe_mask_stem = mask_path.stem.replace(" ", "_")
    return output_dir / f"{title_image.stem}__{safe_mask_stem}.png"


def run_for_mountain(mountain: str, ratio: str) -> int:
    project_root = Path(__file__).resolve().parent
    make_mockups_script = project_root / "make_mockups.py"
    if not make_mockups_script.is_file():
        raise FileNotFoundError(f"Required script not found: {make_mockups_script}")

    mountain_dir = _existing_dir(MOUNTAIN_ROOT / mountain, "Mountain")
    mask_dir = _existing_dir(MASK_ROOT / ratio, "Mask ratio")
    title_image = _latest_title_image(mountain_dir)
    masks = _mask_files(mask_dir)

    output_dir = mountain_dir / ratio
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mountain: {mountain}")
    print(f"Ratio: {ratio}")
    print(f"Title image: {title_image}")
    print(f"Masks found: {len(masks)}")
    print(f"Output directory: {output_dir}")

    failures: list[tuple[Path, str]] = []
    for index, mask_path in enumerate(masks, start=1):
        output_path = _output_path(output_dir, title_image, mask_path)
        cmd = [
            sys.executable,
            str(make_mockups_script),
            "--image",
            str(title_image),
            "--mockup",
            str(mask_path),
            "--output",
            str(output_path),
        ]
        print(f"[{index}/{len(masks)}] Processing mask: {mask_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "").strip()
            failures.append((mask_path, error_text or "Unknown error"))
            print(f"  Failed: {mask_path.name}")
            continue
        print(f"  Saved: {output_path.name}")

    success_count = len(masks) - len(failures)
    print(
        f"Finished {mountain} ({ratio}): {success_count} succeeded, {len(failures)} failed."
    )
    if failures:
        for mask_path, reason in failures:
            print(f"  - {mask_path.name}: {reason}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run_for_mountain(args.mountain, args.ratio)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
