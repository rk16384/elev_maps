"""
Drop shadow editor for mesh render images.

Run after the interactive mesh viewer. Loads a mesh_render_*.png (mostly white
with grayscale object), adds an adjustable light-gray drop shadow, and lets the
user save the composite at full resolution.

Usage:
    python drop_shadow_editor.py <name>
    python drop_shadow_editor.py <name> --auto-accept

    Where <name> is a folder within /Volumes/sandisk1/data_cache/mountain_mesh_data/ (e.g., mt-baker).
    The script will find and use the most recent mesh_render_*.png in that folder.

Controls:
    - Sliders: Offset X, Offset Y, Blur (diffusion)
    - Arrow keys: Move shadow (same as sliders)
    - 's' or Return: Save full-resolution image with shadow
    - 'q' or close window: Quit
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter

# Luminance threshold: pixels darker than this are "object" for the mask
OBJECT_LUMINANCE_THRESHOLD = 255
# Light gray for shadow (RGB)
SHADOW_COLOR = (200, 200, 200)
DEFAULT_SHADOW_SETTINGS = {
    "offset_x": 175,
    "offset_y": 221,
    "blur_radius": 50,
    "shadow_gray": 159,
}
BASE_DIR = Path("/Volumes/sandisk1/data_cache/mountain_mesh_data")


def load_image(path: Path) -> Image.Image:
    """Load image as RGB."""
    img = Image.open(path).convert("RGB")
    return img


def build_object_mask(img: Image.Image) -> Image.Image:
    """Build a mask of non-white (object) pixels. Returns L mode 0/255."""
    import numpy as np
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    mask = (luminance < OBJECT_LUMINANCE_THRESHOLD).astype("uint8") * 255
    return Image.fromarray(mask, mode="L")


def build_image_mask(img: Image.Image, block_size: int = 1) -> Image.Image:
    """Build an image mask using flood-fill from edges with block-based sampling.
    
    This creates a closed polygon mask around the entire object by finding
    all background pixels connected to the image edges. Uses block averaging
    to prevent flood fill from leaking through small gaps.
    
    Args:
        img: Input image
        block_size: Size of pixel blocks for flood fill (default 15x15).
                   Gaps smaller than this won't let flood fill through.
    """
    import numpy as np
    from scipy import ndimage
    
    arr = np.array(img)
    h_orig, w_orig = arr.shape[:2]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Downsample luminance using block averaging
    # Pad to make dimensions divisible by block_size
    h_pad = (block_size - h_orig % block_size) % block_size
    w_pad = (block_size - w_orig % block_size) % block_size
    luminance_padded = np.pad(luminance, ((0, h_pad), (0, w_pad)), mode='edge')
    
    # Reshape and take mean of each block
    h_padded, w_padded = luminance_padded.shape
    h_blocks = h_padded // block_size
    w_blocks = w_padded // block_size
    
    luminance_blocks = luminance_padded.reshape(h_blocks, block_size, w_blocks, block_size)
    luminance_downsampled = luminance_blocks.mean(axis=(1, 3))
    
    # Create a binary image at block resolution: True where block is "background-like"
    is_background_color = luminance_downsampled >= OBJECT_LUMINANCE_THRESHOLD
    
    # Create edge mask at block resolution
    h_ds, w_ds = is_background_color.shape
    edge_mask = np.zeros((h_ds, w_ds), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    
    # Seeds: edge blocks that are background-colored
    seeds = edge_mask & is_background_color
    
    # Label connected components of background-colored blocks
    labeled, num_features = ndimage.label(is_background_color)
    
    # Find which labels touch the edges (are true background)
    edge_labels = set(labeled[seeds].flatten()) - {0}
    
    # Background is any block whose label is in edge_labels
    background_ds = np.isin(labeled, list(edge_labels))
    
    # Upscale back to original resolution
    background_upscaled = np.repeat(np.repeat(background_ds, block_size, axis=0), block_size, axis=1)
    # Crop back to original size
    background = background_upscaled[:h_orig, :w_orig]
    
    # Object mask is everything NOT background
    mask = (~background).astype("uint8") * 255
    return Image.fromarray(mask, mode="L")


def offset_mask(mask: Image.Image, dx: int, dy: int) -> Image.Image:
    """Shift mask by (dx, dy); pixels that shift off edge are lost (0)."""
    w, h = mask.size
    out = Image.new("L", (w, h), 0)
    if dx == 0 and dy == 0:
        return mask.copy()
    box_src = (
        max(0, -dx),
        max(0, -dy),
        min(w, w - dx),
        min(h, h - dy),
    )
    box_dst = (
        max(0, dx),
        max(0, dy),
        max(0, dx) + box_src[2] - box_src[0],
        max(0, dy) + box_src[3] - box_src[1],
    )
    if box_src[2] <= box_src[0] or box_src[3] <= box_src[1]:
        return out
    out.paste(mask.crop(box_src), box_dst)
    return out


def build_shadow_layer(
    original: Image.Image,
    mask: Image.Image,
    offset_x: int,
    offset_y: int,
    blur_radius: float,
    shadow_gray: int = 200,
) -> Image.Image:
    """Build shadow layer: white bg with gray blurred shadow."""
    w, h = original.size
    shifted = offset_mask(mask, offset_x, offset_y)
    if blur_radius > 0.5:
        shifted = shifted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shadow_layer = Image.new("RGB", (w, h), (255, 255, 255))
    shadow_color = (shadow_gray, shadow_gray, shadow_gray)
    gray_img = Image.new("RGB", (w, h), shadow_color)
    shadow_layer.paste(gray_img, (0, 0), mask=shifted)
    return shadow_layer


def composite_with_shadow(
    original: Image.Image,
    offset_x: int,
    offset_y: int,
    blur_radius: float,
    shadow_gray: int = 200,
) -> Image.Image:
    """Full pipeline: object mask for shadow, image mask for compositing."""
    shadow_mask = build_object_mask(original)
    image_mask = build_image_mask(original)
    shadow_layer = build_shadow_layer(
        original, shadow_mask, offset_x, offset_y, blur_radius, shadow_gray
    )
    result = shadow_layer.copy()
    result.paste(original, (0, 0), mask=image_mask)
    return result


def load_mesh_settings(settings_path: Path) -> dict:
    """Load mesh settings JSON if present, otherwise return empty dict."""
    if not settings_path.exists():
        return {}
    try:
        with open(settings_path, "r") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def parse_shadow_settings(mesh_settings: dict) -> dict:
    """Return validated shadow settings from mesh settings with defaults."""
    settings = DEFAULT_SHADOW_SETTINGS.copy()
    raw = mesh_settings.get("shadow_settings")
    if not isinstance(raw, dict):
        return settings

    if isinstance(raw.get("offset_x"), (int, float)):
        settings["offset_x"] = int(raw["offset_x"])
    if isinstance(raw.get("offset_y"), (int, float)):
        settings["offset_y"] = int(raw["offset_y"])
    if isinstance(raw.get("blur_radius"), (int, float)):
        settings["blur_radius"] = float(raw["blur_radius"])
    if isinstance(raw.get("shadow_gray"), (int, float)):
        settings["shadow_gray"] = int(raw["shadow_gray"])

    settings["shadow_gray"] = max(0, min(255, settings["shadow_gray"]))
    return settings


def save_shadow_settings(settings_path: Path, state: dict) -> None:
    """Persist shadow settings into mesh_settings.json without losing other keys."""
    mesh_settings = load_mesh_settings(settings_path)
    mesh_settings["shadow_settings"] = {
        "offset_x": int(state["offset_x"]),
        "offset_y": int(state["offset_y"]),
        "blur_radius": float(state["blur_radius"]),
        "shadow_gray": int(state["shadow_gray"]),
    }
    with open(settings_path, "w") as handle:
        json.dump(mesh_settings, handle, indent=2)


def run_editor(image_path: Path, settings_path: Path) -> None:
    """Run the tkinter GUI for adjusting and saving the drop shadow."""
    import tkinter as tk
    from tkinter import ttk

    original = load_image(image_path)
    w, h = original.size
    shadow_mask = build_object_mask(original)
    image_mask = build_image_mask(original)

    # Preview max size
    PREVIEW_MAX = 800
    scale = min(PREVIEW_MAX / w, PREVIEW_MAX / h, 1.0)
    preview_size = (int(w * scale), int(h * scale))
    ARROW_STEP = 3
    SLIDER_OFFSET_RANGE = 500
    shadow_settings = parse_shadow_settings(load_mesh_settings(settings_path))

    state = {
        "offset_x": max(-SLIDER_OFFSET_RANGE, min(SLIDER_OFFSET_RANGE, int(shadow_settings["offset_x"]))),
        "offset_y": max(-SLIDER_OFFSET_RANGE, min(SLIDER_OFFSET_RANGE, int(shadow_settings["offset_y"]))),
        "blur_radius": float(shadow_settings["blur_radius"]),
        "shadow_gray": int(shadow_settings["shadow_gray"]),
        "original": original,
        "shadow_mask": shadow_mask,
        "image_mask": image_mask,
    }

    def recompute_preview() -> Image.Image:
        shadow_layer = build_shadow_layer(
            state["original"],
            state["shadow_mask"],
            state["offset_x"],
            state["offset_y"],
            state["blur_radius"],
            state["shadow_gray"],
        )
        result = shadow_layer.copy()
        result.paste(state["original"], (0, 0), mask=state["image_mask"])
        return result

    def update_display() -> None:
        comp = recompute_preview()
        preview = comp.resize(preview_size, Image.Resampling.LANCZOS)
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(preview)
        label.configure(image=photo)
        label.image = photo

    def on_offset_x(v: float) -> None:
        state["offset_x"] = int(float(v))
        update_display()

    def on_offset_y(v: float) -> None:
        state["offset_y"] = int(float(v))
        update_display()

    def on_blur(v: float) -> None:
        state["blur_radius"] = float(v)
        update_display()

    def on_shadow_gray(v: float) -> None:
        # Invert: slider 0=light (gray 255), slider 255=dark (gray 0)
        state["shadow_gray"] = 255 - int(float(v))
        update_display()

    def on_key(event: tk.Event) -> None:
        if event.keysym == "Left":
            state["offset_x"] = max(-SLIDER_OFFSET_RANGE, state["offset_x"] - ARROW_STEP)
            slider_x.set(state["offset_x"])
        elif event.keysym == "Right":
            state["offset_x"] = min(SLIDER_OFFSET_RANGE, state["offset_x"] + ARROW_STEP)
            slider_x.set(state["offset_x"])
        elif event.keysym == "Up":
            state["offset_y"] = max(-SLIDER_OFFSET_RANGE, state["offset_y"] - ARROW_STEP)
            slider_y.set(state["offset_y"])
        elif event.keysym == "Down":
            state["offset_y"] = min(SLIDER_OFFSET_RANGE, state["offset_y"] + ARROW_STEP)
            slider_y.set(state["offset_y"])
        elif event.keysym in ("s", "Return"):
            save_result()
            return
        elif event.keysym == "q":
            root.quit()
            return
        update_display()

    def save_result() -> None:
        out = composite_with_shadow(
            state["original"],
            state["offset_x"],
            state["offset_y"],
            state["blur_radius"],
            state["shadow_gray"],
        )
        stem = image_path.stem
        out_path = image_path.parent / f"shadow_{stem}.png"
        out.save(out_path)
        save_shadow_settings(settings_path, state)
        status_var.set(f"Saved: {out_path.name}")
        root.after(3000, lambda: status_var.set("Press 's' or Enter to save | Arrow keys move shadow"))

    root = tk.Tk()
    root.title("Drop shadow editor")
    root.bind("<KeyPress>", on_key)
    root.focus_set()

    main = ttk.Frame(root, padding=10)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Preview
    initial = recompute_preview()
    preview_img = initial.resize(preview_size, Image.Resampling.LANCZOS)
    from PIL import ImageTk
    photo = ImageTk.PhotoImage(preview_img)
    label = ttk.Label(main, image=photo)
    label.image = photo
    label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

    # Sliders
    ttk.Label(main, text="Offset X (left/right):").grid(row=1, column=0, sticky="w", pady=2)
    slider_x = ttk.Scale(
        main,
        from_=-SLIDER_OFFSET_RANGE,
        to=SLIDER_OFFSET_RANGE,
        orient=tk.HORIZONTAL,
        length=300,
        command=lambda v: on_offset_x(v),
    )
    slider_x.set(state["offset_x"])
    slider_x.grid(row=1, column=1, sticky="ew", pady=2)

    ttk.Label(main, text="Offset Y (up/down):").grid(row=2, column=0, sticky="w", pady=2)
    slider_y = ttk.Scale(
        main,
        from_=-SLIDER_OFFSET_RANGE,
        to=SLIDER_OFFSET_RANGE,
        orient=tk.HORIZONTAL,
        length=300,
        command=lambda v: on_offset_y(v),
    )
    slider_y.set(state["offset_y"])
    slider_y.grid(row=2, column=1, sticky="ew", pady=2)

    ttk.Label(main, text="Blur (diffusion):").grid(row=3, column=0, sticky="w", pady=2)
    slider_blur = ttk.Scale(
        main,
        from_=0,
        to=100,
        orient=tk.HORIZONTAL,
        length=300,
        command=lambda v: on_blur(v),
    )
    slider_blur.set(state["blur_radius"])
    slider_blur.grid(row=3, column=1, sticky="ew", pady=2)

    ttk.Label(main, text="Shadow darkness:").grid(row=4, column=0, sticky="w", pady=2)
    slider_gray = ttk.Scale(
        main,
        from_=0,
        to=255,
        orient=tk.HORIZONTAL,
        length=300,
        command=lambda v: on_shadow_gray(v),
    )
    # Invert for display: gray 180 -> slider position 75
    slider_gray.set(255 - state["shadow_gray"])
    slider_gray.grid(row=4, column=1, sticky="ew", pady=2)

    main.columnconfigure(1, weight=1)

    status_var = tk.StringVar(value="Press 's' or Enter to save | Arrow keys move shadow | 'q' to quit")
    ttk.Label(main, textvariable=status_var).grid(row=5, column=0, columnspan=2, pady=10)

    root.mainloop()


def run_auto_accept(image_path: Path, settings_path: Path) -> None:
    """Save a shadow image using persisted settings (or defaults if missing)."""
    shadow_settings = parse_shadow_settings(load_mesh_settings(settings_path))
    offset_x = int(shadow_settings["offset_x"])
    offset_y = int(shadow_settings["offset_y"])
    blur_radius = float(shadow_settings["blur_radius"])
    shadow_gray = int(shadow_settings["shadow_gray"])

    original = load_image(image_path)
    out = composite_with_shadow(
        original,
        offset_x,
        offset_y,
        blur_radius,
        shadow_gray,
    )
    out_path = image_path.parent / f"shadow_{image_path.stem}.png"
    out.save(out_path)
    print(f"Saved: {out_path.name}")


def find_most_recent_mesh_render(folder: Path) -> Path | None:
    """Find the most recent mesh_render_*.png in the given folder."""
    renders = list(folder.glob("mesh_render_*.png"))
    if not renders:
        return None
    # Sort by modification time, most recent last
    renders.sort(key=lambda p: p.stat().st_mtime)
    return renders[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add an adjustable drop shadow to a mesh render image.",
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="tetons",
        help="Folder name within /Volumes/sandisk1/data_cache/mountain_mesh_data/ (e.g., mt-baker)",
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Run non-interactively and save output using persisted shadow settings.",
    )
    args = parser.parse_args()
    
    # Build path to the mountain_mesh_data folder
    mesh_folder = BASE_DIR / args.name
    settings_path = mesh_folder / "mesh_settings.json"
    
    if not mesh_folder.exists():
        print(f"Error: folder not found: {mesh_folder}")
        return 1
    
    # Find most recent mesh render
    path = find_most_recent_mesh_render(mesh_folder)
    if path is None:
        print(f"Error: no mesh_render_*.png found in {mesh_folder}")
        return 1
    
    print(f"Using: {path.name}")
    if args.auto_accept:
        run_auto_accept(path, settings_path)
    else:
        run_editor(path, settings_path)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
