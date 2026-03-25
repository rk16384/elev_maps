"""
Wrapper script for Blender clay/plaster hillshade rendering.

Loads a DEM TIF, normalizes it, saves as a numpy heightmap,
then shells out to Blender for physically-based rendering.

Usage:
    python blender_hillshade_test.py
    python blender_hillshade_test.py --dem_path /path/to/dem.tif
    python blender_hillshade_test.py --render_resolution_x 8000 --render_resolution_y 12000
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import rasterio

BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_clay_render.py")

CONFIG = {
    "dem_path": "/Volumes/sandisk1/data_cache/mountain_mesh_data/glacier-peak/extracted_dem.tif",
    "output_dir": "./blender_output",

    # Light settings
    "sun_azimuth": 315.0,       # degrees, 0=North, clockwise
    "sun_elevation": 30.0,      # degrees above horizon (lower = longer shadows)
    "sun_intensity": 4.0,       # sun energy
    "sun_angle": 8.0,           # angular diameter in degrees (larger = softer shadow edges)
    "ambient_strength": 0.20,   # world background light (fills shadows, prevents pure black)

    # Material (plaster/clay look)
    "base_color": [0.85, 0.85, 0.85],
    "roughness": 0.8,
    "subsurface": 0.05,

    # Geometry
    "vertical_exaggeration": 2.0,
    "z_scale": 0.25,  # max terrain height as fraction of plane width (0.15-0.30 typical)

    # Mesh downsampling: max vertices along the longest edge.
    # None = use DEM at full resolution. Set lower to reduce memory.
    "max_mesh_dimension": None,

    # Render (start small for iteration, scale up for final)
    "render_resolution_x": 4000,
    "render_resolution_y": None,  # None = auto from DEM aspect ratio
    "render_samples": 128,
    "use_denoiser": True,
}


def load_and_normalize_dem(dem_path: str, vertical_exaggeration: float = 1.0,
                           max_mesh_dimension: int | None = None) -> tuple[np.ndarray, dict]:
    """Load DEM TIF, normalize elevation to 0-1, optionally downsample."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        crs = str(src.crs)
        bounds = src.bounds

    print(f"  Raw DEM shape: {dem.shape}")
    print(f"  Elevation range: {np.nanmin(dem):.1f} to {np.nanmax(dem):.1f} m")

    nan_mask = np.isnan(dem) | (dem < -9000)
    dem[nan_mask] = np.nanmin(dem[~nan_mask]) if (~nan_mask).any() else 0.0

    if max_mesh_dimension is not None:
        h, w = dem.shape
        longest = max(h, w)
        if longest > max_mesh_dimension:
            scale = max_mesh_dimension / longest
            new_h, new_w = int(h * scale), int(w * scale)
            from scipy.ndimage import zoom
            dem = zoom(dem, (new_h / h, new_w / w), order=1)
            print(f"  Downsampled to: {dem.shape}")

    dem *= vertical_exaggeration

    elev_min = np.nanmin(dem)
    elev_max = np.nanmax(dem)
    elev_range = elev_max - elev_min
    if elev_range == 0:
        normalized = np.zeros_like(dem)
    else:
        normalized = (dem - elev_min) / elev_range

    aspect_ratio = dem.shape[0] / dem.shape[1]

    metadata = {
        "original_shape": list(dem.shape),
        "elev_min": float(elev_min),
        "elev_max": float(elev_max),
        "elev_range": float(elev_range),
        "aspect_ratio": float(aspect_ratio),
        "crs": crs,
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
    }

    print(f"  Normalized shape: {normalized.shape}")
    print(f"  Aspect ratio (H/W): {aspect_ratio:.4f}")

    return normalized, metadata


def estimate_render_time(total_pixels: int, mesh_verts: int) -> str:
    """Rough estimate based on observed M4 Metal benchmarks.

    Calibrated from:
      2000x1335 (2.7Mpx), 10.7M verts → 49s total (8s mesh + 39s render)
      2000x2002 (4.0Mpx), 16.0M verts → 93s total (12s mesh + 78s render)
      6000x6006 (36Mpx),  25.0M verts → 869s total (19s mesh + 804s render)
    """
    render_s = total_pixels * 22.3e-6  # ~22 us per pixel
    mesh_s = mesh_verts * 0.76e-6      # ~0.76 us per vertex
    total_s = render_s + mesh_s + 5
    if total_s < 60:
        return f"~{total_s:.0f}s"
    elif total_s < 3600:
        return f"~{total_s / 60:.0f} min"
    else:
        return f"~{total_s / 3600:.1f} hr"


def run_blender(config: dict, heightmap_path: str, metadata: dict):
    """Shell out to Blender in background mode with the render script."""
    import re

    render_config = {**config}
    render_config["heightmap_path"] = heightmap_path
    render_config["dem_metadata"] = metadata

    if render_config["render_resolution_y"] is None:
        render_config["render_resolution_y"] = int(
            render_config["render_resolution_x"] * metadata["aspect_ratio"]
        )

    config_path = os.path.join(config["output_dir"], "_render_config.json")
    with open(config_path, "w") as f:
        json.dump(render_config, f, indent=2)

    res_x = render_config["render_resolution_x"]
    res_y = render_config["render_resolution_y"]
    total_pixels = res_x * res_y
    mesh_shape = metadata["original_shape"]
    mesh_verts = mesh_shape[0] * mesh_shape[1]
    est = estimate_render_time(total_pixels, mesh_verts)

    print(f"\nRender resolution: {res_x} x {res_y} ({total_pixels / 1e6:.1f} Mpx)")
    print(f"Mesh vertices: {mesh_verts:,}")
    print(f"Estimated total time: {est}")
    print(f"Config written to: {config_path}")

    cmd = [
        BLENDER_PATH,
        "--background",
        "--python", RENDER_SCRIPT,
        "--", config_path,
    ]

    print(f"\nLaunching Blender...")
    t0 = time.time()

    progress_re = re.compile(r"PROGRESS elapsed=([\d.]+)s")
    render_started = False

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True
    )

    for line in proc.stdout:
        elapsed = time.time() - t0
        line = line.rstrip()

        progress_match = progress_re.search(line)
        if progress_match:
            render_elapsed = float(progress_match.group(1))
            print(f"  [{elapsed:6.0f}s] Rendering... {render_elapsed:.0f}s elapsed", flush=True)
        elif "Rendering" in line and "Mpx" in line:
            render_started = True
            print(f"  [{elapsed:6.0f}s] {line.strip()}", flush=True)
        elif any(k in line for k in ["Building mesh", "Mesh built", "Render completed",
                                      "Using GPU", "Heightmap shape", "Z range",
                                      "Clearing scene", "Loading heightmap",
                                      "Setting up", "[", "Done"]):
            print(f"  [{elapsed:6.0f}s] {line.strip()}", flush=True)
        elif "Saved" in line and ("rgb" in line.lower() or "grayscale" in line.lower()):
            print(f"  [{elapsed:6.0f}s] {line.strip()}", flush=True)

    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\nBlender exited with code {proc.returncode} after {elapsed:.1f}s")
        sys.exit(1)

    print(f"\nBlender finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Blender clay hillshade renderer")
    for key, default in CONFIG.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", type=lambda x: x.lower() == "true", default=default)
        elif isinstance(default, list):
            parser.add_argument(f"--{key}", type=float, nargs="+", default=default)
        elif default is None:
            parser.add_argument(f"--{key}", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)
    args = parser.parse_args()
    config = {k: getattr(args, k) for k in CONFIG}

    if config["render_resolution_y"] is not None:
        config["render_resolution_y"] = int(config["render_resolution_y"])
    if config["max_mesh_dimension"] is not None:
        config["max_mesh_dimension"] = int(config["max_mesh_dimension"])

    os.makedirs(config["output_dir"], exist_ok=True)

    print("=" * 60)
    print("Blender Clay Hillshade Renderer")
    print("=" * 60)

    print(f"\nLoading DEM: {config['dem_path']}")
    heightmap, metadata = load_and_normalize_dem(
        config["dem_path"],
        vertical_exaggeration=config["vertical_exaggeration"],
        max_mesh_dimension=config["max_mesh_dimension"],
    )

    heightmap_path = os.path.join(config["output_dir"], "_heightmap.npy")
    np.save(heightmap_path, heightmap)
    print(f"  Saved heightmap: {heightmap_path} ({os.path.getsize(heightmap_path) / 1024**2:.1f} MB)")

    run_blender(config, os.path.abspath(heightmap_path), metadata)

    rgb_path = os.path.join(config["output_dir"], "clay_render_rgb.png")
    gray_path = os.path.join(config["output_dir"], "clay_render_grayscale.png")
    for path in [rgb_path, gray_path]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024**2
            print(f"  Output: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  WARNING: Expected output not found: {path}")


if __name__ == "__main__":
    main()
