"""
Blender Python script for clay/plaster hillshade rendering.

Run via Blender's embedded Python:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_clay_render.py -- config.json

Reads a normalized heightmap (.npy), builds a 3D mesh, and renders
with Cycles using a plaster/clay material under directional lighting.
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np


def load_config():
    argv = sys.argv
    separator_idx = argv.index("--") if "--" in argv else -1
    if separator_idx < 0 or separator_idx + 1 >= len(argv):
        raise RuntimeError("No config file path provided after '--'")
    config_path = argv[separator_idx + 1]
    with open(config_path, "r") as f:
        return json.load(f)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam, do_unlink=True)


def build_terrain_mesh(heightmap: np.ndarray, config: dict) -> bpy.types.Object:
    """Build a mesh from the heightmap array. One vertex per pixel."""
    h, w = heightmap.shape
    print(f"  Building mesh: {w} x {h} = {w * h:,} vertices", flush=True)
    t0 = time.time()

    plane_width = 10.0
    plane_height = plane_width * (h / w)

    z_fraction = config.get("z_scale", 0.20)
    z_scale = plane_width * z_fraction

    x_coords = np.linspace(-plane_width / 2, plane_width / 2, w)
    y_coords = np.linspace(plane_height / 2, -plane_height / 2, h)
    xx, yy = np.meshgrid(x_coords, y_coords)
    zz = heightmap * z_scale
    print(f"  Z range: 0 to {z_scale:.3f} (plane width={plane_width}, z_fraction={z_fraction})", flush=True)

    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    rows = np.arange(h - 1)[:, None]
    cols = np.arange(w - 1)[None, :]
    tl = rows * w + cols
    tr = tl + 1
    bl = tl + w
    br = bl + 1
    faces = np.column_stack([tl.ravel(), bl.ravel(), br.ravel(), tr.ravel()])

    mesh = bpy.data.meshes.new("terrain_mesh")
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel().astype(np.float32))

    mesh.loops.add(len(faces) * 4)
    mesh.loops.foreach_set("vertex_index", faces.ravel().astype(np.int32))

    mesh.polygons.add(len(faces))
    loop_starts = np.arange(0, len(faces) * 4, 4, dtype=np.int32)
    loop_totals = np.full(len(faces), 4, dtype=np.int32)
    mesh.polygons.foreach_set("loop_start", loop_starts)
    mesh.polygons.foreach_set("loop_total", loop_totals)

    mesh.update()
    mesh.validate()

    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)

    elapsed = time.time() - t0
    print(f"  Mesh built in {elapsed:.1f}s ({len(verts):,} verts, {len(faces):,} faces)", flush=True)
    return obj


def create_clay_material(config: dict) -> bpy.types.Material:
    """Create a Principled BSDF material with plaster/clay appearance."""
    mat = bpy.data.materials.new("ClayMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    base_color = config.get("base_color", [0.85, 0.85, 0.85])
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = config.get("roughness", 0.8)

    subsurface = config.get("subsurface", 0.02)
    bsdf.inputs["Subsurface Weight"].default_value = subsurface
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.8, 0.6)

    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.3

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def setup_camera(config: dict, terrain_obj: bpy.types.Object):
    """Place an orthographic camera looking straight down, framing the terrain.

    When padding info is present in the config, the camera frames only the
    original (unpadded) portion of the mesh so the padding stays off-screen
    and provides natural lighting context without appearing in the output.
    """
    cam_data = bpy.data.cameras.new("OrthoCamera")
    cam_data.type = "ORTHO"

    bbox = terrain_obj.bound_box
    xs = [v[0] for v in bbox]
    ys = [v[1] for v in bbox]
    terrain_width = max(xs) - min(xs)
    terrain_height = max(ys) - min(ys)

    padding = config.get("padding")
    if padding:
        padded_h = padding["original_h"] + 2 * padding["pad_h"]
        padded_w = padding["original_w"] + 2 * padding["pad_w"]
        orig_frac_w = padding["original_w"] / padded_w
        orig_frac_h = padding["original_h"] / padded_h
        orig_width = terrain_width * orig_frac_w
        orig_height = terrain_height * orig_frac_h
        cam_data.ortho_scale = max(orig_width, orig_height)
        print(f"  Camera frames original area: {orig_width:.3f} x {orig_height:.3f} "
              f"(mesh: {terrain_width:.3f} x {terrain_height:.3f})", flush=True)
    else:
        cam_data.ortho_scale = max(terrain_width, terrain_height)

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    zs = [v[2] for v in bbox]
    cam_z = max(zs) + 10.0
    cam_obj.location = (0, 0, cam_z)
    cam_obj.rotation_euler = (0, 0, 0)

    bpy.context.scene.camera = cam_obj
    return cam_obj


def azimuth_elevation_to_direction(azimuth_deg: float, elevation_deg: float):
    """Convert geographic azimuth (0=N, CW) + elevation to a 3D direction vector."""
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)
    x = math.sin(az_rad) * math.cos(el_rad)
    y = math.cos(az_rad) * math.cos(el_rad)
    z = math.sin(el_rad)
    return (x, y, z)


def setup_lighting(config: dict, terrain_obj: bpy.types.Object):
    """Add sun lamp and optional fill light."""
    sun_data = bpy.data.lights.new("SunLight", "SUN")
    sun_data.energy = config.get("sun_intensity", 3.5)
    sun_data.angle = math.radians(config.get("sun_angle", 5.0))

    sun_obj = bpy.data.objects.new("SunLight", sun_data)
    bpy.context.collection.objects.link(sun_obj)

    azimuth = config.get("sun_azimuth", 315.0)
    elevation = config.get("sun_elevation", 35.0)
    dx, dy, dz = azimuth_elevation_to_direction(azimuth, elevation)

    rot_x = math.acos(dz)
    rot_z = math.atan2(dx, -dy)
    sun_obj.rotation_euler = (rot_x, 0, rot_z)

    sun_obj.location = (0, 0, 20)

    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = config.get("ambient_strength", 0.05)

    return sun_obj


def setup_renderer(config: dict):
    """Configure Cycles renderer with Metal GPU."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL"
        if device.use:
            print(f"  Using GPU: {device.name}", flush=True)

    scene.cycles.device = "GPU"
    scene.cycles.samples = config.get("render_samples", 128)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = config.get("use_denoiser", True)
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

    scene.render.resolution_x = config.get("render_resolution_x", 4000)
    scene.render.resolution_y = config.get("render_resolution_y", 2671)
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 15

    scene.render.film_transparent = False

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"


_render_start_time = 0.0
_last_progress_time = 0.0
_progress_interval = 10.0


def _render_progress_handler(scene):
    """Called by Blender during rendering to report progress."""
    global _last_progress_time
    now = time.time()
    if now - _last_progress_time < _progress_interval:
        return
    _last_progress_time = now
    elapsed = now - _render_start_time
    # Blender doesn't expose remaining time directly in handlers,
    # but we can show elapsed time so the wrapper can track progress
    print(f"  PROGRESS elapsed={elapsed:.1f}s", flush=True)


def render_and_save(config: dict):
    """Render the scene and save RGB + grayscale outputs."""
    global _render_start_time, _last_progress_time

    output_dir = config.get("output_dir", "./blender_output")
    os.makedirs(output_dir, exist_ok=True)

    rgb_path = os.path.join(output_dir, "clay_render_rgb.png")
    scene = bpy.context.scene
    scene.render.filepath = rgb_path

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    samples = scene.cycles.samples
    total_mpx = res_x * res_y / 1e6

    print(f"\n  Rendering {res_x}x{res_y} ({total_mpx:.1f} Mpx) "
          f"@ {samples} samples...", flush=True)

    bpy.app.handlers.render_init.append(_render_progress_handler)
    bpy.app.handlers.render_pre.append(_render_progress_handler)
    bpy.app.handlers.render_post.append(_render_progress_handler)
    bpy.app.handlers.render_stats.append(_render_progress_handler)

    _render_start_time = time.time()
    _last_progress_time = 0.0
    bpy.ops.render.render(write_still=True)
    elapsed = time.time() - _render_start_time

    for handler_list in [bpy.app.handlers.render_init,
                         bpy.app.handlers.render_pre,
                         bpy.app.handlers.render_post,
                         bpy.app.handlers.render_stats]:
        if _render_progress_handler in handler_list:
            handler_list.remove(_render_progress_handler)

    print(f"  Render completed in {elapsed:.1f}s", flush=True)

    gray_path = os.path.join(output_dir, "clay_render_grayscale.png")
    try:
        img = bpy.data.images.load(rgb_path)
        pixels = np.array(img.pixels[:]).reshape(
            scene.render.resolution_y, scene.render.resolution_x, 4
        )
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = np.flipud(gray)
        gray_uint16 = (np.clip(gray, 0, 1) * 65535).astype(np.uint16)

        import struct
        import zlib

        def write_png_gray16(path, data):
            height, width = data.shape

            def make_chunk(chunk_type, chunk_data):
                c = chunk_type + chunk_data
                crc = zlib.crc32(c) & 0xFFFFFFFF
                return struct.pack(">I", len(chunk_data)) + c + struct.pack(">I", crc)

            sig = b"\x89PNG\r\n\x1a\n"
            ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
            raw_rows = b""
            for row in data:
                raw_rows += b"\x00" + row.astype(">u2").tobytes()
            idat = zlib.compress(raw_rows, 9)

            with open(path, "wb") as f:
                f.write(sig)
                f.write(make_chunk(b"IHDR", ihdr))
                f.write(make_chunk(b"IDAT", idat))
                f.write(make_chunk(b"IEND", b""))

        write_png_gray16(gray_path, gray_uint16)
        print(f"  Saved grayscale: {gray_path}", flush=True)
    except Exception as e:
        print(f"  Warning: grayscale conversion failed: {e}", flush=True)

    print(f"  Saved RGB: {rgb_path}", flush=True)


def log(msg):
    print(msg, flush=True)


def main():
    log("\n" + "=" * 60)
    log("Blender Clay/Plaster Hillshade Renderer")
    log("=" * 60)

    config = load_config()

    log("\n[1/6] Clearing scene...")
    clear_scene()

    log("\n[2/6] Loading heightmap...")
    heightmap_path = config["heightmap_path"]
    heightmap = np.load(heightmap_path)
    log(f"  Heightmap shape: {heightmap.shape}")

    log("\n[3/6] Building terrain mesh...")
    terrain = build_terrain_mesh(heightmap, config)

    log("\n[4/6] Setting up material...")
    material = create_clay_material(config)
    terrain.data.materials.append(material)

    log("\n[5/6] Setting up camera and lighting...")
    setup_camera(config, terrain)
    setup_lighting(config, terrain)
    setup_renderer(config)

    log("\n[6/6] Rendering...")
    render_and_save(config)

    log("\nDone!")


if __name__ == "__main__":
    main()
