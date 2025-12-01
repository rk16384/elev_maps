import os
import rioxarray
import xdem
from PIL import Image
import numpy as np
from PIL import ImageEnhance
import cv2
import math
# Load dem_subset.tif file
dem_subset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dem_subset.tif")
full_dem_path = "/Volumes/sandisk1/data_cache/Utah/elevation_dem/elevation_dem_Utah_-114.043_37.000_-109.046_42.000_mosaic.tif"

def load_dem(dem_path):
    
    dem = rioxarray.open_rasterio(dem_path)
    dem = dem.rio.reproject("EPSG:3857", resolution=30)
    dem_subset_dem = xdem.DEM.from_array(
        dem.values.squeeze(),
        transform=dem.rio.transform(),
        crs=dem.rio.crs,
    )

    return dem_subset_dem

def save_hillshade_to_png(hillshade, filename):

    hillshade_uint8 = ((hillshade - hillshade.min()) * 255 / 
                       (hillshade.max() - hillshade.min())).astype(np.uint8)
    Image.fromarray(hillshade_uint8).save(filename, format="PNG", optimize=True)


def norm_hillshade(dem, sun_azimuth, sun_altitude):
    hillshade = xdem.terrain.hillshade(
            dem,
            azimuth=sun_azimuth,
            altitude=sun_altitude,
        )
    hillshade_data = hillshade.data.squeeze()

    return hillshade_data

def custom_rvt_hillshade(dem):
        # Parameters you can tweak
    num_directions = 4
    primary_azimuth = 330
    secondary_altitude = 80  # high-altitude fill light
    primary_altitude = 30
    primary_weight = 0.6
    high_altitude_weight = 0.4
    slope_high = 25.0
    flat_power = 1.8
    flat_boost = 0.09
    shadow_push = 0.04
    highlight_push = 0.05
    contrast_gain = 1.3

    slope_degrees = xdem.terrain.slope(dem).data.squeeze()
    slope_norm = np.clip(slope_degrees / slope_high, 0.0, 1.0)
    flat_weight = np.power(1.0 - slope_norm, flat_power)

    def _stack_for_altitude(sun_altitude: float, primary_azimuth: float, num_directions: int):
        azimuth_list = (primary_azimuth + np.arange(num_directions) * 360 / num_directions) % 360
        bands = []
        for azimuth in azimuth_list:
            bands.append(norm_hillshade(dem, azimuth, sun_altitude))
        return np.asarray(bands, dtype=np.float32)

    base_stack = _stack_for_altitude(primary_altitude, primary_azimuth, num_directions)
    fill_stack = _stack_for_altitude(secondary_altitude, primary_azimuth, 1)

    weighted = primary_weight * base_stack[0]
    if len(base_stack) > 1:
        residual_weight = (1 - primary_weight) / (len(base_stack) - 1)
        for i in range(1, len(base_stack)):
            weighted += residual_weight * base_stack[i]

    fill_mean = np.nanmean(fill_stack, axis=0)

    # Percentile normalization for both components
    def _normalize(arr, lo=2, hi=98):
        low_cut = np.nanpercentile(arr, lo)
        high_cut = np.nanpercentile(arr, hi)
        if not np.isfinite(low_cut):
            low_cut = 0.0
        if not np.isfinite(high_cut) or high_cut <= low_cut:
            high_cut = low_cut + 1e-6
        return np.clip((arr - low_cut) / (high_cut - low_cut), 0.0, 1.0)

    weighted_norm = _normalize(weighted)
    fill_norm = _normalize(fill_mean)

    combined = np.clip(weighted_norm * (1 - high_altitude_weight) + fill_norm * high_altitude_weight, 0.0, 1.0)

    combined = np.clip(combined + flat_weight * flat_boost, 0.0, 1.0)

    combined = np.power(combined, 0.9)

    shadow_weight = np.power(1.0 - combined, 2.5)
    combined = np.clip(combined + shadow_weight * shadow_push, 0.0, 1.0)

    highlight_weight = np.power(combined, 3.0)
    combined = np.clip(combined + highlight_weight * highlight_push, 0.0, 1.0)

    hillshade_uint8 = np.clip(combined * 255.0, 0, 255).astype(np.uint8)

    hillshade_img = Image.fromarray(hillshade_uint8, mode="L")
    hillshade_img = ImageEnhance.Contrast(hillshade_img).enhance(contrast_gain)
    hillshade_img.save("test_rvt_option.png", format="PNG", optimize=True)

def lift_light_grays(img_uint8, pivot=190, softness=0.55):
    lut = np.arange(256, dtype=np.float32)
    span = 255 - pivot + 1e-6
    mask = lut >= pivot
    lut[mask] = pivot + span * (1.0 - np.power((255 - lut[mask]) / span, softness))
    return cv2.LUT(img_uint8, lut.astype(np.uint8))

def push_light_grays(enhanced, gamma=0.55, mix=0.7, preserve_top=0.97, preserve_bottom=0.05):
    norm = enhanced.astype(np.float32) / 255.0

    lifted = np.power(np.clip(norm, 0.0, 1.0), gamma)  # gamma < 1 brightens midtones
    blended = norm + mix * (lifted - norm)             # mix controls strength

    light_mask = norm > preserve_top                          # keep highlight detail
    dark_mask = norm < preserve_bottom
    mask = light_mask | dark_mask
    blended[mask] = norm[mask] + 0.2 * (blended[mask] - norm[mask])

    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

def enhance_nominal_hillshade(hillshade, filename):
    data = np.asarray(hillshade).astype(np.float32)
    finite_max = np.nanmax(data)
    data = np.nan_to_num(data, nan=finite_max)
    data = ((data - data.min()) * 255.0 / (np.ptp(data) or 1)).astype(np.uint8)


    enhanced = cv2.equalizeHist(data)
    gamma = 0.5
    #bright = ((enhanced / 255.0) ** gamma * 255).astype(np.uint8)
    #bright = lift_light_grays(enhanced, pivot=250, softness=0.9)
    bright = push_light_grays(enhanced, gamma=0.25, mix=0.9, preserve_top=0.97, preserve_bottom=0.10)
    Image.fromarray(bright, mode="L").save(filename, format="PNG", optimize=True)

def combine_hillshades(hillshade1, hillshade2):
    return 0.6*hillshade1 + 0.4*hillshade2


# ------------------------------------------------------------------------------------------------
def ray_trace_test(dem_path):
    import rasterio
    import pyvista as pv
    import numpy as np

    diffuse = 0.95 # [0, 1.0] more light/shadow contrast vs flatter
    ambient = 0.30 # [0, 1.0] base illumination, higher - shadows are lighter, lower - shadows are darker
    total_intensity = 1.5 # higher, brighter image
    num_lights = 8 # 5-8 is usually enough for static images
    spread_angle = 0.015 # Angle in radians (Sun is ~0.009, use 0.02-0.05 for visible softness)
    base_dir = np.array([-1, 0.3, 2]) # vector in direction of the sun

    # sun altitude
    norm = np.linalg.norm(base_dir)
    arctan = np.arctan(base_dir[2] / norm)
    sun_altitude = np.rad2deg(arctan)
    print(f"Sun altitude: {sun_altitude}")


    # 1. Load the DEM Data
    # Replace with the path to your actual DEM file
    #dem_path = "path/to/your/dem_file.tif"

    scaling_factor = 1
    with rasterio.open(dem_path) as src:
        # Read the first band (elevation data)
        elevation = src.read(1)
        
        # Handle 'NoData' values (often -9999 or extremely low numbers)
        # Mask them to 0 or the minimum valid height to prevent spikes
        if src.nodata is not None:
            elevation[elevation == src.nodata] = np.nan
            
        # Optional: Downsample if the file is massive (e.g., take every 10th point)
          # Try 2 or 4 first. 1 is too heavy for full DEMs.
        elevation = elevation[::scaling_factor, ::scaling_factor]


    check_dem = load_dem(dem_path)
    max_pixels = 24 * 30 * 300 * 400
    dem_height, dem_width = check_dem.data.shape
    dem_pixels = dem_width * dem_height

    scaling = math.sqrt(max_pixels / dem_pixels)
    
    new_width = int(dem_width * scaling)
    new_height = int(dem_height * scaling)
    
    #elevation2 = check_dem.data[::scaling_factor, ::scaling_factor].data
    #e2_h, e2_w = elevation2.shape
    #replace nan with -9999
    #elevation = np.where(np.isnan(elevation), -9999, elevation)
    #reshape elevation to match elevation2 shape
    print(f"elevation shape: {elevation.shape}")
    #print(f"elevation2 shape: {elevation2.shape}")
    elevation = cv2.resize(elevation, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    elevation = elevation * 1

    # After resize, shape is (e2_h, e2_w)
    rows, cols = elevation.shape
    
    elevation_data = elevation.astype(np.float32)
    grid = pv.ImageData()
    
    # CORRECT: VTK (x, y, z) = (cols, rows, depth)
    grid.dimensions = (cols, rows, 1)
    
    grid.point_data["Elevation"] = np.flipud(elevation_data).flatten()
    warped = grid.warp_by_scalar("Elevation")

    # --- 3. SETUP THE RENDERER (OFF-SCREEN) ---
    # off_screen=True prevents the window from popping up
    plotter = pv.Plotter(off_screen=True, lighting=None)

    # CRITICAL: Add the mesh back to the scene!
    plotter.add_mesh(
        warped,
        color="#FFFFFF",
        smooth_shading=True,
        show_scalar_bar=False,
        diffuse=diffuse, # [0, 1.0] more light/shadow contrast vs flatter
        ambient=ambient, # [0, 1.0] base illumination, higher - shadows are lighter, lower - shadows are darker
    )

    # --- 4. LIGHTING (Soft Shadows via Multi-Light) ---
    # Configuration for the sun
    # Direction (-1, 0.4, 1)
    
    # Normalize direction
    base_dir = base_dir / np.linalg.norm(base_dir)

    # Center Camera on Mesh Center
    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    center_z = elevation_data.max() * 2 + 1000
    
    # Distance must be outside the mesh!
    mesh_center = np.array([center_x, center_y, 0])
    light_dist = warped.length * 1000
    base_pos = mesh_center + base_dir * light_dist
    
    # Create a local coordinate system to offset perpendicular to light direction
    # Arbitrary vector not parallel to base_dir
    up = np.array([0, 0, 1]) if abs(base_dir[2]) < 0.9 else np.array([0, 1, 0])
    right = np.cross(base_dir, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, base_dir)
    
    # Offsets in a circle pattern
    offsets = [(0,0)] # Center light
    for i in range(num_lights - 1):
        angle = (2 * np.pi * i) / (num_lights - 1)
        # Radius is roughly tan(spread_angle) * light_dist
        radius = np.tan(spread_angle) * light_dist
        
        off_vec = right * np.cos(angle) * radius + up * np.sin(angle) * radius
        
        # Create light at slightly offset position
        pos = base_pos + off_vec
        
        light = pv.Light(
            position=pos,
            focal_point=(center_x, center_y, 0),
            color='white',
            intensity=total_intensity / num_lights
        )
        light.positional = False # False - light is directional like the sun rays are parallel, True - light is positional
        plotter.add_light(light)

    plotter.enable_shadows()

    # --- 5. APPLY 3D REALISM EFFECTS ---
    # Eye Dome Lighting adds immediate depth and contours
    plotter.enable_eye_dome_lighting()

    # Screen Space Ambient Occlusion (SSAO) is great for realistic shadows
    # darkens crevices and corners
    # plotter.enable_ssao(
    #     radius=5,       # pixel radius
    #     bias=0.025     # prevent self occlusion, low - may be noisy, high - small valleys may lose ambient shadows
    # )

    # Anti-aliasing for smooth edges
    plotter.enable_anti_aliasing() # smooths out jagged edges, limit stairstepping

    # --- 5. EXACT CAMERA SETUP (No Padding) ---
    
    plotter.camera_position = [(center_x, center_y, center_z),
                               (center_x, center_y, 0),
                               (0, 1, 0)]
    plotter.camera.SetParallelProjection(True)
    plotter.camera.clipping_range = (1, 1_000_000)
    
    # 2. Set Scale to EXACTLY match the vertical height of the data
    # parallel_scale = half the world-space height of the viewport
    plotter.camera.parallel_scale = rows / 2.0
    
    # --- 6. TILED RENDERING WITH ASPECT RATIO MATCH ---
    
    # Set Window Aspect Ratio to match Data Aspect Ratio
    aspect_ratio = cols / rows
    
    # Choose a base tile size that respects this ratio
    max_tile_dim = 2048
    
    if aspect_ratio > 1:
        # Wider than tall
        win_w = max_tile_dim
        win_h = int(max_tile_dim / aspect_ratio)
    else:
        # Taller than wide
        win_h = max_tile_dim
        win_w = int(max_tile_dim * aspect_ratio)
        
    plotter.window_size = [win_w, win_h]
    
    # Calculate Magnification to reach full resolution
    mag = int(np.ceil(cols / win_w))
    
    print(f"Rendering: Data {cols}x{rows}, Window {win_w}x{win_h}, Mag {mag}x")
    
    big_img = plotter.screenshot(
        transparent_background=False, 
        return_img=True, 
        window_size=[win_w, win_h],
        scale=mag 
    )
    
    plotter.close()
    
    # Resize/Crop to exact pixel dimensions
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(big_img)
    pil_img = pil_img.resize((cols, rows), PILImage.LANCZOS)
    
    hillshade_array = np.array(pil_img)

    # Convert RGB to Grayscale
    if hillshade_array.ndim == 3:
        hillshade_array = np.dot(hillshade_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
    # save hillshade_array as a png
    Image.fromarray(hillshade_array, mode="L").save("test_ray_trace.png", format="PNG", optimize=True)

    print(f"Ray trace complete. Output shape: {hillshade_array.shape}")
    return hillshade_array


ray_trace_test(full_dem_path)

dem = load_dem(full_dem_path)

# scaling factor for the dem to exaggerate the terrain features
scaling_factor = 3
dem = dem * scaling_factor

# Nominal hillshade
hillshade = norm_hillshade(dem, 315, 70)
save_hillshade_to_png(hillshade, "test_nominal_hillshade.png")

hillshade2 = norm_hillshade(dem, 270, 45)
hillshade_combined = combine_hillshades(hillshade, hillshade2)

enhance_nominal_hillshade(hillshade, "test_combo_hillshade_enhanced.png")
# Custom RVT hillshade
custom_rvt_hillshade(dem)


