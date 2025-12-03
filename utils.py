import numpy as np
from typing import Tuple, Optional, List, Union
import xarray
import pyproj
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
from functools import partial
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import math
import geopandas as gpd
import shapely
import rasterio
import pyvista as pv
import numpy as np
import cv2


def bbox_to_land_area(polygon_or_bbox: Union[Tuple[float, float, float, float], Polygon, MultiPolygon]) -> float:
    """
    Calculate the land area of a bounding box or polygon in square meters.

    Args:
        polygon_or_bbox: Either a shapely Polygon, MultiPolygon, or a (west, south, east, north) tuple

    Returns:
        float: The area in square meters
    """
    if isinstance(polygon_or_bbox, (Polygon, MultiPolygon)):
        polygon_wgs84 = polygon_or_bbox
    else:
        # Assume (west, south, east, north) tuple
        bbox = polygon_or_bbox
        polygon_wgs84 = Polygon([
            (bbox[0], bbox[3]),  # NW corner
            (bbox[2], bbox[3]),  # NE corner
            (bbox[2], bbox[1]),  # SE corner
            (bbox[0], bbox[1])   # SW corner
        ])

    # Use a local Azimuthal Equidistant projection centered on the polygon
    lon_c = polygon_wgs84.centroid.x
    lat_c = polygon_wgs84.centroid.y

    project = partial(
        pyproj.transform,
        pyproj.Proj(proj='latlong', datum='WGS84'),
        pyproj.Proj(proj='aeqd', lat_0=lat_c, lon_0=lon_c)
    )

    projected_polygon = transform(project, polygon_wgs84)
    return projected_polygon.area

def land_area_to_resolution(land_area: float) -> List[int]:
    """
    Output the scan resolutions to check based on the pixelation from the total land area.
    
    Args:
        land_area (float): Area in square meters
        
    Returns:
        List[int]: List of valid resolutions in meters, sorted from highest to lowest quality
    """
    # Target pixel area for high-quality visualization (4800x6000 pixels)
    # max print size
    max_print_size = 24 * 36 # square inches
    dpi = 400 # high quality print
    dpsi = dpi * dpi
    pixel_area = max_print_size * dpsi

    # pixel_area = 4800 * 6000

    # Calculate minimum pixel size
    minimum_pixel_size = np.sqrt(land_area / pixel_area)

    # Available resolutions in meters (from highest to lowest quality)
    valid_search_resolutions = [100, 30, 10, 3, 1]

    # Create boolean mask for valid resolutions
    valid_mask = np.array(valid_search_resolutions) <= minimum_pixel_size
    
    # Get valid resolutions
    valid_resolutions = np.array(valid_search_resolutions)[valid_mask]
    
    # Get invalid resolutions (those NOT in valid_mask)
    invalid_resolutions = np.array(valid_search_resolutions)[~valid_mask]
    rev_invalid_resolutions = invalid_resolutions[::-1]
    # Add the invalid resolutions in reverse to the end of the valid resolutions
    # This is for worst case scenario where the ideal resolutions are invalid
    valid_resolutions = np.concatenate((valid_resolutions, rev_invalid_resolutions))
    
    
    # If no valid resolutions found, return the highest available resolution
    if len(valid_resolutions) == 0:
        return [valid_search_resolutions[0]]
    else:
        return valid_resolutions

def reproject_to_web_mercator(dem_data: xarray.DataArray) -> xarray.DataArray:
    """Reproject DEM data to Web Mercator (EPSG:3857) if needed."""
    if dem_data.rio.crs.is_geographic:
        return dem_data.rio.reproject("EPSG:3857")
    return dem_data

def format_bbox_string(bbox: Tuple[float, float, float, float]) -> str:
    """
    Format bbox coordinates for filename with 3 decimal precision.
    
    Args:
        bbox (tuple): (west, south, east, north) coordinates
        
    Returns:
        str: Formatted string of coordinates
    """
    return "_".join(f"{coord:.3f}" for coord in bbox) 

def create_extrusion_shadow(
    dem_data: xarray.DataArray,
    light_elevation_deg: float = 65.0,
    light_azimuth: float = 0.0, # 0 North, 90 East, 180 South, 270 West
    scale: float = .1,
    shadow_color: tuple = (120, 120, 120, 178), # RGBA Gray
    downsample_factor: int = 16,
    blur_radius: float = 5.0,
    transparent_background: bool = False,
    state_mask: Optional[np.ndarray] = None,
    print_margin_pixels: int = 5
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Generates an extrusion shadow effect for a Digital Elevation Model (DEM).

    This function simulates a light source and casts shadows from the DEM onto
    a surrounding plane, creating the illusion of a 3D extruded map.

    Args:
        dem_data (xarray.DataArray): The 2D xarray DataArray of the DEM.
            Lighter pixels are treated as higher elevation.
        light_elevation_deg (float): The elevation angle of the light source
            above the horizon, in degrees (1-89).
        light_azimuth (float): The direction of the light source in degrees,
            where 0 is North, 90 is East, 180 is South, and 270 is West.
        scale (float): A multiplier to exaggerate or reduce the shadow length.
        shadow_color (tuple): The RGBA color to use for the shadow.
        downsample_factor (int): Factor to downsample the DEM for performance.
            A factor of 4 reduces the image to 1/4 of its width and height.
        blur_radius (float): The radius for the Gaussian blur applied to the shadow.
        transparent_background (bool): Whether to use a transparent background.
        state_mask (np.ndarray): A mask of the state to be used for the background.
        print_margin_pixels (int): The gap in pixels between the hillshade image and the edge of the final printable area.
    Returns:
        tuple[Image.Image, tuple[int, int]]: A tuple containing:
            - An RGB image with the shadow on a white background.
            - A tuple with the (x, y) offset of the DEM area within the image.
    """
    # Get the raw numpy data, removing any single-dimensional entries
    dem_numpy = dem_data.data.squeeze()

    z_bump = 800
    dem_numpy = dem_numpy + z_bump

    if downsample_factor < 1:
        downsample_factor = 1

    original_height, original_width = dem_numpy.shape

    # --- 1. Downsample DEM for performance ---
    if downsample_factor > 1:
        scaled_width = original_width // downsample_factor
        scaled_height = original_height // downsample_factor
        dem_image = Image.fromarray(dem_numpy)
        dem_image_scaled = dem_image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)
        dem_data_scaled = np.array(dem_image_scaled)
    else:
        dem_data_scaled = dem_numpy

    dem_height, dem_width = dem_data_scaled.shape

    # --- 2. Calculate Lighting and Canvas Parameters ---
    theta_rad = math.radians(light_elevation_deg)
    
    # Convert geographic azimuth (0=N, 90=E, clockwise) to standard math
    # angle (0=E, 90=N, counter-clockwise) for the shadow direction.
    # The shadow direction is opposite to the light source.
    phi_deg = (270 - (light_azimuth-90)) % 360 
    phi_rad = math.radians(phi_deg)

    tan_theta = math.tan(theta_rad)
    if tan_theta == 0:
        raise ValueError("Light elevation is 0, cannot cast finite shadow.")

    # Find max elevation from the to calculate the necessary padding around the image
    # if phi deg between 0 and 90, check the left and bottom, if 90 to 180, check the right and bottom, if 180 to 270, check the right and top, if 270 to 360, check the left and top
    if phi_deg >= 180 and phi_deg < 270:
        max_z = np.nanmax(np.concatenate((dem_data_scaled[:, -1], dem_data_scaled[-1, :]))) # left and bottom
    elif phi_deg >= 270 and phi_deg < 360:
        max_z = np.nanmax(np.concatenate((dem_data_scaled[:, -1], dem_data_scaled[0, :]))) # left and top
    elif phi_deg >= 90 and phi_deg < 180:
        max_z = np.nanmax(np.concatenate((dem_data_scaled[:, 0], dem_data_scaled[0, :]))) # right and top
    elif phi_deg >= 0 and phi_deg < 90:
        max_z = np.nanmax(np.concatenate((dem_data_scaled[:, 0], dem_data_scaled[-1, :]))) # right and bottom

    if max_z is np.nan or np.isnan(max_z):
        max_z = np.nanmax(dem_data_scaled)

    max_shadow_len = (max_z * scale) / tan_theta

    width_buffer = dem_width * 0.03
    if max_z > 0:
        scale = width_buffer * tan_theta / max_z * 1
    else:
        scale = 0 # Avoid division by zero if map is flat

    # Calculate padding and final canvas size
    # Ensure we have enough space for the print margin plus shadow padding
    shadow_padding = math.ceil(width_buffer) + 2
    total_padding = max(shadow_padding, print_margin_pixels)
    
    canvas_width = dem_width + total_padding * 2
    canvas_height = dem_height + total_padding * 2
    dem_offset_x = total_padding
    dem_offset_y = total_padding

    # Create the output image, fully transparent initially
    output_image = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(output_image)

    # --- 3. Cast and Draw Shadows ---
    # Iterate through every pixel of the DEM from back to front, relative to the light
    # This helps ensure shadows from taller objects correctly overlap shorter ones.
    # We create an iterator that scans pixels based on the light direction.
    x_coords = range(dem_width)
    y_coords = range(dem_height)
    if math.cos(phi_rad) < 0:
        x_coords = reversed(x_coords)
    if math.sin(phi_rad) < 0:
        y_coords = reversed(y_coords)
    
    # Create a list of all coordinates to iterate over
    all_coords = [(y, x) for y in y_coords for x in x_coords]

    print(f"Processing {len(all_coords)} downsampled pixels for shadow...")

    for y, x in all_coords:
        z = dem_data_scaled[y, x]
        if z is np.nan:
            continue
        if z <= 0:
            continue
        if state_mask is not None and state_mask[y, x] == 1:
            pass
        if z.item() is np.nan:
            z = 0


        shadow_len = (z * scale) / tan_theta

        # Calculate start and end points in the output canvas coordinates
        start_x = x + dem_offset_x
        start_y = y + dem_offset_y
        
        end_x = start_x + shadow_len * math.cos(phi_rad)
        end_y = start_y + shadow_len * math.sin(phi_rad)
        
        # We only need to draw the line if it has length
        if abs(start_x - end_x) > 0.1 or abs(start_y - end_y) > 0.1:
            # This line represents the extruded "wall"
            draw.line([(start_x, start_y), (end_x, end_y)], fill=shadow_color, width=1)

    # --- 4. Composite Final Image ---

    # Extract the shadow's shape (alpha channel) and blur it.
    shadow_alpha = output_image.getchannel('A')
    if blur_radius > 0:
        shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Create a mask for the DEM footprint to punch a hole in the shadow
    dem_mask = Image.new('L', (canvas_width, canvas_height), 255)
    draw_mask = ImageDraw.Draw(dem_mask)

    if state_mask is not None:
        # If a state mask is provided, create a hole in the shape of the state
        mask_image = Image.fromarray((state_mask * 255).astype(np.uint8), 'L')
        if downsample_factor > 1:
            mask_image = mask_image.resize((dem_width, dem_height), Image.Resampling.NEAREST)
        
        # Invert the mask: we want the state area to be the hole (0)
        inverted_mask = Image.eval(mask_image, lambda x: 255 - x)
        
        # Paste the shaped hole into the main dem_mask
        dem_mask.paste(inverted_mask, (dem_offset_x, dem_offset_y))
    else:
        # Original behavior: punch a rectangular hole
        draw_mask.rectangle(
            (dem_offset_x, dem_offset_y, dem_offset_x + dem_width - 1, dem_offset_y + dem_height - 1),
            fill=0  # 0 makes this area transparent in the final alpha
        )

    # Combine the blurred shadow alpha with the DEM mask
    final_alpha = ImageChops.darker(shadow_alpha, dem_mask)

    # Create the final shadow image with a white background and gray shadow
    output_image = Image.new('RGB', (canvas_width, canvas_height), 'white')
    shadow_layer = Image.new('RGB', (canvas_width, canvas_height), shadow_color[:3])
    output_image.paste(shadow_layer, (0, 0), final_alpha)

    # --- 5. Upscale and Finalize ---
    if downsample_factor > 1:
        final_canvas_width = canvas_width * downsample_factor
        final_canvas_height = canvas_height * downsample_factor
        final_dem_offset_x = dem_offset_x * downsample_factor
        final_dem_offset_y = dem_offset_y * downsample_factor

        final_image = output_image.resize((final_canvas_width, final_canvas_height), Image.Resampling.BICUBIC)
    else:
        final_image = output_image
        final_dem_offset_x = dem_offset_x
        final_dem_offset_y = dem_offset_y

    print("Shadow generation complete.")
    return final_image, (final_dem_offset_x, final_dem_offset_y)

def _convert_coastline_to_polygon(self, coastline_features: gpd.GeoDataFrame, bbox_polygon: Polygon) -> gpd.GeoDataFrame:
    """
    Convert coastline features into water body polygons using reference land geometry.
    
    Args:
        coastline_features: GeoDataFrame containing coastline and other water features
        bbox_polygon: Shapely Polygon representing the bounding box
        
    Returns:
        GeoDataFrame with converted water body polygons
    """
    if coastline_features.empty:
        return coastline_features

    try:
        # Ensure all geometries are in the same CRS
        if self.reference_land_geometry is not None:
            # Create a GeoDataFrame for the reference land with WGS84 CRS
            ref_gdf = gpd.GeoDataFrame(geometry=[self.reference_land_geometry], crs="EPSG:4326")
            
            # Transform reference land to match coastline features CRS
            if coastline_features.crs != ref_gdf.crs:
                ref_gdf = ref_gdf.to_crs(coastline_features.crs)
            reference_land = ref_gdf.geometry.iloc[0]
            
            # Create a GeoDataFrame for the bbox_polygon with the same CRS as coastline features
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=coastline_features.crs)
        else:
            raise ValueError("Reference land geometry is required but not available")

        # Separate coastlines from other water features
        coastlines = coastline_features[coastline_features['FCODE'] == 56600].copy()
        other_water = coastline_features[coastline_features['FCODE'] != 56600].copy()

        if coastlines.empty:
            print("No coastline features found, returning original water features")
            return coastline_features

        # Merge all coastlines and clip to bbox
        merged_coastline = coastlines.geometry.unary_union
        clipped_coastline = merged_coastline.intersection(bbox_polygon)

        if clipped_coastline.is_empty:
            print("No coastline intersection with bounding box")
            return other_water

        # Split the bbox using the coastline
        split_result = shapely.ops.split(bbox_polygon, clipped_coastline)
        
        # Handle the split result based on its type
        if isinstance(split_result, shapely.geometry.GeometryCollection):
            # Extract only the Polygon and MultiPolygon geometries
            split_polygons = [geom for geom in split_result.geoms 
                            if isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))]
        else:
            split_polygons = list(split_result)
        
        if len(split_polygons) < 2:
            print("Warning: Coastline did not properly split the bounding box")
            return other_water

        # Identify the ocean polygon using the reference land geometry
        ocean_polygon = None
        for polygon in split_polygons:
            # Handle both Polygon and MultiPolygon
            if isinstance(polygon, shapely.geometry.MultiPolygon):
                # For MultiPolygon, check the largest component
                largest = max(polygon.geoms, key=lambda p: p.area)
                rep_point = largest.representative_point()
            else:
                rep_point = polygon.representative_point()
            
            if not reference_land.contains(rep_point):
                ocean_polygon = polygon
                break

        if ocean_polygon is None:
            print("Warning: Could not identify ocean polygon")
            return other_water

        # Create final water geometry by merging ocean with other water features
        all_water_geometries = [ocean_polygon]
        if not other_water.empty:
            all_water_geometries.extend(other_water.geometry.tolist())

        final_water_geometry = shapely.ops.unary_union(all_water_geometries)

        # Create new GeoDataFrame with the final water geometry
        water_gdf = gpd.GeoDataFrame(
            {
                'geometry': [final_water_geometry],
                'FCODE': [44500],  # SeaOcean code
                'FTYPE': ['SeaOcean']
            },
            crs=coastline_features.crs
        )

        print(f"Created unified water geometry with area: {final_water_geometry.area}")
        return water_gdf

    except Exception as e:
        print(f"Error in coastline conversion: {e}")
        import traceback
        traceback.print_exc()
        return coastline_features


def create_ray_traced_hillshade(dem_data: xarray.DataArray):

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

    scaling_factor = 1
    dem_numpy = dem_data.data.squeeze()

    max_pixels = 24 * 30 * 300 * 300
    dem_height, dem_width = dem_numpy.shape
    dem_pixels = dem_width * dem_height

    scaling = math.sqrt(max_pixels / dem_pixels)
    
    new_width = int(dem_width * scaling)
    new_height = int(dem_height * scaling)
    
    #elevation2 = check_dem.data[::scaling_factor, ::scaling_factor].data
    #e2_h, e2_w = elevation2.shape
    #replace nan with -9999
    dem_numpy = np.where(np.isnan(dem_numpy), -9999, dem_numpy)
    #reshape elevation to match elevation2 shape
    print(f"dem_numpy shape: {dem_numpy.shape}")
    #print(f"elevation2 shape: {elevation2.shape}")
    
    #dem_numpy = cv2.resize(dem_numpy, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    dem_numpy = dem_numpy * 1

    # After resize, shape is (e2_h, e2_w)
    rows, cols = dem_numpy.shape
    
    #dem_numpy_data = dem_numpy.astype(np.float32)
    grid = pv.ImageData()
    
    # CORRECT: VTK (x, y, z) = (cols, rows, depth)
    grid.dimensions = (cols, rows, 1)
    
    grid.point_data["Elevation"] = np.flipud(dem_numpy).flatten()
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
    center_z = dem_numpy.max() * 2 + 1000
    
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
    #Image.fromarray(hillshade_array, mode="L").save("test_ray_trace.png", format="PNG", optimize=True)

    print(f"Ray trace complete. Output shape: {hillshade_array.shape}")
    return hillshade_array