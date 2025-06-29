import numpy as np
from typing import Tuple, Optional, List
import xarray
import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform
from functools import partial
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import math


def bbox_to_land_area(nw_corner: Tuple[float, float], se_corner: Tuple[float, float]) -> float:
    """
    Calculate the land area of a bounding box in square meters.

    Args:
        nw_corner (tuple): (longitude, latitude) of the northwest corner
        se_corner (tuple): (longitude, latitude) of the southeast corner

    Returns:
        float: The area in square meters
    """
    # Define the polygon from bounding box corners in WGS84 (latitude/longitude)
    polygon_wgs84 = Polygon([
        (nw_corner[0], nw_corner[1]),  # NW corner
        (se_corner[0], nw_corner[1]),  # NE corner
        (se_corner[0], se_corner[1]),  # SE corner
        (nw_corner[0], se_corner[1])   # SW corner
    ])

    # Use a local Azimuthal Equidistant projection centered on the polygon
    # This projection is well-suited for preserving area for a local region
    lon_c = polygon_wgs84.centroid.x
    lat_c = polygon_wgs84.centroid.y

    # Define the transformation from WGS84 to the custom AEQD projection
    project = partial(
        pyproj.transform,
        pyproj.Proj(proj='latlong', datum='WGS84'),
        pyproj.Proj(proj='aeqd', lat_0=lat_c, lon_0=lon_c)
    )

    # Apply the projection and get the area in square meters
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
    pixel_area = 4800 * 6000

    # Calculate minimum pixel size with a 2.5x safety factor
    minimum_pixel_size = np.sqrt(land_area / pixel_area) * 2.5

    # Available resolutions in meters (from highest to lowest quality)
    valid_search_resolutions = [1, 3, 10, 30, 100]

    # Return all resolutions that are greater than or equal to the minimum pixel size
    valid_resolutions = [x for x in valid_search_resolutions if x >= minimum_pixel_size]
    
    # If no valid resolutions found, return the highest available resolution
    if not valid_resolutions:
        return [valid_search_resolutions[-1]]
    
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
    downsample_factor: int = 4,
    blur_radius: float = 5.0
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

    Returns:
        tuple[Image.Image, tuple[int, int]]: A tuple containing:
            - An RGB image with the shadow on a white background.
            - A tuple with the (x, y) offset of the DEM area within the image.
    """
    # Get the raw numpy data, removing any single-dimensional entries
    dem_numpy = dem_data.data.squeeze()

    dem_numpy = dem_numpy + 800

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
        max_z = np.max(np.concatenate((dem_data_scaled[:, -1], dem_data_scaled[-1, :]))) # left and bottom
    elif phi_deg >= 270 and phi_deg < 360:
        max_z = np.max(np.concatenate((dem_data_scaled[:, -1], dem_data_scaled[0, :]))) # left and top
    elif phi_deg >= 90 and phi_deg < 180:
        max_z = np.max(np.concatenate((dem_data_scaled[:, 0], dem_data_scaled[0, :]))) # right and top
    elif phi_deg >= 0 and phi_deg < 90:
        max_z = np.max(np.concatenate((dem_data_scaled[:, 0], dem_data_scaled[-1, :]))) # right and bottom
    

    max_shadow_len = (max_z * scale) / tan_theta

    width_buffer = dem_width * 0.1
    if max_z > 0:
        scale = width_buffer * tan_theta / max_z * 1
    else:
        scale = 0 # Avoid division by zero if map is flat

    # Calculate padding and final canvas size
    padding = math.ceil(width_buffer) + 5
    canvas_width = dem_width + padding * 2
    canvas_height = dem_height + padding * 2
    dem_offset_x = padding
    dem_offset_y = padding

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
        if z <= 0:
            continue

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