import os
from shapely.geometry import Polygon
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import xdem
from map_layers import ElevationLayer, WaterLayer, StateBoundaryLayer
from utils import reproject_to_web_mercator, create_extrusion_shadow
import math
import geopandas as gpd
from typing import Optional, Tuple

def create_debug_grid_image(hillshade_img, tiles, dem_crs, dem_transform):
    """
    Creates a transparent image with tile boundaries and numbers overlaid.
    """
    grid_img = Image.new('RGBA', hillshade_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Create a single GeoDataFrame for reprojection
    tile_geometries = [tile for tile in tiles]
    gdf = gpd.GeoDataFrame(geometry=tile_geometries, crs="EPSG:4326")
    gdf_proj = gdf.to_crs(dem_crs)

    for i, proj_geom in enumerate(gdf_proj.geometry):
        # Get bounds of the reprojected geometry
        minx, miny, maxx, maxy = proj_geom.bounds

        # Convert map coordinates to pixel coordinates
        # ~transform converts from world to pixel coordinates
        px_minx, px_miny = ~dem_transform * (minx, miny)
        px_maxx, px_maxy = ~dem_transform * (maxx, maxy)
        
        # Draw rectangle, ensuring coordinates are correctly ordered for PIL
        draw.rectangle(
            [min(px_minx, px_maxx), min(px_miny, px_maxy), max(px_minx, px_maxx), max(px_miny, px_maxy)],
            outline='red',
            width=3
        )
        # Draw tile number at the top-left corner of the pixel box
        draw.text((min(px_minx, px_maxx) + 10, min(px_miny, px_maxy) + 10), str(i + 1), fill='red', font=font)

    return grid_img


def create_hillshade(dem_data, 
                    sun_azimuth=315,
                    sun_altitude=45,
                    water_mask=None,
                    water_color=70,
                    gaussian_blur_sigma=0,
                    vertical_exaggeration=1.0):
    """
    Generate a hillshade from DEM data with optional water mask.
    
    Args:
        dem_data: xdem.DEM object containing elevation data
        sun_azimuth: Sun direction in degrees (0=North, 90=East)
        sun_altitude: Sun elevation in degrees (0=horizon, 90=overhead)
        vertical_exaggeration: Factor to exaggerate elevation differences
        water_mask: Optional binary mask of water features
        water_color: Grayscale color (0-255) for water features
        gaussian_blur_sigma: Sigma for Gaussian blur (0 for no blur)
    Returns:
        PIL.Image: The generated hillshade image
    """
    print("Creating hillshade")
    dem = xdem.DEM.from_array(
        dem_data.values.squeeze(),
        transform=dem_data.rio.transform(),
        crs=dem_data.rio.crs
    )

    # Apply vertical exaggeration to the DEM data before hillshade calculation
    if vertical_exaggeration != 1.0:
        dem.data = dem.data * vertical_exaggeration

    # Generate hillshade
    hillshade = xdem.terrain.hillshade(
        dem,
        azimuth=sun_azimuth,
        altitude=sun_altitude
    )

    # Convert to 8-bit grayscale with enhanced contrast
    hillshade_data = hillshade.data.squeeze()
    # Fill NaN values with a value that will correspond to white after normalization
    # This prevents black backgrounds in transparent areas
    hillshade_data = np.nan_to_num(hillshade_data, nan=np.nanmax(hillshade_data))
    
    # Apply contrast enhancement
    hillshade_uint8 = ((hillshade_data - hillshade_data.min()) * 255 / 
                       (hillshade_data.max() - hillshade_data.min())).astype(np.uint8)
    
    # Apply gamma correction to shift light grays toward white while preserving dark areas
    gamma = 1  # Values < 1 brighten mid-tones, > 1 darken mid-tones
    hillshade_uint8 = np.power(hillshade_uint8 / 255.0, gamma) * 255
    hillshade_uint8 = hillshade_uint8.astype(np.uint8)

    # Apply water mask if provided
    if water_mask is not None:
        hillshade_uint8[water_mask] = water_color

    # Apply Gaussian blur if requested
    if gaussian_blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        hillshade_uint8 = gaussian_filter(hillshade_uint8, sigma=gaussian_blur_sigma)

    # Convert to PIL Image
    return Image.fromarray(hillshade_uint8)

def create_elevation_coloring(dem_data, 
                               water_mask=None,
                               water_color=(0, 0, 139),  # Default to dark blue
                               gaussian_blur_sigma=0,
                               vertical_exaggeration=1.0,
                               low_color: Optional[Tuple[int, int, int]] = None,
                               high_color: Optional[Tuple[int, int, int]] = None,
                               use_percentiles: bool = True,
                               percentile_range: Tuple[int, int] = (0.0001, 99),
                               color_method: str = 'grayscale'):
    """
    Create an elevation coloring map from DEM data with optional water mask.
    Can create a grayscale map or a custom color gradient map.
    """
    print("Creating elevation coloring...")
    elevation_data = dem_data.values.squeeze() * vertical_exaggeration

    # Create a mask of valid land elevation data ( > 0 and not NaN)
    valid_elevation_mask = (~np.isnan(elevation_data)) & (elevation_data > -1000)
    valid_elevation_data = elevation_data[valid_elevation_mask]

    if use_percentiles and len(valid_elevation_data) > 0:
        print(f"Using percentiles {percentile_range} for color scaling.")
        min_elev = np.percentile(valid_elevation_data, percentile_range[0])
        max_elev = np.percentile(valid_elevation_data, percentile_range[1])
    else:
        min_elev = np.nanmin(valid_elevation_data) if len(valid_elevation_data) > 0 else 0
        max_elev = np.nanmax(valid_elevation_data) if len(valid_elevation_data) > 0 else 1

    print(f"Elevation range for coloring: {min_elev:.2f}m to {max_elev:.2f}m")

    if color_method == 'gradient':
        # --- COLOR GRADIENT LOGIC ---
        print(f"Applying color gradient: {low_color} -> White -> {high_color}")
        mid_elev = (min_elev + max_elev) / 2
        
        # Create a white canvas
        output_image_data = np.full((elevation_data.shape[0], elevation_data.shape[1], 3), 255, dtype=np.uint8)

        # Interpolate colors only for the valid data points (this is fast)
        r_interp = np.interp(valid_elevation_data, [min_elev, mid_elev, max_elev], [low_color[0], 255, high_color[0]])
        g_interp = np.interp(valid_elevation_data, [min_elev, mid_elev, max_elev], [low_color[1], 255, high_color[1]])
        b_interp = np.interp(valid_elevation_data, [min_elev, mid_elev, max_elev], [low_color[2], 255, high_color[2]])

        # Use the mask to assign the interpolated color channels to the canvas
        output_image_data[valid_elevation_mask, 0] = r_interp.astype(np.uint8)
        output_image_data[valid_elevation_mask, 1] = g_interp.astype(np.uint8)
        output_image_data[valid_elevation_mask, 2] = b_interp.astype(np.uint8)

        if water_mask is not None:
            output_image_data[water_mask] = water_color

        if gaussian_blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            output_image_data = gaussian_filter(output_image_data, sigma=(gaussian_blur_sigma, gaussian_blur_sigma, 0))

        return Image.fromarray(output_image_data, 'RGB')
    elif color_method == 'grayscale':
        # --- GRAYSCALE LOGIC ---
        print("Applying grayscale elevation coloring.")
        
        # Interpolate colors across the entire 2D array
        normalized_data = np.interp(elevation_data, [min_elev, max_elev], [0, 255])

        # Start with a white canvas
        output_image_data = np.full(elevation_data.shape, 255, dtype=np.uint8)

        # Use the mask to place the colored pixels onto the canvas
        output_image_data[valid_elevation_mask] = normalized_data[valid_elevation_mask].astype(np.uint8)

        if water_mask is not None:
            gray_water_color = int(sum(water_color) / 3) if isinstance(water_color, tuple) else water_color
            output_image_data[water_mask] = gray_water_color

        if gaussian_blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            output_image_data = gaussian_filter(output_image_data, sigma=gaussian_blur_sigma)

        return Image.fromarray(output_image_data)

def create_printable_map(hillshade_img: Image.Image,
                        title: str,
                        output_prefix: str,
                        shadow_img: Image.Image,
                        shadow_offsets: tuple[int, int],
                        state_mask: Optional[np.ndarray] = None,
                        width_inches: float = 5.0,
                        height_inches: float = 7.0,
                        margin_inches: float = 0.25,
                        dpi: int = 600,
                        border: bool = False,
                        shadow: bool = True,
                        transparent_background: bool = False) -> tuple[str, str]:
    """
    Create printable versions (PDF and high-res image) of a hillshade map with title and drop shadow.
    
    Args:
        hillshade_img: Input hillshade PIL Image
        title: Map title text
        output_prefix: Prefix for output filenames (without extension)
        shadow_img: Pre-rendered shadow layer image.
        shadow_offsets: (x, y) offset of the hillshade within the shadow image.
        width_inches: Output width in inches
        height_inches: Output height in inches
        margin_inches: Margin between hillshade and edge in inches
        dpi: Output resolution in dots per inch
    
    Returns:
        tuple[str, str]: Paths to the generated PDF and image files
    """
    # Calculate dimensions in pixels
    width_px = int(width_inches * dpi)
    height_px = int(height_inches * dpi)
    margin_px = int(margin_inches * dpi)
    
    # Create white background
    background = Image.new('RGB', (width_px, height_px), 'white')
    draw = ImageDraw.Draw(background)

    # Calculate title space (x% of height or minimum y inch)
    title_height_px = max(int(height_px * 0.05), int(.6 * dpi))
    
    # Calculate available space for hillshade
    available_width = width_px - (2 * margin_px)
    available_height = height_px - (2 * margin_px) - title_height_px
    
    # Resize hillshade to fit while maintaining aspect ratio
    hillshade_ratio = hillshade_img.width / hillshade_img.height
    available_ratio = available_width / available_height
    
    if hillshade_ratio > available_ratio:
        # Width limited
        new_width = available_width
        new_height = int(new_width / hillshade_ratio)
    else:
        # Height limited
        new_height = available_height
        new_width = int(new_height * hillshade_ratio)
    
    hillshade_resized = hillshade_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    state_mask_resized = None
    if state_mask is not None:
        # Convert numpy boolean mask to a PIL 'L' mode image for the alpha channel
        mask_img = Image.fromarray(state_mask.astype('uint8') * 255, 'L')
        # Resize mask to match resized hillshade
        state_mask_resized = mask_img.resize((new_width, new_height), Image.Resampling.NEAREST)

    # If transparent background is requested, use the state mask as the alpha channel
    if transparent_background and state_mask_resized:
        # Convert hillshade to RGBA to be able to add an alpha channel
        hillshade_resized = hillshade_resized.convert("RGBA")
        # Apply the mask to the alpha channel
        hillshade_resized.putalpha(state_mask_resized)
    
    # Calculate positioning for the hillshade image
    x_offset = (width_px - new_width) // 2
    y_offset = title_height_px + ((height_px - title_height_px - new_height) // 2)
    
    # --- New Shadow Logic ---
    # Resize the shadow image and place it on the canvas
    dem_width_orig, _ = hillshade_img.size
    resize_factor = new_width / dem_width_orig

    if shadow:
        shadow_img_resized = shadow_img.resize(
            (int(shadow_img.width * resize_factor), int(shadow_img.height * resize_factor)),
            Image.Resampling.LANCZOS
        )
        
        # Calculate the offset of the DEM area within the *resized* shadow image
        shadow_dem_offset_x = int(shadow_offsets[0] * resize_factor)
        shadow_dem_offset_y = int(shadow_offsets[1] * resize_factor)

        # Calculate the paste position for the entire shadow image on the background
        shadow_pos_x = x_offset - shadow_dem_offset_x
        shadow_pos_y = y_offset - shadow_dem_offset_y
        
        # Paste the shadow image (which includes the white buffer) onto the main canvas
        background.paste(shadow_img_resized, (shadow_pos_x, shadow_pos_y))
    
    # Paste hillshade onto background, over the shadow layer
    if border:
        border_width = int(new_width / 200)
        bordered_hillshade = ImageOps.expand(hillshade_resized, border=border_width, fill='black')
        paste_mask = None
        if state_mask_resized:
            paste_mask = ImageOps.expand(state_mask_resized, border=border_width, fill=True)
        background.paste(bordered_hillshade, (x_offset - border_width, y_offset - border_width), paste_mask)
    else:
        # If the image has an alpha channel, use it as the mask
        if hillshade_resized.mode == 'RGBA':
            background.paste(hillshade_resized, (x_offset, y_offset), hillshade_resized)
        else:
            background.paste(hillshade_resized, (x_offset, y_offset), state_mask_resized)
    
    # Calculate font size based on available title space and title length
    # Start with a larger percentage of title height for shorter titles
    base_font_size = int(title_height_px * 0.7)  # Start larger than before
    
    # Try to find the optimal font size that fits within the width
    font = None
    font_size = base_font_size
    while font_size > 12:  # Don't go smaller than 12px
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Bodoni 72 OS.ttc", font_size)
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            if title_width <= (width_px - 2 * margin_px):
                break
            font_size = int(font_size * 0.9)  # Reduce by 10% and try again
        except:
            # Fallback to default font if Bodoni not available
            print("Warning: Bodoni 72 Oldstyle font not found, falling back to default font")
            font = ImageFont.load_default()
            break
    
    # Add title
    if font:
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        title_x = (width_px - title_width) // 2
        # Center title vertically in the title space
        # if the image was width limited, center vertically
        if new_width < new_height:
            title_y = (y_offset - title_height_px) // 2 + 20
        else:
            title_y = (title_height_px - title_height) // 2 + margin_px + 20
        draw.text((title_x, title_y), title, fill='black', font=font)
    
    # Save outputs
    pdf_path = f"{output_prefix}.pdf"
    png_path = f"{output_prefix}.png"
    jpeg_path = f"{output_prefix}.jpeg"
    
    # Save high-res PNG
    background.save(png_path, "PNG", dpi=(dpi, dpi))
    
    # Convert to RGB for JPEG and PDF saving if it has an alpha channel
    save_img = background
    if save_img.mode == 'RGBA':
        save_img = Image.new("RGB", background.size, (255, 255, 255))
        save_img.paste(background, mask=background.split()[3]) # 3 is the alpha channel
        
    save_img.save(jpeg_path, "JPEG", dpi=(dpi, dpi))
    
    # Save PDF
    save_img.save(pdf_path, "PDF", resolution=dpi)
    
    return pdf_path, png_path, jpeg_path


def generate_map(coordinates,
                state_polygon: Optional[Polygon] = None,
                location_name="unnamed",
                resolution=None,
                sun_azimuth=315,
                sun_altitude=45,
                vertical_exaggeration=1.0,
                use_cache=True,
                include_water=True,
                water_color=70,
                gaussian_blur_sigma=0,
                create_debug_grid=False,
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
                transparent_background=False,
                low_color: Optional[Tuple[int, int, int]] = None,
                high_color: Optional[Tuple[int, int, int]] = None,
                use_percentiles: bool = True):
    """
    Generate a hillshade map for the given polygon coordinates or state name.
    Args:
        coordinates: List of (lat, lon) tuples defining the polygon in clockwise order
        state_name: Name of the state (e.g., 'Oregon')
        land_only: If True, clip state to landmass (no ocean/Great Lakes)
        location_name: Name for the location (used in cache filenames)
        resolution: Optional specific resolution in meters
        sun_azimuth: Sun direction in degrees (0=North, 90=East)
        sun_altitude: Sun elevation in degrees (0=horizon, 90=overhead)
        vertical_exaggeration: Factor to exaggerate elevation differences
        use_cache: Whether to use cached data
        include_water: Whether to include water features
        water_color: Grayscale color (0-255) for water features
        gaussian_blur_sigma: Sigma for Gaussian blur (0 for no blur)
        create_debug_grid: If True, overlay a numbered grid of tiles for debugging.
        output_dir: Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create polygon from coordinates, swapping lat/lon to lon/lat
    polygon = Polygon([(lon, lat) for lat, lon in coordinates])
    
    # Initialize layers
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", location_name)
    elevation_layer = ElevationLayer(cache_dir=cache_dir, location_name=location_name)
    water_layer = WaterLayer(cache_dir=cache_dir, location_name=location_name) if include_water else None
    
    # Get raw DEM data without vertical exaggeration
    dem_data, metadata, tiles = elevation_layer.get_mosaic_dem(
        polygon_geom=polygon,
        use_cache=use_cache,
        vertical_exaggeration=1.0,
        state_polygon=state_polygon
    )
    
    # Save the full mosaic to cache
    mosaic_cache_path, metadata = elevation_layer._get_cache_filepath(polygon_geom=polygon, resolution='mosaic')
    dem_data.rio.to_raster(mosaic_cache_path)
    print(f"Saved mosaic DEM to {mosaic_cache_path}")
    
    # Reproject DEM to Web Mercator for accurate distance calculations
    dem_data = reproject_to_web_mercator(dem_data)

    state_mask = None
    if transparent_background and state_polygon:
        # Create a StateBoundaryLayer instance to generate the mask
        state_layer = StateBoundaryLayer(cache_dir=cache_dir, location_name=location_name)
        # Create the state mask *after* reprojection to ensure CRS alignment
        state_mask = state_layer.create_state_mask(state_polygon, dem_data)
        
        # Mask the DEM data itself to prevent shadows from outside the state
        # and to make the hillshade transparent later on.
        dem_data = dem_data.where(state_mask)

    shadow_img, shadow_offsets = create_extrusion_shadow(dem_data,
                                        light_elevation_deg=65,
                                        light_azimuth=sun_azimuth,
                                        scale=0.2,
                                        transparent_background=transparent_background,
                                        state_mask=state_mask)
    
    #

    # Get water mask if requested
    water_mask = None
    if include_water and water_layer is not None:
        water_mask = water_layer.get_water_mask(
            polygon_geom=polygon,
            dem_data=dem_data,
            use_cache=use_cache
        )

    do_hillshade = False
    do_elevation_coloring = True
    if do_hillshade:
        # Create hillshade
        hillshade = create_hillshade(
            dem_data=dem_data,
            sun_azimuth=sun_azimuth,
            sun_altitude=sun_altitude,
            water_mask=water_mask,
            water_color=water_color,
            gaussian_blur_sigma=gaussian_blur_sigma,
            vertical_exaggeration=vertical_exaggeration
        )
    elif do_elevation_coloring:
        hillshade = create_elevation_coloring(
            dem_data=dem_data,
            water_mask=water_mask,
            water_color=water_color,
            gaussian_blur_sigma=gaussian_blur_sigma,
            vertical_exaggeration=vertical_exaggeration,
            low_color=low_color,
            high_color=high_color,
            use_percentiles=use_percentiles,
            color_method='gradient'
        )

    # Apply transparency mask if requested
    if transparent_background and state_mask is not None:
        # Ensure mask is the same size as the hillshade
        if state_mask.shape != (hillshade.height, hillshade.width):
             mask_img = Image.fromarray(state_mask.astype('uint8') * 255, 'L')
             mask_resized = mask_img.resize(hillshade.size, Image.Resampling.NEAREST)
             state_mask = np.array(mask_resized) > 0 # Convert back to boolean mask

        # Convert hillshade to RGBA and apply the mask to the alpha channel
        hillshade = hillshade.convert("RGBA")
        alpha_channel = np.where(state_mask, 255, 0).astype('uint8')
        hillshade.putalpha(Image.fromarray(alpha_channel, 'L'))

    if create_debug_grid and tiles is not None:
        debug_grid = create_debug_grid_image(
            hillshade_img=hillshade,
            tiles=tiles,
            dem_crs=dem_data.rio.crs,
            dem_transform=dem_data.rio.transform()
        )
        # Convert hillshade to RGBA to allow alpha compositing
        hillshade = Image.alpha_composite(hillshade.convert("RGBA"), debug_grid)

    # Save basic outputs
    bbox = polygon.bounds
    output_base = f"{location_name}_{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
    
    dem_path = os.path.join(output_dir, f"{output_base}_dem.tif")
    hillshade_path = os.path.join(output_dir, f"{output_base}_hillshade.png")
    
    # Save DEM
    dem_data.rio.to_raster(dem_path)
    
    # Save basic hillshade
    hillshade.save(hillshade_path)
    
    
    print(f"Generated hillshade map at {hillshade_path}")
    print(f"DEM resolution: {metadata['resolution']}m")
    print(f"Image dimensions: {hillshade.size}")
    
    return hillshade, dem_data, metadata, shadow_img, shadow_offsets

def get_state_bbox_and_mask(state_name, land_only=True, location_name="unnamed"):
    """
    Get the state polygon, bounding box, and (optionally) a mask for a DEM.
    If dem_data is provided, returns the mask as well.
    """
    state_layer = StateBoundaryLayer(cache_dir="data_cache", location_name=location_name)
    state_polygon = state_layer.get_state_polygon(state_name, land_only=land_only)
    bbox = state_polygon.bounds

    return state_polygon, bbox, state_layer

if __name__ == "__main__":
    # Example usage for a state
    state_name = "Oregon"
    title = state_name
    sun_azimuth = 300
    sun_altitude = 65
    land_only = True  # Set to False to include state marine/Great Lakes territory
    #location_name = state_name.lower()

    if state_name is not None:
        location_name = state_name.lower()
        # Get state polygon and bbox
        state_polygon, bbox, state_layer = get_state_bbox_and_mask(state_name, land_only=land_only, location_name=location_name)
            # Use bbox to create a polygon for generate_map
        minx, miny, maxx, maxy = bbox
        bbox_coords = [
            (miny, minx),
            (miny, maxx),
            (maxy, maxx),
            (maxy, minx),
            (miny, minx)
    ]
    else:
        state_polygon = None
        # TODO read args from a config file
        # Example usage
        # Define a polygon around Mount Rainier
        rainier_coords = [
            (46.9, -121.8),  # Northwest
            (46.9, -121.7),  # Northeast
            (46.8, -121.7),  # Southeast
            (46.8, -121.8),  # Southwest
            (46.9, -121.8),  # Close the polygon
        ]
        
        # Astoria coordinates are already in lat/lon format
        astoria_coords = [
            (46.394496, -124.094430),
            (46.394496, -123.33293),
            (45.725537, -123.33293),
            (45.725537, -124.094430),
            (46.394496, -124.094430)
        ]

        astoria2_coords = [
            (46.394496, -124.094430),
            (46.394496, -123.414352),
            (45.725537, -123.414352),
            (45.725537, -124.094430),
            (46.394496, -124.094430)]
        
        hood_coords = [
            (45.521676, -121.844886),
            (45.521676, -121.472096),
            (45.0799, -121.472096),
            (45.0799, -121.844886),
            (45.521676, -121.844886)
        ]
        location_name = "mt_hood"
        title = "Mt. Hood"
        bbox_coords = hood_coords



    hillshade_img, dem_data, metadata, shadow_img, shadow_offsets = generate_map(
        coordinates=bbox_coords,
        state_polygon=state_polygon,
        location_name=location_name,
        vertical_exaggeration=7.0,
        sun_azimuth=sun_azimuth,
        sun_altitude=sun_altitude,
        include_water=True,
        water_color=(2, 6, 83),
        use_cache=True,
        create_debug_grid=False,
        transparent_background=True,
        low_color=(0, 0, 255),    # Dark blue/purple for low elevations
        high_color=(255, 239, 160) # Creamy yellow for high elevations
    )

    if state_name is not None:
        # Now get the state mask for the DEM
        state_mask = state_layer.create_state_mask(state_polygon, dem_data)
    else:
        state_mask = None

    # save state mask to rgbimage
    # if state_mask is not None:
    #     # Ensure the array is C-contiguous before passing to fromarray
    #     contiguous_mask = np.ascontiguousarray(state_mask.astype('uint8') * 255)
    #     state_mask_img = Image.fromarray(contiguous_mask, 'L')
    #     state_mask_img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"{title}_state_mask.png"))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    printable_base = os.path.join(output_dir, f"{title}_print")
    pdf_path, png_path, jpeg_path = create_printable_map(
        hillshade_img=hillshade_img,
        shadow_img=shadow_img,
        shadow_offsets=shadow_offsets,
        state_mask=state_mask,
        title=title,
        output_prefix=printable_base,
        width_inches=5.0,
        height_inches=7.0,
        margin_inches=0.5,
        dpi=600, # 2400
        border=True,
        shadow=False,
        transparent_background=True
    )
    print(f"Generated printable map versions:")
    print(f"PDF: {pdf_path}")
    print(f"High-res image: {png_path}")
    print(f"High-res image: {jpeg_path}")