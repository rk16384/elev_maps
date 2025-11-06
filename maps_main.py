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
from map_data import MapData
import xarray
from affine import Affine

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
                    vertical_exaggeration=1.0):
    """
    Generate a hillshade from DEM data with optional water mask.
    
    Args:
        dem_data: xdem.DEM object containing elevation data
        sun_azimuth: Sun direction in degrees (0=North, 90=East)
        sun_altitude: Sun elevation in degrees (0=horizon, 90=overhead)
        vertical_exaggeration: Factor to exaggerate elevation differences
        water_mask: Optional binary mask of water features
        water_color: Grayscale color (0-255) or RGB tuple (r, g, b) for water features
        gaussian_blur_sigma: Sigma for Gaussian blur (0 for no blur)
    Returns:
        PIL.Image: The generated hillshade image
    """
    
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
    
    return hillshade_uint8
    
def add_water_to_hillshade(hillshade_uint8, water_mask=None, water_color=70, gaussian_blur_sigma=0):
       
    # Check if water color is RGB or grayscale
    is_rgb_water = isinstance(water_color, tuple) and len(water_color) == 3 

    # Apply water mask if provided
    if water_mask is not None:
        if is_rgb_water:
            # Convert hillshade to RGB mode to support colored water
            hillshade_rgb = np.stack([hillshade_uint8] * 3, axis=-1)
            # Apply RGB water color
            hillshade_rgb[water_mask] = water_color
            hillshade_uint8 = hillshade_rgb
        else:
            # Apply grayscale water color
            hillshade_uint8[water_mask] = water_color

    # Apply Gaussian blur if requested
    if gaussian_blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        if is_rgb_water:
            # Apply blur to each RGB channel separately
            hillshade_uint8 = gaussian_filter(hillshade_uint8, sigma=(gaussian_blur_sigma, gaussian_blur_sigma, 0))
        else:
            hillshade_uint8 = gaussian_filter(hillshade_uint8, sigma=gaussian_blur_sigma)

    # Convert to PIL Image
    if is_rgb_water:
        return Image.fromarray(hillshade_uint8, 'RGB')
    else:
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
        mid_elev = np.mean(valid_elevation_data)
        
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

def create_land_mask_smart_water_detection(dem_data: xarray.DataArray, 
                                          elevation_threshold: float = -200.0,  # Much lower threshold
                                          flat_threshold: float = 0.1,
                                          min_flat_area_pixels: int = 100) -> np.ndarray:
    """
    Smart water detection that preserves low-lying land areas.
    """
    elevation = dem_data.values.squeeze()
    
    # Start with a very permissive elevation threshold (preserve low-lying land)
    land_mask = elevation > elevation_threshold  # -10m to preserve most land
    
    # Method 1: Remove areas that are exactly flat AND at or near sea level
    # Water typically has elevation = 0 or very close to 0
    sea_level_mask = np.abs(elevation) < 0.5  # Within 0.5m of sea level
    land_mask = land_mask & ~sea_level_mask
    
    # Method 2: Remove large flat areas that are likely water bodies
    # Use simple local variation to find flat areas
    diff_x = np.abs(np.diff(elevation, axis=1))
    diff_y = np.abs(np.diff(elevation, axis=0))
    
    # Pad to match original size
    diff_x = np.pad(diff_x, ((0, 0), (0, 1)), mode='edge')
    diff_y = np.pad(diff_y, ((0, 1), (0, 0)), mode='edge')
    
    # Areas with very small differences are likely flat
    local_variation = np.maximum(diff_x, diff_y)
    flat_mask = local_variation < flat_threshold
    
    # Only remove flat areas that are also near sea level (likely water)
    water_flat_mask = flat_mask & sea_level_mask
    
    # Remove small isolated flat areas (likely noise)
    from scipy.ndimage import binary_opening
    water_flat_mask = binary_opening(water_flat_mask, structure=np.ones((3,3)))
    
    land_mask = land_mask & ~water_flat_mask
    
    return land_mask

def create_printable_map(map_data: MapData,
                        title: str,
                        output_prefix: str,
                        width_inches: float = 5.0,
                        height_inches: float = 7.0,
                        margin_inches: float = 0.15,
                        dpi: int = 600,
                        border: bool = False,
                        shadow: bool = True,
                        input_font: str = "Bodoni 72 OS") -> tuple[str, str]:
    """
    Create printable versions (PDF and high-res image) of a hillshade map with title and drop shadow.
    
    Args:
        map_data: MapData object containing all map data
        title: Map title text
        output_prefix: Prefix for output filenames (without extension)
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
    title_height_px = max(int(height_px * 0.04), int(.4 * dpi))
    
    # Calculate available space for hillshade
    available_width = width_px - (2 * margin_px)
    available_height = height_px - (2 * margin_px) - title_height_px
    
    # Resize hillshade to fit while maintaining aspect ratio
    hillshade_ratio = map_data.hillshade.width / map_data.hillshade.height
    available_ratio = available_width / available_height
    
    if hillshade_ratio > available_ratio:
        # Width limited
        new_width = available_width
        new_height = int(new_width / hillshade_ratio)
    else:
        # Height limited
        new_height = available_height
        new_width = int(new_height * hillshade_ratio)
    
    hillshade_resized = map_data.hillshade.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    state_mask_resized = None
    if map_data.location_mask is not None:
        # Convert numpy boolean mask to a PIL 'L' mode image for the alpha channel
        mask_img = Image.fromarray(map_data.location_mask.astype('uint8') * 255, 'L')
        # Resize mask to match resized hillshade
        state_mask_resized = mask_img.resize((new_width, new_height), Image.Resampling.NEAREST)

    # If transparent background is requested, use the state mask as the alpha channel
    if map_data.common_config['transparent_background'] and state_mask_resized:
        # Convert hillshade to RGBA to be able to add an alpha channel
        hillshade_resized = hillshade_resized.convert("RGBA")
        # Apply the mask to the alpha channel
        hillshade_resized.putalpha(state_mask_resized)
    
    # Calculate positioning for the hillshade image
    x_offset = (width_px - new_width) // 2
    y_offset = title_height_px + ((height_px - title_height_px - new_height) // 3) # tittle height + margin/2
    y_offset = (height_px - new_height) // 2
    
    # --- New Shadow Logic ---
    # Resize the shadow image and place it on the canvas
    dem_width_orig, _ = map_data.hillshade.size
    resize_factor = new_width / dem_width_orig

    if shadow:
        shadow_img_resized = map_data.shadow_img.resize(
            (int(map_data.shadow_img.width * resize_factor), int(map_data.shadow_img.height * resize_factor)),
            Image.Resampling.LANCZOS
        )
        
        # Calculate the offset of the DEM area within the *resized* shadow image
        shadow_dem_offset_x = int(map_data.shadow_offsets[0] * resize_factor)
        shadow_dem_offset_y = int(map_data.shadow_offsets[1] * resize_factor)

        # Calculate the paste position for the entire shadow image on the background
        shadow_pos_x = x_offset - shadow_dem_offset_x
        shadow_pos_y = y_offset - shadow_dem_offset_y
        
        # Paste the shadow image (which includes the white buffer) onto the main canvas
        background.paste(shadow_img_resized, (shadow_pos_x, shadow_pos_y))
    
    # Paste hillshade onto background, over the shadow layer
    if border:
        border_width = int(new_width / 500)
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
    
    # Try to find the optimal font size that fits within both width and height
    font = None
    font_size = base_font_size
    while font_size > 12:  # Don't go smaller than 12px
        try:
            font = ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{input_font}", font_size)
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]
            
            # Check both width and height constraints
            width_ok = title_width <= (width_px - 2 * margin_px)
            height_ok = title_height <= (title_height_px * 0.9)  # Leave 10% margin for vertical spacing
            
            if width_ok and height_ok:
                break
            font_size = int(font_size * 0.9)  # Reduce by 10% and try again
        except:
            # Fallback to default font if Bodoni not available
            print("Warning: Bodoni 72 Oldstyle font not found, falling back to default font")
            font = ImageFont.load_default()
            break
    
    # If we couldn't find a font that fits, use the smallest size and let it overflow
    if font is None:
        try:
            font = ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{input_font}.ttc", 12)
        except:
            font = ImageFont.load_default()
    
    # Add title
    if font:
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        title_x = (width_px - title_width) // 2
        # Center title vertically in the title space
        # if the image was width limited, center vertically
        if new_width < new_height:
            title_y = (y_offset - title_height_px) // 2 + 120
        else:
            title_y = (title_height_px - title_height) // 2 + margin_px + 120
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

def save_to_image(hillshade, filename: str, map_data: MapData = None):
    """
    Save a hillshade (NumPy array or PIL Image) to disk for debugging.
    """
    base_path = os.path.join(map_data.output_dir, f"{map_data.location_name}_{filename}")

    if isinstance(hillshade, Image.Image):
        hillshade_img = hillshade
    else:
        hillshade_img = Image.fromarray(np.asarray(hillshade))

    # PNG debug copy
    #hillshade_img.save(f"{base_path}.png", format="PNG", optimize=True)
    width, height = hillshade_img.size
    total_pixels = width * height
    print(f"{filename} original size: {width}x{height} ({total_pixels/1e6:.1f} Mpx)")
    max_pixels = 100_000_000
    if total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
        new_size = (int(width * scale), int(height * scale))
        hillshade_img = hillshade_img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Downscaled to: {new_size[0]}x{new_size[1]} ({new_size[0]*new_size[1]/1e6:.1f} Mpx")
        hillshade_img.save(f"{base_path}.png", format="PNG", optimize=True)

    # JPEG copy compatible with upload sites
    jpeg_img = hillshade_img.convert("RGB")
    jpeg_img.save(
        f"{base_path}.jpg",
        format="JPEG",
        quality=95,
        subsampling=0,      # keeps max detail, still baseline
        optimize=True
    )

    print(f"Saved debug images at {base_path}.png and {base_path}.jpg")
    print(f"{filename} size: {hillshade_img.size[0]}x{hillshade_img.size[1]}")

def generate_map(create_debug_grid=False,
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
                map_data: MapData = None):
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
    map_data.output_dir = output_dir
    
    # Create polygon from coordinates, swapping lat/lon to lon/lat
    #polygon = Polygon([(lon, lat) for lat, lon in coordinates])
    
    # Initialize layers
    #cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", location_name)
    elevation_layer = ElevationLayer(cache_dir=map_data.cache_dir, location_name=map_data.location_name)
    water_layer = WaterLayer(cache_dir=map_data.cache_dir, location_name=map_data.location_name) if map_data.include_water else None
    
    # --- DEM DATA ---
    if map_data.use_cache_dem and map_data.cache_exists('dem'):
        print("Loading DEM from cache...")
        map_data.dem_data = map_data.load_from_cache('dem')
        # Load metadata separately or reconstruct it
        map_data.metadata = {'resolution': 'cached', 'crs': map_data.dem_data.rio.crs}
        map_data.tiles = []  # Tiles not cached separately
    else:
        print("Fetching DEM data...")
        map_data.dem_data, map_data.metadata, map_data.tiles = elevation_layer.get_mosaic_dem(
            polygon_geom=map_data.location_polygon,
            use_cache=map_data.use_cache_dem,
            vertical_exaggeration=map_data.vertical_exaggeration,
            state_polygon=map_data.location_polygon
        )
        
        if map_data.use_cache_dem:
            map_data.save_to_cache('dem', map_data.dem_data)
    
    # Check if DEM needs downsampling
    max_pixels = 50_000_000  # 50 million pixels (adjust as needed)
    dem_shape = map_data.dem_data.shape
    if len(dem_shape) == 3: 
        current_pixels = dem_shape[0] * dem_shape[1] * dem_shape[2]
    else:
        current_pixels = dem_shape[0] * dem_shape[1]
    
    if False: #current_pixels > max_pixels:
        print(f"DEM is very large ({current_pixels:,} pixels). Consider downsampling for better performance.")
        
        # Calculate downsampling factor
        downsample_factor = int(np.sqrt(current_pixels / max_pixels))
        print(f"Recommended downsampling factor: {downsample_factor}")
        downsample_factor = 0
        
        # Ask user or use automatic downsampling
        if hasattr(map_data, 'auto_downsample') and map_data.auto_downsample:
            print(f"Auto-downsampling DEM by factor {downsample_factor}...")
            map_data.dem_data = downsample_dem(map_data.dem_data, downsample_factor)
            print(f"Downsampled DEM shape: {map_data.dem_data.shape}")
            print(f"Downsampled DEM size: {map_data.dem_data.nbytes / 1024**3:.2f} GB")

   

    # Reproject DEM to Web Mercator for accurate distance calculations
    map_data.dem_data = reproject_to_web_mercator(map_data.dem_data)

    # --- STATE BOUNDARY/MASK ---
    if map_data.common_config['transparent_background'] and map_data.location_type == 'state':
        print("Creating state mask...")
        state_layer = map_data.location_layer
        map_data.location_mask = state_layer.create_state_mask(map_data.location_polygon, map_data.dem_data)
        # Use the new function to create a land mask
        land_mask = create_land_mask_smart_water_detection(map_data.dem_data, elevation_threshold=-1000)
        map_data.location_mask = map_data.location_mask & land_mask
        # Mask the DEM data itself
        map_data.dem_data = map_data.dem_data.where(map_data.location_mask)

    # --- SHADOW ---
    if map_data.use_cache_shadow and map_data.cache_exists('shadow'):
        print("Loading shadow from cache...")
        map_data.shadow_img = map_data.load_from_cache('shadow')
        # Load shadow offsets separately
        shadow_offsets_path = map_data.get_cache_path('shadow_offsets', '.pkl')
        if os.path.exists(shadow_offsets_path):
            import pickle
            with open(shadow_offsets_path, 'rb') as f:
                map_data.shadow_offsets = pickle.load(f)
    else:
        print("Creating shadow...")
        map_data.shadow_img, map_data.shadow_offsets = create_extrusion_shadow(
            map_data.dem_data,
            light_elevation_deg=map_data.shadow_config['light_elevation_deg'],
            light_azimuth=map_data.shadow_config['light_azimuth'],
            scale=map_data.shadow_config['scale'],
            transparent_background=map_data.shadow_config['transparent_background'],
            state_mask=map_data.location_mask
        )
        
        if map_data.use_cache_shadow:
            map_data.save_to_cache('shadow', map_data.shadow_img)
            # Save shadow offsets separately
            shadow_offsets_path = map_data.get_cache_path('shadow_offsets', '.pkl')
            import pickle
            with open(shadow_offsets_path, 'wb') as f:
                pickle.dump(map_data.shadow_offsets, f)

    # --- WATER MASK ---
    if map_data.include_water and water_layer is not None:
        print("Getting water mask...")
        
        # Get water filter config from location config
        water_filter_config = map_data.location_config.get('water_filter_config', None)
        
        # Use the original method that handles its own internal caching
        map_data.water_mask = water_layer.get_water_mask(
            polygon_geom=map_data.location_polygon,
            dem_data=map_data.dem_data,
            use_cache=map_data.use_cache_water,
            state_polygon=map_data.location_polygon,
            water_filter_config=water_filter_config
        )

    # --- HILLSHADE OR ELEVATION COLORING ---
    do_hillshade = map_data.common_config['do_hillshade']
    do_elevation_coloring = map_data.common_config['do_elevation_coloring']
    
    if do_hillshade:
        if map_data.use_cache_hillshade and map_data.cache_exists('hillshade'):
            print("Loading hillshade from cache...")
            map_data.hillshade = map_data.load_from_cache('hillshade')
        else:
            print("Creating hillshade...")
            map_data.hillshade = create_hillshade(
                dem_data=map_data.dem_data,
                sun_azimuth=map_data.hillshade_config['sun_azimuth'],
                sun_altitude=map_data.hillshade_config['sun_altitude'],
                vertical_exaggeration=map_data.hillshade_config['vertical_exaggeration']
            )
            
            if map_data.use_cache_hillshade:
                map_data.save_to_cache('hillshade', map_data.hillshade)

        save_to_image(map_data.hillshade, 'hillshade_debug', map_data)

        # Add water to hillshade
        if map_data.include_water:
            map_data.hillshade = add_water_to_hillshade(map_data.hillshade, map_data.water_mask, map_data.hillshade_config['water_color'], map_data.hillshade_config['gaussian_blur_sigma'])
        else:
            map_data.hillshade = Image.fromarray(map_data.hillshade)

        save_to_image(map_data.hillshade, 'hillshade_water_debug', map_data)

    elif do_elevation_coloring:
        if map_data.use_cache_elevation_coloring and map_data.cache_exists('elevation_coloring'):
            print("Loading elevation coloring from cache...")
            map_data.hillshade = map_data.load_from_cache('elevation_coloring')
        else:
            print("Creating elevation coloring...")
            map_data.hillshade = create_elevation_coloring(
                dem_data=map_data.dem_data,
                water_mask=map_data.water_mask,
                water_color=map_data.elevation_coloring_config['water_color'],
                gaussian_blur_sigma=map_data.elevation_coloring_config['gaussian_blur_sigma'],
                vertical_exaggeration=map_data.elevation_coloring_config['vertical_exaggeration'],
                low_color=map_data.elevation_coloring_config['low_color'],
                high_color=map_data.elevation_coloring_config['high_color'],
                use_percentiles=map_data.elevation_coloring_config['use_percentiles'],
                color_method=map_data.elevation_coloring_config['color_method']
            )
            
            if map_data.use_cache_elevation_coloring:
                map_data.save_to_cache('elevation_coloring', map_data.hillshade)

    # Apply transparency mask if requested
    if map_data.hillshade_config['transparent_background'] and map_data.location_mask is not None:
        # Ensure mask is the same size as the hillshade
        if map_data.location_mask.shape != (map_data.hillshade.height, map_data.hillshade.width):
             mask_img = Image.fromarray(map_data.location_mask.astype('uint8') * 255, 'L')
             mask_resized = mask_img.resize(map_data.hillshade.size, Image.Resampling.NEAREST)
             map_data.location_mask = np.array(mask_resized) > 0 # Convert back to boolean mask

        # Convert hillshade to RGBA and apply the mask to the alpha channel
        map_data.hillshade = map_data.hillshade.convert("RGBA")
        alpha_channel = np.where(map_data.location_mask, 255, 0).astype('uint8')
        map_data.hillshade.putalpha(Image.fromarray(alpha_channel, 'L'))

    if create_debug_grid and map_data.tiles is not None:
        map_data.debug_grid = create_debug_grid_image(
            hillshade_img=map_data.hillshade,
            tiles=map_data.tiles,
            dem_crs=map_data.dem_data.rio.crs,
            dem_transform=map_data.dem_data.rio.transform()
        )
        # Convert hillshade to RGBA to allow alpha compositing
        map_data.hillshade = Image.alpha_composite(map_data.hillshade.convert("RGBA"), map_data.debug_grid)

    # Save basic outputs
    bbox = map_data.location_polygon.bounds
    output_base = f"{map_data.location_name}_{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
    
    map_data.dem_path = os.path.join(map_data.output_dir, f"{output_base}_dem.tif")
    map_data.hillshade_path = os.path.join(map_data.output_dir, f"{output_base}_hillshade.png")
    
    # Save DEM
    map_data.dem_data.rio.to_raster(map_data.dem_path)
    
    # Save basic hillshade
    map_data.hillshade.save(map_data.hillshade_path)
    
    print(f"Generated hillshade map at {map_data.hillshade_path}")
    print(f"DEM resolution: {map_data.metadata['resolution']}m")
    print(f"Image dimensions: {map_data.hillshade.size}")
    
    return map_data

def get_state_bbox_and_mask(state_name, land_only=True, location_name="unnamed"):
    """
    Get the state polygon, bounding box, and (optionally) a mask for a DEM.
    If dem_data is provided, returns the mask as well.
    """
    state_layer = StateBoundaryLayer(cache_dir="data_cache", location_name=location_name)
    state_polygon = state_layer.get_state_polygon(state_name, land_only=land_only)
    bbox = state_polygon.bounds

    return state_polygon, bbox, state_layer

from scipy.ndimage import zoom
import numpy as np

def downsample_dem(dem_data: xarray.DataArray, factor: int) -> xarray.DataArray:
    if factor <= 1:
        return dem_data

    print(f"Downsampling DEM by factor {factor}...")

    nodata_val = dem_data.rio.nodata
    if nodata_val is None and np.any(np.isclose(dem_data.values, -9999)):
        nodata_val = -9999.0
        dem_data = dem_data.rio.write_nodata(nodata_val)
        print(f"Setting nodata to {nodata_val} before downsampling")

    elevation = dem_data.values.squeeze().astype(np.float64)
    original_shape = elevation.shape

    if nodata_val is not None:
        nodata_mask = np.isclose(elevation, nodata_val)
        elevation[nodata_mask] = np.nan
    else:
        nodata_mask = np.zeros_like(elevation, dtype=bool)

    new_height = original_shape[0] // factor
    new_width = original_shape[1] // factor

    elevation_downsampled = zoom(
        elevation,
        (new_height / original_shape[0], new_width / original_shape[1]),
        order=0,  # nearest neighbour preserves nodata
        mode="nearest",
        cval=np.nan,
    )

    if nodata_val is not None:
        elevation_downsampled[np.isnan(elevation_downsampled)] = nodata_val

    original_transform = dem_data.rio.transform()
    new_transform = Affine(
        original_transform.a * factor,
        original_transform.b,
        original_transform.c,
        original_transform.d,
        original_transform.e * factor,
        original_transform.f,
    )

    new_y_coords = np.linspace(dem_data.y.values[0], dem_data.y.values[-1], new_height)
    new_x_coords = np.linspace(dem_data.x.values[0], dem_data.x.values[-1], new_width)

    dem_downsampled = xarray.DataArray(
        elevation_downsampled,
        dims=("y", "x"),
        coords={"y": new_y_coords, "x": new_x_coords},
        attrs=dem_data.attrs,
    ).rio.write_crs(dem_data.rio.crs).rio.write_transform(new_transform)

    if nodata_val is not None:
        dem_downsampled = dem_downsampled.rio.write_nodata(nodata_val)

    return dem_downsampled

if __name__ == "__main__":
    # Example usage for a state
    title = " "
    sun_azimuth = 300
    sun_altitude = 65
    land_only = True  # Set to False to include state marine/Great Lakes territory
    #location_name = state_name.lower()

    # provide state, county, or coordinates
    """
    Class will init the bounding box, location name, etc
    Call each function and save the output to the class
    Save hillshade, dem, etc
    """
    from map_data import MapData

    location = "Utah"
    map_data = MapData(location_name=location)


    font_list = [
        "Bodoni 72 OS.ttc",
        "Bodoni 72 Smallcaps Book.ttf",
        "Hoefler Text.ttc",
    ]
    font_index = 2
    font = font_list[font_index]
    
    map_data = generate_map(
        create_debug_grid=False,
        map_data=map_data
    )

    # if map_data.location_type == 'state':
    #     # Now get the state mask for the DEM
    #     state_mask = map_data.location_layer.create_state_mask(map_data.location_polygon, map_data.dem_data)
    # else:
    #     state_mask = None

    # save state mask to rgbimage
    # if state_mask is not None:
    #     # Ensure the array is C-contiguous before passing to fromarray
    #     contiguous_mask = np.ascontiguousarray(state_mask.astype('uint8') * 255)
    #     state_mask_img = Image.fromarray(contiguous_mask, 'L')
    #     state_mask_img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"{title}_state_mask.png"))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    printable_base = os.path.join(output_dir, f"{title}_print")
    pdf_path, png_path, jpeg_path = create_printable_map(
        map_data=map_data,
        title=title,
        output_prefix=printable_base,
        width_inches=20.0,
        height_inches=30.0,
        margin_inches=0.65,
        dpi=400, # 2400
        border=True,
        shadow=True,
        input_font=font
    )
    print(f"Generated printable map versions:")
    print(f"PDF: {pdf_path}")
    print(f"High-res image: {png_path}")
    print(f"High-res image: {jpeg_path}")