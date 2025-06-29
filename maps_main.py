import os
from shapely.geometry import Polygon
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import xdem
from map_layers import ElevationLayer, WaterLayer
from utils import reproject_to_web_mercator, create_extrusion_shadow
import math
import geopandas as gpd
from typing import Optional

def create_hillshade(dem_data, 
                    sun_azimuth=315,
                    sun_altitude=45,
                    water_mask=None,
                    water_color=70,
                    gaussian_blur_sigma=0):
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
    # Apply vertical exaggeration
    # if vertical_exaggeration != 1.0:
    #     dem_data = dem_data * vertical_exaggeration

    dem = xdem.DEM.from_array(
        dem_data.values.squeeze(),  # Get the numpy array and remove single dimensions
        transform=dem_data.rio.transform(),
        crs=dem_data.rio.crs
    )

    # Generate hillshade
    hillshade = xdem.terrain.hillshade(
        dem,
        azimuth=sun_azimuth,
        altitude=sun_altitude
    )

    # Convert to 8-bit grayscale more robustly
    hillshade_data = hillshade.data.squeeze()
    # replace nan with min
    hillshade_data = np.nan_to_num(hillshade_data, nan=hillshade_data.min())
    hillshade_uint8 = ((hillshade_data - hillshade_data.min()) * 255 / 
                       (hillshade_data.max() - hillshade_data.min())).astype(np.uint8)

    # Apply water mask if provided
    if water_mask is not None:
        hillshade_uint8[water_mask] = water_color

    # Apply Gaussian blur if requested
    if gaussian_blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        hillshade_uint8 = gaussian_filter(hillshade_uint8, sigma=gaussian_blur_sigma)

    # Convert to PIL Image
    return Image.fromarray(hillshade_uint8)

def create_printable_map(hillshade_img: Image.Image,
                        title: str,
                        output_prefix: str,
                        shadow_img: Image.Image,
                        shadow_offsets: tuple[int, int],
                        width_inches: float = 5.0,
                        height_inches: float = 7.0,
                        margin_inches: float = 0.25,
                        dpi: int = 300) -> tuple[str, str]:
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
    
    # Calculate positioning for the hillshade image
    x_offset = (width_px - new_width) // 2
    y_offset = title_height_px + ((height_px - title_height_px - new_height) // 2)
    
    # --- New Shadow Logic ---
    # Resize the shadow image and place it on the canvas
    dem_width_orig, _ = hillshade_img.size
    resize_factor = new_width / dem_width_orig

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
    background.paste(hillshade_resized, (x_offset, y_offset))
    
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
            title_y = (y_offset - title_height) // 2
        else:
            title_y = (title_height_px - title_height) // 2 + margin_px
        draw.text((title_x, title_y), title, fill='black', font=font)
    
    # Save outputs
    pdf_path = f"{output_prefix}.pdf"
    img_path = f"{output_prefix}.png"
    
    # Save high-res PNG
    background.save(img_path, "PNG", dpi=(dpi, dpi))
    
    # Save PDF
    background.save(pdf_path, "PDF", resolution=dpi)
    
    return pdf_path, img_path


def generate_map(coordinates, 
                location_name="unnamed",
                resolution=None,
                sun_azimuth=315,
                sun_altitude=45,
                vertical_exaggeration=1.0,
                use_cache=True,
                include_water=True,
                water_color=70,
                gaussian_blur_sigma=0,
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")):
    """
    Generate a hillshade map for the given polygon coordinates.
    
    Args:
        coordinates: List of (lat, lon) tuples defining the polygon in clockwise order
        location_name: Name for the location (used in cache filenames)
        title: Optional title for printable output
        resolution: Optional specific resolution in meters
        sun_azimuth: Sun direction in degrees (0=North, 90=East)
        sun_altitude: Sun elevation in degrees (0=horizon, 90=overhead)
        vertical_exaggeration: Factor to exaggerate elevation differences
        use_cache: Whether to use cached data
        include_water: Whether to include water features
        water_color: Grayscale color (0-255) for water features
        gaussian_blur_sigma: Sigma for Gaussian blur (0 for no blur)
        output_dir: Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create polygon from coordinates, swapping lat/lon to lon/lat
    polygon = Polygon([(lon, lat) for lat, lon in coordinates])
    
    # Initialize layers
    elevation_layer = ElevationLayer(cache_dir="data_cache", location_name=location_name)
    water_layer = WaterLayer(cache_dir="data_cache", location_name=location_name) if include_water else None
    
    # Get DEM data
    dem_data, metadata = elevation_layer.get_dem(
        polygon_geom=polygon,
        resolution=resolution,
        use_cache=use_cache,
        vertical_exaggeration=vertical_exaggeration
    )
    
    # Reproject DEM to Web Mercator for accurate distance calculations
    dem_data = reproject_to_web_mercator(dem_data)

    shadow_img, shadow_offsets = create_extrusion_shadow(dem_data,
                                        light_elevation_deg=65,
                                        light_azimuth=sun_azimuth,
                                        scale=0.2)
    
    #

    # Get water mask if requested
    water_mask = None
    if include_water and water_layer is not None:
        water_mask = water_layer.get_water_mask(
            polygon_geom=polygon,
            dem_data=dem_data,
            use_cache=use_cache
        )
    
    # Create hillshade
    hillshade = create_hillshade(
        dem_data=dem_data,
        sun_azimuth=sun_azimuth,
        sun_altitude=sun_altitude,
        water_mask=water_mask,
        water_color=water_color,
        gaussian_blur_sigma=gaussian_blur_sigma
    )
    
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

if __name__ == "__main__":
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
        (46.394496, -124.094430)
    ]
    
    title = "Astoria"
    sun_azimuth = 315
    sun_altitude = 45

    hillshade_img, dem_data, metadata, shadow_img, shadow_offsets = generate_map(
        coordinates=astoria2_coords,
        location_name="astoria",
        vertical_exaggeration=3.0,
        sun_azimuth=sun_azimuth,  # Light from northwest
        sun_altitude=sun_altitude,
        include_water=True,
        water_color=50,
        use_cache=True
    )

        # Create printable versions if title is provided
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    printable_base = os.path.join(output_dir, f"{title}_print")
    pdf_path, print_img_path = create_printable_map(
        hillshade_img=hillshade_img,
        shadow_img=shadow_img,
        shadow_offsets=shadow_offsets,
        title=title,
        output_prefix=printable_base,
        width_inches=5.0,
        height_inches=7.0,
        margin_inches=0.5,
        dpi=300
    )
    print(f"Generated printable map versions:")
    print(f"PDF: {pdf_path}")
    print(f"High-res image: {print_img_path}")