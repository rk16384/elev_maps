import os
from shapely.geometry import Polygon
import numpy as np
from PIL import Image
import xdem
from map_layers import ElevationLayer, WaterLayer
from utils import reproject_to_web_mercator

def create_hillshade(dem_data, 
                    sun_azimuth=315,
                    sun_altitude=45,
                    vertical_exaggeration=1.0,
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
    # Convert the xarray.DataArray to xdem.DEM
    if vertical_exaggeration != 1.0:
        dem_data = dem_data * vertical_exaggeration

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

    # Convert to 8-bit grayscale
    hillshade_data = hillshade.data.squeeze()
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
        coordinates: List of (lon, lat) tuples defining the polygon in clockwise order
        location_name: Name for the location (used in cache filenames)
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
    
    # Create polygon from coordinates
    polygon = Polygon(coordinates)
    
    # Initialize layers
    elevation_layer = ElevationLayer(cache_dir="data_cache", location_name=location_name)
    water_layer = WaterLayer(cache_dir="data_cache", location_name=location_name) if include_water else None
    
    # Get DEM data
    dem_data, metadata = elevation_layer.get_dem(
        polygon_geom=polygon,
        resolution=resolution,
        use_cache=use_cache
    )
    
    # Reproject DEM to Web Mercator for accurate distance calculations
    dem_data = reproject_to_web_mercator(dem_data)
    
    # Get water mask if requested
    water_mask = None
    if include_water:
        water_mask = water_layer.get_water_mask(
            polygon_geom=polygon,
            dem_data=dem_data,
            use_cache=use_cache
        )
    
    # Generate hillshade
    hillshade_img = create_hillshade(
        dem_data=dem_data,
        sun_azimuth=sun_azimuth,
        sun_altitude=sun_altitude,
        vertical_exaggeration=vertical_exaggeration,
        water_mask=water_mask,
        water_color=water_color,
        gaussian_blur_sigma=gaussian_blur_sigma
    )
    
    # Save outputs
    bbox = polygon.bounds
    output_base = f"{location_name}_{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
    
    dem_path = os.path.join(output_dir, f"{output_base}_dem.tif")
    dem_data.rio.to_raster(dem_path)
    
    hillshade_path = os.path.join(output_dir, f"{output_base}_hillshade.png")
    hillshade_img.save(hillshade_path)
    
    print(f"Generated hillshade map at {hillshade_path}")
    print(f"DEM resolution: {metadata['resolution']}m")
    print(f"Image dimensions: {hillshade_img.size}")
    
    return hillshade_img, dem_data, metadata

if __name__ == "__main__":
    # Example usage
    # Define a polygon around Mount Rainier
    rainier_coords = [
        (-121.8, 46.9),  # Northwest
        (-121.7, 46.9),  # Northeast
        (-121.7, 46.8),  # Southeast
        (-121.8, 46.8),  # Southwest
        (-121.8, 46.9),  # Close the polygon
    ]
    astoria_coords = [
        (46.394496, -124.094430),
        (46.394496, -123.33293),
        (45.725537, -123.33293),
        (45.725537, -124.094430)
    ]
    
    hillshade_img, dem_data, metadata = generate_map(
        coordinates=rainier_coords,
        location_name="astoria",
        vertical_exaggeration=2.0,
        sun_azimuth=315,  # Light from northwest
        sun_altitude=45,
        include_water=True,
        water_color=50,
        gaussian_blur_sigma=0.5
    )