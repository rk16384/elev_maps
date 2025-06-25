import numpy as np
from typing import Tuple, Optional, List
import xarray
import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform
from functools import partial

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