"""
Extract DEM data for a polygon defined in GeoJSON format.

Downloads the highest available resolution DEM from USGS 3DEP service
and saves it as a GeoTIFF file.

Can be used as a standalone script or imported as a module.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import requests
import rioxarray
from shapely.geometry import shape

from utils import bbox_to_land_area, land_area_to_resolution


def fetch_dem_from_3dep(bbox: tuple, resolution: int, timeout: int = 60):
    """
    Fetch DEM data from USGS 3DEP service.
    
    Args:
        bbox: (west, south, east, north) bounding box in EPSG:4326
        resolution: Resolution in meters
        timeout: Request timeout in seconds
        
    Returns:
        xarray.DataArray of DEM data
        
    Raises:
        ValueError: If the request fails or returns invalid data
    """
    base_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    
    # Calculate image size from resolution
    deg_to_m = 111320  # Approximation for degrees to meters at mid-latitudes
    width = int((bbox[2] - bbox[0]) * deg_to_m / resolution)
    height = int((bbox[3] - bbox[1]) * deg_to_m / resolution)
    
    # Limit to ~16 million pixels to avoid server errors
    if width * height > 16000000:
        raise ValueError(f"Requested image size ({width}x{height}) is too large for {resolution}m resolution.")
    
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": "4326",
        "size": f"{width},{height}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "noData": -9999,
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    
    print(f"  Requesting {width}x{height} pixels...")
    response = requests.get(base_url, params=params, timeout=timeout)
    response.raise_for_status()
    
    # Check if the response is actually an image
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise ValueError(f"Server did not return an image. Content-Type: {content_type}")
    
    # Read the image data into rioxarray
    with rioxarray.open_rasterio(io.BytesIO(response.content)) as rds:
        dem_data = rds.squeeze(drop=True).copy()
    
    # Validate the data
    if dem_data.ndim < 2 or min(dem_data.shape) < 2:
        raise ValueError(f"Invalid DEM data shape: {dem_data.shape}")
    
    return dem_data


def extract_dem_for_polygon(polygon, output_path: str):
    """
    Download DEM data for a polygon and save to file.
    
    Args:
        polygon: Shapely Polygon geometry
        output_path: Path to save the output GeoTIFF
        
    Returns:
        Path to the saved file, or None if failed
    """
    bbox = polygon.bounds  # (minx, miny, maxx, maxy)
    print(f"Bounding box: {bbox}")
    
    # Calculate land area and determine resolutions to try
    land_area = bbox_to_land_area(polygon)
    print(f"Estimated area: {land_area / 1e6:.2f} km²")
    
    resolutions = land_area_to_resolution(land_area)
    print(f"Resolutions to try (in order): {list(resolutions)}")
    
    dem_data = None
    used_resolution = None
    
    for resolution in resolutions:
        try:
            print(f"\nAttempting to fetch DEM at {resolution}m resolution...")
            dem_data = fetch_dem_from_3dep(bbox, resolution)
            used_resolution = resolution
            print(f"Success! Downloaded DEM at {resolution}m resolution")
            print(f"  Shape: {dem_data.shape}")
            print(f"  Elevation range: {float(np.nanmin(dem_data.values)):.1f} to {float(np.nanmax(dem_data.values)):.1f} meters")
            break
            
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP error: {e}")
            continue
        except ValueError as e:
            print(f"  Validation error: {e}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Network error: {e}")
            continue
        except Exception as e:
            print(f"  Unexpected error: {e}")
            continue
    
    if dem_data is None:
        print("\nFailed to fetch DEM at any resolution.")
        return None
    
    # Save to file
    print(f"\nSaving DEM to: {output_path}")
    dem_data.rio.to_raster(output_path)
    print(f"Saved successfully. File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path


def extract_dem_from_geojson(geojson_data: dict, output_path: str):
    """
    Extract DEM data from a GeoJSON FeatureCollection.
    
    Args:
        geojson_data: GeoJSON dict with FeatureCollection containing a Polygon
        output_path: Path to save the output GeoTIFF
        
    Returns:
        Tuple of (output_path, polygon) or (None, None) if failed
    """
    # Extract the first polygon from the GeoJSON
    first_feature = geojson_data["features"][0]
    polygon = shape(first_feature["geometry"])
    
    print("=" * 60)
    print("DEM Extraction")
    print("=" * 60)
    print(f"Output path: {output_path}")
    print(f"Polygon vertices: {len(polygon.exterior.coords)}")
    print()
    
    result = extract_dem_for_polygon(polygon, output_path)
    
    if result:
        print("\n" + "=" * 60)
        print("Extraction complete!")
        print(f"DEM saved to: {Path(result).absolute()}")
        print("=" * 60)
        return result, polygon
    else:
        print("\nExtraction failed.")
        return None, None


# ============================================================================
# Standalone script functionality
# ============================================================================

# https://geojson.io/
# Hardcoded GeoJSON polygon for standalone use
STANDALONE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "coordinates": [
                    [
                        [-111.58072504396681, 40.60547782150721],
                        [-111.59448985040217, 40.59977301879928],
                        [-111.60087443557845, 40.59714453532348],
                        [-111.59190143550231, 40.58468146058149],
                        [-111.57402039001227, 40.578819784950355],
                        [-111.5614048491994, 40.583129435795826],
                        [-111.55732946278928, 40.58923175584971],
                        [-111.5574483052459, 40.59509878407255],
                        [-111.58072504396681, 40.60547782150721],
                    ]
                ],
                "type": "Polygon",
            },
        }
    ],
}

STANDALONE_OUTPUT_DIR = "mountain_mesh_data/standalone"
STANDALONE_OUTPUT_FILENAME = "extracted_dem.tif"


def main():
    """Standalone script entry point."""
    # Create output directory
    output_dir = Path(STANDALONE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / STANDALONE_OUTPUT_FILENAME
    
    result, polygon = extract_dem_from_geojson(STANDALONE_GEOJSON, str(output_path))
    
    if result:
        # Save polygon.json for reference
        polygon_path = output_dir / "polygon.json"
        with open(polygon_path, "w") as f:
            json.dump(STANDALONE_GEOJSON, f, indent=2)
        print(f"Saved polygon.json to: {polygon_path}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
