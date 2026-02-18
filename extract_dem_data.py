"""
Extract DEM data for a polygon defined in GeoJSON format.

Downloads the highest available resolution DEM from USGS 3DEP service
and saves it as a GeoTIFF file.

Can be used as a standalone script or imported as a module.
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
import rioxarray
import xarray
from rasterio.merge import merge as rio_merge
from shapely.geometry import box, shape

from utils import bbox_to_land_area, land_area_to_resolution


def fetch_dem_from_3dep(bbox: tuple, resolution: int, timeout: int = 120, max_retries: int = 3):
    """
    Fetch DEM data from USGS 3DEP service with retry logic.
    
    Args:
        bbox: (west, south, east, north) bounding box in EPSG:4326
        resolution: Resolution in meters
        timeout: Request timeout in seconds (default 120s for larger requests)
        max_retries: Maximum number of retry attempts for timeout/network errors
        
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
    ignore_server_errors = True
    if not ignore_server_errors:
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
    
    # Retry loop for timeout and network errors
    last_exception = None
    for attempt in range(max_retries):
        try:
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
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                print(f"  Timeout/connection error, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"  All {max_retries} attempts failed.")
                raise
        except requests.exceptions.HTTPError as e:
            # Retry on 502 Bad Gateway and 504 Gateway Timeout (transient server issues)
            if e.response is not None and e.response.status_code in (502, 504):
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10  # Longer backoff for server errors: 10s, 20s, 40s
                    print(f"  Server timeout ({e.response.status_code}), retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"  All {max_retries} attempts failed with server timeout.")
                    raise
            else:
                # Don't retry other HTTP errors (4xx, 500, etc.)
                raise
        except ValueError:
            # Don't retry validation errors - the data is bad, not the connection
            raise
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


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


def extract_dem_from_geojson(geojson_data: dict, output_path: str, polygon_coords: list = None):
    """
    Extract DEM data from a GeoJSON FeatureCollection.
    
    Args:
        geojson_data: GeoJSON dict with FeatureCollection containing a Polygon
        output_path: Path to save the output GeoTIFF
        polygon_coords: Optional list of (lon, lat) tuples defining the extraction bounds.
                        If provided, uses these coordinates instead of geojson_data for bounds.
        
    Returns:
        Tuple of (output_path, polygon) or (None, None) if failed
    """
    from shapely.geometry import Polygon
    
    # Use provided polygon_coords if available, otherwise extract from geojson
    if polygon_coords is not None:
        polygon = Polygon(polygon_coords)
    else:
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
# Mosaic DEM extraction for small areas (adaptive tiling)
# ============================================================================

def extract_dem_mosaic(polygon, output_path: str, resolutions: list = None, 
                       min_tile_size: float = 0.01, use_cache: bool = True,
                       max_dimension: int = 4000):
    """
    Extract DEM using adaptive tiling/mosaicking approach.
    
    This function tries to fetch the highest resolution data available by:
    1. Attempting to fetch the whole area at each resolution (1m, 3m, 10m)
    2. If a fetch fails, subdividing into smaller tiles
    3. Merging all tiles together at the end
    4. Downsampling if the result exceeds max_dimension
    
    Args:
        polygon: Shapely Polygon geometry
        output_path: Path to save the output GeoTIFF
        resolutions: List of resolutions to try in order (default: [1, 3, 10])
        min_tile_size: Minimum tile size in degrees before giving up (~1km)
        use_cache: Whether to use/save cached tiles
        max_dimension: Maximum dimension (width or height) for final output.
                       If the merged DEM exceeds this, it will be downsampled.
                       Set to None to disable downsampling. Default: 4000 pixels.
        
    Returns:
        Path to the saved file, or None if failed
    """
    from shapely.geometry import Polygon
    
    bbox = polygon.bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bbox
    
    # Default resolutions prioritizing highest first
    if resolutions is None:
        resolutions = [1, 3, 10]
    
    print(f"Bounding box: {bbox}")
    land_area = bbox_to_land_area(polygon)
    print(f"Estimated area: {land_area / 1e6:.2f} km²")
    print(f"Resolutions to try (in order): {resolutions}")
    print(f"Min tile size: {min_tile_size}° (~{min_tile_size * 111:.1f} km)")
    
    # Cache directory for tiles
    cache_dir = Path(output_path).parent / "tile_cache"
    if use_cache:
        cache_dir.mkdir(exist_ok=True)
    
    processed_tiles = []
    
    def _get_tile_cache_path(tile_bounds, resolution):
        """Generate cache path for a tile."""
        bbox_str = f"{tile_bounds[0]:.6f}_{tile_bounds[1]:.6f}_{tile_bounds[2]:.6f}_{tile_bounds[3]:.6f}"
        return cache_dir / f"tile_{bbox_str}_{resolution}m.tif"
    
    def _process_tile(tile_geom, tile_id, depth=0):
        """
        Recursively process a tile, subdividing if necessary.
        
        Returns list of successfully fetched DEM DataArrays.
        """
        tile_bounds = tile_geom.bounds
        tile_width = tile_bounds[2] - tile_bounds[0]
        tile_height = tile_bounds[3] - tile_bounds[1]
        is_min_size = tile_width < min_tile_size or tile_height < min_tile_size
        
        indent = "  " * depth
        print(f"{indent}Tile {tile_id}: {tile_width:.4f}° x {tile_height:.4f}°")
        
        for resolution in resolutions:
            try:
                # Check cache first
                cache_path = _get_tile_cache_path(tile_bounds, resolution)
                if use_cache and cache_path.exists():
                    print(f"{indent}  Loading cached tile at {resolution}m...")
                    with rioxarray.open_rasterio(cache_path) as rds:
                        dem_data = rds.squeeze(drop=True).copy()
                    print(f"{indent}  Loaded from cache: {dem_data.shape}")
                    return [dem_data]
                
                print(f"{indent}  Trying {resolution}m resolution...")
                dem_data = fetch_dem_from_3dep(tile_bounds, resolution)
                
                # Validate the data
                data_values = dem_data.values
                nodata_val = dem_data.rio.nodata or -9999
                
                # Check for nodata/invalid pixels
                valid_mask = ~np.isnan(data_values) & ~np.isclose(data_values, nodata_val)
                valid_ratio = np.count_nonzero(valid_mask) / data_values.size
                
                if valid_ratio < 0.90:  # Less than 90% valid data
                    raise ValueError(f"Tile has only {valid_ratio*100:.1f}% valid data")
                
                # Check for flat/empty tiles
                valid_data = data_values[valid_mask]
                if len(valid_data) > 0 and (np.max(valid_data) - np.min(valid_data)) < 1:
                    raise ValueError("Tile appears flat/empty")
                
                # Success! Save to cache and return
                if use_cache:
                    dem_data.rio.to_raster(str(cache_path))
                
                print(f"{indent}  Success at {resolution}m! Shape: {dem_data.shape}")
                return [dem_data]
                
            except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
                error_code = getattr(getattr(e, 'response', None), 'status_code', None)
                
                # Subdivide on server errors (500, 502, 504) or timeouts if not at min size
                should_subdivide = (
                    not is_min_size and 
                    (error_code in (500, 502, 504) or 
                     isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)))
                )
                
                if should_subdivide:
                    print(f"{indent}  Failed at {resolution}m (error: {error_code or type(e).__name__}), subdividing...")
                    return _subdivide_tile(tile_geom, tile_id, depth)
                else:
                    print(f"{indent}  Failed at {resolution}m: {type(e).__name__}")
                    continue
                    
            except ValueError as e:
                # Validation errors - try next resolution or subdivide
                if not is_min_size and "valid data" in str(e):
                    print(f"{indent}  {e}, subdividing...")
                    return _subdivide_tile(tile_geom, tile_id, depth)
                else:
                    print(f"{indent}  {e}")
                    continue
                    
            except Exception as e:
                print(f"{indent}  Unexpected error: {e}")
                continue
        
        # All resolutions failed
        if is_min_size:
            print(f"{indent}  Tile at minimum size, skipping.")
        else:
            print(f"{indent}  All resolutions failed, subdividing as last resort...")
            return _subdivide_tile(tile_geom, tile_id, depth)
        
        return []
    
    def _subdivide_tile(tile_geom, tile_id, depth):
        """Subdivide a tile into 4 quadrants and process each."""
        t_minx, t_miny, t_maxx, t_maxy = tile_geom.bounds
        midx = (t_minx + t_maxx) / 2
        midy = (t_miny + t_maxy) / 2
        
        child_dems = []
        # NW, NE, SW, SE quadrants
        child_dems.extend(_process_tile(box(t_minx, midy, midx, t_maxy), f"{tile_id}.1", depth + 1))
        child_dems.extend(_process_tile(box(midx, midy, t_maxx, t_maxy), f"{tile_id}.2", depth + 1))
        child_dems.extend(_process_tile(box(t_minx, t_miny, midx, midy), f"{tile_id}.3", depth + 1))
        child_dems.extend(_process_tile(box(midx, t_miny, t_maxx, midy), f"{tile_id}.4", depth + 1))
        return child_dems
    
    # Start processing from the whole area
    print("\n" + "-" * 40)
    print("Starting mosaic extraction...")
    print("-" * 40)
    
    initial_tile = box(minx, miny, maxx, maxy)
    processed_tiles = _process_tile(initial_tile, "1", depth=0)
    
    if not processed_tiles:
        print("\nFailed to fetch any DEM tiles.")
        return None
    
    print(f"\nSuccessfully fetched {len(processed_tiles)} tile(s)")
    
    # Merge tiles if we have multiple
    if len(processed_tiles) == 1:
        final_dem = processed_tiles[0]
        print("Single tile, no merging needed.")
    else:
        print(f"Merging {len(processed_tiles)} tiles...")
        final_dem = _merge_tiles(processed_tiles)
    
    # Clip to original bounding box
    final_dem = final_dem.rio.clip_box(minx, miny, maxx, maxy)
    
    # Downsample if exceeds max_dimension
    if max_dimension is not None:
        current_max = max(final_dem.shape)
        if current_max > max_dimension:
            scale_factor = max_dimension / current_max
            new_height = int(final_dem.shape[0] * scale_factor)
            new_width = int(final_dem.shape[1] * scale_factor)
            
            print(f"\nDownsampling from {final_dem.shape[1]}x{final_dem.shape[0]} to {new_width}x{new_height}...")
            print(f"  (max_dimension={max_dimension}, scale={scale_factor:.3f})")
            
            final_dem = _downsample_dem(final_dem, new_width, new_height)
    
    # Save to file
    print(f"\nSaving DEM to: {output_path}")
    final_dem.rio.to_raster(output_path)
    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved successfully. File size: {file_size:.1f} KB")
    print(f"  Shape: {final_dem.shape}")
    print(f"  Elevation range: {float(np.nanmin(final_dem.values)):.1f} to {float(np.nanmax(final_dem.values)):.1f} meters")
    
    return output_path


def _merge_tiles(tiles: list):
    """
    Merge multiple DEM tiles into a single DataArray.
    
    Handles tiles of different resolutions by resampling to the highest.
    """
    if len(tiles) == 1:
        return tiles[0]
    
    # Check if we have multiple resolutions
    resolutions = {abs(dem.rio.resolution()[0]) for dem in tiles}
    if len(resolutions) > 1:
        target_res = min(resolutions)
        print(f"  Multiple resolutions found, resampling to highest ({target_res:.8f}°)")
    else:
        target_res = None
    
    # Use temporary files for rasterio merge
    temp_files = []
    try:
        for i, dem in enumerate(tiles):
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_f:
                dem.rio.to_raster(temp_f.name)
                temp_files.append(temp_f.name)
        
        # Open and merge
        srcs = [rasterio.open(f) for f in temp_files]
        try:
            if target_res:
                mosaic_arr, out_trans = rio_merge(srcs, method='first', res=target_res)
            else:
                mosaic_arr, out_trans = rio_merge(srcs, method='first')
        finally:
            for src in srcs:
                src.close()
        
        # Convert back to xarray DataArray
        first_tile = tiles[0]
        merged_da = xarray.DataArray(
            mosaic_arr[0],
            dims=("y", "x"),
            coords={
                "y": (("y",), out_trans.f + np.arange(mosaic_arr.shape[1]) * out_trans.e),
                "x": (("x",), out_trans.c + np.arange(mosaic_arr.shape[2]) * out_trans.a),
            },
            attrs=first_tile.attrs
        ).rio.write_crs(first_tile.rio.crs).rio.write_transform(out_trans)
        
        return merged_da
        
    finally:
        # Clean up temporary files
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass


def _downsample_dem(dem_data, new_width: int, new_height: int):
    """
    Downsample a DEM DataArray to the specified dimensions.
    
    Uses bilinear interpolation to preserve elevation accuracy.
    """
    from scipy.ndimage import zoom
    from affine import Affine
    
    original_shape = dem_data.shape
    data = dem_data.values
    
    # Handle nodata values
    nodata_val = dem_data.rio.nodata or -9999
    nodata_mask = np.isclose(data, nodata_val) | np.isnan(data)
    
    # Replace nodata with nan for interpolation
    data_float = data.astype(np.float64)
    data_float[nodata_mask] = np.nan
    
    # Calculate zoom factors
    zoom_y = new_height / original_shape[0]
    zoom_x = new_width / original_shape[1]
    
    # Downsample using bilinear interpolation (order=1)
    # Use mode='nearest' for edges
    downsampled = zoom(data_float, (zoom_y, zoom_x), order=1, mode='nearest')
    
    # Restore nodata values
    downsampled = np.nan_to_num(downsampled, nan=nodata_val)
    
    # Update transform
    original_transform = dem_data.rio.transform()
    new_transform = Affine(
        original_transform.a / zoom_x,  # pixel width
        original_transform.b,
        original_transform.c,            # origin x
        original_transform.d,
        original_transform.e / zoom_y,  # pixel height (negative)
        original_transform.f             # origin y
    )
    
    # Create new coordinates
    new_y_coords = np.linspace(dem_data.y.values[0], dem_data.y.values[-1], new_height)
    new_x_coords = np.linspace(dem_data.x.values[0], dem_data.x.values[-1], new_width)
    
    # Create new DataArray
    downsampled_da = xarray.DataArray(
        downsampled.astype(np.float32),
        dims=("y", "x"),
        coords={"y": new_y_coords, "x": new_x_coords},
        attrs=dem_data.attrs
    ).rio.write_crs(dem_data.rio.crs).rio.write_transform(new_transform)
    
    if nodata_val is not None:
        downsampled_da = downsampled_da.rio.write_nodata(nodata_val)
    
    return downsampled_da


def extract_dem_mosaic_from_geojson(geojson_data: dict, output_path: str, 
                                    polygon_coords: list = None,
                                    resolutions: list = None,
                                    max_dimension: int = 4000):
    """
    Extract DEM using mosaic approach from a GeoJSON FeatureCollection.
    
    This is the recommended function for small mountain areas as it will
    attempt to fetch the highest resolution data available using adaptive tiling.
    
    Args:
        geojson_data: GeoJSON dict with FeatureCollection containing a Polygon
        output_path: Path to save the output GeoTIFF
        polygon_coords: Optional list of (lon, lat) tuples defining the extraction bounds.
        resolutions: List of resolutions to try (default: [1, 3, 10])
        max_dimension: Maximum dimension (width or height) for final output.
                       If the merged DEM exceeds this, it will be downsampled.
                       Set to None to disable downsampling. Default: 4000 pixels.
        
    Returns:
        Tuple of (output_path, polygon) or (None, None) if failed
    """
    from shapely.geometry import Polygon
    
    # Use provided polygon_coords if available, otherwise extract from geojson
    if polygon_coords is not None:
        polygon = Polygon(polygon_coords)
    else:
        first_feature = geojson_data["features"][0]
        polygon = shape(first_feature["geometry"])
    
    print("=" * 60)
    print("DEM Mosaic Extraction")
    print("=" * 60)
    print(f"Output path: {output_path}")
    print(f"Polygon vertices: {len(polygon.exterior.coords)}")
    print(f"Max dimension: {max_dimension} pixels")
    print()
    
    result = extract_dem_mosaic(polygon, output_path, resolutions=resolutions, 
                                max_dimension=max_dimension)
    
    if result:
        print("\n" + "=" * 60)
        print("Mosaic extraction complete!")
        print(f"DEM saved to: {Path(result).absolute()}")
        print("=" * 60)
        return result, polygon
    else:
        print("\nMosaic extraction failed.")
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
