from abc import ABC, abstractmethod
import os
import rioxarray
import xdem
import py3dep
import numpy as np
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, gaussian_filter, binary_opening, label
from affine import Affine
from PIL import Image
import re
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
import pynhd
import pygeohydro as gh
import requests
import io
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from utils import bbox_to_land_area, land_area_to_resolution, reproject_to_web_mercator, format_bbox_string
from shapely.ops import transform
import pyproj
from functools import partial
from urllib.parse import quote
import xarray
import shapely
import tempfile
import rasterio
from rasterio.merge import merge as rio_merge
from concurrent.futures import ThreadPoolExecutor, as_completed

class FetchError(Exception):
    """Custom exception for DEM fetching failures."""
    pass

class BaseLayer(ABC):

    def __init__(self, cache_dir="data_cache"):
        # Common: all layers will have a name and a cache directory.
        self.cache_dir = os.path.join(cache_dir, self.name)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.data = None

    @property
    @abstractmethod
    def name(self):
        """
        RULE: You MUST provide a name.
        (Implemented by the subclass)
        """
        pass

    @abstractmethod
    def _get_cache_filepath(self, bbox):
        """
        RULE: You MUST know how to name your own cache files.
        Example: A vector layer might return a '.geojson' path,
                 a raster layer might return a '.tif' path.
        (Implemented by the subclass)
        """
        pass

    @abstractmethod
    def _fetch_from_source(self, polygon_geom):
        """
        RULE: You MUST know how to download your own data.
        I don't know your API, your parameters, or your data format,
        but you must have this method.
        (Implemented by the subclass)
        """
        pass

    @abstractmethod
    def _save_to_cache(self, data, filepath):
        """
        RULE: You MUST know how to save your own data type.
        (Implemented by the subclass)
        """
        pass

    @abstractmethod
    def _load_from_cache(self, filepath):
        """
        RULE: You MUST know how to load your own data type.
        (Implemented by the subclass)
        """
        pass

    # THIS IS THE ONLY METHOD WITH SHARED, REUSABLE LOGIC
    def get_data(self, polygon_geom):
        """
        This orchestration logic is the same for ALL layers.
        It calls the abstract methods which are implemented by the specific subclass.
        """
        filepath = self._get_cache_filepath(bbox)
        if os.path.exists(filepath):
            print(f"Loading '{self.name}' data from cache...")
            self.data = self._load_from_cache(filepath)
        else:
            print(f"Fetching '{self.name}' data from source...")
            self.data = self._fetch_from_source(bbox)
            if self.data is not None:
                print(f"Saving '{self.name}' data to cache...")
                self._save_to_cache(self.data, filepath)
        return self.data
    

class ElevationLayer(BaseLayer):
    def __init__(self, cache_dir: str, location_name: str = "unnamed"):
        super().__init__(cache_dir)
        self.location_name = location_name
        self.land_shapefile_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets", "reference_data", "ne_50m_land.shp"
        )
        self.land_gdf = self._load_land_gdf()

    @property
    def name(self):
        return "elevation_dem"

    def _get_cache_filepath(self, polygon_geom: Polygon, resolution: Optional[any] = None, is_tile: bool = False) -> str:
        """
        Generate cache filepath based on polygon bounds and resolution.
        
        Args:
            polygon_geom: Shapely Polygon defining the area of interest
            resolution: Optional specific resolution in meters, or 'mosaic'
            is_tile: If True, save to the 'tiles' subdirectory
        """
        bbox = polygon_geom.bounds  # (minx, miny, maxx, maxy)
        bbox_str = format_bbox_string(bbox)
        
        res_str = f"{resolution}m" if isinstance(resolution, (int, float, np.number)) else str(resolution)
        
        filename = f"{self.name}_{self.location_name}_{bbox_str}_{res_str}.tif"

        # Determine the directory
        base_dir = self.cache_dir
        if is_tile:
            # Place individual tiles in a subdirectory to keep the main cache clean
            tile_dir = os.path.join(base_dir, "tiles")
            os.makedirs(tile_dir, exist_ok=True)
            cache_path = os.path.join(tile_dir, filename)
        else:
            cache_path = os.path.join(base_dir, filename)

        metadata = {
            "resolution": resolution,
            "crs": None,
            "bbox": bbox_str,
            "shape": None,
            "nodata_value": None
        }

        return cache_path, metadata

    def _fetch_from_source(self, polygon_geom: Polygon, 
                          target_resolution: Optional[int] = None) -> Tuple[xarray.DataArray, Dict[str, Any]]:
        """
        Fetch DEM data from py3dep service.
        
        Args:
            polygon_geom: Shapely Polygon defining the area of interest
            target_resolution: Optional specific resolution to fetch
            
        Returns:
            Tuple of (dem_data, metadata)
        """
        bbox = polygon_geom.bounds
        
        # Calculate appropriate resolutions if not specified
        if target_resolution is None:
            land_area = polygon_geom.area  # Already in square meters if using projected CRS
            resolutions = land_area_to_resolution(land_area)
        else:
            resolutions = [target_resolution]

        dem_data = None
        used_resolution = None
        
        for resolution in resolutions:
            try:
                print(f"Attempting to fetch DEM at {resolution}m resolution...")
                
                # Manually construct the request to bypass py3dep and control the timeout
                base_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
                
                # Calculate image size from resolution and add a safety limit
                deg_to_m = 111320 # Approximation for degrees to meters at mid-latitudes
                width = int((bbox[2] - bbox[0]) * deg_to_m / resolution)
                height = int((bbox[3] - bbox[1]) * deg_to_m / resolution)

                if width * height > 16000000: # Limit to ~16 million pixels to avoid server errors
                    raise ValueError(f"Requested image size ({width}x{height}) is too large.")

                params = {
                    'bbox': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}',
                    'bboxSR': '4326',
                    'size': f'{width},{height}',
                    'imageSR': '4326',
                    'format': 'tiff',
                    'pixelType': 'F32',
                    'noData': -9999,
                    'interpolation': 'RSP_BilinearInterpolation',
                    'f': 'image'
                }
                
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()

                # Check if the response is actually an image before trying to parse it
                if 'image' not in response.headers.get('Content-Type', ''):
                    raise ValueError(f"Server did not return an image. Content-Type: {response.headers.get('Content-Type', 'N/A')}")

                # Read the image data into a rioxarray
                with rioxarray.open_rasterio(io.BytesIO(response.content)) as rds:
                    dem_data_attempt = rds.squeeze(drop=True)
                
                # Validate the data
                if dem_data_attempt.ndim < 2 or min(dem_data_attempt.shape) < 2:
                    raise ValueError(f"Invalid DEM data at {resolution}m resolution")
                
                dem_data = dem_data_attempt
                used_resolution = resolution
                # print(f"Successfully fetched DEM at {resolution}m resolution")
                break
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 500:
                    print(f"INFO: Server error (500) at {resolution}m. The server might be overloaded or the request too complex.")
                else:
                    print(f"INFO: HTTP error {e.response.status_code} at {resolution}m.")
                if target_resolution is not None:
                    raise FetchError from e
                continue
            except ValueError as e:  # Catches our custom validation errors
                print(f"INFO: Validation error at {resolution}m: {e}")
                if target_resolution is not None:
                    raise FetchError from e
                continue
            except requests.exceptions.RequestException as e:  # Catches timeouts, connection errors etc.
                print(f"INFO: Network error at {resolution}m: {e}")
                if target_resolution is not None:
                    raise FetchError from e
                continue
            except Exception as e:
                print(f"INFO: An unexpected error occurred at {resolution}m: {e}")
                if target_resolution is not None:
                    raise FetchError from e
                continue

        if dem_data is None:
            raise ValueError("Could not fetch DEM at any resolution")

        metadata = {
            "resolution": used_resolution,
            "crs": dem_data.rio.crs,
            "bbox": bbox,
            "shape": dem_data.shape,
            "nodata_value": dem_data.rio.nodata
        }

        return dem_data, metadata

    def _save_to_cache(self, data: xarray.DataArray, filepath: str) -> None:
        """Save DEM data to cache."""
        data.rio.to_raster(filepath)

    def _load_from_cache(self, filepath: str) -> Tuple[xarray.DataArray, Dict[str, Any]]:
        """Load DEM data from cache."""
        dem_data = rioxarray.open_rasterio(filepath)
        
        # Squeeze out the band dimension if it exists
        dem_data = dem_data.squeeze()
        
        # Extract resolution from filename
        filename = os.path.basename(filepath)
        resolution = filename.split('_')[-1].replace('m.tif', '')
        
        metadata = {
            "resolution": resolution,
            "crs": dem_data.rio.crs,
            "shape": dem_data.shape,
            "nodata_value": dem_data.rio.nodata
        }
        print(f"Loaded DEM from cache: {filepath}")
        return dem_data, metadata

    def get_dem(self, polygon_geom: Polygon, 
                resolution: Optional[int] = None,
                use_cache: bool = True,
                vertical_exaggeration: float = 1.0) -> Tuple[xarray.DataArray, Dict[str, Any]]:
        """
        Main method to get DEM data.
        
        Args:
            polygon_geom: Shapely Polygon defining the area of interest
            resolution: Optional specific resolution in meters
            use_cache: Whether to use cached data if available
            vertical_exaggeration: Vertical exaggeration factor
        Returns:
            Tuple of (dem_data, metadata)
        """
        cache_path, metadata = self._get_cache_filepath(polygon_geom, resolution)
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached DEM from {cache_path}")
            return self._load_from_cache(cache_path)
        
        dem_data, metadata = self._fetch_from_source(polygon_geom, resolution)
        self._save_to_cache(dem_data, cache_path)

        dem_data = dem_data * vertical_exaggeration
        metadata["vertical_exaggeration"] = vertical_exaggeration
        
        return dem_data, metadata

    def _load_land_gdf(self):
        try:
            gdf = gpd.read_file(self.land_shapefile_path)
            # Create a unified geometry for faster intersection checks
            unified_land = gdf.geometry.unary_union
            # Return a GeoDataFrame with the single unified geometry
            return gpd.GeoDataFrame(geometry=[unified_land], crs=gdf.crs)
        except Exception as e:
            print(f"Warning: Could not load reference land geometry: {e}")
            return None

    def get_mosaic_dem(self, polygon_geom, use_cache=True, vertical_exaggeration=1.0, state_polygon: Optional[Polygon] = None):
        land_area = bbox_to_land_area(polygon_geom)
        valid_resolutions = land_area_to_resolution(land_area)
        minx, miny, maxx, maxy = polygon_geom.bounds

        unavailable_tiles_log = os.path.join(self.cache_dir, "unavailable_tiles.txt")
        unavailable_bounds = set()
        if use_cache and os.path.exists(unavailable_tiles_log):
            with open(unavailable_tiles_log, 'r') as f:
                for line in f:
                    try:
                        bounds_str = line.split(': ')[1].strip()
                        unavailable_bounds.add(bounds_str)
                    except (IndexError, ValueError):
                        continue

        mosaic_cache_path, metadata = self._get_cache_filepath(polygon_geom, resolution='mosaic', is_tile=False)
        if use_cache and os.path.exists(mosaic_cache_path):
            print(f"Loading cached mosaic DEM from {mosaic_cache_path}")
            mosaic = rioxarray.open_rasterio(mosaic_cache_path).squeeze()
            mosaic = mosaic * vertical_exaggeration
            return mosaic, metadata, None

        min_tile_size = 0.005  # Approx 500m, the base case for recursion
        processed_tiles = []

        if self.land_gdf is not None:
            land_geom_proj = self.land_gdf.to_crs("EPSG:4326").geometry.iloc[0]
        else:
            land_geom_proj = None

        def _process_tile(tile_geom, tile_id):
            if state_polygon and not tile_geom.intersects(state_polygon):
                print(f"Skipping tile {tile_id} that is outside of the state boundary.")
                return []

            tile_bounds = tile_geom.bounds
            print(f"--- Processing Tile {tile_id} | BBOX: {tile_bounds} ---")
            
            if format_bbox_string(tile_bounds) in unavailable_bounds:
                print(f"Skipping cached unavailable tile {tile_id}")
                return []

            is_min_size = tile_bounds[2] - tile_bounds[0] < min_tile_size or tile_bounds[3] - tile_bounds[1] < min_tile_size
            if is_min_size:
                print(f"DEBUG: Tile {tile_id} is at or below min size, will not be subdivided further.")

            if land_geom_proj and not tile_geom.intersects(land_geom_proj):
                print(f"Skipping ocean tile {tile_id}")
                return []

            for res in valid_resolutions:
                try:
                    tile_cache_path, _ = self._get_cache_filepath(tile_geom, resolution=res, is_tile=True)
                    if use_cache and os.path.exists(tile_cache_path):
                        dem_data, _ = self._load_from_cache(tile_cache_path)
                    else:
                        dem_data, _ = self._fetch_from_source(tile_geom, target_resolution=res)
                    
                    min_elev, max_elev = np.nanmin(dem_data.values), np.nanmax(dem_data.values)
                    
                    # --- Revised check for empty/invalid data ---
                    nodata_val = dem_data.rio.nodata
                    data_values = dem_data.values

                    # Start by assuming all data is valid and then mask out invalid pixels
                    valid_pixels = np.full(data_values.shape, True, dtype=bool)
                    
                    # Mask out NaN values
                    nan_mask = np.isnan(data_values)
                    valid_pixels[nan_mask] = False
                    
                    # Mask out the specific nodata value if it is present
                    nodata_count = 0
                    if nodata_val is not None:
                        # Use np.isclose for robust floating-point comparison
                        nodata_mask = np.isclose(data_values, nodata_val)
                        valid_pixels[nodata_mask] = False
                        nodata_count = np.count_nonzero(nodata_mask)

                    valid_data_ratio = np.count_nonzero(valid_pixels) / data_values.size
                    nan_count = np.count_nonzero(nan_mask)
                    zero_count = np.count_nonzero(np.isclose(data_values, 0))
                    
                    print(f"DEBUG Tile {tile_id}: nodata_val={nodata_val}, min={min_elev:.2f}, max={max_elev:.2f}, "
                          f"nan_count={nan_count}, zero_count={zero_count}, nodata_count={nodata_count}, "
                          f"valid_ratio={valid_data_ratio:.2f}")

                    if valid_data_ratio < 0.98: # If less than 98% of the tile has data
                        raise ValueError("Tile contains significant no-data areas.")

                    if min_elev is not np.nan and max_elev is not np.nan and (max_elev - min_elev) < 1:
                        raise ValueError("Tile is flat, contains no significant elevation data.")

                    if not use_cache or not os.path.exists(tile_cache_path):
                         self._save_to_cache(dem_data, tile_cache_path)

                    print(f"Successfully processed tile {tile_id} ({tile_bounds}) at {res}m resolution.")
                    return [dem_data]

                except (FetchError, ValueError) as e:
                    error_str = f"{e} {e.__cause__}"
                    if not is_min_size and ("too large" in error_str or "flat" in error_str or "500" in error_str or "no-data" in error_str):
                        print(f"Tile {tile_id} at {res}m failed. Subdividing. Error: {e.__cause__ or e}")
                        t_minx, t_miny, t_maxx, t_maxy = tile_bounds
                        midx, midy = (t_minx + t_maxx) / 2, (t_miny + t_maxy) / 2
                        
                        child_dems = []
                        child_dems.extend(_process_tile(box(t_minx, midy, midx, t_maxy), f"{tile_id}.1"))
                        child_dems.extend(_process_tile(box(midx, midy, t_maxx, t_maxy), f"{tile_id}.2"))
                        child_dems.extend(_process_tile(box(t_minx, t_miny, midx, midy), f"{tile_id}.3"))
                        child_dems.extend(_process_tile(box(midx, t_miny, t_maxx, midy), f"{tile_id}.4"))
                        return child_dems
                    else:
                        print(f"INFO: Tile {tile_id} at {res}m failed. Trying next resolution. Error: {e.__cause__ or e}")
                        continue

            print(f"WARNING: No DEM available for tile {tile_id}. Logging as unavailable.")
            with open(unavailable_tiles_log, 'a') as f:
                f.write(f"Tile: {format_bbox_string(tile_bounds)}\n")
            return []

        # Start with large tiles covering the whole area
        initial_tile_size = 1.0
        x_coords = np.arange(minx, maxx, initial_tile_size)
        y_coords = np.arange(miny, maxy, initial_tile_size)
        
        initial_tiles_to_process = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                tile_id = f"{i*len(y_coords) + j + 1}"
                tile_geom = box(x, y, min(x + initial_tile_size, maxx), min(y + initial_tile_size, maxy))
                initial_tiles_to_process.append({'geom': tile_geom, 'id': tile_id})

        processed_tiles = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_tile = {executor.submit(_process_tile, tile['geom'], tile['id']): tile for tile in initial_tiles_to_process}
            
            total_tiles = len(initial_tiles_to_process)
            for i, future in enumerate(as_completed(future_to_tile)):
                tile_id = future_to_tile[future]['id']
                try:
                    dem_list = future.result()
                    if dem_list:
                        processed_tiles.extend(dem_list)
                    print(f"--- Completed initial tile {tile_id} ({i + 1}/{total_tiles}). Found {len(dem_list)} DEMs. ---")
                except Exception as exc:
                    print(f'Initial tile {tile_id} generated an exception: {exc}')

        if not processed_tiles:
            raise ValueError("Could not fetch any DEM tiles for the given area.")
            
        # Check for multiple resolutions and resample to the highest if necessary
        # to prevent seams in the final output.
        resolutions = {dem.rio.resolution()[0] for dem in processed_tiles}
        if len(resolutions) > 1:
            target_res = min(resolutions)
            print(f"Found multiple resolutions. Resampling all tiles to highest resolution ({target_res:.8f} degrees) to prevent seams.")
        else:
            target_res = None # No resampling needed if all tiles are the same resolution

        # MERGE LOGIC - Progressive merge for large numbers of tiles
        print(f"Stitching {len(processed_tiles)} tiles together using progressive merge...")
        
        def merge_tiles_batch(tiles_batch, target_res=None):
            """Merge a batch of tiles together"""
            if len(tiles_batch) == 1:
                return tiles_batch[0]
            
            temp_files = []
            try:
                # Create temporary files for this batch
                for i, dem in enumerate(tiles_batch):
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_f:
                        dem.rio.to_raster(temp_f.name)
                        temp_files.append(temp_f.name)
                
                # Open and merge the batch
                srcs = [rasterio.open(f) for f in temp_files]
                if target_res:
                    mosaic_arr, out_trans = rio_merge(srcs, method='first', res=target_res)
                else:
                    mosaic_arr, out_trans = rio_merge(srcs, method='first')
                
                # Close sources
                for src in srcs:
                    src.close()
                
                # Convert back to xarray DataArray
                first_tile = tiles_batch[0]
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
                    except Exception as e:
                        print(f"Warning: Could not remove temporary file {f}: {e}")
        
        # Progressive merge strategy
        batch_size = 6  # Merge 10 tiles at a time
        current_tiles = processed_tiles.copy()
        batch_num = 1
        
        while len(current_tiles) > 1:
            print(f"Batch {batch_num}: Merging {len(current_tiles)} tiles in batches of {batch_size}...")
            
            # Split into batches
            batches = [current_tiles[i:i + batch_size] for i in range(0, len(current_tiles), batch_size)]
            merged_batches = []
            
            for i, batch in enumerate(batches):
                print(f"  Processing batch {i+1}/{len(batches)} ({len(batch)} tiles)...")
                merged_batch = merge_tiles_batch(batch, target_res)
                merged_batches.append(merged_batch)
            
            current_tiles = merged_batches
            batch_num += 1
        
        # Final result should be a single merged tile
        if len(current_tiles) == 1:
            mosaic_da = current_tiles[0]
            print("Progressive merge completed successfully!")
        else:
            raise ValueError("Progressive merge failed - no tiles remaining")
        
        # Clip to the original bounding box and apply vertical exaggeration
        mosaic_da = mosaic_da.rio.clip_box(minx, miny, maxx, maxy)
        mosaic_da = mosaic_da * vertical_exaggeration

        print("Saving final mosaic to cache...")
        mosaic_da.rio.to_raster(mosaic_cache_path)
        
        # Return the tiles list for potential debugging/visualization if needed
        initial_tiles = [box(x, y, x + initial_tile_size, y + initial_tile_size) for x in x_coords for y in y_coords]
        
        return mosaic_da, metadata, initial_tiles

class WaterLayer(BaseLayer):
    def __init__(self, cache_dir: str, location_name: str = "unnamed"):
        super().__init__(cache_dir)
        self.location_name = location_name
        self.reference_land_geometry = self._load_reference_land_geometry()

    def _load_reference_land_geometry(self) -> Optional[Polygon]:
        """
        Load and prepare the Natural Earth Data land geometry reference.
        This is loaded once during initialization and used for all subsequent processing.
        
        Returns:
            Optional[Polygon]: A unified geometry representing all land masses, or None if loading fails
        """
        try:
            reference_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets", "reference_data", "ne_50m_land.shp"
            )
            
            if not os.path.exists(reference_path):
                print(f"Warning: Reference land geometry file not found at {reference_path}")
                return None
                
            # Load the land shapefile
            land_gdf = gpd.read_file(reference_path)
            
            # Create a unified geometry from all land masses
            unified_land = land_gdf.geometry.unary_union
            return unified_land
            
        except Exception as e:
            print(f"Error loading reference land geometry: {e}")
            return None

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
                print("Warning: No reference land geometry available. Coastline processing may be inaccurate.")
                return coastline_features

            # Separate coastlines from other water features
            coastlines = coastline_features[coastline_features['FCODE'] == 56600].copy()
            other_water = coastline_features[coastline_features['FCODE'] != 56600].copy()

            if coastlines.empty:
                return coastline_features

            # Merge all coastlines and clip to bbox
            merged_coastline = coastlines.geometry.unary_union
            clipped_coastline = merged_coastline.intersection(bbox_polygon)

            if clipped_coastline.is_empty:
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

            return water_gdf

        except Exception as e:
            print(f"Error in coastline conversion: {e}")
            import traceback
            traceback.print_exc()
            return coastline_features

    @property
    def name(self):
        return "water_features"

    def _get_cache_filepath(self, polygon_geom: Polygon, min_area_km2: float = 0.0) -> str:
        """Generate cache filepath for water features."""
        bbox = polygon_geom.bounds
        bbox_str = format_bbox_string(bbox)
        area_str = f"_min_area_{min_area_km2}" if min_area_km2 > 0 else ""
        return os.path.join(
            self.cache_dir,
            f"{self.name}_{self.location_name}_{bbox_str}{area_str}.geojson"
        )

    def _fetch_from_source(self, polygon_geom: Polygon, min_area_km2: float = 0.0) -> gpd.GeoDataFrame:
        """
        Fetch water features from NHD.
        """
        print(f"Fetching NHD water body data (min area: {min_area_km2:.2f} km^2)...")
        try:
            bbox = polygon_geom.bounds
            # Layer numbers from NHD MapServer:
            # 10: Waterbody - Small Scale (includes oceans)
            # 11: Waterbody - Small Scale HI, pacific territories
            # 12: Waterbody - Large Scale
            # 4,5,6: Flowline - Small/Large Scale (rivers/streams)
            # 7,8,9: Area - Small/Large Scale
            layers = [10, 11, 12, 4, 5, 6, 7, 8, 9]
            layers = [10, 11, 12] # for
            all_water_features = []
            
            # FCodes for water features
            include_fcodes = [
                # Oceans and Seas
                44500,  # SeaOcean
                44501,  # Sea/Ocean - Atlantic
                44502,  # Sea/Ocean - Pacific
                44503,  # Sea/Ocean - Arctic
                44504,  # Sea/Ocean - Indian
                # Lakes and Ponds
                39004,  # Lake/Pond - Perennial
                39009,  # Lake/Pond - Average water elevation
                39010,  # Lake/Pond - Normal pool
                # Rivers and Streams
                #46006,  # StreamRiver - Perennial
                #46003,  # StreamRiver - Intermittent
                # Coastal Features
                31200,  # BayInlet
                #49300,  # Estuary
                36400,  # Foreshore
                56600,  # Coastline
                # Reservoirs
                43615,  # Reservoir - Perennial
                43617,  # Reservoir
                43621,  # Reservoir
            ]

            # Remove coastal features that extend beyond state boundaries
            exclude_fcodes = [
                36400,  # Foreshore - often extends beyond boundaries
                56600,  # Coastline - often extends beyond boundaries
                44500,  # SeaOcean - the Great Lakes themselves
                44501,  # Sea/Ocean - Atlantic
                44502,  # Sea/Ocean - Pacific
                44503,  # Sea/Ocean - Arctic
                44504,  # Sea/Ocean - Indian
            ]

            base_where_clause = f"FCode IN ({','.join(map(str, include_fcodes))})"
            
            for layer in layers:
                # Add area filter for polygon layers, but not for line layers
                if layer in [7, 8, 9, 10, 11, 12] and min_area_km2 > 0:
                    area_clause = f"AreaSqKm >= {min_area_km2}"
                    where_clause = f"({base_where_clause}) AND ({area_clause})"
                else:
                    where_clause = base_where_clause
                
                encoded_where = quote(where_clause)
                
                nhd_url = (
                    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/"
                    f"{layer}/query?"
                    f"geometry={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&"
                    "geometryType=esriGeometryEnvelope&"
                    "spatialRel=esriSpatialRelIntersects&"
                    "outFields=*&"
                    "returnGeometry=true&"
                    "f=geojson&"
                    "inSR=4326&outSR=4326&"
                    f"where={encoded_where}"
                )
                
                response = requests.get(nhd_url, timeout=60) # Increased timeout for large queries
                if response.status_code == 200 and response.text:
                    try:
                        water_features = gpd.read_file(io.StringIO(response.text))
                        if not water_features.empty:
                            all_water_features.append(water_features)
                            print(f"Found {len(water_features)} water features in layer {layer}")
                    except Exception as e:
                        print(f"Could not parse GeoJSON for layer {layer}: {e}")

            if not all_water_features:
                return None

            water_bodies_gdf = pd.concat(all_water_features, ignore_index=True)
            print(f"Total water features found: {len(water_bodies_gdf)}")

            # Filter out these FCodes
            if 'FCODE' in water_bodies_gdf.columns:
                water_bodies_gdf = water_bodies_gdf[~water_bodies_gdf['FCODE'].isin(exclude_fcodes)]
            
            return water_bodies_gdf

        except Exception as e:
            print(f"Error fetching water features: {e}")
            return None

    def fetch_great_lakes(self, polygon_geom: Polygon, use_cache: bool = True) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch Great Lakes/ocean polygons from NHD for shoreline clipping.
        These are the large water bodies that extend beyond state political boundaries.
        
        Args:
            polygon_geom: Shapely Polygon defining the area of interest
            use_cache: Whether to use cached data
            
        Returns:
            GeoDataFrame with Great Lakes polygons, or None if not found
        """
        # Cache path for Great Lakes
        bbox = polygon_geom.bounds
        bbox_str = format_bbox_string(bbox)
        cache_path = os.path.join(self.cache_dir, f"great_lakes_{self.location_name}_{bbox_str}.geojson")
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached Great Lakes data from {cache_path}")
            return gpd.read_file(cache_path)
        
        print("Fetching Great Lakes polygons from NHD...")
        try:
            bbox = polygon_geom.bounds
            # Great Lakes are classified as large lakes in NHD, not seas/oceans
            # We query by name since FCodes for lakes are generic
            # The Great Lakes names in NHD: Lake Superior, Lake Michigan, Lake Huron, Lake Erie, Lake Ontario
            great_lakes_names = [
                "Lake Superior",
                "Lake Michigan", 
                "Lake Huron",
                "Lake Erie",
                "Lake Ontario",
            ]
            
            # Also include sea/ocean codes for coastal states
            great_lakes_fcodes = [
                44500,  # SeaOcean
                44501,  # Sea/Ocean - Atlantic
                44502,  # Sea/Ocean - Pacific
            ]
            
            # Query layers 10, 11 (small scale waterbodies)
            layers = [10, 11]
            all_features = []
            
            # Build WHERE clause: search by name for Great Lakes OR by FCode for oceans
            name_conditions = " OR ".join([f"GNIS_Name LIKE '%{name}%'" for name in great_lakes_names])
            fcode_condition = f"FCode IN ({','.join(map(str, great_lakes_fcodes))})"
            where_clause = f"({name_conditions}) OR ({fcode_condition})"
            encoded_where = quote(where_clause)
            
            for layer in layers:
                nhd_url = (
                    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/"
                    f"{layer}/query?"
                    f"geometry={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&"
                    "geometryType=esriGeometryEnvelope&"
                    "spatialRel=esriSpatialRelIntersects&"
                    "outFields=*&"
                    "returnGeometry=true&"
                    "f=geojson&"
                    "inSR=4326&outSR=4326&"
                    f"where={encoded_where}"
                )
                
                print(f"Querying NHD layer {layer} for Great Lakes...")
                response = requests.get(nhd_url, timeout=120)
                if response.status_code == 200 and response.text:
                    try:
                        features = gpd.read_file(io.StringIO(response.text))
                        if not features.empty:
                            all_features.append(features)
                            print(f"Found {len(features)} Great Lakes features in layer {layer}")
                            if 'GNIS_Name' in features.columns:
                                print(f"  Names: {features['GNIS_Name'].unique().tolist()}")
                    except Exception as e:
                        print(f"Could not parse GeoJSON for layer {layer}: {e}")
                else:
                    print(f"Layer {layer} query failed with status {response.status_code}")
            
            if not all_features:
                print("No Great Lakes features found in area")
                return None
            
            great_lakes_gdf = pd.concat(all_features, ignore_index=True)
            print(f"Total Great Lakes features: {len(great_lakes_gdf)}")
            
            # Save to cache
            great_lakes_gdf.to_file(cache_path, driver='GeoJSON')
            
            return great_lakes_gdf
            
        except Exception as e:
            print(f"Error fetching Great Lakes: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_to_cache(self, data: gpd.GeoDataFrame, filepath: str) -> None:
        """Save water features to GeoJSON cache file."""
        if data is not None:
            data.to_file(filepath, driver='GeoJSON')

    def _load_from_cache(self, filepath: str) -> gpd.GeoDataFrame:
        """Load water features from GeoJSON cache file."""
        return gpd.read_file(filepath)

    def create_water_mask(self, water_features: gpd.GeoDataFrame, dem_data: xarray.DataArray) -> np.ndarray:
        """Create a binary water mask from water features."""
        if water_features is None or water_features.empty:
            return None

        # Ensure the water features are in the same CRS as the DEM
        water_features = water_features.to_crs(dem_data.rio.crs)
        
        # Create a bbox polygon for coastline conversion
        bbox = box(*dem_data.rio.bounds())
        
        # Convert coastlines to polygons
        water_features = self._convert_coastline_to_polygon(water_features, bbox)
        
        # Handle rivers/streams with buffering
        if 'FTYPE' in water_features.columns:
            flowlines = water_features[water_features['FTYPE'] == 'StreamRiver']
            if not flowlines.empty:
                # Use larger buffer for major rivers based on FCODE
                perennial_mask = flowlines['FCODE'].astype(str) == '46006'
                buffer_sizes = np.where(perennial_mask, 100, 50)  # 100m for perennial, 50m for others
                flowlines_buffered = flowlines.geometry.buffer(buffer_sizes)
                water_features.loc[flowlines.index, 'geometry'] = flowlines_buffered

        # Get the shape for rasterization
        out_shape = dem_data.shape[1:] if dem_data.ndim == 3 else dem_data.shape

        # Create water mask with all_touched=True for better coverage
        water_mask = rasterize(
            shapes=water_features.geometry,
            out_shape=out_shape,
            transform=dem_data.rio.transform(),
            fill=0,
            dtype='uint8',
            all_touched=True  # This helps ensure we don't miss any water pixels
        ).astype(bool)

        return water_mask

    def _filter_water_features(self, water_features: gpd.GeoDataFrame, state_polygon: Polygon, 
                              filter_config: dict) -> gpd.GeoDataFrame:
        """
        Filter water features to remove coastal water and large bodies.
        
        Args:
            water_features: GeoDataFrame of water features
            state_polygon: State boundary polygon
            filter_config: Configuration for filtering parameters
            
        Returns:
            Filtered GeoDataFrame of water features
        """
        if water_features is None or water_features.empty:
            return water_features
        
        print(f"Filtering {len(water_features)} water features...")
        
        # Get filtering parameters
        coastline_buffer = filter_config.get('coastline_buffer_degrees', 0.002)
        max_water_area_km2 = filter_config.get('max_water_area_km2', 1000.0)
        remove_large_bodies = filter_config.get('remove_large_bodies', True)
        preserve_rivers = filter_config.get('preserve_rivers', True)
        river_max_width_km = filter_config.get('river_max_width_km', 2.0)
        
        # Step 1: Create inland boundary by buffering from coastline
        inland_boundary = state_polygon.buffer(-coastline_buffer)
        print(f"Created inland boundary with {coastline_buffer} degree buffer")
        
        # Step 2: Filter by area and location
        filtered_features = []
        
        for idx, feature in water_features.iterrows():
            # Calculate feature area in square kilometers
            # Convert from degrees to km (approximate: 1 degree ≈ 111 km)
            feature_area_km2 = feature.geometry.area * (111.32 ** 2)
            
            # Check if feature is within inland boundary
            is_inland = inland_boundary.contains(feature.geometry) or inland_boundary.intersects(feature.geometry)
            
            # Determine if this is likely a river (narrow feature)
            bounds = feature.geometry.bounds
            width_km = (bounds[2] - bounds[0]) * 111.32  # longitude difference
            height_km = (bounds[3] - bounds[1]) * 111.32  # latitude difference
            is_likely_river = max(width_km, height_km) / min(width_km, height_km) > 3  # aspect ratio > 3:1
            
            # Apply filtering rules
            keep_feature = True
            
            # Rule 1: Remove large bodies of water
            if remove_large_bodies and feature_area_km2 > max_water_area_km2:
                if not (preserve_rivers and is_likely_river):
                    keep_feature = False
                    print(f"Removed large water body: {feature_area_km2:.1f} km²")
            
            # Rule 2: Remove features that are primarily coastal
            if not is_inland and feature_area_km2 > 10:  # Only remove larger coastal features
                keep_feature = False
                print(f"Removed coastal water feature: {feature_area_km2:.1f} km²")
            
            # Rule 3: Preserve rivers even if they're large
            if preserve_rivers and is_likely_river and max(width_km, height_km) < river_max_width_km:
                keep_feature = True
                #print(f"Preserved river: {feature_area_km2:.1f} km²")
            
            if keep_feature:
                filtered_features.append(feature)
        
        if filtered_features:
            filtered_gdf = gpd.GeoDataFrame(filtered_features, crs=water_features.crs)
            print(f"Filtered water features: {len(water_features)} → {len(filtered_gdf)}")
            return filtered_gdf
        else:
            print("No water features remaining after filtering")
            return gpd.GeoDataFrame(columns=water_features.columns, crs=water_features.crs)

    def get_water_mask(self, polygon_geom: Polygon, dem_data: xarray.DataArray,
                      use_cache: bool = True, min_area_km2: Optional[float] = None,
                      state_polygon: Optional[Polygon] = None, 
                      water_filter_config: Optional[dict] = None) -> np.ndarray:
        """
        Main method to get water mask with configurable filtering.
        
        Args:
            polygon_geom: Shapely Polygon defining the area of interest
            dem_data: DEM data to match dimensions and transform
            use_cache: Whether to use cached data
            min_area_km2: The minimum area in square kilometers for a water feature to be included.
            state_polygon: State boundary polygon for filtering
            water_filter_config: Configuration for water filtering
            
        Returns:
            numpy.ndarray: Binary mask where True indicates water
        """
        # If no minimum area is specified, calculate a reasonable default
        # based on the total size of the map area to avoid fetching tiny features.
        if min_area_km2 is None:
            total_area_km2 = bbox_to_land_area(polygon_geom) / 1_000_000
            if total_area_km2 > 10000:  # e.g., large state
                min_area_km2 = 1.0
            elif total_area_km2 > 1000: # e.g., large county or small state
                min_area_km2 = 0.5
            else:  # For smaller, local maps
                min_area_km2 = 0.0
            print(f"INFO: Automatically setting min water feature area to {min_area_km2:.2f} km^2 for this map size.")
        
        cache_path = self._get_cache_filepath(polygon_geom, min_area_km2=min_area_km2)
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached water features from {cache_path}")
            water_features = self._load_from_cache(cache_path)
        else:
            water_features = self._fetch_from_source(polygon_geom, min_area_km2=min_area_km2)
            if water_features is not None:
                self._save_to_cache(water_features, cache_path)

        # After getting water features but before rasterizing
        if state_polygon is not None and water_filter_config is not None:
            # Apply advanced filtering
            water_features = self._filter_water_features(water_features, state_polygon, water_filter_config)
        elif state_polygon is not None:
            # Simple clipping (original behavior)
            water_features = water_features.clip(state_polygon)
            print(f"Water features after clipping to state boundary: {len(water_features)}")

        return self.create_water_mask(water_features, dem_data)

class StateBoundaryLayer(BaseLayer):
    def __init__(self, cache_dir: str, location_name: str = "unnamed"):
        super().__init__(cache_dir)
        self.location_name = location_name
        self.state_shapefile_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets", "reference_data", "ne_10m_admin_1_states_provinces_lakes.shp"
        )
        self.land_shapefile_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets", "reference_data", "ne_50m_land.shp"
        )
        self.states_gdf = self._load_states_gdf()
        self.land_gdf = self._load_land_gdf()

    @property
    def name(self):
        return "state_boundary"

    def _get_cache_filepath(self, polygon_geom):
        # Not used for state boundaries, but required by BaseLayer
        return os.path.join(self.cache_dir, f"{self.name}_{self.location_name}.geojson")

    def _fetch_from_source(self, polygon_geom):
        # Not used for state boundaries, but required by BaseLayer
        return None

    def _save_to_cache(self, data, filepath):
        # Not used for state boundaries, but required by BaseLayer
        pass

    def _load_from_cache(self, filepath):
        # Not used for state boundaries, but required by BaseLayer
        return None

    def _load_states_gdf(self):
        try:
            gdf = gpd.read_file(self.state_shapefile_path)
            return gdf
        except Exception as e:
            print(f"Error loading state boundaries: {e}")
            return None

    def _load_land_gdf(self):
        try:
            gdf = gpd.read_file(self.land_shapefile_path)
            return gdf
        except Exception as e:
            print(f"Error loading land polygons: {e}")
            return None

    def get_state_polygon(self, state_name: str, country_name: str = "United States of America", land_only: bool = False) -> Polygon:
        """
        Get the polygon for a given state (optionally land-only, i.e., clipped to landmass).
        """
        if self.states_gdf is None:
            raise RuntimeError("State boundaries not loaded.")
        # Try to match by state name and country
        state_row = self.states_gdf[
            (self.states_gdf['name'].str.lower() == state_name.lower()) &
            (self.states_gdf['admin'].str.lower() == country_name.lower())
        ]
        if state_row.empty:
            # Try matching by abbrev
            state_row = self.states_gdf[
                (self.states_gdf['abbrev'].str.lower() == state_name.lower()) &
                (self.states_gdf['admin'].str.lower() == country_name.lower())
            ]
        if state_row.empty:
            raise ValueError(f"State '{state_name}' not found in shapefile.")
        state_geom = state_row.geometry.iloc[0]
        if not land_only or self.land_gdf is None:
            return state_geom
        # Intersect with land polygon(s)
        land_union = self.land_gdf.geometry.unary_union
        clipped = state_geom.intersection(land_union)
        return clipped

    def get_state_polygon_shoreline_clipped(self, state_name: str, 
                                            great_lakes_gdf: Optional[gpd.GeoDataFrame] = None,
                                            country_name: str = "United States of America") -> Polygon:
        """
        Get the state polygon clipped to the actual shoreline using NHD Great Lakes data.
        
        This removes the portions of the political boundary that extend into 
        large water bodies (Great Lakes, oceans) while preserving inland boundaries.
        
        Args:
            state_name: Name of the state
            great_lakes_gdf: GeoDataFrame with Great Lakes polygons from NHD.
                            If None, returns the unclipped state polygon.
            country_name: Name of the country
            
        Returns:
            Shapely Polygon/MultiPolygon clipped to shoreline
        """
        # Get the political state boundary (not clipped to NE land data)
        state_geom = self.get_state_polygon(state_name, country_name, land_only=False)
        
        if great_lakes_gdf is None or great_lakes_gdf.empty:
            print("No Great Lakes data provided, returning unclipped state polygon")
            return state_geom
        
        try:
            # Ensure same CRS
            if great_lakes_gdf.crs is None:
                great_lakes_gdf = great_lakes_gdf.set_crs("EPSG:4326")
            
            # Union all Great Lakes polygons
            lakes_union = great_lakes_gdf.geometry.unary_union
            
            # Subtract the Great Lakes from the state polygon
            # This clips the state boundary to the shoreline
            clipped = state_geom.difference(lakes_union)
            
            if clipped.is_empty:
                print("Warning: Clipped state polygon is empty, returning original")
                return state_geom
            
            print(f"Successfully clipped state polygon to Great Lakes shoreline")
            return clipped
            
        except Exception as e:
            print(f"Error clipping state to shoreline: {e}")
            import traceback
            traceback.print_exc()
            return state_geom

    def create_state_mask(self, state_polygon: Polygon, dem_data: xarray.DataArray) -> np.ndarray:
        """
        Create a binary mask for the state polygon on the DEM grid.
        """
        # Ensure CRS match
        from shapely.geometry import mapping
        import rasterio
        import rasterio.features
        # Reproject state polygon to DEM CRS if needed
        if hasattr(dem_data, 'rio') and hasattr(dem_data.rio, 'crs'):
            dem_crs = dem_data.rio.crs
            gdf = gpd.GeoDataFrame(geometry=[state_polygon], crs="EPSG:4326")
            gdf = gdf.to_crs(dem_crs)
            poly = gdf.geometry.iloc[0]
        else:
            poly = state_polygon
        out_shape = dem_data.shape[1:] if dem_data.ndim == 3 else dem_data.shape
        mask = rasterio.features.rasterize(
            [(poly, 1)],
            out_shape=out_shape,
            transform=dem_data.rio.transform(),
            fill=0,
            dtype='uint8',
            all_touched=True
        ).astype(bool)
        return mask