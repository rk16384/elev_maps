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

    @property
    def name(self):
        return "elevation_dem"

    def _get_cache_filepath(self, polygon_geom: Polygon, resolution: Optional[int] = None) -> str:
        """
        Generate cache filepath based on polygon bounds and resolution.
        
        Args:
            polygon_geom: Shapely Polygon defining the area of interest
            resolution: Optional specific resolution in meters
        """
        bbox = polygon_geom.bounds  # (minx, miny, maxx, maxy)
        bbox_str = format_bbox_string(bbox)
        res_str = f"{resolution}m" if resolution else "auto"

        metadata = {
            "resolution":resolution,
            "crs": None,
            "bbox": bbox_str,
            "shape": None,
            "nodata_value": None
        }

        return os.path.join(
            self.cache_dir,
            f"{self.name}_{self.location_name}_{bbox_str}_{res_str}.tif"
        ), metadata

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
                dem_data_attempt = py3dep.get_map(
                    layers='DEM',
                    geometry=polygon_geom,  # py3dep accepts Shapely geometries
                    resolution=resolution,
                    crs="EPSG:4326"
                )
                
                # Validate the data
                if dem_data_attempt.ndim < 2 or min(dem_data_attempt.shape) < 2:
                    raise ValueError(f"Invalid DEM data at {resolution}m resolution")
                
                dem_data = dem_data_attempt
                used_resolution = resolution
                print(f"Successfully fetched DEM at {resolution}m resolution")
                break
                
            except Exception as e:
                print(f"Failed to fetch at {resolution}m: {str(e)}")
                if target_resolution is not None:
                    raise ValueError(f"Could not fetch DEM at specified resolution of {target_resolution}m")
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

    def get_mosaic_dem(self, polygon_geom, use_cache=True, vertical_exaggeration=1.0):
        land_area = bbox_to_land_area(polygon_geom)
        valid_resolutions = land_area_to_resolution(land_area)  # e.g., [1, 3, 10, 30, 100]
        minx, miny, maxx, maxy = polygon_geom.bounds

        # Check for cached mosaic
        mosaic_cache_path, metadata = self._get_cache_filepath(polygon_geom, resolution='mosaic')
        if use_cache and os.path.exists(mosaic_cache_path):
            print(f"Loading cached mosaic DEM from {mosaic_cache_path}")
            mosaic = rioxarray.open_rasterio(mosaic_cache_path).squeeze()
            mosaic = mosaic * vertical_exaggeration
            return mosaic, metadata

        # Choose a tile size based on the highest resolution
        if 1 in valid_resolutions:
            tile_size = 0.01  # ~1km
        elif 3 in valid_resolutions:
            tile_size = 0.03  # ~3km
        elif 10 in valid_resolutions:
            tile_size = 0.05  # ~5km
        elif 30 in valid_resolutions:
            tile_size = 0.1   # ~10km
        else:
            tile_size = 0.25  # fallback

        overlap = 0.01
        # Generate tiles
        tiles = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                tile = box(
                            max(minx, x - overlap),
                            max(miny, y - overlap),
                            min(maxx, x + tile_size + overlap),
                            min(maxy, y + tile_size + overlap)
                        )
                tiles.append(tile)
                y += tile_size
            x += tile_size

        # Download highest available resolution for each tile
        tile_dems = []
        fetched_resolutions = {}
        metadata = {}
        for i, tile in enumerate(tiles):
            dem_data = None
            for res in valid_resolutions:
                if res not in fetched_resolutions:
                    fetched_resolutions[res] = 0
                try:
                    dem_data, metadata = self._fetch_from_source(tile, target_resolution=res)
                    fetched_resolutions[res] += 1
                    break
                except Exception as e:
                    continue
            if dem_data is not None:
                tile_dems.append(dem_data)
            else:
                print(f"WARNING: No DEM available for tile {tile.bounds}")

            print(f"Fetched {i}/{len(tiles)} tiles")
            print(f"Fetched resolutions: {fetched_resolutions}")

        # Save each tile to a temporary file
        temp_files = []
        for i, dem in enumerate(tile_dems):
            temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            dem.rio.to_raster(temp_file.name)
            temp_files.append(temp_file.name)

        # Open all tiles as rasterio datasets
        srcs = [rasterio.open(f) for f in temp_files]
        mosaic_arr, out_trans = rio_merge(srcs, method='first')

        # Debug: Check mosaic array for NaN values
        print(f"Mosaic array shape: {mosaic_arr.shape}")
        print(f"Mosaic array dtype: {mosaic_arr.dtype}")
        print(f"NaN values in mosaic: {np.isnan(mosaic_arr).sum()}")
        print(f"Min/Max values: {np.nanmin(mosaic_arr):.2f}/{np.nanmax(mosaic_arr):.2f}")

        # Optionally, close files and clean up
        for src in srcs:
            src.close()
        for f in temp_files:
            os.remove(f)

        # Convert the numpy array back to a DataArray
        # Use the metadata from the first tile for CRS, etc.
        first_tile = tile_dems[0]
        # mosaic_da = rioxarray.open_rasterio(
        #     rasterio.io.MemoryFile().open(
        #         driver='GTiff',
        #         width=mosaic_arr.shape[2],
        #         height=mosaic_arr.shape[1],
        #         count=1,
        #         dtype=mosaic_arr.dtype,
        #         transform=out_trans,
        #         crs=first_tile.rio.crs
        #     )
        # )
        # Or, if you want to avoid MemoryFile, you can use rioxarray's from_array:
        import xarray as xr
        # mosaic_arr shape: (bands, height, width)
        height, width = mosaic_arr.shape[1], mosaic_arr.shape[2]
        transform = out_trans

        # Generate coordinates using the transform
        x_coords = np.array([transform * (i, 0) for i in range(width)])[:, 0]
        y_coords = np.array([transform * (0, j) for j in range(height)])[:, 1]

        mosaic_da = xr.DataArray(
            mosaic_arr[0],  # [0] if single band
            dims=("y", "x"),
            coords={
                "y": y_coords,
                "x": x_coords,
            },
            attrs=first_tile.attrs
        )
        mosaic_da = mosaic_da.rio.write_crs(first_tile.rio.crs)
        mosaic_da = mosaic_da.rio.write_transform(out_trans)

        # Debug: Check DataArray before vertical exaggeration
        print(f"DataArray shape: {mosaic_da.shape}")
        print(f"DataArray NaN values: {np.isnan(mosaic_da.values).sum()}")
        print(f"DataArray Min/Max: {np.nanmin(mosaic_da.values):.2f}/{np.nanmax(mosaic_da.values):.2f}")

        # Apply vertical exaggeration
        mosaic_da = mosaic_da * vertical_exaggeration

        # Debug: Check after vertical exaggeration
        print(f"After vertical exaggeration - NaN values: {np.isnan(mosaic_da.values).sum()}")

        # Save the full mosaic to cache
        mosaic_da.rio.to_raster(mosaic_cache_path)
        print(f"Saved mosaic DEM to {mosaic_cache_path}")

        # After creating mosaic_da
        mosaic_da = mosaic_da.rio.clip_box(minx, miny, maxx, maxy)
        
        # Debug: Check after clipping
        print(f"After clipping - NaN values: {np.isnan(mosaic_da.values).sum()}")
        print(f"After clipping - Min/Max: {np.nanmin(mosaic_da.values):.2f}/{np.nanmax(mosaic_da.values):.2f}")

        return mosaic_da, metadata

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

    def _get_cache_filepath(self, polygon_geom: Polygon) -> str:
        """Generate cache filepath for water features."""
        bbox = polygon_geom.bounds
        bbox_str = format_bbox_string(bbox)
        return os.path.join(
            self.cache_dir,
            f"{self.name}_{self.location_name}_{bbox_str}.geojson"
        )

    def _fetch_from_source(self, polygon_geom: Polygon) -> gpd.GeoDataFrame:
        """
        Fetch water features from NHD.
        """
        print("Fetching NHD water body data...")
        try:
            bbox = polygon_geom.bounds
            # Layer numbers from NHD MapServer:
            # 10: Waterbody - Small Scale (includes oceans)
            # 11: Waterbody - Small Scale HI, pacific territories
            # 12: Waterbody - Large Scale
            # 4,5,6: Flowline - Small/Large Scale (rivers/streams)
            # 7,8,9: Area - Small/Large Scale
            layers = [10, 11, 12, 4, 5, 6, 7, 8, 9]
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
                46006,  # StreamRiver - Perennial
                46003,  # StreamRiver - Intermittent
                # Coastal Features
                31200,  # BayInlet
                49300,  # Estuary
                36400,  # Foreshore
                56600,  # Coastline
                # Reservoirs
                43615,  # Reservoir - Perennial
                43617,  # Reservoir
                43621,  # Reservoir
            ]

            # Include all water feature types
            ftype_clause = "FTYPE IN ('SeaOcean', 'Ocean', 'Sea', 'BayInlet', 'StreamRiver', 'Estuary', 'CanalDitch', 'Reservoir', 'LakePond')"
            #where_clause = f"(FCode IN ({','.join(map(str, include_fcodes))}) OR {ftype_clause})"
            where_clause = f"FCode IN ({','.join(map(str, include_fcodes))})"
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
                
                response = requests.get(nhd_url)
                if response.status_code == 200:
                    try:
                        water_features = gpd.read_file(response.text)
                        if not water_features.empty:
                            # Print the types of features found for debugging
                            if 'FTYPE' in water_features.columns:
                                print(f"Layer {layer} feature types: {water_features['FTYPE'].unique()}")
                            if 'FCODE' in water_features.columns:
                                print(f"Layer {layer} feature codes: {water_features['FCODE'].unique()}")
                            all_water_features.append(water_features)
                            print(f"Found {len(water_features)} water features in layer {layer}")
                    except Exception as e:
                        print(f"Error parsing GeoJSON response for layer {layer}: {e}")

            if not all_water_features:
                return None

            water_bodies_gdf = pd.concat(all_water_features, ignore_index=True)
            print(f"Total water features found: {len(water_bodies_gdf)}")
            
            return water_bodies_gdf

        except Exception as e:
            print(f"Error fetching water features: {e}")
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

    def get_water_mask(self, polygon_geom: Polygon, dem_data: xarray.DataArray,
                      use_cache: bool = True) -> np.ndarray:
        """
        Main method to get water mask.
        
        Args:
            polygon_geom: Shapely Polygon defining the area of interest
            dem_data: DEM data to match dimensions and transform
            use_cache: Whether to use cached data
            
        Returns:
            numpy.ndarray: Binary mask where True indicates water
        """
        cache_path = self._get_cache_filepath(polygon_geom)
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached water features from {cache_path}")
            water_features = self._load_from_cache(cache_path)
        else:
            water_features = self._fetch_from_source(polygon_geom)
            if water_features is not None:
                self._save_to_cache(water_features, cache_path)

        return self.create_water_mask(water_features, dem_data)