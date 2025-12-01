from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

from PIL import Image
import numpy as np
from shapely.geometry import Polygon
import xarray
import os
import json

from map_layers import StateBoundaryLayer


@dataclass
class MapData:
    """
    A data class to hold all configuration and generated data for a single map.
    """
    # --- Input Settings ---
    location_name: str
    location_config: dict = field(init=False, default=None)
    location_type: str = field(init=False, default=None)

    coordinates: Optional[List[Tuple[float, float]]] = None
    land_only: bool = True

    # Rendering settings
    sun_azimuth: float = 315.0
    sun_altitude: float = 45.0
    vertical_exaggeration: float = 1.0
    gaussian_blur_sigma: float = 0.0

    # Shadow settings
    shadow_config: dict = field(init=False, default=None)
    hillshade_config: dict = field(init=False, default=None)
    elevation_coloring_config: dict = field(init=False, default=None)
    common_config: dict = field(init=False, default=None)

    # Coloring settings
    color_method: str = 'hillshade'  # 'hillshade' or 'gradient' or 'grayscale'
    low_color: Optional[Tuple[int, int, int]] = None
    high_color: Optional[Tuple[int, int, int]] = None
    use_percentiles: bool = True
    percentile_range: Tuple[float, float] = (5, 95)
    
    # Feature settings
    include_water: bool = True
    water_color: Union[int, Tuple[int, int, int]] = (0, 0, 139)

    # Output settings
    output_dir: str = 'output'
    base_dir: Optional[str] = None
    use_cache: bool = True
    create_debug_grid: bool = False
    transparent_background: bool = False

    # Print settings
    print_width_inches: float = 5.0
    print_height_inches: float = 7.0
    print_margin_inches: float = 0.5
    print_dpi: int = 600
    print_border: bool = True
    print_shadow: bool = True

    # --- Generated Data (initialized after creation) ---
    polygon_geom: Optional[Polygon] = field(init=False, default=None)
    location_polygon: Optional[Polygon] = field(init=False, default=None)
    location_layer: Optional['StateBoundaryLayer'] = field(init=False, default=None)


    location_mask: Optional[np.ndarray] = field(init=False, default=None)
    # Water Layer Data
    water_mask: Optional[np.ndarray] = field(init=False, default=None)

    # Elevation Layer Data
    dem_data: Optional[xarray.DataArray] = field(init=False, default=None)
    metadata: Optional[dict] = field(init=False, default=None)
    tiles: Optional[list] = field(init=False, default=None)

    # Shadow Layer Data
    shadow_img: Optional[Image.Image] = field(init=False, default=None)
    shadow_offsets: Optional[Tuple[int, int]] = field(init=False, default=None)
    map_image: Optional[Image.Image] = field(init=False, default=None)

    # Cache settings
    use_cache_dem: bool = True
    use_cache_state_boundary: bool = True
    use_cache_hillshade: bool = True
    use_cache_elevation_coloring: bool = True
    use_cache_water: bool = True
    use_cache_shadow: bool = True

    # DEM processing settings
    auto_downsample: bool = True
    max_pixels: int = 50000000
    downsample_factor: int = 2

    def __post_init__(self):
        """Load data from location config"""
        # determine base directory for caches/outputs
        self.base_dir = self.base_dir or os.getenv("ELEV_MAPS_BASE_DIR") or os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.abspath(self.base_dir)

        # load location config
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'location_configs.json'), 'r') as f:
            location_configs = json.load(f)
        self.location_config = location_configs[self.location_name]

        self._prepare_geometry()
        self._init_cache_dir()
        self.location_type = self.location_config['type']
        # Shadow config
        if "shadow_config" in self.location_config:
            self.shadow_config = self.location_config['shadow_config']
        else:
            self.shadow_config = location_configs['default']['shadow_config']
            
        # Hillshade config
        if "hillshade_config" in self.location_config:
            self.hillshade_config = self.location_config['hillshade_config']
        else:
            self.hillshade_config = location_configs['default']['hillshade_config']

        # Elevation coloring config
        if "elevation_coloring_config" in self.location_config:
            self.elevation_coloring_config = self.location_config['elevation_coloring_config']
        else:
            self.elevation_coloring_config = location_configs['default']['elevation_coloring_config']

        # Common config
        if "common" in self.location_config:
            self.common_config = self.location_config['common']
        else:
            self.common_config = location_configs['default']['common']

        # Cache settings
        if "cache" in self.location_config:
            self.set_cache_settings(self.location_config['cache'])
        else:
            self.set_cache_settings(location_configs['default']['cache'])

        self.include_water = self.hillshade_config['include_water']

        if isinstance(self.hillshade_config['water_color'], list):
            self.hillshade_config['water_color'] = tuple(self.hillshade_config['water_color'])


    def _prepare_geometry(self):
        """Load data from location config"""
        # load location config
        if self.location_config['type'] == 'state':
            self.location_polygon, self.bbox, self.location_layer = self.get_state_bbox_and_mask()
            
        elif self.location_config['type'] == 'coordinates':
            # Convert from [lat, lon] to [lon, lat] for Shapely
            coordinates = [(coord[1], coord[0]) for coord in self.location_config['coordinates']]
            self.location_polygon = Polygon(coordinates)
            self.bbox = self.location_polygon.bounds

        elif self.location_config['type'] == 'center_point':
            # make a box around the center point
            center_point = self.location_config['center_point']
            radius = self.location_config['radius']
            self.location_polygon = Polygon([
                (center_point[0] - radius, center_point[1] - radius),
                (center_point[0] - radius, center_point[1] + radius),
                (center_point[0] + radius, center_point[1] + radius),
                (center_point[0] + radius, center_point[1] - radius),
                (center_point[0] - radius, center_point[1] - radius)
            ])
            self.bbox = self.location_polygon.bounds
        else:
            raise ValueError(f"Invalid location type: {self.location_config['type']}")

    def make_output_dir(self):
        """Create the output directory if it doesn't exist."""
        output_root = os.path.join(self.base_dir, self.output_dir)
        os.makedirs(output_root, exist_ok=True)
        self.output_dir = output_root

    def _init_cache_dir(self) -> None:
        """Generate the path for the cache directory."""
        cache_root = os.path.join(self.base_dir, "data_cache")
        os.makedirs(cache_root, exist_ok=True)
        cache_path = os.path.join(cache_root, self.location_name)
        os.makedirs(cache_path, exist_ok=True)
        self.cache_dir = cache_path

    def set_cache_settings(self, cache_config: dict):
        """Set cache settings from configuration"""
        self.use_cache_dem = cache_config.get('dem', True)
        self.use_cache_state_boundary = cache_config.get('state_boundary', True)
        self.use_cache_hillshade = cache_config.get('hillshade', True)
        self.use_cache_elevation_coloring = cache_config.get('elevation_coloring', True)
        self.use_cache_water = cache_config.get('water', True)
        self.use_cache_shadow = cache_config.get('shadow', True)

    def get_cache_path(self, component: str, extension: str = None) -> str:
        """Get cache file path for a specific component"""
        bbox = self.location_polygon.bounds
        bbox_str = f"{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
        
        if extension is None:
            extension = self._get_default_extension(component)
        
        filename = f"{self.location_name}_{component}_{bbox_str}{extension}"
        return os.path.join(self.cache_dir, filename)

    def _get_default_extension(self, component: str) -> str:
        """Get default file extension for component"""
        extensions = {
            'dem': '.tif',
            'hillshade': '.pkl',  # Changed to pickle
            'elevation_coloring': '.pkl',  # Changed to pickle
            'water': '.geojson',
            'shadow': '.pkl',  # Changed to pickle
            'state_boundary': '.geojson',
            'state_mask': '.pkl'  # Changed to pickle
        }
        return extensions.get(component, '.pkl')

    def save_to_cache(self, component: str, data, **kwargs):
        """Save data to cache for a specific component"""
        cache_path = self.get_cache_path(component, kwargs.get('extension'))
        
        try:
            if component == 'dem':
                data.rio.to_raster(cache_path)
            elif component in ['hillshade', 'elevation_coloring', 'shadow', 'state_mask']:
                # Use pickle for images to avoid PIL size limits
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            elif component in ['water', 'state_boundary']:
                data.to_file(cache_path, driver='GeoJSON')
            else:
                # Generic pickle fallback
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            print(f"Saved {component} to cache: {cache_path}")
            return cache_path
        except Exception as e:
            print(f"Error saving {component} to cache: {e}")
            return None

    def load_from_cache(self, component: str, **kwargs):
        """Load data from cache for a specific component"""
        cache_path = self.get_cache_path(component, kwargs.get('extension'))
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            if component == 'dem':
                import rioxarray
                return rioxarray.open_rasterio(cache_path)
            elif component in ['hillshade', 'elevation_coloring', 'shadow', 'state_mask']:
                # Use pickle for images to avoid PIL size limits
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            elif component in ['water', 'state_boundary']:
                import geopandas as gpd
                return gpd.read_file(cache_path)
            else:
                # Generic pickle fallback
                import pickle
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading {component} from cache: {e}")
            return None

    def cache_exists(self, component: str, **kwargs) -> bool:
        """Check if cache exists for a component"""
        cache_path = self.get_cache_path(component, kwargs.get('extension'))
        return os.path.exists(cache_path)

    def clear_cache(self, component: str = None):
        """Clear cache for a specific component or all components"""
        if component:
            cache_path = self.get_cache_path(component)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"Cleared cache for {component}")
        else:
            # Clear all caches for this location
            bbox = self.location_polygon.bounds
            bbox_str = f"{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
            pattern = f"{self.location_name}_*_{bbox_str}.*"
            
            import glob
            cache_files = glob.glob(os.path.join(self.cache_dir, pattern))
            for cache_file in cache_files:
                os.remove(cache_file)
                print(f"Removed cache file: {cache_file}")

    def get_state_bbox_and_mask(self, land_only=True):
        """
        Get the state polygon, bounding box, and (optionally) a mask for a DEM.
        If dem_data is provided, returns the mask as well.
        """
        cache_root = os.path.join(self.base_dir, "data_cache")
        state_layer = StateBoundaryLayer(cache_dir=cache_root, location_name=self.location_name.lower())
        state_polygon = state_layer.get_state_polygon(self.location_name, land_only=land_only)
        bbox = state_polygon.bounds

        return state_polygon, bbox, state_layer