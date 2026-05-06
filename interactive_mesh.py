"""
Interactive 3D DEM mesh viewer with adjustable lighting and camera controls.

Usage:
    python interactive_mesh.py <location_name>
    python interactive_mesh.py <location_name> --auto-accept
    
    First, create a folder: /Volumes/sandisk1/data_cache/mountain_mesh_data/<location_name>/
    Then paste your GeoJSON polygon into: /Volumes/sandisk1/data_cache/mountain_mesh_data/<location_name>/polygon.json
    
Controls:
    - Use mouse to rotate/pan/zoom the view
    - Adjust sliders to modify rendering parameters
    - Press 'p' to save settings and render high-resolution image
    - Press 'q' to quit
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv
import rasterio
from rasterio.warp import transform_geom
from PIL import Image
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, mapping, shape

from dem_raster_utils import fill_invalid_nearest
from extract_dem_data import extract_dem_from_geojson, extract_dem_mosaic_from_geojson


def _dem_invalid_to_nan(arr: np.ndarray, nodata: float | int | None) -> None:
    """In-place: mark raster nodata and common sentinels as NaN (do not use 0 for holes)."""
    if nodata is not None:
        if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
            pass
        else:
            arr[arr == nodata] = np.nan
    arr[arr == -9999] = np.nan


# Base directory for all location data
BASE_DIR = Path("/Volumes/sandisk1/data_cache/mountain_mesh_data")


def resolve_dem_path(location_dir: Path) -> Path:
    """
    Prefer mesh_settings.json 'dem_filename' (basename under location_dir) when present
    and the file exists; otherwise extracted_dem.tif.
    """
    default = location_dir / "extracted_dem.tif"
    sf = location_dir / "mesh_settings.json"
    if not sf.exists():
        return default
    try:
        with open(sf) as f:
            s = json.load(f)
        name = s.get("dem_filename")
        if isinstance(name, str) and name.strip():
            alt = location_dir / name.strip()
            if alt.is_file():
                return alt
            print(
                f"Warning: mesh_settings.json dem_filename {name!r} not found; using {default.name}",
                file=sys.stderr,
            )
    except Exception:
        pass
    return default


class InteractiveMeshViewer:
    def __init__(
        self,
        location_dir: Path,
        polygon_coords: list,
        method: str = "radius",
        *,
        dem_path: Path | None = None,
        vertical_bias_percent: float = 0.0,
        render_max_dimension: int = 8192,
        render_max_mesh_dimension: int = 8000,
    ):
        self.location_dir = location_dir
        self.polygon_coords = polygon_coords
        self.method = method

        self.dem_path = dem_path if dem_path is not None else resolve_dem_path(location_dir)
        self.vertical_bias_percent = float(vertical_bias_percent)
        self.render_max_dimension = int(render_max_dimension)
        self.render_max_mesh_dimension = int(render_max_mesh_dimension)
        self.settings_file = location_dir / "mesh_settings.json"
        
        self.dem_data = None
        self.transform = None
        self.mesh = None
        self.base_mesh = None  # Store unrotated mesh for rotation
        self.plotter = None
        self.light = None
        self.actor = None
        self.save_requested = False
        self._pending_camera_info = None
        self._pending_window_size = None
        self.export_stl = False
        
        # Default parameters
        self.params = {
            "vertical_exaggeration": 0.5,
            "smoothing_sigma": 1.5,
            "light_elevation": 45.0,
            "light_azimuth": 315.0,
            "light_distance": 10000.0,
            "z_rotation": 0.0,  # Z-axis rotation in degrees
            "pixel_min": 2,  # Minimum pixel value for mesh (background stays white)
            "pixel_max": 253,  # Maximum pixel value for mesh
            # Nearest-neighbor fill for small mosaic holes (pixels); 0 = off
            "hole_fill_max_px": 48.0,
        }
        
        # Stored camera settings (loaded from file)
        self.saved_camera = None
        
        # Stored window size (loaded from file)
        self.saved_window_size = None
        
        # Load DEM data
        self._load_dem()
        
        # Load previous settings if available
        self._load_settings()
        
    def _load_dem(self, max_mesh_dimension: int = 2000):
        """
        Load DEM data from file, downsampling if too large for interactive viewing.
        
        Args:
            max_mesh_dimension: Maximum dimension (width or height) for the mesh.
                               Larger DEMs will be downsampled to prevent memory issues.
                               Default: 2000 (4 million point mesh max)
        """
        with rasterio.open(self.dem_path) as src:
            self.dem_data = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.bounds = src.bounds
            self.crs = src.crs
            src_nodata = src.nodata
        
        _dem_invalid_to_nan(self.dem_data, src_nodata)
        
        original_shape = self.dem_data.shape
        
        # Downsample if DEM is too large for interactive mesh viewing
        current_max = max(self.dem_data.shape)
        if current_max > max_mesh_dimension:
            from scipy.ndimage import zoom
            
            scale_factor = max_mesh_dimension / current_max
            new_height = int(self.dem_data.shape[0] * scale_factor)
            new_width = int(self.dem_data.shape[1] * scale_factor)
            
            print(f"Downsampling DEM for interactive viewing: {self.dem_data.shape[1]}x{self.dem_data.shape[0]} -> {new_width}x{new_height}")
            
            # Handle nodata before downsampling
            nodata_mask = ~np.isfinite(self.dem_data)
            self.dem_data[nodata_mask] = np.nan
            
            # Downsample using bilinear interpolation
            self.dem_data = zoom(self.dem_data, scale_factor, order=1, mode='nearest')
            
            # Update transform to reflect new resolution
            from affine import Affine
            self.transform = Affine(
                self.transform.a / scale_factor,
                self.transform.b,
                self.transform.c,
                self.transform.d,
                self.transform.e / scale_factor,
                self.transform.f
            )
            
        # Keep NaN for invalid pixels — converting to 0 creates fake "sea level" inside the polygon
        # and vertical cliff artifacts between real terrain and holes in the mosaic.
        
        print(f"Loaded DEM: {self.dem_path}")
        if original_shape != self.dem_data.shape:
            print(f"  Original shape: {original_shape}")
        print(f"  Shape: {self.dem_data.shape}")
        valid = np.isfinite(self.dem_data)
        if np.any(valid):
            print(
                f"  Elevation range (valid pixels): {np.nanmin(self.dem_data):.1f} to {np.nanmax(self.dem_data):.1f}"
            )
        else:
            print("  Elevation range: (no valid pixels)")
    
    def _load_settings(self):
        """Load previous settings from mesh_settings.json if it exists."""
        if not self.settings_file.exists():
            print("No previous settings found, using defaults.")
            return
        
        try:
            with open(self.settings_file, "r") as f:
                settings = json.load(f)
            
            # Load parameters
            if "parameters" in settings:
                for key, value in settings["parameters"].items():
                    if key in self.params:
                        self.params[key] = value
                print(f"Loaded parameters from: {self.settings_file}")
            
            # Store camera settings to apply after plotter is created
            if "camera" in settings:
                self.saved_camera = settings["camera"]
                print("  Camera settings will be restored on startup.")
            
            # Store window size to apply when creating plotter
            if "window_size" in settings:
                self.saved_window_size = tuple(settings["window_size"])
                print(f"  Window size will be restored: {self.saved_window_size}")
                
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
    
    def _polygon_shape_in_dem_crs(self):
        """GeoJSON vertices are WGS84 lon/lat; DEM may be projected (e.g. EPSG:6428)."""
        polygon = Polygon(self.polygon_coords)
        if self.crs is None or self.crs.is_geographic:
            return polygon
        geom = transform_geom("EPSG:4326", self.crs, mapping(polygon))
        return shape(geom)

    def _meters_per_pixel_xy(self) -> tuple[float, float]:
        """East/north pixel spacing in meters (elevation is meters in typical DEMs)."""
        if self.crs is not None and self.crs.is_geographic:
            center_lat = (self.bounds.top + self.bounds.bottom) / 2
            deg_to_m_lat = 111320.0
            deg_to_m_lon = 111320.0 * np.cos(np.radians(center_lat))
            return abs(self.transform.a) * deg_to_m_lon, abs(self.transform.e) * deg_to_m_lat
        try:
            from pyproj import CRS as PyProjCRS

            p = PyProjCRS.from_user_input(self.crs)
            unit = (p.axis_info[0].unit_name or "metre").lower()
            m_per_u = 0.304800609601219 if "foot" in unit else 1.0
            return abs(self.transform.a) * m_per_u, abs(self.transform.e) * m_per_u
        except Exception:
            return abs(self.transform.a), abs(self.transform.e)

    def _create_polygon_mask(self, erode_pixels: int = 2) -> np.ndarray:
        """Create a boolean mask from polygon coordinates.
        
        Args:
            erode_pixels: Number of pixels to erode from the boundary to remove
                          edge artifacts from lighting calculations.
        """
        from rasterio.features import geometry_mask
        from scipy.ndimage import binary_erosion
        
        if self.polygon_coords is None:
            return np.ones(self.dem_data.shape, dtype=bool)
        
        polygon = self._polygon_shape_in_dem_crs()
        
        # Create mask using rasterio (True = inside polygon)
        mask = ~geometry_mask(
            [polygon],
            out_shape=self.dem_data.shape,
            transform=self.transform,
            invert=False
        )
        
        # Erode mask to remove boundary pixels with incomplete neighbor data
        if erode_pixels > 0:
            mask = binary_erosion(mask, iterations=erode_pixels)
        
        return mask
        
    def _create_mesh(self):
        """Create PyVista mesh from DEM data with current smoothing."""
        elev = np.asarray(self.dem_data, dtype=np.float32)
        mx = float(self.params.get("hole_fill_max_px", 0.0))
        if mx > 0.0:
            elev = fill_invalid_nearest(elev, max_dist_px=mx)

        sigma = float(self.params["smoothing_sigma"])
        if sigma > 0.0:
            valid = np.isfinite(elev)
            if np.any(valid):
                fill = float(np.nanmedian(self.dem_data[valid]))
            else:
                fill = 0.0
            filled = np.where(valid, elev, fill)
            smoothed = gaussian_filter(filled, sigma=sigma)
            smoothed = np.where(valid, smoothed, np.nan)
        else:
            smoothed = elev
        
        # Create coordinate grids in real-world units (meters)
        nrows, ncols = smoothed.shape
        
        pixel_width_m, pixel_height_m = self._meters_per_pixel_xy()
        
        # Create coordinate grids in meters
        x = np.linspace(0, ncols * pixel_width_m, ncols)
        y = np.linspace(0, nrows * pixel_height_m, nrows)
        x, y = np.meshgrid(x, y)
        
        # Flip y to match geographic orientation (north up)
        y = np.flipud(y)
        
        # Apply vertical exaggeration
        valid_elev = np.isfinite(smoothed)
        z = smoothed * self.params["vertical_exaggeration"]
        
        # Apply polygon mask - set outside values to NaN
        if self.polygon_coords is not None:
            mask = self._create_polygon_mask()
            z = np.where(mask & valid_elev, z, np.nan)
        else:
            z = np.where(valid_elev, z, np.nan)
        
        # Create structured grid
        mesh = pv.StructuredGrid(x, y, z)
        
        # Extract only valid (non-NaN) cells if polygon trimming is active
        if self.polygon_coords is not None:
            # Get cell validity based on points
            point_valid = ~np.isnan(mesh.points[:, 2])
            # adjacent_cells=False ensures only cells with ALL valid vertices are kept
            mesh = mesh.extract_points(point_valid, adjacent_cells=False)
        
        if mesh.n_points == 0:
            raise ValueError(
                "Terrain mesh has zero points after polygon clip. "
                "Check that polygon.json (WGS84) overlaps the DEM extent, "
                "or that the DEM CRS matches your workflow."
            )
        
        return mesh
    
    def _calculate_light_position(self) -> tuple:
        """Calculate light position from elevation, azimuth, and distance."""
        # Convert to radians
        elev_rad = np.radians(self.params["light_elevation"])
        azim_rad = np.radians(self.params["light_azimuth"])
        dist = self.params["light_distance"]
        
        # Get mesh center
        if self.mesh is not None:
            center = self.mesh.center
        else:
            center = (self.dem_data.shape[1] / 2, self.dem_data.shape[0] / 2, 0)
        
        # Calculate position (azimuth: 0=North, 90=East, clockwise)
        # In PyVista: x=East, y=North, z=Up
        dx = dist * np.cos(elev_rad) * np.sin(azim_rad)
        dy = dist * np.cos(elev_rad) * np.cos(azim_rad)
        dz = dist * np.sin(elev_rad)
        
        return (center[0] + dx, center[1] + dy, center[2] + dz)
    
    def _update_mesh(self, value=None):
        """Rebuild mesh with current smoothing and vertical exaggeration."""
        if self.plotter is None:
            return
            
        # Remove old mesh actor
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
        
        # Create new mesh and apply Z rotation
        self.mesh = self._create_mesh()
        self.mesh = self.mesh.rotate_z(
            self.params["z_rotation"], point=self.mesh.center, inplace=False
        )
        
        # Add mesh with white color
        self.actor = self.plotter.add_mesh(
            self.mesh,
            color="white",
            smooth_shading=True,
            specular=0.0,
            diffuse=1.0,
            ambient=0.1,
        )
        
        # Update light position
        self._update_light()
        
    def _update_light(self, value=None):
        """Update light position based on current parameters."""
        if self.light is None or self.plotter is None:
            return
            
        pos = self._calculate_light_position()
        self.light.position = pos
        
        # Point light at mesh center
        if self.mesh is not None:
            self.light.focal_point = self.mesh.center
    
    def _on_vertical_exaggeration(self, value):
        """Callback for vertical exaggeration slider."""
        self.params["vertical_exaggeration"] = value
        self._update_mesh()
        
    def _on_smoothing(self, value):
        """Callback for smoothing slider."""
        self.params["smoothing_sigma"] = value
        self._update_mesh()
        
    def _on_light_elevation(self, value):
        """Callback for light elevation slider."""
        self.params["light_elevation"] = value
        self._update_light()
        
    def _on_light_azimuth(self, value):
        """Callback for light azimuth slider."""
        self.params["light_azimuth"] = value
        self._update_light()
        
    def _on_light_distance(self, value):
        """Callback for light distance slider."""
        self.params["light_distance"] = value
        self._update_light()
    
    def _on_z_rotation(self, value):
        """Callback for Z rotation slider."""
        self.params["z_rotation"] = value
        self._update_mesh()
    
    def _capture_current_view_state(self):
        """Capture current camera/window state while the interactive plotter is valid."""
        camera = self.plotter.camera
        camera_info = {
            "position": list(camera.position),
            "focal_point": list(camera.focal_point),
            "view_up": list(camera.up),
            "view_angle": camera.view_angle,
            "clipping_range": list(camera.clipping_range),
            "parallel_scale": camera.parallel_scale,
            "parallel_projection": camera.parallel_projection,
        }
        window_size = list(self.plotter.window_size)
        return camera_info, window_size

    def _save_settings(self, camera_info: dict | None = None, window_size: list | None = None):
        """Save current settings and camera position to file and print to terminal."""
        if camera_info is None or window_size is None:
            camera_info, window_size = self._capture_current_view_state()
        
        # Build core settings payload written by this tool.
        core_settings = {
            "dem_path": str(self.dem_path),
            "parameters": self.params.copy(),
            "camera": camera_info,
            "window_size": window_size,
            "timestamp": datetime.now().isoformat(),
        }

        # Preserve unrelated top-level settings blocks (e.g. title settings).
        preserved = {}
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    for key, value in existing.items():
                        if key not in core_settings:
                            preserved[key] = value
            except Exception:
                # If existing settings are invalid, fall back to writing core settings only.
                preserved = {}

        settings = {**preserved, **core_settings}
        default_dem = self.location_dir / "extracted_dem.tif"
        try:
            if self.dem_path.resolve() == default_dem.resolve():
                settings.pop("dem_filename", None)
            else:
                settings["dem_filename"] = self.dem_path.name
        except Exception:
            pass

        # Print to terminal
        print("\n" + "=" * 60)
        print("CURRENT SETTINGS")
        print("=" * 60)
        print(f"DEM Path: {settings['dem_path']}")
        print("\nRendering Parameters:")
        for key, value in self.params.items():
            print(f"  {key}: {value}")
        print("\nCamera Settings:")
        print(f"  position: {camera_info['position']}")
        print(f"  focal_point: {camera_info['focal_point']}")
        print(f"  view_up: {camera_info['view_up']}")
        print(f"  view_angle: {camera_info['view_angle']}")
        print(f"  parallel_projection: {camera_info['parallel_projection']}")
        print(f"\nWindow Size: {window_size[0]}x{window_size[1]}")
        print("=" * 60 + "\n")
        
        # Save to file
        with open(self.settings_file, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"Settings saved to: {self.settings_file.absolute()}")
        
        # Render high-resolution image
        self._render_high_res(
            camera_info,
            window_size=window_size,
            max_dimension=self.render_max_dimension,
            max_mesh_dimension=self.render_max_mesh_dimension,
            export_stl=self.export_stl,
        )
    
    def _render_high_res(
        self,
        camera_info: dict,
        window_size: list | tuple | None = None,
        max_dimension: int = 8192,
        max_mesh_dimension: int = 8000,
        export_stl: bool = False,
    ):
        """
        Render the current view at high resolution and save to file.
        
        Temporarily loads a higher-resolution version of the DEM for rendering,
        then restores the downsampled version for interactive viewing.
        
        Args:
            camera_info: Dictionary containing camera settings
            window_size: Optional [width, height] to preserve interactive aspect ratio.
                         If not provided, falls back to saved/default size.
            max_dimension: Maximum dimension (width or height) for output image
            max_mesh_dimension: Maximum dimension for the mesh used in rendering.
                               Higher = better quality but more memory. Default: 4000
        """
        print("\nRendering high-resolution image...")
        
        # Save current (downsampled) DEM data
        original_dem_data = self.dem_data
        original_transform = self.transform
        
        # Reload DEM at higher resolution for rendering
        print(f"Loading high-resolution DEM for rendering (max {max_mesh_dimension}px)...")
        with rasterio.open(self.dem_path) as src:
            full_dem = src.read(1).astype(np.float32)
            full_transform = src.transform
            full_nodata = src.nodata
        
        _dem_invalid_to_nan(full_dem, full_nodata)
        
        # Downsample if still too large, but at higher resolution than interactive
        current_max = max(full_dem.shape)
        if current_max > max_mesh_dimension:
            from scipy.ndimage import zoom
            from affine import Affine
            
            scale_factor = max_mesh_dimension / current_max
            new_height = int(full_dem.shape[0] * scale_factor)
            new_width = int(full_dem.shape[1] * scale_factor)
            
            print(f"  Downsampling for render: {full_dem.shape[1]}x{full_dem.shape[0]} -> {new_width}x{new_height}")
            
            # Handle nodata before downsampling
            nodata_mask = ~np.isfinite(full_dem)
            full_dem[nodata_mask] = np.nan
            full_dem = zoom(full_dem, scale_factor, order=1, mode='nearest')
            
            full_transform = Affine(
                full_transform.a / scale_factor,
                full_transform.b,
                full_transform.c,
                full_transform.d,
                full_transform.e / scale_factor,
                full_transform.f
            )
        
        # Temporarily use high-res DEM
        self.dem_data = full_dem
        self.transform = full_transform
        
        print(f"  Render mesh shape: {self.dem_data.shape}")
        
        # Determine output aspect ratio without relying on a potentially closed plotter.
        if (
            window_size is not None
            and len(window_size) == 2
            and window_size[0] > 0
            and window_size[1] > 0
        ):
            current_window_size = [int(window_size[0]), int(window_size[1])]
            window_source = "captured interactive window size"
        elif (
            self.saved_window_size is not None
            and len(self.saved_window_size) == 2
            and self.saved_window_size[0] > 0
            and self.saved_window_size[1] > 0
        ):
            current_window_size = [
                int(self.saved_window_size[0]),
                int(self.saved_window_size[1]),
            ]
            window_source = "saved window size"
        else:
            current_window_size = [1920, 1080]
            window_source = "default fallback"

        print(
            f"  Using aspect ratio source: {window_source} "
            f"({current_window_size[0]}x{current_window_size[1]})"
        )
        aspect_ratio = current_window_size[0] / current_window_size[1]
        
        # Calculate output resolution maintaining aspect ratio
        if aspect_ratio >= 1:
            # Wider than tall
            out_width = max_dimension
            out_height = int(max_dimension / aspect_ratio)
        else:
            # Taller than wide
            out_height = max_dimension
            out_width = int(max_dimension * aspect_ratio)
        
        resolution = (out_width, out_height)
        
        # Recreate mesh with current settings and apply Z rotation
        mesh = self._create_mesh()
        mesh = mesh.rotate_z(self.params["z_rotation"], point=mesh.center, inplace=False)
        self.mesh = mesh  # so _calculate_light_position uses mesh center
        light_pos = self._calculate_light_position()

        if export_stl:
            # Convert to a clean triangular surface mesh for broad CAD compatibility.
            stl_surface = mesh.extract_surface().triangulate()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stl_path = self.location_dir / f"terrain_mesh_{timestamp}.stl"
            stl_surface.save(str(stl_path))
            print(f"STL mesh exported to: {stl_path.absolute()}")
        
        def create_plotter(background_color):
            """Helper to create a configured plotter with given background."""
            plotter = pv.Plotter(off_screen=True, lighting="none", window_size=resolution)
            plotter.set_background(background_color)
            
            plotter.add_mesh(
                mesh,
                color="white",
                smooth_shading=True,
                specular=0.0,
                diffuse=1.0,
                ambient=0.1,
            )
            
            # Add point light with current settings
            light = pv.Light(
                position=light_pos,
                focal_point=mesh.center,
                color="white",
                intensity=1.0,
                positional=True,
            )
            plotter.add_light(light)
            
            # Add ambient light
            ambient_light = pv.Light(
                light_type="headlight",
                intensity=0.1,
            )
            plotter.add_light(ambient_light)
            
            # Set camera to match current view exactly
            plotter.camera.position = camera_info["position"]
            plotter.camera.focal_point = camera_info["focal_point"]
            plotter.camera.up = camera_info["view_up"]
            plotter.camera.view_angle = camera_info["view_angle"]
            plotter.camera.clipping_range = camera_info["clipping_range"]
            
            if camera_info["parallel_projection"]:
                plotter.camera.parallel_projection = True
                plotter.camera.parallel_scale = camera_info["parallel_scale"]
            
            return plotter
        
        # First render: use magenta background to create mask
        # Magenta (255, 0, 255) won't appear in grayscale mesh rendering
        mask_plotter = create_plotter((255, 0, 255))
        mask_img = mask_plotter.screenshot(return_img=True)
        mask_plotter.close()
        
        # Create mask: pixels that are NOT magenta are mesh pixels
        # Check for exact magenta background color
        is_magenta = (
            (mask_img[..., 0] == 255) & 
            (mask_img[..., 1] == 0) & 
            (mask_img[..., 2] == 255)
        )
        mesh_mask = ~is_magenta
        
        # Second render: white background for actual image
        img_plotter = create_plotter("white")
        img = img_plotter.screenshot(return_img=True)
        img_plotter.close()
        
        # Post-process to constrain mesh pixel values to [pixel_min, pixel_max]
        # while keeping background white (255)
        img = self._remap_pixel_values(img, mesh_mask)
        
        # Center the mesh in the frame with equal borders on all sides
        img = self._center_mesh_in_frame(img, mesh_mask)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.location_dir / f"mesh_render_{timestamp}.png"
        
        # Save the processed image
        Image.fromarray(img).save(str(output_path))
        
        print(f"High-resolution image saved to: {output_path.absolute()}")
        print(f"  Resolution: {resolution[0]}x{resolution[1]}")
        print(f"  Mesh pixel range: [{self.params['pixel_min']}, {self.params['pixel_max']}]")
        
        # Restore the original downsampled DEM for interactive viewing
        self.dem_data = original_dem_data
        self.transform = original_transform
        print("Restored interactive-resolution DEM.")
    
    def _remap_pixel_values(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remap mesh pixel values to [pixel_min, pixel_max] range while keeping background white.
        
        Args:
            img: RGB image array from PyVista screenshot
            mask: Boolean array where True = mesh pixel, False = background
            
        Returns:
            Processed image with mesh values remapped
        """
        pixel_min = self.params["pixel_min"]
        pixel_max = self.params["pixel_max"]
        
        # Convert to grayscale for processing (mesh is rendered as grayscale anyway)
        # Use luminosity method: 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        
        if not np.any(mask):
            # No mesh pixels found, return white image
            return np.full_like(img, 255)
        
        # Get the actual value range of mesh pixels
        mesh_values = gray[mask]
        actual_min = mesh_values.min()
        actual_max = mesh_values.max()
        
        print(f"  Original mesh pixel range: [{actual_min:.1f}, {actual_max:.1f}]")
        
        # Remap mesh values from [actual_min, actual_max] to [pixel_min, pixel_max]
        # Avoid division by zero if all mesh pixels have the same value
        if actual_max - actual_min < 0.001:
            remapped = np.full_like(gray, (pixel_min + pixel_max) / 2)
        else:
            remapped = (gray - actual_min) / (actual_max - actual_min)
            remapped = remapped * (pixel_max - pixel_min) + pixel_min
        
        # Set background to white
        remapped[~mask] = 255.0
        
        # Clip and convert to uint8
        remapped = np.clip(remapped, 0, 255).astype(np.uint8)
        
        # Convert back to RGB (grayscale)
        result = np.stack([remapped, remapped, remapped], axis=-1)
        
        # Preserve alpha channel if present
        if img.shape[-1] == 4:
            result = np.concatenate([result, img[..., 3:4]], axis=-1)
        
        return result
    
    def _center_mesh_in_frame(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Center the mesh in the frame with equal borders on all sides.
        
        Args:
            img: RGB image array (already remapped)
            mask: Boolean array where True = mesh pixel, False = background
            
        Returns:
            Image with mesh centered, background filled with white
        """
        if not np.any(mask):
            return img
        
        # Find bounding box of mesh pixels
        rows_with_mesh = np.any(mask, axis=1)
        cols_with_mesh = np.any(mask, axis=0)
        
        row_indices = np.where(rows_with_mesh)[0]
        col_indices = np.where(cols_with_mesh)[0]
        
        top = row_indices[0]
        bottom = row_indices[-1]
        left = col_indices[0]
        right = col_indices[-1]
        
        # Calculate mesh dimensions and current center
        mesh_height = bottom - top + 1
        mesh_width = right - left + 1
        mesh_center_y = (top + bottom) / 2
        mesh_center_x = (left + right) / 2
        
        # Calculate image center
        img_height, img_width = img.shape[:2]
        img_center_y = img_height / 2
        img_center_x = img_width / 2
        bias_px = (self.vertical_bias_percent / 100.0) * img_height
        
        # Positive bias percent moves mesh upward in the frame.
        shift_y = int(round(img_center_y - mesh_center_y - bias_px))
        shift_x = int(round(img_center_x - mesh_center_x))
        
        print(f"  Mesh bounding box: ({left}, {top}) to ({right}, {bottom})")
        print(f"  Mesh size: {mesh_width}x{mesh_height}")
        print(f"  Vertical bias (%): {self.vertical_bias_percent}")
        print(f"  Centering shift: ({shift_x}, {shift_y})")
        
        # Create new white image
        centered = np.full_like(img, 255)
        
        # Calculate source and destination regions
        # Source region (from original image)
        src_y1 = max(0, -shift_y)
        src_y2 = min(img_height, img_height - shift_y)
        src_x1 = max(0, -shift_x)
        src_x2 = min(img_width, img_width - shift_x)
        
        # Destination region (in centered image)
        dst_y1 = max(0, shift_y)
        dst_y2 = min(img_height, img_height + shift_y)
        dst_x1 = max(0, shift_x)
        dst_x2 = min(img_width, img_width + shift_x)
        
        # Copy the content
        centered[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        
        return centered
        
    def _on_key_press(self):
        """Handle key press events without rendering inside the live VTK event loop."""
        self.save_requested = True
        self._pending_camera_info, self._pending_window_size = self._capture_current_view_state()
        print("Save/render queued. Press 'q' to close viewer and run export.")
    
    def run(self, auto_accept: bool = False, export_stl: bool = False):
        """Launch the interactive viewer or run a non-interactive save/render pass."""
        self.export_stl = export_stl

        # Create plotter with saved window size if available
        off_screen = auto_accept
        if self.saved_window_size is not None:
            self.plotter = pv.Plotter(
                lighting="none",
                window_size=self.saved_window_size,
                off_screen=off_screen,
            )
        else:
            self.plotter = pv.Plotter(lighting="none", off_screen=off_screen)
        self.plotter.set_background("white")
        
        # Create initial mesh and apply Z rotation
        self.mesh = self._create_mesh()
        self.mesh = self.mesh.rotate_z(
            self.params["z_rotation"], point=self.mesh.center, inplace=False
        )
        
        # Add mesh
        self.actor = self.plotter.add_mesh(
            self.mesh,
            color="white",
            smooth_shading=True,
            specular=0.0,
            diffuse=1.0,
            ambient=0.1,
        )
        
        # Add point light
        light_pos = self._calculate_light_position()
        self.light = pv.Light(
            position=light_pos,
            focal_point=self.mesh.center,
            color="white",
            intensity=1.0,
            positional=True,  # Point light
        )
        self.plotter.add_light(self.light)
        
        # Add ambient light for fill
        ambient_light = pv.Light(
            light_type="headlight",
            intensity=0.1,
        )
        self.plotter.add_light(ambient_light)
        
        # Set camera view
        if self.saved_camera is not None:
            # Restore saved camera position
            self.plotter.camera.position = self.saved_camera["position"]
            self.plotter.camera.focal_point = self.saved_camera["focal_point"]
            self.plotter.camera.up = self.saved_camera["view_up"]
            self.plotter.camera.view_angle = self.saved_camera["view_angle"]
            self.plotter.camera.clipping_range = self.saved_camera["clipping_range"]
            if self.saved_camera.get("parallel_projection"):
                self.plotter.camera.parallel_projection = True
                self.plotter.camera.parallel_scale = self.saved_camera["parallel_scale"]
            print("Restored camera position from saved settings.")
        else:
            # Default: looking down at terrain
            self.plotter.camera_position = "xy"
            self.plotter.reset_camera()

        if auto_accept:
            print("\nAuto-accept enabled: saving settings + rendering high-res image...\n")
            self._save_settings()
            self.plotter.close()
            return

        # Add sliders (left column)
        slider_y_positions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

        self.plotter.add_slider_widget(
            self._on_vertical_exaggeration,
            rng=[0.5, 10.0],
            value=self.params["vertical_exaggeration"],
            title="Vertical Exaggeration",
            pointa=(0.02, slider_y_positions[0]),
            pointb=(0.25, slider_y_positions[0]),
            style="modern",
        )

        self.plotter.add_slider_widget(
            self._on_smoothing,
            rng=[0.0, 10.0],
            value=self.params["smoothing_sigma"],
            title="Smoothing Sigma",
            pointa=(0.02, slider_y_positions[1]),
            pointb=(0.25, slider_y_positions[1]),
            style="modern",
        )

        self.plotter.add_slider_widget(
            self._on_light_elevation,
            rng=[0.0, 90.0],
            value=self.params["light_elevation"],
            title="Light Elevation",
            pointa=(0.02, slider_y_positions[2]),
            pointb=(0.25, slider_y_positions[2]),
            style="modern",
        )

        self.plotter.add_slider_widget(
            self._on_light_azimuth,
            rng=[0.0, 360.0],
            value=self.params["light_azimuth"],
            title="Light Azimuth",
            pointa=(0.02, slider_y_positions[3]),
            pointb=(0.25, slider_y_positions[3]),
            style="modern",
        )

        self.plotter.add_slider_widget(
            self._on_light_distance,
            rng=[100.0, 100000.0],
            value=self.params["light_distance"],
            title="Light Distance",
            pointa=(0.02, slider_y_positions[4]),
            pointb=(0.25, slider_y_positions[4]),
            style="modern",
        )

        self.plotter.add_slider_widget(
            self._on_z_rotation,
            rng=[0.0, 360.0],
            value=self.params["z_rotation"],
            title="Z Rotation",
            pointa=(0.02, slider_y_positions[5]),
            pointb=(0.25, slider_y_positions[5]),
            style="modern",
        )

        # Add key binding for saving settings
        self.plotter.add_key_event("p", self._on_key_press)

        # Add text instructions
        self.plotter.add_text(
            "Press 'p' to queue save+render, then 'q' to quit",
            position="lower_right",
            font_size=10,
        )

        # Show
        print("\nStarting interactive viewer...")
        print("  - Use mouse to rotate/pan/zoom")
        print("  - Adjust sliders to modify parameters")
        print("  - Press 'p' to queue save+render")
        print("  - Press 'q' to close viewer and run queued render")
        print("  - Press 'q' to quit\n")

        self.plotter.show()
        if self.save_requested:
            self._save_settings(
                camera_info=self._pending_camera_info,
                window_size=self._pending_window_size,
            )


def load_polygon_from_geojson(geojson_path: Path) -> list:
    """
    Load polygon coordinates from a GeoJSON file.
    
    Args:
        geojson_path: Path to the GeoJSON file
        
    Returns:
        List of (lon, lat) coordinate tuples
    """
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)
    
    first_feature = geojson_data["features"][0]
    polygon = shape(first_feature["geometry"])
    return list(polygon.exterior.coords)

def load_radius_from_geojson(polygon_coords: list) -> list:
    """
    Create a circular polygon from center point and radius point.
    
    Accounts for latitude-dependent scaling of longitude to produce
    a true circle in real-world coordinates.
    """
    # First two points: center and a point on the radius
    center = polygon_coords[0]  # (lon, lat)
    radius_point = polygon_coords[1]
    
    center_lon, center_lat = center
    
    # Conversion factors (meters per degree)
    deg_to_m_lat = 111320
    deg_to_m_lon = 111320 * np.cos(np.radians(center_lat))
    
    # Calculate radius in meters
    delta_lon = radius_point[0] - center_lon
    delta_lat = radius_point[1] - center_lat
    delta_x_m = delta_lon * deg_to_m_lon
    delta_y_m = delta_lat * deg_to_m_lat
    radius_m = np.sqrt(delta_x_m**2 + delta_y_m**2)
    
    # Generate circle points in meters, then convert back to degrees
    N = 100
    theta = np.linspace(0, 2 * np.pi, N)
    
    # Offsets in meters
    x_offset_m = radius_m * np.cos(theta)
    y_offset_m = radius_m * np.sin(theta)
    
    # Convert back to degrees (with proper scaling)
    lon = center_lon + x_offset_m / deg_to_m_lon
    lat = center_lat + y_offset_m / deg_to_m_lat
    
    return list(zip(lon, lat))


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D DEM mesh viewer with adjustable lighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    1. Create folder: /Volumes/sandisk1/data_cache/mountain_mesh_data/my_mountain/
    2. Paste GeoJSON polygon into: /Volumes/sandisk1/data_cache/mountain_mesh_data/my_mountain/polygon.json
    3. Run: python interactive_mesh.py my_mountain
        """
    )
    parser.add_argument(
        "location_name",
        nargs="?",
        default="mt-rainier",
        help="Name of the location folder inside /Volumes/sandisk1/data_cache/mountain_mesh_data/ (default: brighton)",
    )
    parser.add_argument(
        "method",
        nargs="?",
        default="radius",
        help="Method of defining the area: polygon, radius"
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Run non-interactively and immediately save settings + render output.",
    )
    parser.add_argument(
        "--export-stl",
        action="store_true",
        help="Also export the generated terrain mesh as STL during save/render.",
    )
    parser.add_argument(
        "--dem-file",
        type=str,
        default=None,
        help=(
            "DEM filename under the location folder (e.g. extracted_dem_denoised.tif) "
            "or absolute path; overrides mesh_settings dem_filename."
        ),
    )
    parser.add_argument(
        "--dem-path",
        type=str,
        default=None,
        metavar="ABSOLUTE_PATH",
        help=(
            "Full absolute path to any DEM GeoTIFF, regardless of the location folder "
            "(e.g. /Volumes/sandisk1/data_cache/mountain_mesh_data/aspen/extracted_dem_denoised.tif). "
            "Takes precedence over --dem-file and mesh_settings dem_filename."
        ),
    )
    parser.add_argument(
        "--vertical-bias-percent",
        type=float,
        default=0.0,
        help="Shift mesh vertically as percent of image height; positive moves mesh up.",
    )
    parser.add_argument(
        "--render-max-dimension",
        type=int,
        default=8192,
        help="Max pixels on the longest side of the rendered PNG.",
    )
    parser.add_argument(
        "--render-max-mesh-dimension",
        type=int,
        default=8000,
        help="Max DEM pixels on the longest side used for the render mesh.",
    )
    args = parser.parse_args()
    
    # Set up paths
    location_dir = BASE_DIR / args.location_name
    polygon_file = location_dir / "polygon.json"
    if args.dem_path:
        dem_file = Path(args.dem_path).expanduser().resolve()
    elif args.dem_file:
        raw = Path(args.dem_file)
        dem_file = raw if raw.is_absolute() else location_dir / raw
    else:
        dem_file = resolve_dem_path(location_dir)
    
    # Check if location folder exists
    if not location_dir.exists():
        print(f"Creating location folder: {location_dir}")
        location_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nNext steps:")
        print(f"  1. Paste your GeoJSON polygon into: {polygon_file}")
        print(f"  2. Run this script again: python interactive_mesh.py {args.location_name}")
        return 0
    
    # Check if polygon.json exists
    if not polygon_file.exists():
        print(f"Error: polygon.json not found at: {polygon_file}")
        print(f"\nPlease create polygon.json with your GeoJSON polygon data.")
        print(f"You can use https://geojson.io/ to create a polygon and copy the JSON.")
        return 1
    
    # Load polygon coordinates
    print(f"Loading polygon from: {polygon_file}")
    with open(polygon_file, "r") as f:
        geojson_data = json.load(f)
    polygon_coords = load_polygon_from_geojson(polygon_file)
    if args.method == "radius":
        polygon_coords = load_radius_from_geojson(polygon_coords)
    print(f"Loaded polygon with {len(polygon_coords)} vertices")
    
    # Check if DEM exists; only auto-fetch USGS mosaic for default extracted_dem.tif
    default_dem = location_dir / "extracted_dem.tif"
    if not dem_file.exists():
        if dem_file.resolve() != default_dem.resolve():
            print(f"Error: DEM not found: {dem_file}", file=sys.stderr)
            print(
                "  Create it (e.g. pitkin_dem_pipeline / pitkin_dem_postprocess) or omit dem_filename.",
                file=sys.stderr,
            )
            return 1
        print(f"\nDEM not found. Extracting DEM data with mosaic approach...")
        # Use mosaic extraction to get highest available resolution (1m > 3m > 10m)
        # max_dimension=4000 ensures the final DEM is manageable for interactive viewing
        result, _ = extract_dem_mosaic_from_geojson(
            geojson_data,
            str(default_dem),
            polygon_coords,
            resolutions=[1, 3, 10],  # Prioritize highest resolution
            max_dimension=4000,  # Limit to 4000px on longest side for interactive use
        )
        if result is None:
            print("Error: Failed to extract DEM data.")
            return 1
        dem_file = default_dem
    else:
        print(f"Using existing DEM: {dem_file}")
    
    # Launch viewer
    viewer = InteractiveMeshViewer(
        location_dir,
        polygon_coords,
        dem_path=dem_file,
        vertical_bias_percent=args.vertical_bias_percent,
        render_max_dimension=args.render_max_dimension,
        render_max_mesh_dimension=args.render_max_mesh_dimension,
    )
    viewer.run(auto_accept=args.auto_accept, export_stl=args.export_stl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
