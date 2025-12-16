"""
Interactive 3D DEM mesh viewer with adjustable lighting and camera controls.

Usage:
    python interactive_mesh.py <location_name>
    
    First, create a folder: mountain_mesh_data/<location_name>/
    Then paste your GeoJSON polygon into: mountain_mesh_data/<location_name>/polygon.json
    
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
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape

from extract_dem_data import extract_dem_from_geojson


# Base directory for all location data
BASE_DIR = Path(__file__).parent / "mountain_mesh_data"


class InteractiveMeshViewer:
    def __init__(self, location_dir: Path, polygon_coords: list):
        self.location_dir = location_dir
        self.polygon_coords = polygon_coords
        
        self.dem_path = location_dir / "extracted_dem.tif"
        self.settings_file = location_dir / "mesh_settings.json"
        
        self.dem_data = None
        self.transform = None
        self.mesh = None
        self.plotter = None
        self.light = None
        self.actor = None
        
        # Default parameters
        self.params = {
            "vertical_exaggeration": 0.5,
            "smoothing_sigma": 1.5,
            "light_elevation": 45.0,
            "light_azimuth": 315.0,
            "light_distance": 10000.0,
        }
        
        # Stored camera settings (loaded from file)
        self.saved_camera = None
        
        # Load DEM data
        self._load_dem()
        
        # Load previous settings if available
        self._load_settings()
        
    def _load_dem(self):
        """Load DEM data from file."""
        with rasterio.open(self.dem_path) as src:
            self.dem_data = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.bounds = src.bounds
            self.crs = src.crs
            
        # Handle nodata values
        self.dem_data = np.nan_to_num(self.dem_data, nan=0.0)
        
        print(f"Loaded DEM: {self.dem_path}")
        print(f"  Shape: {self.dem_data.shape}")
        print(f"  Elevation range: {np.nanmin(self.dem_data):.1f} to {np.nanmax(self.dem_data):.1f}")
    
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
                
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
    
    def _create_polygon_mask(self) -> np.ndarray:
        """Create a boolean mask from polygon coordinates."""
        from rasterio.features import geometry_mask
        from shapely.geometry import Polygon
        
        if self.polygon_coords is None:
            return np.ones(self.dem_data.shape, dtype=bool)
        
        polygon = Polygon(self.polygon_coords)
        
        # Create mask using rasterio (True = inside polygon)
        mask = ~geometry_mask(
            [polygon],
            out_shape=self.dem_data.shape,
            transform=self.transform,
            invert=False
        )
        return mask
        
    def _create_mesh(self):
        """Create PyVista mesh from DEM data with current smoothing."""
        # Apply gaussian smoothing
        smoothed = gaussian_filter(self.dem_data, sigma=self.params["smoothing_sigma"])
        
        # Create coordinate grids
        nrows, ncols = smoothed.shape
        
        # Use pixel coordinates scaled to preserve aspect ratio
        x = np.linspace(0, ncols, ncols)
        y = np.linspace(0, nrows, nrows)
        x, y = np.meshgrid(x, y)
        
        # Flip y to match geographic orientation (north up)
        y = np.flipud(y)
        
        # Apply vertical exaggeration
        z = smoothed * self.params["vertical_exaggeration"]
        
        # Apply polygon mask - set outside values to NaN
        if self.polygon_coords is not None:
            mask = self._create_polygon_mask()
            z = np.where(mask, z, np.nan)
        
        # Create structured grid
        mesh = pv.StructuredGrid(x, y, z)
        
        # Extract only valid (non-NaN) cells if polygon trimming is active
        if self.polygon_coords is not None:
            # Get cell validity based on points
            point_valid = ~np.isnan(mesh.points[:, 2])
            mesh = mesh.extract_points(point_valid)
        
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
        
        # Create new mesh
        self.mesh = self._create_mesh()
        
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
    
    def _save_settings(self):
        """Save current settings and camera position to file and print to terminal."""
        # Get camera information
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
        
        # Full settings dict
        settings = {
            "dem_path": str(self.dem_path),
            "parameters": self.params.copy(),
            "camera": camera_info,
            "timestamp": datetime.now().isoformat(),
        }
        
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
        print("=" * 60 + "\n")
        
        # Save to file
        with open(self.settings_file, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"Settings saved to: {self.settings_file.absolute()}")
        
        # Render high-resolution image
        self._render_high_res(camera_info)
    
    def _render_high_res(self, camera_info: dict, max_dimension: int = 8192):
        """
        Render the current view at high resolution and save to file.
        
        Args:
            camera_info: Dictionary containing camera settings
            max_dimension: Maximum dimension (width or height) for output
        """
        print("\nRendering high-resolution image...")
        
        # Get current window aspect ratio to match the interactive view
        current_window_size = self.plotter.window_size
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
        
        # Create off-screen plotter with matching aspect ratio
        plotter = pv.Plotter(off_screen=True, lighting="none", window_size=resolution)
        plotter.set_background("white")
        
        # Recreate mesh with current settings
        mesh = self._create_mesh()
        
        # Add mesh
        plotter.add_mesh(
            mesh,
            color="white",
            smooth_shading=True,
            specular=0.0,
            diffuse=1.0,
            ambient=0.1,
        )
        
        # Add point light with current settings
        light_pos = self._calculate_light_position()
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
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.location_dir / f"mesh_render_{timestamp}.png"
        
        # Take screenshot at high resolution
        plotter.screenshot(
            str(output_path),
            return_img=False,
        )
        
        plotter.close()
        
        print(f"High-resolution image saved to: {output_path.absolute()}")
        print(f"  Resolution: {resolution[0]}x{resolution[1]}")
        
    def _on_key_press(self):
        """Handle key press events."""
        self._save_settings()
    
    def run(self):
        """Launch the interactive viewer."""
        # Create plotter
        self.plotter = pv.Plotter(lighting="none")
        self.plotter.set_background("white")
        
        # Create initial mesh
        self.mesh = self._create_mesh()
        
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
        
        # Add sliders
        slider_y_positions = [0.9, 0.8, 0.7, 0.6, 0.5]
        
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
        
        # Add key binding for saving settings
        self.plotter.add_key_event("p", self._on_key_press)
        
        # Add text instructions
        self.plotter.add_text(
            "Press 'p' to save settings + render | 'q' to quit",
            position="lower_right",
            font_size=10,
        )
        
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
        
        # Show
        print("\nStarting interactive viewer...")
        print("  - Use mouse to rotate/pan/zoom")
        print("  - Adjust sliders to modify parameters")
        print("  - Press 'p' to save settings and render high-res image")
        print("  - Press 'q' to quit\n")
        
        self.plotter.show()


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


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D DEM mesh viewer with adjustable lighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    1. Create folder: mountain_mesh_data/my_mountain/
    2. Paste GeoJSON polygon into: mountain_mesh_data/my_mountain/polygon.json
    3. Run: python interactive_mesh.py my_mountain
        """
    )
    parser.add_argument(
        "location_name",
        nargs="?",
        default="breck",
        help="Name of the location folder inside mountain_mesh_data/ (default: brighton)",
    )
    args = parser.parse_args()
    
    # Set up paths
    location_dir = BASE_DIR / args.location_name
    polygon_file = location_dir / "polygon.json"
    dem_file = location_dir / "extracted_dem.tif"
    
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
    print(f"Loaded polygon with {len(polygon_coords)} vertices")
    
    # Check if DEM exists, if not extract it
    if not dem_file.exists():
        print(f"\nDEM not found. Extracting DEM data...")
        result, _ = extract_dem_from_geojson(geojson_data, str(dem_file))
        if result is None:
            print("Error: Failed to extract DEM data.")
            return 1
    else:
        print(f"Using existing DEM: {dem_file}")
    
    # Launch viewer
    viewer = InteractiveMeshViewer(location_dir, polygon_coords)
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
