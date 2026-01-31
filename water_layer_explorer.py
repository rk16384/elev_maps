"""
Interactive Water Layer Explorer

A GUI tool to explore and configure water layer settings for elevation maps.
Allows toggling FCodes, layers, and filters to quickly iterate on water configurations.
"""

import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import requests
import io
from urllib.parse import quote
import pandas as pd

from map_layers import StateBoundaryLayer
from map_data import MapData


# NHD Layer definitions
NHD_LAYERS = {
    4: "Flowline - Small Scale",
    5: "Flowline - Medium Scale", 
    6: "Flowline - Large Scale",
    7: "Area - Small Scale",
    8: "Area - Medium Scale",
    9: "Area - Large Scale",
    10: "Waterbody - Small Scale",
    11: "Waterbody - Small Scale HI/Pacific",
    12: "Waterbody - Large Scale",
}

# FCode definitions with categories
FCODE_DEFINITIONS = {
    # Oceans and Seas
    44500: ("SeaOcean", "Oceans/Seas"),
    44501: ("Sea/Ocean - Atlantic", "Oceans/Seas"),
    44502: ("Sea/Ocean - Pacific", "Oceans/Seas"),
    44503: ("Sea/Ocean - Arctic", "Oceans/Seas"),
    44504: ("Sea/Ocean - Indian", "Oceans/Seas"),
    # Lakes and Ponds
    39000: ("Lake/Pond", "Lakes/Ponds"),
    39001: ("Lake/Pond - Intermittent", "Lakes/Ponds"),
    39004: ("Lake/Pond - Perennial", "Lakes/Ponds"),
    39009: ("Lake/Pond - Avg Water Elev", "Lakes/Ponds"),
    39010: ("Lake/Pond - Normal Pool", "Lakes/Ponds"),
    39011: ("Lake/Pond - Date of Photography", "Lakes/Ponds"),
    39012: ("Lake/Pond - Spillway Elev", "Lakes/Ponds"),
    # Rivers and Streams
    46000: ("StreamRiver", "Rivers/Streams"),
    46003: ("StreamRiver - Intermittent", "Rivers/Streams"),
    46006: ("StreamRiver - Perennial", "Rivers/Streams"),
    46007: ("StreamRiver - Perennial Navigable", "Rivers/Streams"),
    # Coastal Features
    31200: ("BayInlet", "Coastal"),
    36400: ("Foreshore", "Coastal"),
    49300: ("Estuary", "Coastal"),
    56600: ("Coastline", "Coastal"),
    # Reservoirs
    43600: ("Reservoir", "Reservoirs"),
    43615: ("Reservoir - Perennial", "Reservoirs"),
    43617: ("Reservoir - Avg Water Elev", "Reservoirs"),
    43621: ("Reservoir - Normal Pool", "Reservoirs"),
    # Canals/Ditches
    33600: ("CanalDitch", "Canals"),
    33601: ("CanalDitch - Aqueduct", "Canals"),
    # Swamps/Marshes
    46600: ("SwampMarsh", "Wetlands"),
    46601: ("SwampMarsh - Intermittent", "Wetlands"),
    46602: ("SwampMarsh - Perennial", "Wetlands"),
}


class WaterLayerExplorer:
    def __init__(self, location_name: str):
        self.location_name = location_name
        self.water_features = None
        self.state_polygon = None
        self.state_gdf = None
        
        # Load location config
        self.config_path = os.path.join(os.path.dirname(__file__), "location_configs.json")
        with open(self.config_path, 'r') as f:
            self.all_configs = json.load(f)
        
        self.location_config = self.all_configs.get(location_name, self.all_configs.get("default", {}))
        
        # Initialize state for toggles
        self.layer_vars = {}
        self.fcode_vars = {}
        self.filter_vars = {}
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title(f"Water Layer Explorer - {location_name}")
        self.root.geometry("1400x900")
        
        self._setup_gui()
        self._fetch_initial_data()
        
    def _fetch_initial_data(self):
        """Fetch state boundary and all water features once."""
        print(f"Fetching data for {self.location_name}...")
        
        # Get state boundary
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_root = os.path.join(base_dir, "data_cache")
        state_layer = StateBoundaryLayer(cache_dir=cache_root, location_name=self.location_name.lower())
        
        # Get the state polygon
        map_data = MapData(location_name=self.location_name)
        self.state_polygon = map_data.location_polygon
        self.state_gdf = gpd.GeoDataFrame(geometry=[self.state_polygon], crs="EPSG:4326")
        
        # Fetch ALL water features (no filtering at fetch time)
        self._fetch_all_water_features()
        
        # Also fetch Great Lakes by name (they may not match FCodes)
        self._fetch_great_lakes_by_name()
        
        # Initial plot
        self._update_plot()
        
    def _fetch_all_water_features(self, min_area_km2: float = 0.5):
        """Fetch water features from all layers with all FCodes.
        
        Args:
            min_area_km2: Minimum area in km² for polygon layers (server-side filter).
                          Set to 0 to fetch all features (slow).
        """
        print(f"Fetching water features (min area: {min_area_km2} km²)...")
        bbox = self.state_polygon.bounds
        
        all_features = []
        all_fcodes = list(FCODE_DEFINITIONS.keys())
        all_layers = list(NHD_LAYERS.keys())
        
        # Polygon layers that support AreaSqKm filter
        polygon_layers = [7, 8, 9, 10, 11, 12]
        # Line layers (flowlines) - no area filter
        line_layers = [4, 5, 6]
        
        for layer in all_layers:
            try:
                # Build WHERE clause for all FCodes
                fcode_str = ','.join(map(str, all_fcodes))
                base_where = f"FCode IN ({fcode_str})"
                
                # Add area filter for polygon layers
                if layer in polygon_layers and min_area_km2 > 0:
                    where_clause = f"({base_where}) AND (AreaSqKm >= {min_area_km2})"
                else:
                    where_clause = base_where
                
                encoded_where = quote(where_clause)
                
                # Paginate to get all features (API limit is 2000 per request)
                layer_features = []
                offset = 0
                batch_size = 2000
                
                while True:
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
                        f"where={encoded_where}&"
                        f"resultOffset={offset}&"
                        f"resultRecordCount={batch_size}"
                    )
                    
                    response = requests.get(nhd_url, timeout=60)
                    if response.status_code == 200 and response.text:
                        try:
                            features = gpd.read_file(io.StringIO(response.text))
                            if not features.empty:
                                layer_features.append(features)
                                print(f"  Layer {layer}: fetched {len(features)} features (offset {offset})")
                                
                                # If we got less than batch_size, we've reached the end
                                if len(features) < batch_size:
                                    break
                                offset += batch_size
                            else:
                                break
                        except Exception as e:
                            print(f"  Layer {layer}: parse error at offset {offset} - {e}")
                            break
                    else:
                        break
                
                # Combine all pages for this layer
                if layer_features:
                    combined = pd.concat(layer_features, ignore_index=True)
                    combined['NHD_LAYER'] = layer
                    all_features.append(combined)
                    print(f"  Layer {layer} total: {len(combined)} features")
                    
            except Exception as e:
                print(f"  Layer {layer}: fetch error - {e}")
        
        if all_features:
            self.water_features = pd.concat(all_features, ignore_index=True)
            # Ensure FCODE column exists and is numeric
            if 'FCODE' in self.water_features.columns:
                self.water_features['FCODE'] = pd.to_numeric(self.water_features['FCODE'], errors='coerce')
            print(f"Total water features fetched: {len(self.water_features)}")
            
            # Get unique FCodes found
            if 'FCODE' in self.water_features.columns:
                found_fcodes = self.water_features['FCODE'].dropna().unique()
                print(f"FCodes found: {sorted(found_fcodes)}")
        else:
            self.water_features = gpd.GeoDataFrame()
            print("No water features found")
    
    def _fetch_great_lakes_by_name(self):
        """Fetch Great Lakes by name (they may not match standard FCodes)."""
        print("Fetching Great Lakes by name...")
        bbox = self.state_polygon.bounds
        
        great_lakes_names = [
            "Lake Superior",
            "Lake Michigan",
            "Lake Huron", 
            "Lake Erie",
            "Lake Ontario",
        ]
        
        # Query layers 10, 11 (small scale waterbodies)
        layers = [10, 11]
        all_features = []
        
        # Build WHERE clause: search by name for Great Lakes
        name_conditions = " OR ".join([f"GNIS_Name LIKE '%{name}%'" for name in great_lakes_names])
        where_clause = f"({name_conditions})"
        encoded_where = quote(where_clause)
        
        for layer in layers:
            try:
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
                
                response = requests.get(nhd_url, timeout=120)
                if response.status_code == 200 and response.text:
                    try:
                        features = gpd.read_file(io.StringIO(response.text))
                        if not features.empty:
                            features['NHD_LAYER'] = layer
                            features['SOURCE'] = 'Great Lakes (by name)'
                            all_features.append(features)
                            print(f"  Layer {layer}: {len(features)} Great Lakes features")
                            if 'GNIS_Name' in features.columns:
                                print(f"    Names: {features['GNIS_Name'].unique().tolist()}")
                    except Exception as e:
                        print(f"  Layer {layer}: parse error - {e}")
            except Exception as e:
                print(f"  Layer {layer}: fetch error - {e}")
        
        if all_features:
            great_lakes_gdf = pd.concat(all_features, ignore_index=True)
            if 'FCODE' in great_lakes_gdf.columns:
                great_lakes_gdf['FCODE'] = pd.to_numeric(great_lakes_gdf['FCODE'], errors='coerce')
            
            # Merge with existing water features (avoid duplicates by geometry)
            if self.water_features is not None and not self.water_features.empty:
                # Simple concat - duplicates will be handled by the user visually
                self.water_features = pd.concat([self.water_features, great_lakes_gdf], ignore_index=True)
                print(f"Total water features after adding Great Lakes: {len(self.water_features)}")
            else:
                self.water_features = great_lakes_gdf
        else:
            print("No Great Lakes found in area")
    
    def _setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (controls) with scrollbar
        left_container = ttk.Frame(main_frame, width=350)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_container.pack_propagate(False)
        
        # Canvas for scrolling
        canvas = tk.Canvas(left_container)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        self.left_panel = ttk.Frame(canvas)
        
        self.left_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.left_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel (map)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup controls
        self._setup_layer_controls()
        self._setup_fcode_controls()
        self._setup_filter_controls()
        self._setup_name_filter()
        self._setup_export_button()
        
        # Setup matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_layer_controls(self):
        """Setup NHD layer toggle controls."""
        layer_frame = ttk.LabelFrame(self.left_panel, text="NHD Layers", padding=5)
        layer_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for layer_id, layer_name in NHD_LAYERS.items():
            var = tk.BooleanVar(value=True)
            self.layer_vars[layer_id] = var
            cb = ttk.Checkbutton(
                layer_frame, 
                text=f"{layer_id}: {layer_name}",
                variable=var,
                command=self._update_plot
            )
            cb.pack(anchor=tk.W)
    
    def _setup_fcode_controls(self):
        """Setup FCode toggle controls grouped by category."""
        fcode_frame = ttk.LabelFrame(self.left_panel, text="FCodes", padding=5)
        fcode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Group by category
        categories = {}
        for fcode, (name, category) in FCODE_DEFINITIONS.items():
            if category not in categories:
                categories[category] = []
            categories[category].append((fcode, name))
        
        for category, fcodes in categories.items():
            cat_frame = ttk.LabelFrame(fcode_frame, text=category, padding=2)
            cat_frame.pack(fill=tk.X, pady=2)
            
            for fcode, name in fcodes:
                var = tk.BooleanVar(value=True)
                self.fcode_vars[fcode] = var
                cb = ttk.Checkbutton(
                    cat_frame,
                    text=f"{fcode}: {name}",
                    variable=var,
                    command=self._update_plot
                )
                cb.pack(anchor=tk.W)
    
    def _setup_filter_controls(self):
        """Setup filter sliders."""
        filter_frame = ttk.LabelFrame(self.left_panel, text="Filters", padding=5)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Min area slider
        ttk.Label(filter_frame, text="Min Area (km²):").pack(anchor=tk.W)
        self.min_area_var = tk.DoubleVar(value=0.0)
        min_area_slider = ttk.Scale(
            filter_frame, 
            from_=0, 
            to=100, 
            variable=self.min_area_var,
            command=lambda _: self._update_plot()
        )
        min_area_slider.pack(fill=tk.X)
        self.min_area_label = ttk.Label(filter_frame, text="0.0 km²")
        self.min_area_label.pack(anchor=tk.W)
        
        # Max area slider
        ttk.Label(filter_frame, text="Max Area (km²):").pack(anchor=tk.W, pady=(10, 0))
        self.max_area_var = tk.DoubleVar(value=1000000.0)
        max_area_slider = ttk.Scale(
            filter_frame,
            from_=0,
            to=10000,
            variable=self.max_area_var,
            command=lambda _: self._update_plot()
        )
        max_area_slider.pack(fill=tk.X)
        self.max_area_label = ttk.Label(filter_frame, text="1000000.0 km²")
        self.max_area_label.pack(anchor=tk.W)
        
        # River width slider
        ttk.Label(filter_frame, text="River Max Width (km):").pack(anchor=tk.W, pady=(10, 0))
        self.river_width_var = tk.DoubleVar(value=10.0)
        river_width_slider = ttk.Scale(
            filter_frame,
            from_=0.1,
            to=50,
            variable=self.river_width_var,
            command=lambda _: self._update_plot()
        )
        river_width_slider.pack(fill=tk.X)
        self.river_width_label = ttk.Label(filter_frame, text="10.0 km")
        self.river_width_label.pack(anchor=tk.W)
        
        # Aspect ratio for river detection
        ttk.Label(filter_frame, text="River Aspect Ratio:").pack(anchor=tk.W, pady=(10, 0))
        self.aspect_ratio_var = tk.DoubleVar(value=3.0)
        aspect_slider = ttk.Scale(
            filter_frame,
            from_=1.5,
            to=10,
            variable=self.aspect_ratio_var,
            command=lambda _: self._update_plot()
        )
        aspect_slider.pack(fill=tk.X)
        self.aspect_label = ttk.Label(filter_frame, text="3.0")
        self.aspect_label.pack(anchor=tk.W)
        
    def _setup_name_filter(self):
        """Setup name filter text input."""
        name_frame = ttk.LabelFrame(self.left_panel, text="Name Filter", padding=5)
        name_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(name_frame, text="Filter by name (comma-separated):").pack(anchor=tk.W)
        self.name_filter_var = tk.StringVar(value="")
        name_entry = ttk.Entry(name_frame, textvariable=self.name_filter_var)
        name_entry.pack(fill=tk.X)
        name_entry.bind('<Return>', lambda _: self._update_plot())
        
        ttk.Button(name_frame, text="Apply", command=self._update_plot).pack(pady=5)
        
        # Show matching features count
        self.match_count_label = ttk.Label(name_frame, text="")
        self.match_count_label.pack(anchor=tk.W)
        
    def _setup_export_button(self):
        """Setup export configuration button."""
        export_frame = ttk.LabelFrame(self.left_panel, text="Export", padding=5)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            export_frame,
            text="Export Config to location_configs.json",
            command=self._export_config
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            export_frame,
            text="Export Filtered Features as GeoJSON",
            command=self._export_geojson
        ).pack(fill=tk.X, pady=5)
        
        # Feature count display
        self.feature_count_label = ttk.Label(export_frame, text="Visible: 0 features")
        self.feature_count_label.pack(anchor=tk.W, pady=5)
    
    def _get_filtered_features(self):
        """Get water features based on current filter settings."""
        import time
        t0 = time.time()
        
        if self.water_features is None or self.water_features.empty:
            return gpd.GeoDataFrame()
        
        print(f"Filtering {len(self.water_features)} features...")
        filtered = self.water_features.copy()
        
        # Filter by enabled layers
        enabled_layers = [lid for lid, var in self.layer_vars.items() if var.get()]
        if 'NHD_LAYER' in filtered.columns:
            filtered = filtered[filtered['NHD_LAYER'].isin(enabled_layers)]
        print(f"  After layer filter: {len(filtered)} ({time.time()-t0:.2f}s)")
        
        # Filter by enabled FCodes
        enabled_fcodes = [fc for fc, var in self.fcode_vars.items() if var.get()]
        if 'FCODE' in filtered.columns:
            filtered = filtered[filtered['FCODE'].isin(enabled_fcodes)]
        print(f"  After FCode filter: {len(filtered)} ({time.time()-t0:.2f}s)")
        
        # Filter by area
        min_area = self.min_area_var.get()
        max_area = self.max_area_var.get()
        
        if not filtered.empty:
            # Calculate area in km² (vectorized, should be fast)
            print(f"  Calculating areas...")
            areas_km2 = filtered.geometry.area * (111.32 ** 2)
            filtered = filtered[(areas_km2 >= min_area) & (areas_km2 <= max_area)]
        print(f"  After area filter: {len(filtered)} ({time.time()-t0:.2f}s)")
        
        # Filter by name if specified
        name_filter = self.name_filter_var.get().strip()
        if name_filter and not filtered.empty:
            print(f"  Applying name filter: '{name_filter}'...")
            names = [n.strip().lower() for n in name_filter.split(',')]
            name_col = None
            for col in ['GNIS_NAME', 'GNIS_Name', 'NAME', 'Name', 'name']:
                if col in filtered.columns:
                    name_col = col
                    break
            
            if name_col:
                # Use vectorized string operations instead of apply
                name_series = filtered[name_col].fillna('').str.lower()
                mask = name_series.str.contains('|'.join(names), regex=True, na=False)
                filtered = filtered[mask]
            print(f"  After name filter: {len(filtered)} ({time.time()-t0:.2f}s)")
        
        print(f"  Filter complete: {len(filtered)} features ({time.time()-t0:.2f}s)")
        return filtered
    
    def _update_plot(self):
        """Update the map plot with current filter settings."""
        import time
        t0 = time.time()
        print("Updating plot...")
        
        self.ax.clear()
        
        # Update slider labels
        self.min_area_label.config(text=f"{self.min_area_var.get():.1f} km²")
        self.max_area_label.config(text=f"{self.max_area_var.get():.1f} km²")
        self.river_width_label.config(text=f"{self.river_width_var.get():.1f} km")
        self.aspect_label.config(text=f"{self.aspect_ratio_var.get():.1f}")
        
        # Plot state boundary
        if self.state_gdf is not None:
            self.state_gdf.boundary.plot(ax=self.ax, color='black', linewidth=1.5)
        print(f"  State boundary plotted ({time.time()-t0:.2f}s)")
        
        # Get filtered features
        filtered = self._get_filtered_features()
        
        # Plot water features
        if not filtered.empty:
            print(f"  Plotting {len(filtered)} water features...")
            # Color by category - assign colors first, then plot once
            if 'FCODE' in filtered.columns:
                category_colors = {
                    "Oceans/Seas": "#1f77b4",
                    "Lakes/Ponds": "#2ca02c",
                    "Rivers/Streams": "#17becf",
                    "Coastal": "#9467bd",
                    "Reservoirs": "#8c564b",
                    "Canals": "#e377c2",
                    "Wetlands": "#7f7f7f",
                }
                
                # Assign colors based on FCode category
                def get_color(fcode):
                    if fcode in FCODE_DEFINITIONS:
                        _, category = FCODE_DEFINITIONS[fcode]
                        return category_colors.get(category, "#1f77b4")
                    return "#1f77b4"
                
                colors = filtered['FCODE'].apply(get_color)
                
                # Plot all at once with colors
                for color in colors.unique():
                    subset = filtered[colors == color]
                    subset.plot(ax=self.ax, color=color, alpha=0.6, edgecolor='darkblue', linewidth=0.3)
                    print(f"    Plotted {len(subset)} features with color {color}")
            else:
                filtered.plot(ax=self.ax, color='blue', alpha=0.6)
            print(f"  Features plotted ({time.time()-t0:.2f}s)")
        
        # Set bounds to state
        if self.state_polygon:
            bounds = self.state_polygon.bounds
            padding = 0.1
            x_pad = (bounds[2] - bounds[0]) * padding
            y_pad = (bounds[3] - bounds[1]) * padding
            self.ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
            self.ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)
        
        self.ax.set_aspect('equal')
        self.ax.set_title(f"{self.location_name} - {len(filtered)} water features")
        
        # Update feature count
        self.feature_count_label.config(text=f"Visible: {len(filtered)} features")
        
        # Update name match count
        name_filter = self.name_filter_var.get().strip()
        if name_filter:
            self.match_count_label.config(text=f"Name matches: {len(filtered)}")
        else:
            self.match_count_label.config(text="")
        
        print(f"  Drawing canvas...")
        self.canvas.draw()
        print(f"  Plot update complete ({time.time()-t0:.2f}s)")
    
    def _export_config(self):
        """Export current settings to location_configs.json."""
        # Build water config
        enabled_fcodes = [fc for fc, var in self.fcode_vars.items() if var.get()]
        enabled_layers = [lid for lid, var in self.layer_vars.items() if var.get()]
        
        water_config = {
            "include_fcodes": enabled_fcodes,
            "include_layers": enabled_layers,
            "min_area_km2": round(self.min_area_var.get(), 2),
            "max_area_km2": round(self.max_area_var.get(), 2),
            "river_max_width_km": round(self.river_width_var.get(), 2),
            "river_aspect_ratio": round(self.aspect_ratio_var.get(), 2),
        }
        
        name_filter = self.name_filter_var.get().strip()
        if name_filter:
            water_config["include_names"] = [n.strip() for n in name_filter.split(',')]
        
        # Update location config
        if self.location_name not in self.all_configs:
            self.all_configs[self.location_name] = {}
        
        self.all_configs[self.location_name]["water_layer_config"] = water_config
        
        # Save to file
        with open(self.config_path, 'w') as f:
            json.dump(self.all_configs, f, indent=4)
        
        messagebox.showinfo("Export Complete", f"Water config saved to location_configs.json for {self.location_name}")
    
    def _export_geojson(self):
        """Export filtered features as GeoJSON."""
        filtered = self._get_filtered_features()
        if filtered.empty:
            messagebox.showwarning("No Features", "No features to export with current filters.")
            return
        
        output_path = os.path.join(
            os.path.dirname(__file__),
            "output",
            f"{self.location_name}_filtered_water.geojson"
        )
        filtered.to_file(output_path, driver='GeoJSON')
        messagebox.showinfo("Export Complete", f"Exported {len(filtered)} features to:\n{output_path}")
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    import sys
    # if len(sys.argv) < 2:
    #     print("Usage: python water_layer_explorer.py <location_name>")
    #     print("Example: python water_layer_explorer.py Minnesota")
    #     sys.exit(1)
    
    location_name = "Minnesota"
    app = WaterLayerExplorer(location_name)
    app.run()


if __name__ == "__main__":
    main()

