#!/usr/bin/env python3
"""
Debug script to visualize tile coverage for a specific location.
Shows which tiles have data and which are missing/unavailable.
"""

import os
import sys
import glob
import re
from PIL import Image, ImageDraw, ImageFont
import json
import rioxarray
import numpy as np

# Add this import at the top of debug_map.py
from map_layers import StateBoundaryLayer
import geopandas as gpd
from shapely.geometry import Point

def get_state_boundary(location_name):
    """
    Get the actual state boundary polygon for a location.
    Returns the state polygon and its bounds.
    """
    try:
        # Initialize the StateBoundaryLayer
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
        state_layer = StateBoundaryLayer(cache_dir=cache_dir, location_name=location_name)
        
        # Get the state polygon (land-only for better visualization)
        state_polygon = state_layer.get_state_polygon(location_name, land_only=True)
        
        # Get bounds from the polygon
        bounds = state_polygon.bounds  # (minx, miny, maxx, maxy)
        
        return state_polygon, bounds
        
    except Exception as e:
        print(f"Error getting state boundary for {location_name}: {e}")
        # Fallback to rectangular bounds
        return None, get_location_bounds(location_name)

def validate_cached_tile(tile_info):
    """
    Validate a cached tile to ensure it contains valid elevation data.
    Returns (is_valid, validation_info)
    """
    try:
        # Load the DEM data
        with rioxarray.open_rasterio(tile_info['file_path']) as dem_data:
            data_values = dem_data.values.squeeze()
            
            # Basic shape validation
            if data_values.ndim < 2 or min(data_values.shape) < 2:
                return False, {"error": "Invalid data shape", "shape": data_values.shape}
            
            # Check for valid data (not all NaN or nodata)
            valid_pixels = ~np.isnan(data_values)
            valid_count = np.count_nonzero(valid_pixels)
            total_pixels = data_values.size
            valid_ratio = valid_count / total_pixels
            
            if valid_ratio < 0.01:  # Less than 1% valid data
                return False, {
                    "error": "Insufficient valid data",
                    "valid_ratio": valid_ratio,
                    "valid_pixels": valid_count,
                    "total_pixels": total_pixels
                }
            
            # Get elevation statistics from valid data
            valid_data = data_values[valid_pixels]
            min_elev = np.min(valid_data)
            max_elev = np.max(valid_data)
            elevation_range = max_elev - min_elev
            mean_elev = np.mean(valid_data)
            
            # Check for reasonable elevation range
            if elevation_range < 0.1:  # Less than 0.1m range
                return False, {
                    "error": "Terrain too flat",
                    "elevation_range": elevation_range,
                    "min_elev": min_elev,
                    "max_elev": max_elev
                }
            
            # Check for suspiciously constant data
            if len(valid_data) > 10 and np.std(valid_data) < 0.01:
                return False, {
                    "error": "Data appears constant",
                    "std": np.std(valid_data),
                    "elevation_range": elevation_range
                }
            
            # Check for reasonable elevation values (not ocean floor or mountain peaks)
            if min_elev < -1000 or max_elev > 50000:  # Unreasonable elevation values
                return False, {
                    "error": "Unreasonable elevation values",
                    "min_elev": min_elev,
                    "max_elev": max_elev
                }
            
            return True, {
                "valid_ratio": valid_ratio,
                "min_elev": min_elev,
                "max_elev": max_elev,
                "elevation_range": elevation_range,
                "mean_elev": mean_elev,
                "std_elev": np.std(valid_data),
                "shape": data_values.shape
            }
            
    except Exception as e:
        return False, {"error": f"Failed to load tile: {str(e)}"}

def validate_all_cached_tiles(cache_dir, location_name):
    """
    Validate all cached tiles for a location.
    """
    cached_tiles = load_cached_tiles(cache_dir, location_name)
    
    print(f"\nVALIDATING {len(cached_tiles)} CACHED TILES")
    print("=" * 50)
    
    valid_tiles = []
    invalid_tiles = []
    
    for i, tile in enumerate(cached_tiles):
        print(f"Validating tile {i+1}/{len(cached_tiles)}: {tile['bounds']}")
        
        is_valid, validation_info = validate_cached_tile(tile)
        
        if is_valid:
            valid_tiles.append((tile, validation_info))
            print(f"  ✓ VALID - Range: {validation_info['min_elev']:.1f}m to {validation_info['max_elev']:.1f}m")
        else:
            invalid_tiles.append((tile, validation_info))
            print(f"  ✗ INVALID - {validation_info.get('error', 'Unknown error')}")
            if 'valid_ratio' in validation_info:
                print(f"    Valid data: {validation_info['valid_ratio']:.1%}")
            if 'elevation_range' in validation_info:
                print(f"    Elevation range: {validation_info['elevation_range']:.2f}m")
    
    # Summary
    print(f"\nVALIDATION SUMMARY")
    print("=" * 30)
    print(f"Total tiles: {len(cached_tiles)}")
    print(f"Valid tiles: {len(valid_tiles)} ({len(valid_tiles)/len(cached_tiles)*100:.1f}%)")
    print(f"Invalid tiles: {len(invalid_tiles)} ({len(invalid_tiles)/len(cached_tiles)*100:.1f}%)")
    
    if invalid_tiles:
        print(f"\nINVALID TILE DETAILS:")
        for tile, info in invalid_tiles:
            print(f"  {tile['bounds']} ({tile['resolution']}m): {info.get('error', 'Unknown error')}")
    
    # Analyze valid tile statistics
    if valid_tiles:
        print(f"\nVALID TILE STATISTICS:")
        elevations = [info['mean_elev'] for _, info in valid_tiles]
        ranges = [info['elevation_range'] for _, info in valid_tiles]
        
        print(f"  Mean elevation: {np.mean(elevations):.1f}m (range: {np.min(elevations):.1f}m to {np.max(elevations):.1f}m)")
        print(f"  Average elevation range: {np.mean(ranges):.1f}m")
        print(f"  Resolution distribution:")
        
        res_counts = {}
        for tile, _ in valid_tiles:
            res = tile['resolution']
            res_counts[res] = res_counts.get(res, 0) + 1
        
        for res in sorted(res_counts.keys()):
            print(f"    {res}m: {res_counts[res]} tiles")
    
    return valid_tiles, invalid_tiles

def create_validation_report(location_name, output_file="tile_validation_report.txt"):
    """
    Create a detailed validation report for all cached tiles.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
    valid_tiles, invalid_tiles = validate_all_cached_tiles(cache_dir, location_name)
    
    # Save detailed report
    elevation_dem_dir = os.path.join(cache_dir, location_name, "elevation_dem")
    report_path = os.path.join(elevation_dem_dir, output_file)
    
    with open(report_path, 'w') as f:
        f.write(f"TILE VALIDATION REPORT FOR {location_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total tiles: {len(valid_tiles) + len(invalid_tiles)}\n")
        f.write(f"Valid tiles: {len(valid_tiles)}\n")
        f.write(f"Invalid tiles: {len(invalid_tiles)}\n")
        f.write(f"Success rate: {len(valid_tiles)/(len(valid_tiles) + len(invalid_tiles))*100:.1f}%\n\n")
        
        if invalid_tiles:
            f.write("INVALID TILES:\n")
            f.write("-" * 20 + "\n")
            for tile, info in invalid_tiles:
                f.write(f"Tile: {tile['bounds']} ({tile['resolution']}m)\n")
                f.write(f"  Error: {info.get('error', 'Unknown error')}\n")
                if 'valid_ratio' in info:
                    f.write(f"  Valid data: {info['valid_ratio']:.1%}\n")
                if 'elevation_range' in info:
                    f.write(f"  Elevation range: {info['elevation_range']:.2f}m\n")
                f.write("\n")
        
        if valid_tiles:
            f.write("VALID TILES:\n")
            f.write("-" * 20 + "\n")
            for tile, info in valid_tiles:
                f.write(f"Tile: {tile['bounds']} ({tile['resolution']}m)\n")
                f.write(f"  Elevation: {info['min_elev']:.1f}m to {info['max_elev']:.1f}m\n")
                f.write(f"  Range: {info['elevation_range']:.1f}m\n")
                f.write(f"  Valid data: {info['valid_ratio']:.1%}\n")
                f.write("\n")
    
    print(f"Validation report saved to: {report_path}")
    return valid_tiles, invalid_tiles

def parse_tile_filename(filename):
    """
    Parse tile filename to extract coordinates and resolution.
    Format: elevation_dem_Utah_<minx>_<miny>_<maxx>_<maxy>_<resolution>m.tif
    """
    pattern = r'elevation_dem_(\w+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_(\d+)m\.tif'
    match = re.match(pattern, filename)
    
    if match:
        location, minx, miny, maxx, maxy, resolution = match.groups()
        return {
            'location': location,
            'bounds': (float(minx), float(miny), float(maxx), float(maxy)),
            'resolution': int(resolution)
        }
    return None

def load_unavailable_tiles(cache_dir, location_name):
    """
    Load the list of unavailable tiles.
    """
    unavailable_file = os.path.join(cache_dir, location_name, "elevation_dem", "unavailable_tiles.txt")
    unavailable_tiles = set()
    
    if os.path.exists(unavailable_file):
        with open(unavailable_file, 'r') as f:
            for line in f:
                if line.startswith("Tile: "):
                    bounds_str = line.replace("Tile: ", "").strip()
                    unavailable_tiles.add(bounds_str)
    
    return unavailable_tiles

def load_cached_tiles(cache_dir, location_name):
    """
    Load information about cached tiles.
    """
    tiles_dir = os.path.join(cache_dir, location_name, "elevation_dem", "tiles")
    cached_tiles = []
    
    if os.path.exists(tiles_dir):
        for tile_file in glob.glob(os.path.join(tiles_dir, "*.tif")):
            filename = os.path.basename(tile_file)
            tile_info = parse_tile_filename(filename)
            if tile_info:
                tile_info['file_path'] = tile_file
                tile_info['file_size'] = os.path.getsize(tile_file)
                cached_tiles.append(tile_info)
    
    return cached_tiles

def get_location_bounds(location_name):
    """
    Get the bounds for a location from the location configs.
    """
    config_file = "location_configs.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        if location_name in configs:
            bounds = configs[location_name].get('bounds')
            if bounds:
                return bounds
    
    # Default Utah bounds if not found
    return [-114.043, 37.0, -109.046, 42.0]

def create_tile_visualization(location_name, output_file="tile_coverage_debug.png"):
    """
    Create a visualization showing tile coverage with actual state boundary.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
    
    # Load data
    unavailable_tiles = load_unavailable_tiles(cache_dir, location_name)
    cached_tiles = load_cached_tiles(cache_dir, location_name)
    
    # Get actual state boundary
    state_polygon, location_bounds = get_state_boundary(location_name)
    
    print(f"Debug visualization for {location_name}")
    print(f"Location bounds: {location_bounds}")
    print(f"Unavailable tiles: {len(unavailable_tiles)}")
    print(f"Cached tiles: {len(cached_tiles)}")
    
    # Calculate aspect ratio from bounds
    minx, miny, maxx, maxy = location_bounds
    width = maxx - minx
    height = maxy - miny
    aspect_ratio = width / height
    
    # Set desired image dimensions with padding
    padding = 50
    max_image_width = 1200
    max_image_height = 800
    
    # Calculate image dimensions that maintain aspect ratio
    available_width = max_image_width - (2 * padding)
    available_height = max_image_height - (2 * padding)
    
    scale_x = available_width / width
    scale_y = available_height / height
    scale = min(scale_x, scale_y)
    
    img_width = int(width * scale) + (2 * padding)
    img_height = int(height * scale) + (2 * padding)
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Calculate scale factors for coordinate conversion
    scale_x = (img_width - 2 * padding) / width
    scale_y = (img_height - 2 * padding) / height
    
    def coord_to_pixel(lon, lat):
        """Convert geographic coordinates to pixel coordinates."""
        x = padding + int((lon - minx) * scale_x)
        y = padding + int((maxy - lat) * scale_y)  # Flip Y axis
        return x, y
    
    # Draw state boundary (actual polygon if available, otherwise rectangle)
    outline_color = (0, 0, 0)
    outline_width = 3
    
    if state_polygon is not None:
        # Draw actual state boundary
        try:
            # Convert polygon to pixel coordinates
            if hasattr(state_polygon, 'exterior'):
                # Single polygon
                coords = list(state_polygon.exterior.coords)
                pixel_coords = [coord_to_pixel(lon, lat) for lon, lat in coords]
                draw.polygon(pixel_coords, outline=outline_color, width=outline_width)
            else:
                # MultiPolygon - draw each part
                for geom in state_polygon.geoms:
                    coords = list(geom.exterior.coords)
                    pixel_coords = [coord_to_pixel(lon, lat) for lon, lat in coords]
                    draw.polygon(pixel_coords, outline=outline_color, width=outline_width)
        except Exception as e:
            print(f"Error drawing state boundary: {e}")
            # Fallback to rectangle
            x1, y1 = coord_to_pixel(minx, miny)
            x2, y2 = coord_to_pixel(maxx, maxy)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=outline_width)
    else:
        # Draw rectangular boundary
        x1, y1 = coord_to_pixel(minx, miny)
        x2, y2 = coord_to_pixel(maxx, maxy)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=outline_width)
    
    # Draw cached tiles with validation
    cached_count = 0
    invalid_cached_count = 0
    
    print(f"\nValidating and drawing {len(cached_tiles)} cached tiles...")
    
    for i, tile in enumerate(cached_tiles):
        if i % 50 == 0:  # Progress indicator
            print(f"  Processing tile {i+1}/{len(cached_tiles)}")
            
        minx_tile, miny_tile, maxx_tile, maxy_tile = tile['bounds']
        x1, y1 = coord_to_pixel(minx_tile, miny_tile)
        x2, y2 = coord_to_pixel(maxx_tile, maxy_tile)
        
        # Ensure coordinates are properly ordered
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Validate the tile
        is_valid, validation_info = validate_cached_tile(tile)
        
        # Color by validation status and resolution
        if not is_valid:
            # Invalid cached tile - use orange/red color
            color = (255, 165, 0)  # Orange for invalid cached tiles
            invalid_cached_count += 1
        else:
            # Valid cached tile - color by resolution
            if tile['resolution'] == 10:
                color = (0, 255, 0)  # Bright green for 10m
            elif tile['resolution'] == 30:
                color = (0, 200, 0)   # Medium green for 30m
            elif tile['resolution'] == 100:
                color = (0, 150, 0)  # Dark green for 100m
            else:
                color = (0, 100, 0)  # Very dark green for other resolutions
        
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=1)
        cached_count += 1

    # Draw unavailable tiles (red)
    unavailable_count = 0
    for bounds_str in unavailable_tiles:
        try:
            # Parse bounds string: "minx_miny_maxx_maxy"
            parts = bounds_str.split('_')
            if len(parts) == 4:
                minx_tile, miny_tile, maxx_tile, maxy_tile = map(float, parts)
                x1, y1 = coord_to_pixel(minx_tile, miny_tile)
                x2, y2 = coord_to_pixel(maxx_tile, maxy_tile)
                
                # Ensure coordinates are properly ordered
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Ensure minimum size for visibility (at least 2x2 pixels)
                if x2 - x1 < 2:
                    center_x = (x1 + x2) // 2
                    x1 = center_x - 1
                    x2 = center_x + 1
                if y2 - y1 < 2:
                    center_y = (y1 + y2) // 2
                    y1 = center_y - 1
                    y2 = center_y + 1
                
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0), outline=(0, 0, 0), width=1)
                unavailable_count += 1
        except (ValueError, IndexError):
            continue
    
    # Add legend
    legend_y = 20
    legend_x = 20
    
    # Title
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    draw.text((legend_x, legend_y), f"Tile Coverage Debug: {location_name}", fill=(0, 0, 0), font=font_large)
    legend_y += 30
    
    # Legend items
    legend_items = [
        ((0, 255, 0), "Valid cached tiles (10m)"),
        ((0, 200, 0), "Valid cached tiles (30m)"),
        ((0, 150, 0), "Valid cached tiles (100m)"),
        ((255, 165, 0), "Invalid cached tiles"),
        ((255, 0, 0), "Unavailable tiles"),
        ((0, 0, 0), "State boundary")
    ]
    
    for color, label in legend_items:
        # Draw colored square
        draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=color, outline=(0, 0, 0))
        # Draw label
        draw.text((legend_x + 20, legend_y + 2), label, fill=(0, 0, 0), font=font_small)
        legend_y += 20
    
    # Add statistics
    legend_y += 10
    stats_text = [
        f"Valid cached tiles: {cached_count - invalid_cached_count}",
        f"Invalid cached tiles: {invalid_cached_count}",
        f"Total unavailable tiles: {unavailable_count}",
        f"Coverage ratio: {(cached_count - invalid_cached_count)/(cached_count + unavailable_count)*100:.1f}%" if (cached_count + unavailable_count) > 0 else "No tiles found"
    ]
    
    for stat in stats_text:
        draw.text((legend_x, legend_y), stat, fill=(0, 0, 0), font=font_small)
        legend_y += 15
    
    # Save image
    elevation_dem_dir = os.path.join(cache_dir, location_name, "elevation_dem")
    output_path = os.path.join(elevation_dem_dir, output_file)
    img.save(output_path)
    print(f"Debug visualization saved to: {output_path}")
    
    return {
        'cached_tiles': cached_count,
        'invalid_cached_tiles': invalid_cached_count,
        'unavailable_tiles': unavailable_count,
        'coverage_ratio': (cached_count - invalid_cached_count)/(cached_count + unavailable_count) if (cached_count + unavailable_count) > 0 else 0
    }


def analyze_tile_patterns(location_name):
    """
    Analyze patterns in the tile data.
    """
    cache_dir = "data_cache"
    unavailable_tiles = load_unavailable_tiles(cache_dir, location_name)
    cached_tiles = load_cached_tiles(cache_dir, location_name)
    
    print(f"\nTILE ANALYSIS FOR {location_name}")
    print("=" * 50)
    
    # Analyze by resolution
    resolution_counts = {}
    for tile in cached_tiles:
        res = tile['resolution']
        resolution_counts[res] = resolution_counts.get(res, 0) + 1
    
    print("Cached tiles by resolution:")
    for res in sorted(resolution_counts.keys()):
        print(f"  {res}m: {resolution_counts[res]} tiles")
    
    # Analyze unavailable tile patterns
    if unavailable_tiles:
        print(f"\nUnavailable tiles: {len(unavailable_tiles)}")
        print("Sample unavailable tiles:")
        for i, bounds_str in enumerate(list(unavailable_tiles)[:5]):
            print(f"  {i+1}: {bounds_str}")
        if len(unavailable_tiles) > 5:
            print(f"  ... and {len(unavailable_tiles) - 5} more")
    
    # Calculate coverage statistics
    total_tiles = len(cached_tiles) + len(unavailable_tiles)
    if total_tiles > 0:
        coverage_ratio = len(cached_tiles) / total_tiles
        print(f"\nCoverage statistics:")
        print(f"  Total tiles: {total_tiles}")
        print(f"  Cached: {len(cached_tiles)} ({len(cached_tiles)/total_tiles*100:.1f}%)")
        print(f"  Unavailable: {len(unavailable_tiles)} ({len(unavailable_tiles)/total_tiles*100:.1f}%)")
        print(f"  Coverage ratio: {coverage_ratio:.1%}")

def main():
    """
    Main function to run the debug visualization.
    """
    if len(sys.argv) > 1:
        location_name = sys.argv[1]
    else:
        location_name = "Utah"  # Default
    
    print(f"Creating debug visualization for {location_name}...")
    
    # Create visualization
    stats = create_tile_visualization(location_name)

    # Validate cached tiles
    print(f"\nValidating cached tiles...")
    valid_tiles, invalid_tiles = create_validation_report(location_name)
    
    # Analyze patterns
    analyze_tile_patterns(location_name)
    
    print(f"\nDebug visualization complete!")
    print(f"Check the generated image to see tile coverage patterns.")

if __name__ == "__main__":
    main()
