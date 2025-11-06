#!/usr/bin/env python3
"""
Debug script to test tile stitching issues.
Identifies gaps, overlaps, and data quality issues in the merged mosaic.
"""

import os
import sys
import glob
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rioxarray
from shapely.geometry import box
import geopandas as gpd
from rasterio.transform import from_bounds
import json
import rasterio
from rasterio.merge import merge as rio_merge
import tempfile
import xdem

def parse_tile_filename(filename):
    """
    Parse tile filename to extract coordinates and resolution.
    Format: elevation_dem_<location>_<minx>_<miny>_<maxx>_<maxy>_<resolution>m.tif
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

def load_mosaic_dem(cache_dir, location_name):
    """Load the final merged mosaic DEM"""
    cache_path = os.path.join(cache_dir, location_name, "elevation_dem")
    
    # Find mosaic file (contains 'mosaic' in filename)
    mosaic_pattern = os.path.join(cache_path, f"elevation_dem_{location_name}_*_mosaic.tif")
    mosaic_files = glob.glob(mosaic_pattern)
    
    if not mosaic_files:
        # Try without location name prefix
        mosaic_pattern = os.path.join(cache_path, "*_mosaic.tif")
        mosaic_files = glob.glob(mosaic_pattern)
    
    if not mosaic_files:
        raise FileNotFoundError(f"No mosaic file found in {cache_path}")
    
    if len(mosaic_files) > 1:
        print(f"Warning: Found {len(mosaic_files)} mosaic files, using: {mosaic_files[0]}")
    
    print(f"Loading mosaic from: {mosaic_files[0]}")
    mosaic = rioxarray.open_rasterio(mosaic_files[0]).squeeze()
    return mosaic

def load_all_tiles(cache_dir, location_name):
    """Load all individual tile files"""
    tiles_dir = os.path.join(cache_dir, location_name, "elevation_dem", "tiles")
    
    if not os.path.exists(tiles_dir):
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")
    
    tile_files = glob.glob(os.path.join(tiles_dir, "*.tif"))
    print(f"Found {len(tile_files)} tile files")
    
    tiles = []
    for tile_file in tile_files:
        filename = os.path.basename(tile_file)
        tile_info = parse_tile_filename(filename)
        
        if tile_info:
            try:
                tile_data = rioxarray.open_rasterio(tile_file).squeeze()
                tile_info['data'] = tile_data
                tile_info['file_path'] = tile_file
                tiles.append(tile_info)
            except Exception as e:
                print(f"Warning: Could not load tile {filename}: {e}")
    
    print(f"Successfully loaded {len(tiles)} tiles")
    return tiles

def get_tile_value_at_coord(tile_data, lon, lat):
    """Get the value from a tile at a specific geographic coordinate"""
    tile_transform = tile_data.rio.transform()
    tile_px, tile_py = ~tile_transform * (lon, lat)
    tile_px, tile_py = int(tile_px), int(tile_py)
    
    if 0 <= tile_px < tile_data.shape[1] and 0 <= tile_py < tile_data.shape[0]:
        value = tile_data.values[tile_py, tile_px]
        return value, (tile_px, tile_py), True
    return None, (tile_px, tile_py), False

def find_tiles_for_region(mosaic, tiles, center_y, center_x, radius=50):
    """Find all tiles that should contribute to a region around a pixel"""
    mosaic_transform = mosaic.rio.transform()
    
    # Convert pixel coordinates to geographic
    center_lon, center_lat = mosaic_transform * (center_x, center_y)
    
    # Get bounding box in geographic coordinates
    # Approximate pixel size
    res_x, res_y = mosaic.rio.resolution()
    radius_lon = abs(radius * res_x)
    radius_lat = abs(radius * res_y)
    
    region_minx = center_lon - radius_lon
    region_maxx = center_lon + radius_lon
    region_miny = center_lat - radius_lat
    region_maxy = center_lat + radius_lat
    
    contributing_tiles = []
    
    for tile in tiles:
        tile_bounds = tile['bounds']
        minx, miny, maxx, maxy = tile_bounds
        
        # Check if tile overlaps with region
        if not (maxx < region_minx or minx > region_maxx or maxy < region_miny or miny > region_maxy):
            tile_data = tile['data']
            
            # Get tile's actual bounds
            tile_actual_bounds = tile_data.rio.bounds()
            
            # Check actual overlap
            if not (tile_actual_bounds[2] < region_minx or tile_actual_bounds[0] > region_maxx or 
                    tile_actual_bounds[3] < region_miny or tile_actual_bounds[1] > region_maxy):
                contributing_tiles.append(tile)
    
    return contributing_tiles, (center_lon, center_lat)

def analyze_nan_region(mosaic, tiles, nan_mask, output_dir):
    """Deep dive analysis of NaN regions that have tile coverage"""
    print("\n=== DETAILED NaN REGION ANALYSIS ===")
    
    from scipy.ndimage import label, find_objects
    
    # Label all NaN regions
    labeled_nans, num_nan_regions = label(nan_mask)
    
    # Find regions to analyze (focus on medium-sized ones, not tiny or huge)
    nan_region_sizes = [np.sum(labeled_nans == i) for i in range(1, num_nan_regions + 1)]
    
    # Filter for significant regions (100-10000 pixels)
    significant_regions = [(i+1, size) for i, size in enumerate(nan_region_sizes) 
                          if 100 <= size <= 10000]
    
    if not significant_regions:
        print("No significant NaN regions found for detailed analysis")
        return
    
    # Sort by size, analyze largest ones
    significant_regions.sort(key=lambda x: x[1], reverse=True)
    
    mosaic_transform = mosaic.rio.transform()
    mosaic_values = mosaic.values
    
    print(f"\nAnalyzing {min(5, len(significant_regions))} largest NaN regions...")
    
    for region_idx, (region_id, region_size) in enumerate(significant_regions[:5]):
        print(f"\n--- Region {region_idx + 1}: {region_size:,} pixels ---")
        
        # Get bounding box of this region
        region_mask = (labeled_nans == region_id)
        region_slices = find_objects(region_mask)
        
        if not region_slices:
            continue
            
        slice_y, slice_x = region_slices[0]
        center_y = int((slice_y.start + slice_y.stop) / 2)
        center_x = int((slice_x.start + slice_x.stop) / 2)
        
        # Convert to geographic coordinates
        center_lon, center_lat = mosaic_transform * (center_x, center_y)
        
        print(f"  Center pixel: ({center_x}, {center_y})")
        print(f"  Center coordinates: ({center_lon:.6f}, {center_lat:.6f})")
        
        # Find tiles that should cover this region
        contributing_tiles, _ = find_tiles_for_region(mosaic, tiles, center_y, center_x, radius=100)
        print(f"  Tiles that should cover this region: {len(contributing_tiles)}")
        
        if len(contributing_tiles) == 0:
            print("  WARNING: No tiles found covering this region!")
            continue
        
        # Check what values tiles actually have at center point
        print(f"\n  Tile data at center point:")
        valid_tile_values = []
        
        for i, tile in enumerate(contributing_tiles):
            value, pixel_coord, in_bounds = get_tile_value_at_coord(tile['data'], center_lon, center_lat)
            
            if in_bounds:
                is_nan = np.isnan(value) if value is not None else True
                print(f"    Tile {i+1} ({tile['bounds']}):")
                print(f"      Pixel in tile: {pixel_coord}")
                print(f"      Value: {value:.2f}" if not is_nan else f"      Value: NaN")
                print(f"      Resolution: {tile['resolution']}m")
                print(f"      Tile CRS: {tile['data'].rio.crs}")
                print(f"      Tile transform: {tile['data'].rio.transform()}")
                print(f"      Tile shape: {tile['data'].shape}")
                print(f"      Tile bounds: {tile['data'].rio.bounds()}")
                
                if not is_nan:
                    valid_tile_values.append((i, tile, value))
        
        if len(valid_tile_values) > 0:
            print(f"\n  PROBLEM IDENTIFIED: {len(valid_tile_values)} tile(s) have valid data but mosaic shows NaN!")
            print(f"  This suggests a rio_merge issue.")
            
            # Try to manually merge just these tiles to see what happens
            print(f"\n  Testing manual merge of {len(valid_tile_values)} tiles...")
            test_merge_result = test_manual_merge([t[1] for t in valid_tile_values], center_lon, center_lat, output_dir, region_idx)
            
            if test_merge_result:
                print(f"    Manual merge result: {test_merge_result}")
        else:
            print(f"\n  All tiles have NaN at this location (data issue, not stitching issue)")
        
        # Check for resolution/transform mismatches
        if len(contributing_tiles) > 1:
            print(f"\n  Checking for resolution/transform mismatches:")
            resolutions = [t['resolution'] for t in contributing_tiles]
            if len(set(resolutions)) > 1:
                print(f"    WARNING: Multiple resolutions found: {set(resolutions)}")
            
            # Check transform differences
            transforms = [t['data'].rio.transform() for t in contributing_tiles]
            first_transform = transforms[0]
            for i, transform in enumerate(transforms[1:], 1):
                if transform != first_transform:
                    print(f"    WARNING: Transform mismatch between tile 0 and tile {i}")
                    print(f"      Tile 0: {first_transform}")
                    print(f"      Tile {i}: {transform}")

def identify_tiles_with_extreme_values(mosaic, tiles, extreme_mask, extreme_low_mask, extreme_high_mask, mosaic_transform):
    """Identify which tiles contain extremely low values"""
    problematic_tiles = []
    
    for tile in tiles:
        tile_bounds = tile['bounds']
        tile_data = tile['data']
        minx, miny, maxx, maxy = tile_bounds
        
        # Convert tile bounds to mosaic pixel coordinates
        px_minx, px_miny = ~mosaic_transform * (minx, maxy)
        px_maxx, px_maxy = ~mosaic_transform * (maxx, miny)
        
        # Ensure within bounds
        px_minx = max(0, int(px_minx))
        px_miny = max(0, int(px_miny))
        px_maxx = min(mosaic.shape[1], int(px_maxx))
        px_maxy = min(mosaic.shape[0], int(px_maxy))
        
        if px_maxx <= px_minx or px_maxy <= px_miny:
            continue
        
        # Check if this tile region contains extremely low values
        tile_extreme_low_region = extreme_low_mask[px_miny:px_maxy, px_minx:px_maxx]
        extreme_low_count = np.sum(tile_extreme_low_region)
        
        # Only flag tiles with extremely low values
        if extreme_low_count > 0:
            tile_values = tile_data.values
            valid_tile_values = tile_values[~np.isnan(tile_values)]
            min_value = np.min(valid_tile_values) if len(valid_tile_values) > 0 else np.nan
            max_value = np.max(valid_tile_values) if len(valid_tile_values) > 0 else np.nan
            
            problematic_tiles.append({
                'bounds': tile_bounds,
                'file_path': tile['file_path'],
                'extreme_low_count': extreme_low_count,
                'extreme_high_count': 0,  # Not checking high values
                'total_extreme_count': extreme_low_count,
                'min_value': min_value,
                'max_value': max_value
            })
    
    return problematic_tiles

def test_manual_merge(test_tiles, center_lon, center_lat, output_dir, region_idx):
    """Test manually merging a subset of tiles to see what rio_merge produces"""
    try:
        # Save tiles to temp files
        temp_files = []
        for tile in test_tiles:
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_f:
                tile['data'].rio.to_raster(temp_f.name)
                temp_files.append(temp_f.name)
        
        # Open with rasterio
        srcs = [rasterio.open(f) for f in temp_files]
        
        # Try merge with method='first'
        mosaic_arr, out_trans = rio_merge(srcs, method='first')
        
        # Convert center coordinates to pixel coordinates in merged result
        merged_px, merged_py = ~out_trans * (center_lon, center_lat)
        merged_px, merged_py = int(merged_px), int(merged_py)
        
        result_value = None
        if 0 <= merged_px < mosaic_arr.shape[2] and 0 <= merged_py < mosaic_arr.shape[1]:
            result_value = mosaic_arr[0, merged_py, merged_px]
        
        # Close sources
        for src in srcs:
            src.close()
        
        # Clean up temp files
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
        
        if result_value is not None:
            is_nan = np.isnan(result_value)
            return f"{result_value:.2f}" if not is_nan else "NaN (merge failed!)"
        else:
            return "Outside merged bounds"
            
    except Exception as e:
        return f"Merge test failed: {e}"

def analyze_stitching(mosaic, tiles, location_name, output_dir):
    """Perform comprehensive stitching analysis"""
    print("\n=== STITCHING ANALYSIS ===")
    
    # Get mosaic bounds and transform
    mosaic_bounds = mosaic.rio.bounds()
    mosaic_transform = mosaic.rio.transform()
    mosaic_shape = mosaic.shape
    
    print(f"Mosaic bounds: {mosaic_bounds}")
    print(f"Mosaic shape: {mosaic_shape}")
    print(f"Mosaic CRS: {mosaic.rio.crs}")
    print(f"Mosaic resolution: {mosaic.rio.resolution()}")
    
    # 1. Check for NaN/NoData values
    print("\n1. Checking for NaN/NoData values...")
    mosaic_values = mosaic.values
    nan_mask = np.isnan(mosaic_values)
    nan_count = np.sum(nan_mask)
    total_pixels = mosaic_values.size
    nan_ratio = nan_count / total_pixels
    
    print(f"  NaN pixels: {nan_count:,} ({nan_ratio:.2%})")
    
    # Check nodata value
    nodata_val = mosaic.rio.nodata
    if nodata_val is not None:
        nodata_mask = np.isclose(mosaic_values, nodata_val)
        nodata_count = np.sum(nodata_mask)
        print(f"  Nodata pixels: {nodata_count:,} ({nodata_count/total_pixels:.2%})")
        nan_mask = nan_mask | nodata_mask
    
    # Check for extreme values (very low only)
    print("\n1b. Checking for extremely low values...")
    valid_mask = ~nan_mask
    if np.any(valid_mask):
        valid_data = mosaic_values[valid_mask]
        
        # Calculate statistics
        p01 = np.percentile(valid_data, 1)
        p05 = np.percentile(valid_data, 5)
        median = np.median(valid_data)
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        
        print(f"  Valid data statistics:")
        print(f"    Min: {valid_data.min():.2f}")
        print(f"    1st percentile: {p01:.2f}")
        print(f"    5th percentile: {p05:.2f}")
        print(f"    Median: {median:.2f}")
        print(f"    Mean: {mean:.2f}")
        print(f"    Max: {valid_data.max():.2f}")
        print(f"    Std dev: {std:.2f}")
        
        # Define extremely low values as below 1st percentile or 3 standard deviations below mean
        # Increased threshold slightly (higher value = catches more values, less restrictive)
        extreme_low_threshold_1 = p01
        extreme_low_threshold_2 = mean - 3 * std
        
        # Use a slightly higher threshold to catch more extremely low values
        # Using 1.5th percentile (instead of 1st) and 2.5 std (instead of 3) makes threshold higher
        p015 = np.percentile(valid_data, 1.5)
        extreme_low_threshold_3 = p015
        extreme_low_threshold_4 = mean - 2.5 * std
        
        # Use the highest threshold (least restrictive) to catch more extremely low values
        extreme_low_threshold = max(extreme_low_threshold_1, extreme_low_threshold_2,
                                     extreme_low_threshold_3, extreme_low_threshold_4)
        
        extreme_low_mask = (mosaic_values < extreme_low_threshold) & valid_mask
        extreme_high_mask = np.zeros_like(nan_mask, dtype=bool)  # Not checking high values
        extreme_mask = extreme_low_mask  # Only low values
        extreme_high_threshold = None  # Not checking high values
        
        extreme_low_count = np.sum(extreme_low_mask)
        extreme_count = extreme_low_count
        
        print(f"\n  Extremely low value threshold: {extreme_low_threshold:.2f}")
        print(f"  Extremely low values detected: {extreme_low_count:,} ({extreme_low_count/total_pixels:.2%})")
        
        # Check for regions of extreme values (could be the black rectangles)
        from scipy.ndimage import label
        if extreme_count > 0:
            labeled_extremes, num_extreme_regions = label(extreme_mask)
            extreme_region_sizes = [np.sum(labeled_extremes == i) for i in range(1, num_extreme_regions + 1)]
            if extreme_region_sizes:
                print(f"    Extreme low value regions: {num_extreme_regions}")
                print(f"    Largest region: {max(extreme_region_sizes):,} pixels")
                print(f"    Average region size: {np.mean(extreme_region_sizes):.0f} pixels")
            
            # Identify which tiles contain extreme values (only low)
            problematic_tiles = identify_tiles_with_extreme_values(mosaic, tiles, extreme_mask, extreme_low_mask, extreme_high_mask, mosaic_transform)
            if problematic_tiles:
                print(f"\n  Tiles containing extremely low values:")
                for tile_info in problematic_tiles:
                    print(f"    {tile_info['bounds']} - {tile_info['file_path']}")
                    print(f"      Extreme low pixels: {tile_info['extreme_low_count']:,}")
                    print(f"      Min value: {tile_info['min_value']:.2f}, Max value: {tile_info['max_value']:.2f}")
    else:
        extreme_mask = np.zeros_like(nan_mask, dtype=bool)
        extreme_low_mask = np.zeros_like(nan_mask, dtype=bool)
        extreme_high_mask = np.zeros_like(nan_mask, dtype=bool)
        extreme_low_threshold = None
        extreme_high_threshold = None
    
    # 2. Check tile coverage (COMMENTED OUT - only need extreme value detection)
    # print("\n2. Checking tile coverage...")
    
    # Create a coverage map showing which tiles contribute to each pixel
    coverage_map = np.zeros(mosaic_shape, dtype=int)  # 0 = no coverage, >0 = number of tiles
    tile_boundary_map = np.zeros(mosaic_shape, dtype=bool)
    
    # Simplified coverage map just for identifying problematic tiles
    for tile_idx, tile in enumerate(tiles):
        tile_bounds = tile['bounds']
        minx, miny, maxx, maxy = tile_bounds
        
        # Convert to pixel coordinates in mosaic
        px_minx, px_miny = ~mosaic_transform * (minx, maxy)  # Note: Y is flipped
        px_maxx, px_maxy = ~mosaic_transform * (maxx, miny)
        
        # Ensure within bounds
        px_minx = max(0, int(px_minx))
        px_miny = max(0, int(px_miny))
        px_maxx = min(mosaic_shape[1], int(px_maxx))
        px_maxy = min(mosaic_shape[0], int(px_maxy))
        
        # Mark covered pixels (simplified - just mark the bounding box)
        if px_maxx > px_minx and px_maxy > px_miny:
            coverage_map[px_miny:px_maxy, px_minx:px_maxx] += 1
    
    # # Find gaps (pixels with no coverage)
    # gap_mask = (coverage_map == 0) & ~nan_mask
    # gap_count = np.sum(gap_mask)
    # print(f"  Gaps (no tile coverage): {gap_count:,} pixels ({gap_count/total_pixels:.2%})")
    
    # # Find overlaps (pixels with multiple tiles)
    # overlap_mask = coverage_map > 1
    # overlap_count = np.sum(overlap_mask)
    # print(f"  Overlaps (multiple tiles): {overlap_count:,} pixels ({overlap_count/total_pixels:.2%})")
    
    # # Check NaN at tile boundaries
    # print("\n3. Checking NaN at tile boundaries...")
    # boundary_nan_mask = tile_boundary_map & nan_mask
    # boundary_nan_count = np.sum(boundary_nan_mask)
    # boundary_total = np.sum(tile_boundary_map)
    # if boundary_total > 0:
    #     boundary_nan_ratio = boundary_nan_count / boundary_total
    #     print(f"  NaN pixels at boundaries: {boundary_nan_count:,} / {boundary_total:,} ({boundary_nan_ratio:.2%})")
    
    # # NEW: Find NaN regions that have tile coverage (the main problem!)
    # print("\n4. Checking NaN regions WITH tile coverage (stitching bug indicator)...")
    # nan_with_coverage = nan_mask & (coverage_map > 0)
    # nan_with_coverage_count = np.sum(nan_with_coverage)
    # print(f"  NaN pixels WITH tile coverage: {nan_with_coverage_count:,} ({nan_with_coverage_count/total_pixels:.2%})")
    # print(f"  This indicates tiles exist but rio_merge is failing!")
    
    # if nan_with_coverage_count > 0:
    #     analyze_nan_region(mosaic, tiles, nan_with_coverage, output_dir)
    
    # Create dummy masks for return
    gap_mask = np.zeros_like(nan_mask, dtype=bool)
    overlap_mask = np.zeros_like(nan_mask, dtype=bool)
    nan_with_coverage = np.zeros_like(nan_mask, dtype=bool)
    
    # Store problematic tiles info if found
    problematic_tiles_info = []
    if np.any(extreme_mask):
        problematic_tiles_info = identify_tiles_with_extreme_values(
            mosaic, tiles, extreme_mask, extreme_low_mask, extreme_high_mask, mosaic_transform
        )
    
    return {
        'nan_mask': nan_mask,
        'gap_mask': gap_mask,
        'overlap_mask': overlap_mask,
        'tile_boundary_map': tile_boundary_map,
        'coverage_map': coverage_map,
        'mosaic': mosaic,
        'nan_with_coverage': nan_with_coverage,
        'extreme_mask': extreme_mask,
        'extreme_low_mask': extreme_low_mask,
        'extreme_high_mask': extreme_high_mask,
        'extreme_thresholds': (extreme_low_threshold, extreme_high_threshold),
        'problematic_tiles': problematic_tiles_info
    }

def create_stitching_visualization(analysis_results, tiles, location_name, output_dir):
    """Create visualization images showing stitching issues"""
    print("\n=== Creating visualizations ===")
    
    mosaic = analysis_results['mosaic']
    nan_mask = analysis_results['nan_mask']
    gap_mask = analysis_results['gap_mask']
    overlap_mask = analysis_results['overlap_mask']
    tile_boundary_map = analysis_results['tile_boundary_map']
    coverage_map = analysis_results['coverage_map']
    nan_with_coverage = analysis_results.get('nan_with_coverage', np.zeros_like(nan_mask))
    extreme_mask = analysis_results.get('extreme_mask', np.zeros_like(nan_mask))
    extreme_low_mask = analysis_results.get('extreme_low_mask', np.zeros_like(nan_mask))
    extreme_high_mask = analysis_results.get('extreme_high_mask', np.zeros_like(nan_mask))
    extreme_thresholds = analysis_results.get('extreme_thresholds', (None, None))
    
    mosaic_shape = mosaic.shape
    
    # Create normalized mosaic using percentile-based scaling to avoid outlier effects
    print("  Creating normalized mosaic (percentile-based)...")
    mosaic_normalized = np.full(mosaic_shape, 255, dtype=np.uint8)
    valid_mask = ~nan_mask
    if np.any(valid_mask):
        valid_data = mosaic.values[valid_mask]
        if len(valid_data) > 0:
            # Use 5th and 95th percentiles for normalization to avoid extreme outliers
            p05 = np.percentile(valid_data, 5)
            p95 = np.percentile(valid_data, 95)
            
            if p95 > p05:
                # Normalize valid data
                normalized = np.clip(
                    ((mosaic.values - p05) / (p95 - p05) * 255),
                    0, 255
                ).astype(np.uint8)
                mosaic_normalized = normalized
            else:
                # All values are the same
                mosaic_normalized[valid_mask] = 128
    
    # 1. Mosaic with NaN areas highlighted
    print("  Creating NaN visualization...")
    nan_img = np.stack([mosaic_normalized] * 3, axis=-1)
    nan_img[nan_mask] = [255, 0, 0]  # Red
    
    Image.fromarray(nan_img).save(os.path.join(output_dir, "stitching_debug_nan.png"))
    print(f"    Saved: stitching_debug_nan.png")
    
    # 1a. NEW: Extreme low values visualization
    if np.any(extreme_low_mask):
        print("  Creating extreme low values visualization...")
        extreme_img = np.stack([mosaic_normalized] * 3, axis=-1)
        extreme_img[extreme_low_mask] = [0, 0, 255]  # Blue for very low values
        
        Image.fromarray(extreme_img).save(os.path.join(output_dir, "stitching_debug_extreme_values.png"))
        print(f"    Saved: stitching_debug_extreme_values.png")
    
    # 2. Gap visualization
    print("  Creating gap visualization...")
    gap_img = np.stack([mosaic_normalized] * 3, axis=-1)
    gap_img[gap_mask] = [255, 165, 0]  # Orange for gaps
    
    Image.fromarray(gap_img).save(os.path.join(output_dir, "stitching_debug_gaps.png"))
    print(f"    Saved: stitching_debug_gaps.png")
    
    # NEW: NaN with coverage visualization
    print("  Creating NaN-with-coverage visualization...")
    nan_coverage_img = np.stack([mosaic_normalized] * 3, axis=-1)
    nan_coverage_img[nan_with_coverage] = [255, 0, 255]  # Magenta for NaN with coverage (stitching bug)
    
    Image.fromarray(nan_coverage_img).save(os.path.join(output_dir, "stitching_debug_nan_and_gaps.png"))
    print(f"    Saved: stitching_debug_nan_and_gaps.png")
    
    # 3. Tile boundaries
    print("  Creating tile boundary visualization...")
    boundary_img = np.stack([mosaic_normalized] * 3, axis=-1)
    boundary_img[tile_boundary_map] = [0, 255, 0]  # Green for boundaries
    
    Image.fromarray(boundary_img).save(os.path.join(output_dir, "stitching_debug_boundaries.png"))
    print(f"    Saved: stitching_debug_boundaries.png")
    
    # 4. Coverage map (number of tiles per pixel)
    print("  Creating coverage map...")
    coverage_normalized = (coverage_map / coverage_map.max() * 255).astype(np.uint8) if coverage_map.max() > 0 else np.zeros_like(coverage_map, dtype=np.uint8)
    Image.fromarray(coverage_normalized).save(os.path.join(output_dir, "stitching_debug_coverage.png"))
    print(f"    Saved: stitching_debug_coverage.png")
    
    # 5. Combined visualization (including extreme low values)
    print("  Creating combined visualization...")
    combined_img = np.stack([mosaic_normalized] * 3, axis=-1)
    combined_img[gap_mask] = [255, 165, 0]  # Orange gaps
    combined_img[tile_boundary_map] = [0, 255, 0]  # Green boundaries
    combined_img[nan_with_coverage] = [255, 0, 255]  # Magenta for NaN with coverage (stitching bug)
    combined_img[extreme_low_mask] = [0, 0, 255]  # Blue for very low values
    combined_img[nan_mask & ~gap_mask & ~nan_with_coverage] = [255, 0, 0]  # Red NaN (no coverage)
    
    Image.fromarray(combined_img).save(os.path.join(output_dir, "stitching_debug_combined.png"))
    print(f"    Saved: stitching_debug_combined.png")
    
    # 6. Tile index map (which tile contributes to each pixel)
    print("  Creating tile index visualization...")
    # This is simplified - in reality would need proper spatial matching
    tile_index_map = np.zeros(mosaic_shape, dtype=int)
    for i, tile in enumerate(tiles):
        tile_bounds = tile['bounds']
        minx, miny, maxx, maxy = tile_bounds
        mosaic_transform = mosaic.rio.transform()
        
        px_minx, px_miny = ~mosaic_transform * (minx, maxy)
        px_maxx, px_maxy = ~mosaic_transform * (maxx, miny)
        
        px_minx = max(0, int(px_minx))
        px_miny = max(0, int(px_miny))
        px_maxx = min(mosaic_shape[1], int(px_maxx))
        px_maxy = min(mosaic_shape[0], int(px_maxy))
        
        if px_maxx > px_minx and px_maxy > px_miny:
            # Use modulo to create distinct colors
            tile_index_map[px_miny:px_maxy, px_minx:px_maxx] = (i % 10) + 1
    
    # Create colormap
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab10', 10)
    tile_index_rgb = cmap(tile_index_map / 10.0)[:, :, :3]
    tile_index_rgb = (tile_index_rgb * 255).astype(np.uint8)
    Image.fromarray(tile_index_rgb).save(os.path.join(output_dir, "stitching_debug_tile_indices.png"))
    print(f"    Saved: stitching_debug_tile_indices.png")

def analyze_hillshade_vs_dem(mosaic, output_dir):
    """Compare DEM data with hillshade to find where hillshade fails"""
    print("\n=== HILLSHADE vs DEM ANALYSIS ===")
    
    # Check resolution first
    res_x, res_y = mosaic.rio.resolution()
    print(f"DEM resolution: ({res_x:.10f}, {res_y:.10f})")
    
    # xdem requires equal X and Y resolution, so we need to reproject
    # Check if CRS is geographic (EPSG:4326) - if so, reproject to Web Mercator
    if str(mosaic.rio.crs) == 'EPSG:4326':
        print("  Reprojecting to Web Mercator for hillshade calculation...")
        from utils import reproject_to_web_mercator
        mosaic_proj = reproject_to_web_mercator(mosaic)
        original_crs = 'EPSG:4326'
    else:
        mosaic_proj = mosaic
        original_crs = None
    
    # Check resolution after reprojection
    res_x, res_y = mosaic_proj.rio.resolution()
    print(f"Projected DEM resolution: ({abs(res_x):.10f}, {abs(res_y):.10f})")
    
    # Create a test hillshade from the reprojected mosaic
    import xdem
    dem = xdem.DEM.from_array(
        mosaic_proj.values.squeeze(),
        transform=mosaic_proj.rio.transform(),
        crs=mosaic_proj.rio.crs
    )
    
    # Generate hillshade - use the same parameters as your actual hillshade
    print("  Generating hillshade...")
    hillshade = xdem.terrain.hillshade(dem, azimuth=315, altitude=45)
    hillshade_data = hillshade.data.squeeze()
    hillshade_transform = hillshade.transform
    
    # For comparison, we'll work in the projected space (simpler)
    # Just analyze the projected hillshade against the projected DEM
    print(f"  Hillshade shape: {hillshade_data.shape}")
    print(f"  Projected DEM shape: {mosaic_proj.shape}")
    
    # Check for NaN in hillshade
    hillshade_nan_mask = np.isnan(hillshade_data)
    hillshade_nan_count = np.sum(hillshade_nan_mask)
    
    print(f"Hillshade NaN pixels: {hillshade_nan_count:,} ({hillshade_nan_count/hillshade_data.size:.2%})")
    
    # Check if projected DEM has valid data where hillshade is NaN
    dem_proj_values = mosaic_proj.values
    dem_proj_valid = ~np.isnan(dem_proj_values)
    
    # Handle shape mismatch - hillshade might have slightly different shape due to edge effects
    if hillshade_data.shape != dem_proj_values.shape:
        # Use smaller shape for comparison
        min_h = min(hillshade_data.shape[0], dem_proj_values.shape[0])
        min_w = min(hillshade_data.shape[1], dem_proj_values.shape[1])
        
        hillshade_nan_mask = hillshade_nan_mask[:min_h, :min_w]
        dem_proj_valid = dem_proj_valid[:min_h, :min_w]
        hillshade_data = hillshade_data[:min_h, :min_w]
        dem_proj_values = dem_proj_values[:min_h, :min_w]
        
        print(f"  Adjusted to common shape: {hillshade_nan_mask.shape} for comparison")
    
    dem_valid_but_hillshade_nan = dem_proj_valid & hillshade_nan_mask
    problem_count = np.sum(dem_valid_but_hillshade_nan)
    print(f"DEM valid but hillshade NaN: {problem_count:,} pixels")
    
    if problem_count > 0:
        print(f"  PROBLEM: Hillshade algorithm is producing NaN where DEM has valid data!")
        
        # Enhanced visualization - show problem areas overlaid on DEM
        dem_normalized = np.full(dem_proj_values.shape, 255, dtype=np.uint8)
        if np.any(dem_proj_valid):
            valid_data = dem_proj_values[dem_proj_valid]
            if valid_data.max() > valid_data.min():
                normalized = ((dem_proj_values - valid_data.min()) / (valid_data.max() - valid_data.min()) * 255).astype(np.uint8)
                dem_normalized[dem_proj_valid] = normalized[dem_proj_valid]
        
        # Create RGB image with problem areas highlighted
        problem_img = np.stack([dem_normalized] * 3, axis=-1)
        problem_img[dem_valid_but_hillshade_nan] = [255, 0, 0]  # Red for problem areas
        
        Image.fromarray(problem_img).save(
            os.path.join(output_dir, "hillshade_problem_areas.png")
        )
        print(f"  Saved: hillshade_problem_areas.png")
        
        # Also check if these are just edge effects (hillshade often produces NaN at edges)
        from scipy.ndimage import label
        labeled, num_regions = label(dem_valid_but_hillshade_nan)
        region_sizes = [np.sum(labeled == i) for i in range(1, num_regions + 1)]
        print(f"  Number of problem regions: {num_regions}")
        if region_sizes:
            print(f"  Largest problem region: {max(region_sizes):,} pixels")
            print(f"  Average region size: {np.mean(region_sizes):.0f} pixels")
    
    # Also check for very low hillshade values (might appear black but aren't NaN)
    if np.any(~hillshade_nan_mask):
        valid_hillshade = hillshade_data[~hillshade_nan_mask]
        print(f"\nHillshade value range (valid pixels):")
        print(f"  Min: {valid_hillshade.min():.2f}")
        print(f"  Max: {valid_hillshade.max():.2f}")
        print(f"  Mean: {valid_hillshade.mean():.2f}")
        
        # Check for very low values that might appear black
        very_low_threshold = valid_hillshade.min() + (valid_hillshade.max() - valid_hillshade.min()) * 0.01
        very_low_mask = (hillshade_data < very_low_threshold) & ~hillshade_nan_mask
        very_low_count = np.sum(very_low_mask)
        print(f"  Very low values (<1% of range): {very_low_count:,} pixels")    

def generate_hillshade_debug(mosaic, output_dir, sun_azimuth=315, sun_altitude=45, vertical_exaggeration=1.0):
    """Generate hillshade from mosaic DEM to verify it works correctly"""
    print("\nGenerating hillshade...")
    
    try:
        # Check if we need to reproject (xdem requires equal X and Y resolution)
        res_x, res_y = mosaic.rio.resolution()
        print(f"DEM resolution: ({abs(res_x):.10f}, {abs(res_y):.10f})")
        
        # Check if CRS is geographic (EPSG:4326) - if so, reproject to Web Mercator
        dem_data = mosaic
        if str(mosaic.rio.crs) == 'EPSG:4326':
            print("  Reprojecting to Web Mercator for hillshade calculation...")
            from utils import reproject_to_web_mercator
            dem_data = reproject_to_web_mercator(mosaic)
            res_x, res_y = dem_data.rio.resolution()
            print(f"  Projected resolution: ({abs(res_x):.10f}, {abs(res_y):.10f})")
        
        # Convert to xdem.DEM
        dem = xdem.DEM.from_array(
            dem_data.values.squeeze(),
            transform=dem_data.rio.transform(),
            crs=dem_data.rio.crs
        )
        
        # Apply vertical exaggeration
        if vertical_exaggeration != 1.0:
            dem.data = dem.data * vertical_exaggeration
            print(f"  Applied vertical exaggeration: {vertical_exaggeration}")
        
        # Generate hillshade
        print(f"  Generating hillshade (azimuth={sun_azimuth}, altitude={sun_altitude})...")
        hillshade = xdem.terrain.hillshade(
            dem,
            azimuth=sun_azimuth,
            altitude=sun_altitude
        )
        
        hillshade_data = hillshade.data.squeeze()
        print(f"  Hillshade shape: {hillshade_data.shape}")
        
        # Check for NaN in hillshade
        hillshade_nan_mask = np.isnan(hillshade_data)
        hillshade_nan_count = np.sum(hillshade_nan_mask)
        print(f"  Hillshade NaN pixels: {hillshade_nan_count:,} ({hillshade_nan_count/hillshade_data.size:.2%})")
        
        # Convert to 8-bit grayscale (matching maps_main.py logic)
        # Fill NaN values with max value (white) to prevent black backgrounds
        hillshade_data_clean = np.nan_to_num(hillshade_data, nan=np.nanmax(hillshade_data) if not np.all(np.isnan(hillshade_data)) else 255)
        
        # Normalize to 0-255
        if hillshade_data_clean.max() > hillshade_data_clean.min():
            hillshade_uint8 = ((hillshade_data_clean - hillshade_data_clean.min()) * 255 / 
                              (hillshade_data_clean.max() - hillshade_data_clean.min())).astype(np.uint8)
        else:
            hillshade_uint8 = np.full_like(hillshade_data_clean, 128, dtype=np.uint8)
        
        # Apply gamma correction (same as maps_main.py)
        gamma = 1
        hillshade_uint8 = np.power(hillshade_uint8 / 255.0, gamma) * 255
        hillshade_uint8 = hillshade_uint8.astype(np.uint8)
        
        # Check if DEM has valid data where hillshade is NaN
        dem_values = dem_data.values.squeeze()
        dem_valid = ~np.isnan(dem_values)
        
        # Handle shape mismatch
        if hillshade_data.shape != dem_values.shape:
            min_h = min(hillshade_data.shape[0], dem_values.shape[0])
            min_w = min(hillshade_data.shape[1], dem_values.shape[1])
            hillshade_nan_mask = hillshade_nan_mask[:min_h, :min_w]
            dem_valid = dem_valid[:min_h, :min_w]
            print(f"  Adjusted shapes for comparison: {hillshade_nan_mask.shape}")
        
        dem_valid_but_hillshade_nan = dem_valid & hillshade_nan_mask
        problem_count = np.sum(dem_valid_but_hillshade_nan)
        
        if problem_count > 0:
            print(f"  WARNING: {problem_count:,} pixels have valid DEM data but hillshade is NaN!")
        
        # Save hillshade image
        hillshade_img = Image.fromarray(hillshade_uint8, 'L')
        hillshade_path = os.path.join(output_dir, "stitching_debug_hillshade.png")
        hillshade_img.save(hillshade_path)
        print(f"  Saved hillshade: {hillshade_path}")
        print(f"  Hillshade size: {hillshade_img.size[0]}x{hillshade_img.size[1]}")
        print(f"  DEM size: {dem_values.shape[1]}x{dem_values.shape[0]}")
        
        # Compare shapes
        if hillshade_img.size[0] != dem_values.shape[1] or hillshade_img.size[1] != dem_values.shape[0]:
            print(f"  WARNING: Size mismatch between DEM and hillshade!")
            print(f"    This could indicate cropping issues.")
        
        return hillshade_uint8
        
    except Exception as e:
        print(f"ERROR generating hillshade: {e}")
        import traceback
        traceback.print_exc()
        return None

        
def delete_problematic_tiles(cache_dir, location_name):
    """Delete tiles with extremely low values to force redownload"""
    output_dir = os.path.join(cache_dir, location_name, "elevation_dem")
    
    print(f"Identifying problematic tiles (extremely low values) for: {location_name}")
    print("=" * 60)
    
    try:
        # Load analysis to get problematic tiles
        mosaic = load_mosaic_dem(cache_dir, location_name)
        tiles = load_all_tiles(cache_dir, location_name)
        
        # Quick analysis to find problematic tiles (only low values)
        mosaic_values = mosaic.values
        nan_mask = np.isnan(mosaic_values)
        valid_mask = ~nan_mask
        
        if np.any(valid_mask):
            valid_data = mosaic_values[valid_mask]
            p01 = np.percentile(valid_data, 1)
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            extreme_low_threshold = min(p01, mean - 3 * std)
            
            extreme_low_mask = (mosaic_values < extreme_low_threshold) & valid_mask
            extreme_high_mask = np.zeros_like(nan_mask, dtype=bool)
            extreme_mask = extreme_low_mask
            
            problematic_tiles = identify_tiles_with_extreme_values(
                mosaic, tiles, extreme_mask, extreme_low_mask, extreme_high_mask, mosaic.rio.transform()
            )
            
            if not problematic_tiles:
                print("No problematic tiles found.")
                return
            
            # Show tiles to be deleted and get approval
            print(f"\n{'=' * 60}")
            print(f"TILES TO BE DELETED ({len(problematic_tiles)} tile(s)):")
            print("=" * 60)
            print()
            
            tiles_to_delete = []
            for i, tile_info in enumerate(problematic_tiles):
                print(f"{i+1}. Tile bounds: {tile_info['bounds']}")
                print(f"   File: {os.path.basename(tile_info['file_path'])}")
                print(f"   Path: {tile_info['file_path']}")
                print(f"   Extreme low pixels: {tile_info['extreme_low_count']:,}")
                print(f"   Min value: {tile_info['min_value']:.2f}, Max value: {tile_info['max_value']:.2f}")
                print()
                tiles_to_delete.append(tile_info['file_path'])
            
            # Get user approval
            print("=" * 60)
            response = input(f"\nDelete these {len(tiles_to_delete)} tile(s)? (yes/no): ").strip().lower()
            
            if response not in ['yes', 'y']:
                print("Deletion cancelled.")
                return
            
            # Delete tiles
            print("\nDeleting tiles...")
            deleted_count = 0
            for tile_path in tiles_to_delete:
                if os.path.exists(tile_path):
                    try:
                        os.remove(tile_path)
                        print(f"  Deleted: {os.path.basename(tile_path)}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  Error deleting {os.path.basename(tile_path)}: {e}")
                else:
                    print(f"  Not found (already deleted?): {os.path.basename(tile_path)}")
            
            print(f"\nDeleted {deleted_count} tile(s).")
            print("These tiles will be redownloaded on next run.")
            
            # Also delete the mosaic cache to force regeneration
            print("\nDeleting mosaic cache to force regeneration...")
            mosaic_pattern = os.path.join(output_dir, "*_mosaic.tif")
            mosaic_files = glob.glob(mosaic_pattern)
            for mosaic_file in mosaic_files:
                try:
                    os.remove(mosaic_file)
                    print(f"  Deleted mosaic cache: {os.path.basename(mosaic_file)}")
                except Exception as e:
                    print(f"  Error deleting mosaic: {e}")
        else:
            print("No valid data found in mosaic.")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    # Check for --delete-tiles flag
    if len(sys.argv) > 1 and sys.argv[1] == "--delete-tiles":
        if len(sys.argv) > 2:
            location_name = sys.argv[2]
        else:
            location_name = "Utah"  # Default
        
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
        delete_problematic_tiles(cache_dir, location_name)
        return
    
    if len(sys.argv) > 1:
        location_name = sys.argv[1]
    else:
        location_name = "Utah"  # Default
    
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
    output_dir = os.path.join(cache_dir, location_name, "elevation_dem")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Stitching Debug Analysis for: {location_name}")
    print("=" * 60)
    
    try:
        # Load mosaic
        mosaic = load_mosaic_dem(cache_dir, location_name)
        
        # Load tiles
        tiles = load_all_tiles(cache_dir, location_name)
        
        if len(tiles) == 0:
            print("ERROR: No tiles found. Cannot perform stitching analysis.")
            return
        
        # Perform analysis
        analysis_results = analyze_stitching(mosaic, tiles, location_name, output_dir)
        
        # Create visualizations
        # create_stitching_visualization(analysis_results, tiles, location_name, output_dir)
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Output directory: {output_dir}")

        # Generate hillshade to verify it works
        print("\n" + "=" * 60)
        print("GENERATING HILLSHADE FOR VERIFICATION")
        print("=" * 60)
        generate_hillshade_debug(mosaic, output_dir)
        
        # Compare hillshade with DEM (COMMENTED OUT)
        # analyze_hillshade_vs_dem(mosaic, output_dir)
        
        # Report problematic tiles and offer to delete them
        problematic_tiles = analysis_results.get('problematic_tiles', [])
        if problematic_tiles:
            print("\n" + "=" * 60)
            print("PROBLEMATIC TILES DETECTED (Extremely Low Values)")
            print("=" * 60)
            print(f"Found {len(problematic_tiles)} tile(s) with extremely low values:")
            print()
            
            tiles_to_delete = []
            for i, tile_info in enumerate(problematic_tiles):
                print(f"{i+1}. Tile bounds: {tile_info['bounds']}")
                print(f"   File: {os.path.basename(tile_info['file_path'])}")
                print(f"   Path: {tile_info['file_path']}")
                print(f"   Extreme low pixels: {tile_info['extreme_low_count']:,}")
                print(f"   Min value: {tile_info['min_value']:.2f}, Max value: {tile_info['max_value']:.2f}")
                print()
                tiles_to_delete.append(tile_info['file_path'])
            
            # Get user approval and delete
            print("=" * 60)
            response = input(f"\nDelete these {len(tiles_to_delete)} tile(s)? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                print("\nDeleting tiles...")
                deleted_count = 0
                for tile_path in tiles_to_delete:
                    if os.path.exists(tile_path):
                        try:
                            os.remove(tile_path)
                            print(f"  Deleted: {os.path.basename(tile_path)}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"  Error deleting {os.path.basename(tile_path)}: {e}")
                    else:
                        print(f"  Not found (already deleted?): {os.path.basename(tile_path)}")
                
                print(f"\nDeleted {deleted_count} tile(s).")
                print("These tiles will be redownloaded on next run.")
                
                # Also delete the mosaic cache to force regeneration
                print("\nDeleting mosaic cache to force regeneration...")
                mosaic_pattern = os.path.join(output_dir, "*_mosaic.tif")
                mosaic_files = glob.glob(mosaic_pattern)
                for mosaic_file in mosaic_files:
                    try:
                        os.remove(mosaic_file)
                        print(f"  Deleted mosaic cache: {os.path.basename(mosaic_file)}")
                    except Exception as e:
                        print(f"  Error deleting mosaic: {e}")
            else:
                print("Deletion cancelled.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()