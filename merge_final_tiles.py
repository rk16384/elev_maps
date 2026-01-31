"""
Standalone script to merge and downsample the final large intermediate tiles.
"""
import os
import gc
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.enums import Resampling
import rioxarray

# Configuration
INTERMEDIATE_DIR = "/Volumes/sandisk1/data_cache/Oregon/elevation_dem/tiles/merge_intermediate"
OUTPUT_DIR = "/Volumes/sandisk1/data_cache/Oregon/elevation_dem"
LOCATION_NAME = "Oregon"

# Downsample factor (2 = half resolution, 4 = quarter resolution)
DOWNSAMPLE_FACTOR = 3

def get_batch_files(batch_num):
    """Get all files from a specific batch"""
    files = []
    for f in os.listdir(INTERMEDIATE_DIR):
        if f.startswith(f'merge_b{batch_num}_') and f.endswith('.tif') and not f.startswith('._'):
            path = os.path.join(INTERMEDIATE_DIR, f)
            # Skip 0-byte files
            if os.path.getsize(path) > 0:
                files.append(path)
    return sorted(files)

def downsample_tile(input_path, output_path, factor):
    """Downsample a single tile by the given factor"""
    print(f"  Downsampling {os.path.basename(input_path)}...")
    
    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        new_height = src.height // factor
        new_width = src.width // factor
        
        # Read and resample
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )
        
        # Update transform
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Write output
        profile = src.profile.copy()
        profile.update(
            height=new_height,
            width=new_width,
            transform=transform
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    
    print(f"    Done: {new_width}x{new_height}")
    gc.collect()

def merge_tiles(file_paths, output_dir, location_name):
    """Merge multiple tiles into one and save with correct naming convention"""
    print(f"Merging {len(file_paths)} tiles...")
    
    srcs = [rasterio.open(f) for f in file_paths]
    try:
        mosaic_arr, out_trans = rio_merge(srcs, method='first')
        
        # Calculate bounds from transform and shape
        height = mosaic_arr.shape[1]
        width = mosaic_arr.shape[2]
        minx = out_trans.c
        maxy = out_trans.f
        maxx = minx + width * out_trans.a
        miny = maxy + height * out_trans.e  # e is negative
        
        # Create bbox string matching the naming convention
        bbox_str = f"{minx:.3f}_{miny:.3f}_{maxx:.3f}_{maxy:.3f}"
        output_filename = f"elevation_dem_{location_name}_{bbox_str}_mosaic.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Get metadata from first source
        profile = srcs[0].profile.copy()
        profile.update(
            height=height,
            width=width,
            transform=out_trans
        )
        
        print(f"  Output size: {width}x{height}")
        print(f"  Bounds: ({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f})")
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mosaic_arr)
            
    finally:
        for src in srcs:
            src.close()
    
    gc.collect()
    print(f"  Saved to {output_path}")
    return output_path

def main():
    # Get batch 5 files (the large ones)
    batch5_files = get_batch_files(5)
    print(f"Found {len(batch5_files)} files from batch 5:")
    for f in batch5_files:
        size_gb = os.path.getsize(f) / (1024**3)
        print(f"  {os.path.basename(f)}: {size_gb:.1f} GB")
    
    if not batch5_files:
        print("No batch 5 files found!")
        return
    
    # Step 1: Downsample each tile
    print(f"\nStep 1: Downsampling by factor of {DOWNSAMPLE_FACTOR}...")
    downsampled_files = []
    for i, f in enumerate(batch5_files):
        output_name = f"downsampled_{i}.tif"
        output_path = os.path.join(INTERMEDIATE_DIR, output_name)
        downsample_tile(f, output_path, DOWNSAMPLE_FACTOR)
        downsampled_files.append(output_path)
        
        # Check new file size
        size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"    New size: {size_gb:.1f} GB")
    
    # Step 2: Merge downsampled tiles
    print(f"\nStep 2: Merging {len(downsampled_files)} downsampled tiles...")
    output_path = merge_tiles(downsampled_files, OUTPUT_DIR, LOCATION_NAME)
    
    # Step 3: Report final size
    final_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"\nFinal mosaic: {output_path}")
    print(f"Final size: {final_size_gb:.1f} GB")
    
    # Optional: Clean up downsampled intermediates
    print("\nCleaning up intermediate downsampled files...")
    for f in downsampled_files:
        try:
            os.remove(f)
            print(f"  Removed {os.path.basename(f)}")
        except Exception as e:
            print(f"  Could not remove {f}: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

