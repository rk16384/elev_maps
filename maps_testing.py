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
from shapely.geometry import box
import pynhd
import pygeohydro as gh
import requests
import pandas as pd

def get_cache_filename(bbox, resolution, location_name):
    """
    Generate a cache filename based on the bounding box and resolution.
    
    Args:
        bbox (tuple): Bounding box coordinates (west, south, east, north)
        resolution (int): Resolution in meters
    
    Returns:
        str: Filename in format 'dem_west_south_east_north_resolution.tif'
    """
    # Format coordinates to 3 decimal places
    coords = [f"{coord:.3f}" for coord in bbox]
    return f"dem_{location_name}_{'_'.join(coords)}_{resolution}m.tif"

def is_matching_bbox(bbox, search_directory):
    coords = [f"{coord:.3f}" for coord in bbox]
    bbox_str = "_".join(coords)

    # get all .tif files in search directory
    tif_files = [f for f in os.listdir(search_directory) if f.endswith('.tif')]

    # check if any of the tif files have the same bbox
    for tif_file in tif_files:
        if bbox_str in tif_file:
            return True, tif_file

    return False, None

def bbox_to_land_area(nw_corner, se_corner):
    """
    Calculate the land area of a bounding box in square meters.

    This function reprojects geographic coordinates (lat/lon) to a local
    azimuthal equidistant projection to accurately calculate the area, as
    calculating area from lat/lon coordinates directly results in an
    incorrect value in "square degrees".

    Args:
        nw_corner (tuple): (longitude, latitude) of the northwest corner.
        se_corner (tuple): (longitude, latitude) of the southeast corner.

    Returns:
        float: The area in square meters.
    """
    from shapely.geometry import Polygon
    from shapely.ops import transform
    import pyproj
    from functools import partial

    # Define the polygon from bounding box corners in WGS84 (latitude/longitude)
    polygon_wgs84 = Polygon([
        (nw_corner[0], nw_corner[1]),
        (se_corner[0], nw_corner[1]),
        (se_corner[0], se_corner[1]),
        (nw_corner[0], se_corner[1])
    ])

    # Use a local Azimuthal Equidistant projection centered on the polygon.
    # This projection is well-suited for preserving area for a local region.
    lon_c = polygon_wgs84.centroid.x
    lat_c = polygon_wgs84.centroid.y

    # Define the transformation from WGS84 to the custom AEQD projection
    project = partial(
        pyproj.transform,
        pyproj.Proj(proj='latlong', datum='WGS84'),
        pyproj.Proj(proj='aeqd', lat_0=lat_c, lon_0=lon_c)
    )

    # Apply the projection and get the area in square meters
    projected_polygon = transform(project, polygon_wgs84)
    area_in_sq_meters = projected_polygon.area

    # Convert area to square meters
    return area_in_sq_meters


def land_area_to_resolution(land_area):
    """
    Output the scan resolutions to check based on the pixelation from the total land area
    """
    # pixel area for hd 1080 * 720 = 777600 pixels
    # square meters = land_area
    pixel_area = 4800 * 6000  # Or 4800x6000 or 10800 x 7200

    minimum_pixel_size = np.sqrt(land_area / pixel_area) * 2.5 # 2.5 safety factor

    valid_search_resolutions = [100, 30, 10, 3, 1]

    # 1000000 square meters per square kilometer
    return [x for x in valid_search_resolutions if x <= minimum_pixel_size]



    # 1000000 square meters per square kilometer
    return land_area / 1000000


def normalize_to_uint8(array):
    """
    Normalize a numpy array to uint8 range (0-255), handling NaN values.
    
    Args:
        array (numpy.ndarray or Raster): Input array to normalize
    
    Returns:
        numpy.ndarray: Normalized array in uint8 format
    """
    # Convert Raster to numpy array if needed
    if hasattr(array, 'data'):
        data = array.data
    else:
        data = array

    # Replace NaN values with the minimum value
    data = np.nan_to_num(data, nan=np.nanmin(data))
    
    # Get min and max values
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Normalize to 0-255 range
    if data_max > data_min:
        normalized = 255 * (data - data_min) / (data_max - data_min)
    else:
        normalized = np.zeros_like(data)
    
    # Convert to uint8
    return normalized.astype(np.uint8)

def load_dem_from_cache(dem_cache_path, dem_cache_dir):
    # look for _<resolution>m using regex
    resolution = re.search(r'_(\d+)m.tif', dem_cache_path).group(1)
    try:
        dem_data = rioxarray.open_rasterio(os.path.join(dem_cache_dir, dem_cache_path))
        used_resolution = resolution
        print(f"Successfully loaded cached DEM data from {dem_cache_path}")

    except Exception as e:
        raise Exception(f"Error loading cached data: {e}")
    
    return dem_data, used_resolution

def check_resolutions(resolutions, bbox, dem_cache_dir, location_name, output_dem_file):
    dem_data = None
    used_resolution = None
    output_dem_path = os.path.join(os.path.dirname(output_dem_file), os.path.basename(output_dem_file))

    for resolution in resolutions:
        # Check if we have a cached version
        dem_cache_filename = get_cache_filename(bbox, resolution, location_name)
        dem_cache_path = os.path.join(dem_cache_dir, dem_cache_filename)

        try:
            print(f"\nTrying to download DEM data at {resolution}m resolution...")
            dem_data_attempt = py3dep.get_map(
                layers='DEM',
                geometry=bbox,
                resolution=resolution,
                crs="EPSG:4326"
            )
            
            # Add validation to ensure the downloaded data is a valid 2D grid.
            # py3dep can return a 1D array on failure without raising an exception.
            if dem_data_attempt.ndim < 2 or min(dem_data_attempt.shape) < 2:
                raise ValueError(f"Downloaded data at {resolution}m resolution is not a valid 2D grid.")

            # Save the DEM data to both the cache and the output location
            dem_data_attempt.rio.to_raster(dem_cache_path)
            dem_data_attempt.rio.to_raster(output_dem_path)
            
            dem_data = dem_data_attempt
            used_resolution = resolution
            print(f"Success! DEM data downloaded at {resolution}m resolution and saved to {dem_cache_path}")
            break  # Exit the loop on successful download
        except Exception as e:
            print(f"Failed to download at {resolution}m resolution. This is often expected for high resolutions in areas with no coverage.")
            # Clean up if a file was partially created
            if os.path.exists(output_dem_path):
                os.remove(output_dem_path)
            if os.path.exists(dem_cache_path):
                os.remove(dem_cache_path)

    if dem_data is None:
        raise Exception("\nCould not download DEM data at any of the attempted resolutions.")
        
    return dem_data, used_resolution
        

def get_nhd_water_mask(bbox, dem):
    """
    Fetch water body data from the National Hydrography Dataset and create a water mask.
    """
    print("Fetching NHD water body data for a high-accuracy water mask...")
    try:
        west, south, east, north = bbox
        layers = [9, 2, 3, 4]
        all_water_features = []
        # FCodes for perennial lakes, ponds, and reservoirs
        include_fcodes = [
            39004,  # Lake/Pond - Perennial
            39009,  # Lake/Pond - Average water elevation
            39010,  # Lake/Pond - Normal pool
            43615,  # Reservoir - Earthen, perennial
            43617,  # Reservoir - Perennial
            43621,  # Reservoir - Perennial
        ]

        where_clause = f"FCode IN ({','.join(map(str, include_fcodes))})"
        from urllib.parse import quote
        encoded_where = quote(where_clause)
        
        for layer in layers:
            nhd_url = (
                "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/"
                f"{layer}/query?"
                f"geometry={west},{south},{east},{north}&"
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
                        all_water_features.append(water_features)
                        print(f"Found {len(water_features)} water features in layer {layer}")
                except Exception as e:
                    print(f"Error parsing GeoJSON response for layer {layer}: {e}")

        if not all_water_features:
            return None

        # Combine all water features into a single GeoDataFrame
        water_bodies_gdf = pd.concat(all_water_features, ignore_index=True)
        print(f"Total water features found: {len(water_bodies_gdf)}")

        # Debug: Print CRS information
        print(f"DEM CRS: {dem.crs}")
        print(f"Water features CRS before transformation: {water_bodies_gdf.crs}")

        # Ensure the hydrography data is in the same projection as our DEM
        water_bodies_gdf = water_bodies_gdf.to_crs(dem.crs)
        print(f"Water features CRS after transformation: {water_bodies_gdf.crs}")
        
        # Debug: Print dimensions
        print(f"DEM shape: {dem.shape}")
        
        # For flowlines (rivers), buffer them to give them width
        if 'FTYPE' in water_bodies_gdf.columns:
            flowlines = water_bodies_gdf[water_bodies_gdf['FTYPE'] == 'StreamRiver']
            if not flowlines.empty:
                # Buffer size in the same units as the CRS
                buffer_size = 30  # meters if using EPSG:3857
                flowlines_buffered = flowlines.geometry.buffer(buffer_size)
                water_bodies_gdf.loc[flowlines.index, 'geometry'] = flowlines_buffered

        # Rasterize the vector polygons into a pixel mask
        water_mask = rasterize(
            shapes=water_bodies_gdf.geometry,
            out_shape=dem.shape,
            transform=dem.transform,
            fill=0,
            dtype='uint8'
        ).astype(bool)
        
        # Debug: Print mask information
        print(f"Water mask shape: {water_mask.shape}")
        print(f"Number of water pixels: {np.sum(water_mask)}")
        print(f"Percentage of area covered by water: {(np.sum(water_mask) / water_mask.size) * 100:.2f}%")
        
        # Save debug images
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save the water mask as a PNG for visual inspection
        water_mask_img = Image.fromarray(water_mask.astype(np.uint8) * 255)
        water_mask_img.save(os.path.join(debug_dir, "water_mask_debug.png"))
        
        return water_mask

    except Exception as e:
        print(f"Warning: Could not fetch or process NHD data. Error: {e}")
        print("Falling back to slope-based water detection...")
        return None

def find_best_dem_and_hillshade(bbox, output_dem_file="dem.tif", output_hillshade_file="hillshade.tif",
                               output_png_file="hillshade.png", location_name="location_name",
                               sun_azimuth=315, sun_altitude=45, use_cache=True, resolutions=None,
                               vertical_exaggeration=1.0, use_nhd_water=False,
                               flatten_water_by_slope=False, water_slope_threshold=1.0,
                               water_color=50, water_min_area_pixels=100, water_dilation_pixels=2, gaussian_blur_sigma=0):
    """
    Tries multiple data sources and resolutions to find the best available DEM,
    downloads it, and generates a hillshade.

    Args:
        bbox (tuple): A tuple defining the bounding box as (west, south, east, north).
        output_dem_file (str): The path to save the downloaded DEM file.
        output_hillshade_file (str): The path to save the generated hillshade file.
        output_png_file (str): The path to save the generated hillshade as a PNG file.
        sun_azimuth (float): The azimuth angle of the sun in degrees (0-360).
        sun_altitude (float): The altitude angle of the sun in degrees (0-90).
        use_cache (bool): Whether to use cached DEM data if available.
        resolutions (list): A list of resolutions to attempt, in meters.
        vertical_exaggeration (float): Factor to exaggerate vertical elevation. Default is 1.0 (no exaggeration).
        use_nhd_water (bool): Whether to use the National Hydrography Dataset for an accurate water mask.
        flatten_water_by_slope (bool): Whether to identify and color water bodies based on slope (less accurate).
        water_slope_threshold (float): The slope threshold in degrees to identify water.
        water_color (int): The grayscale color (0-255) to use for water bodies.
        water_min_area_pixels (int): The minimum number of pixels for a flat area to be considered water.
        water_dilation_pixels (int): The number of pixels to dilate the water mask to cover edge artifacts.
    """
    print(f"Searching for DEM for bounding box: {bbox}")

    # Create dem_downloads directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dem_cache_dir = os.path.join(current_dir, "dem_downloads")
    os.makedirs(dem_cache_dir, exist_ok=True)

    # make output folder
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    output_dem_path = os.path.join(output_dir, output_dem_file)
    output_hillshade_path = os.path.join(output_dir, output_hillshade_file)
    output_png_path = os.path.join(output_dir, output_png_file)

    dem_data = None
    used_resolution = None

    if use_cache:
        is_matching, matching_file = is_matching_bbox(bbox, dem_cache_dir)
        if is_matching:
            dem_data, used_resolution = load_dem_from_cache(matching_file, dem_cache_dir)
        else:
            dem_data, used_resolution = check_resolutions(resolutions, bbox, dem_cache_dir, location_name, output_dem_path)
    else:
        dem_data, used_resolution = check_resolutions(resolutions, bbox, dem_cache_dir, location_name, output_dem_path)

    
    # --- Generate Hillshade ---
    print("\nGenerating hillshade...")
    try:
        # Ensure the CRS is projected for accurate calculations
        if dem_data.rio.crs.is_geographic:
            print("Reprojecting to a projected CRS (EPSG:3857) for hillshade calculation.")
            dem_data = dem_data.rio.reproject("EPSG:3857")

        # Extract the elevation data as a numpy array
        # if dem_data is 2d dont take values[0] just take values
        if dem_data.data.ndim == 2:
            elevation = dem_data.values
        else:
            elevation = dem_data.values[0]  # Get the first band

        # Apply vertical exaggeration to enhance terrain features
        if vertical_exaggeration != 1.0:
            print(f"Applying vertical exaggeration of {vertical_exaggeration}x")
            elevation = elevation * vertical_exaggeration

        # Replace NaNs in elevation data with minimum value
        elevation = np.nan_to_num(elevation, nan=np.nanmin(elevation))

        # Create a DEM object from the numpy array
        dem = xdem.DEM.from_array(
            elevation,
            transform=dem_data.rio.transform(),
            crs=dem_data.rio.crs
        )

        # --- Generate a water mask ---
        water_mask = None
        # NHD is the preferred, more accurate method
        if use_nhd_water:
            water_mask = get_nhd_water_mask(bbox, dem)
        
        # Fallback to slope-based method if NHD is not used or failed
        elif flatten_water_by_slope:
            print("Identifying potential water bodies based on slope...")
            # Calculate slope in degrees
            slope_map = xdem.terrain.slope(dem)
            
            # Mask areas with a slope of less than the threshold (typical for water)
            # We use the raw data array for boolean masking and squeeze it to 2D
            mask = (slope_map.data < water_slope_threshold).squeeze()

            # Clean the mask with morphological operations to remove noise.
            # binary_opening removes small, isolated flat areas that aren't lakes.
            print("Cleaning up water mask (initial pass)...")
            opened_mask = binary_opening(mask, iterations=2)

            print(f"Filtering water bodies smaller than {water_min_area_pixels} pixels...")
            # Label connected regions in the binary mask
            labeled_mask, num_features = label(opened_mask)
            
            # Calculate the size of each labeled feature
            feature_sizes = np.bincount(labeled_mask.ravel())
            
            # Identify features (labels) that are smaller than the threshold
            too_small = np.where(feature_sizes < water_min_area_pixels)[0]
            
            # Create a boolean mask of the same shape as labeled_mask
            # It will be True for pixels belonging to small features
            remove_mask = np.isin(labeled_mask, too_small)
            
            # Remove the small features from the original mask
            water_mask_filtered = opened_mask.copy()
            water_mask_filtered[remove_mask] = False

            # Dilate the final mask to cover up edge artifacts from DEM tiles
            if water_dilation_pixels > 0:
                print(f"Dilating water mask by {water_dilation_pixels} pixels to cover edges...")
                water_mask = binary_dilation(water_mask_filtered, iterations=water_dilation_pixels)
            else:
                water_mask = water_mask_filtered

        # Generate the hillshade with custom sun parameters
        hillshade = xdem.terrain.hillshade(
            dem,
            azimuth=sun_azimuth,
            altitude=sun_altitude
        )

        # --- FIX: Crop the hillshade to remove NaN edges ---
        # The hillshade algorithm produces NaN on border pixels. We crop them
        # by slicing the numpy array and updating the geotransform.
        crop_size = 1

        # Crop the underlying numpy array
        cropped_data = hillshade.data.data[crop_size:-crop_size, crop_size:-crop_size]

        # --- Crop the water mask identically ---
        if water_mask is not None:
            cropped_water_mask = water_mask[crop_size:-crop_size, crop_size:-crop_size]

        # Update the georeferencing transform for the cropped raster
        transform = hillshade.transform
        new_transform = Affine(transform.a, transform.b, transform.c + crop_size * transform.a,
                               transform.d, transform.e, transform.f + crop_size * transform.e)

        # Create a new, cropped DEM object
        hillshade_cropped = xdem.DEM.from_array(
            data=cropped_data,
            transform=new_transform,
            crs=hillshade.crs,
        )

        # --- FIX: Normalize the cropped data and save it ---
        # Convert the cropped hillshade data to a normalized uint8 array
        hillshade_uint8_data = normalize_to_uint8(hillshade_cropped.data.data)

        # --- Apply the water mask to the final image ---
        if water_mask is not None and 'cropped_water_mask' in locals():
            print(f"Applying water mask with color {water_color}")
            print(f"Final image shape: {hillshade_uint8_data.shape}")
            print(f"Water mask shape: {cropped_water_mask.shape}")
            # Make sure the mask matches the dimensions
            if hillshade_uint8_data.shape == cropped_water_mask.shape:
                hillshade_uint8_data[cropped_water_mask] = water_color
            else:
                print(f"Warning: Water mask shape {cropped_water_mask.shape} doesn't match final image shape {hillshade_uint8_data.shape}")

        # Create a new raster object with the final uint8 data for saving
        hillshade_to_save = xdem.DEM.from_array(
            data=hillshade_uint8_data,
            transform=hillshade_cropped.transform,
            crs=hillshade_cropped.crs
        )
        hillshade_to_save.set_nodata(0)

        # Save the final, cropped, and normalized hillshade
        hillshade_to_save.save(
            output_hillshade_path,
            compress='LZW',
            dtype='uint8'
        )

        print(f"\nHillshade successfully generated and saved to {output_hillshade_path}")

        # --- Save a PNG for easy viewing ---
        try:
            # Apply a slight Gaussian blur to smooth the hillshade and reduce artifacts
            blurred_data = gaussian_filter(hillshade_uint8_data, sigma=gaussian_blur_sigma)
            png_image = Image.fromarray(blurred_data)
            png_image.save(output_png_path)
            print(f"Successfully saved PNG preview as '{output_png_path}'")
        except Exception as e:
            print(f"An error occurred while saving the PNG image: {e}")

        print(f"Sun parameters: Azimuth={sun_azimuth}°, Altitude={sun_altitude}°")
        print(f"Using DEM data at {used_resolution}m resolution")

    except Exception as e:
        print(f"An error occurred during hillshade generation: {e}")
        # Clean up if a file was partially created
        if os.path.exists(output_hillshade_file):
            os.remove(output_hillshade_file)

if __name__ == "__main__":
    # Define the bounding box for your area of interest
    # Example: A mountainous area in Colorado
    # Format: (West, South, East, North)

    bbs = {
        "cottonwoods": {
            "NW_corner": (-111.7925, 40.677),
            "SE_corner": (-111.542348, 40.483556),
            "location_name": "cottonwoods"
        },
        "tc": {
            "NW_corner": (-86.296739, 45.209229),
            "SE_corner": (-85.233928, 44.578334),
            "location_name": "tc"
        }
    }
    location = "cottonwoods"

    loc_data = bbs[location]
    location_name = loc_data["location_name"]
    nw_corner = loc_data["NW_corner"]
    se_corner = loc_data["SE_corner"]
    land_area = bbox_to_land_area(nw_corner, se_corner)
    resolutions = land_area_to_resolution(land_area)

    # Bounding box format is (west, south, east, north)
    bounding_box = (nw_corner[0], se_corner[1], se_corner[0], nw_corner[1])

    output_png_filename = f"{location_name}_{nw_corner[0]}_{nw_corner[1]}_{se_corner[0]}_{se_corner[1]}_hillshade.png"

    # Example with custom sun parameters
    # Try different combinations to get the desired effect:
    # - sun_azimuth: 0-360 degrees (0=North, 90=East, 180=South, 270=West)
    # - sun_altitude: 0-90 degrees (0=horizon, 90=overhead)
    find_best_dem_and_hillshade(
        bounding_box,
        sun_azimuth=0,  # Northwest sun
        sun_altitude=45,  # 45 degrees above horizon
        use_cache=True,    # Enable caching
        output_png_file=output_png_filename, # Specify a PNG output file
        location_name=location_name,
        resolutions=resolutions,
        vertical_exaggeration=5.0,  # Exaggerate the vertical dimension by 2x
        use_nhd_water=True, # Use the accurate NHD data source
        water_color=70  # Set water to a dark gray color

    )