# the script is organized in 4 phases:
# 1. config and setup
#   1.1 setup of the bounding box to slice the Alaska from satellite data
#   1.2 setup of the time window to slice a period before the fires peak (this after cleaning will be our pre fire data) and 
#   after the fire peak (this after some cleaning will be our post fire data)
#   1.3 inizialization and authentication into GEE
# 2. defining helper functions for sentinel2
#   2.1 defining of a function clod_mask_s2 to remove clouds and shadows from optical images. This is crucial for Alaska because it's cloudy 
#   2.2 defining of a function add_spectral_indices to calculates features(NDVI, NBR, NDMI) from satellite bands 
#   2.3 defining of a function get_sentinel2_composite to get the images from sentinel2 for the specified time window and bounding box
#       remove the clouds with clouds_mask_s2 and then calculate features with add_spectral_indices
#   2.4 defining of a function get_sentinel1_composite that do something similar but for radar data that can see through clouds
# 3. main execution
#   3.1 take the images from sentinel2 of the time windows that is the period before fires peak, create a map of vegetations without clouds thanks
#       to the functions in 2 and start a download of this pre fire data to google drive
#   3.2 take the images from sentinel2 of the time windows that is the periood after fires peak, create a map of vegetations without clouds thanks
#       to the functions in 2 and start a download of this post fire data to google drive
#   3.3 we calculate the damage as a difference map: (Pre-Fire NBR)-(Post-Fire NBR) 
#   3.4 & 3.5 get the radar data from sentinel-1 for pre and post fire. Radar sees thour smoke and clouds and calculate vh_vv_ratio = VH - VV
#             where VV is the bounces off surface moisture and HV is the bounces off complex structures (branches, leaves).
#             in this way we can estimate the biomass volume (how much stuff is there to burn) regardless of the cloud cover.
#   3.6 get the MTBS data, this would be the ground truth
#   3.7 download the SRMT satellites to get elevation, slope and the direction the hill faces.
#   3.8 downaload the NLCD map to tell the model is a pixel is a Forest or a City 
# 4. download weather data from European weather center (copernicus)
#   4.1 connect do CDS
#   4.2 dowlndload ERA5 hourly weather data for temperature, wind, precitipation, soil water
#   4.3 save the data locally in data/raw/era5/

import ee
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# PHASE 1 CONFIG AND SETUP

# IMPORTANT: Store bounds as a plain Python list.
# The ee.Geometry object is created AFTER ee.Initialize() inside export_all_data().
# Creating ee objects at module level causes "not initialized" errors.
STUDY_BOUNDS = [-150.5, 63.5, -149.0, 64.5]  # [west, south, east, north]
STUDY_NAME = "interior_alaska_fairbanks"

# time periods
FIRE_YEAR = 2022
PRE_FIRE_START = f"{FIRE_YEAR}-05-01"
PRE_FIRE_END = f"{FIRE_YEAR}-06-30"
POST_FIRE_START = f"{FIRE_YEAR}-08-01"
POST_FIRE_END = f"{FIRE_YEAR}-09-30"

# output directory
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project="gen-lang-client-0326473018")
        print("✓ GEE initialized successfully")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="gen-lang-client-0326473018")
        print("✓ GEE authenticated and initialized")


# PHASE 2 HELPER FUNCTIONS FOR SENTINEL2
def cloud_mask_s2(image):
    scl = image.select("SCL")
    cloud_mask = (
        scl.neq(3)   # cloud shadow
        .And(scl.neq(8))   # cloud medium probability
        .And(scl.neq(9))   # cloud high probability
        .And(scl.neq(10))  # thin cirrus
    )
    return image.updateMask(cloud_mask)

def add_spectral_indices(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    
    # Bare Soil Index
    bsi = image.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {
            "SWIR": image.select("B11"),
            "RED": image.select("B4"),
            "NIR": image.select("B8"),
            "BLUE": image.select("B2"),
        }
    ).rename("BSI")
    
    return image.addBands([ndvi, nbr, ndmi, bsi])


def get_sentinel2_composite(aoi, start_date, end_date, max_cloud_pct=30):
    """
    Get a cloud-free Sentinel-2 composite for a given area and time range.
    """
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(cloud_mask_s2)
        .map(add_spectral_indices)
    )
    
    # Select key bands + indices
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
             "NDVI", "NBR", "NDMI", "BSI"]
    
    composite = s2.select(bands).median().clip(aoi)
    
    scene_count = s2.size().getInfo()
    print(f"  Sentinel-2: {scene_count} scenes found ({start_date} to {end_date})")
    
    return composite


def get_sentinel1_composite(aoi, start_date, end_date):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(["VV", "VH"])
    )
    
    composite = s1.median().clip(aoi)
    
    # Add VH/VV ratio — good indicator of vegetation density
    vh_vv_ratio = composite.select("VH").subtract(composite.select("VV")).rename("VH_VV_ratio")
    composite = composite.addBands(vh_vv_ratio)
    
    scene_count = s1.size().getInfo()
    print(f"  Sentinel-1: {scene_count} scenes found ({start_date} to {end_date})")
    
    return composite

def get_fire_perimeters(aoi, year):
    """
    Get historical fire perimeters from MTBS (Monitoring Trends in Burn Severity).
    These serve as ground truth labels for training.
    """
    # MTBS burn severity
    mtbs = (
        ee.ImageCollection("USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
        .first()
    )
    
    if mtbs is not None:
        burn_mask = mtbs.select("Severity").gte(2).And(mtbs.select("Severity").lte(4))
        print(f"  MTBS: Fire perimeters loaded for {year}")
        return burn_mask.clip(aoi).rename("burned")
    
    # Fallback: MODIS burned area
    modis_ba = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
        .select("BurnDate")
    )
    
    burn_mask = modis_ba.max().gt(0).clip(aoi).rename("burned")
    print(f"  MODIS Burned Area: Loaded for {year}")
    return burn_mask


def get_active_fires(aoi, start_date, end_date):
    """Get MODIS and VIIRS active fire detections."""
    firms = (
        ee.ImageCollection("FIRMS")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select("T21")
    )
    
    fire_count = firms.count().clip(aoi).rename("fire_frequency")
    fire_max_temp = firms.max().clip(aoi).rename("max_fire_temp")
    
    print(f"  FIRMS active fires: loaded for {start_date} to {end_date}")
    return fire_count.addBands(fire_max_temp)



def get_terrain_features(aoi):
    """Get terrain features from SRTM DEM."""
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    terrain = ee.Terrain.products(dem)
    return terrain.select(["elevation", "slope", "aspect"])


def get_land_cover(aoi, year):
    """Get NLCD land cover classification."""
    nlcd = (
        ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
        .filterBounds(aoi)
        .sort("system:time_start", False)
        .first()
        .select("landcover")
        .clip(aoi)
    )
    return nlcd


def export_to_drive(image, description, aoi, scale=30, folder="wildfire_data"):
    """Export an ee.Image to Google Drive as GeoTIFF."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=aoi,
        scale=scale,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"  → Export started: {description} (scale={scale}m)")
    return task

# PHASE 3 MAIN EXECUTION
def export_all_data():
    # Initialize GEE first 
    initialize_gee()
    
    # NOW create the ee.Geometry (after initialization) ──
    STUDY_AREA = ee.Geometry.Rectangle(STUDY_BOUNDS)
    
    print(f"\n{'='*60}")
    print(f"STUDY AREA: {STUDY_NAME}")
    print(f"FIRE YEAR: {FIRE_YEAR}")
    print(f"{'='*60}\n")
    
    tasks = []
    
    # 1. Pre-fire Sentinel-2
    print("[1/8] Pre-fire Sentinel-2 composite...")
    s2_pre = get_sentinel2_composite(STUDY_AREA, PRE_FIRE_START, PRE_FIRE_END)
    tasks.append(export_to_drive(s2_pre, f"s2_pre_fire_{STUDY_NAME}_{FIRE_YEAR}", STUDY_AREA, scale=10))
    
    # 2. Post-fire Sentinel-2
    print("\n[2/8] Post-fire Sentinel-2 composite...")
    s2_post = get_sentinel2_composite(STUDY_AREA, POST_FIRE_START, POST_FIRE_END)
    tasks.append(export_to_drive(s2_post, f"s2_post_fire_{STUDY_NAME}_{FIRE_YEAR}", STUDY_AREA, scale=10))
    
    # 3. dNBR (burn severity change)
    print("\n[3/8] Computing dNBR (burn severity)...")
    dnbr = s2_pre.select("NBR").subtract(s2_post.select("NBR")).rename("dNBR")
    tasks.append(export_to_drive(dnbr, f"dnbr_{STUDY_NAME}_{FIRE_YEAR}", STUDY_AREA, scale=10))
    
    # 4. Pre-fire Sentinel-1 SAR
    print("\n[4/8] Pre-fire Sentinel-1 SAR composite...")
    s1_pre = get_sentinel1_composite(STUDY_AREA, PRE_FIRE_START, PRE_FIRE_END)
    tasks.append(export_to_drive(s1_pre, f"s1_pre_fire_{STUDY_NAME}_{FIRE_YEAR}", STUDY_AREA, scale=10))
    
    # 5. Post-fire Sentinel-1 SAR
    print("\n[5/8] Post-fire Sentinel-1 SAR composite...")
    s1_post = get_sentinel1_composite(STUDY_AREA, POST_FIRE_START, POST_FIRE_END)
    tasks.append(export_to_drive(s1_post, f"s1_post_fire_{STUDY_NAME}_{FIRE_YEAR}", STUDY_AREA, scale=10))
    
    # 6. Fire perimeters (labels)
    print("\n[6/8] Fire perimeters (ground truth)...")
    fire_labels = get_fire_perimeters(STUDY_AREA, FIRE_YEAR)
    tasks.append(export_to_drive(fire_labels, f"fire_labels_{STUDY_NAME}_{FIRE_YEAR}", STUDY_AREA, scale=30))
    
    # 7. Terrain
    print("\n[7/8] Terrain features...")
    terrain = get_terrain_features(STUDY_AREA)
    tasks.append(export_to_drive(terrain, f"terrain_{STUDY_NAME}", STUDY_AREA, scale=30))
    
    # 8. Land cover
    print("\n[8/8] Land cover...")
    lc = get_land_cover(STUDY_AREA, FIRE_YEAR)
    tasks.append(export_to_drive(lc, f"landcover_{STUDY_NAME}", STUDY_AREA, scale=30))
    
    print(f"\n{'='*60}")
    print(f"✓ {len(tasks)} export tasks submitted to Google Earth Engine.")
    print(f"  Check GEE Task Manager or Google Drive folder 'wildfire_data'")
    print(f"  Task manager: https://code.earthengine.google.com/tasks")
    print(f"{'='*60}")
    
    return tasks


# 4. DOWNLOAD WEATHER DATA

def download_era5_weather(year, months, bbox, output_path="data/raw/era5"):
    """
    Download ERA5 reanalysis weather data from Copernicus Climate Data Store.
    
    Requires: pip install cdsapi + ~/.cdsapirc configured
    """
    import cdsapi
    
    os.makedirs(output_path, exist_ok=True)
    c = cdsapi.Client()
    
    variables = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation",
        "volumetric_soil_water_layer_1",
    ]
    
    for month in months:
        output_file = os.path.join(output_path, f"era5_{year}_{month:02d}.nc")
        
        if os.path.exists(output_file):
            print(f"  ERA5 {year}-{month:02d}: already downloaded, skipping")
            continue
        
        print(f"  ERA5 {year}-{month:02d}: downloading...")
        
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                "area": bbox,
            },
            output_file,
        )
        print(f"    → Saved: {output_file}")
    
    print(f"  ✓ ERA5 download complete for {year}")


if __name__ == "__main__":
    # Export satellite data via GEE
    export_all_data()
    
    # Download ERA5 weather data (optional — needs CDS API key)
    alaska_bbox = [65.0, -152.0, 63.0, -148.0]
    
    try:
        download_era5_weather(
            year=FIRE_YEAR,
            months=[4, 5, 6, 7, 8, 9],
            bbox=alaska_bbox,
        )
    except Exception as e:
        print(f"\n  ⚠ ERA5 download skipped: {e}")
        print("  Set up CDS API key to enable ERA5 downloads.")
        print("  See: https://cds.climate.copernicus.eu/")