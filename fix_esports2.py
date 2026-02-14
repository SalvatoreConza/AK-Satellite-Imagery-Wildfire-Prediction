# the data acquisition script failed because the original script tried to use SRTM (Shuttle Radar Topography Mission) 
# for elevation data. SRTM only collected data between 60°N and 56°S. Alaska (64°N) is too far north, so SRTM returns 
# empty/null values there.

# the fix is ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2") which switch to 
# ALOS World 3D dataset from JAXA (Japanese Space Ahengy)

# the sentile2 data resolution of 10 m is to small and we have a massive file, we
# decide to pass to a coarser scale of 30 m. Also all the 13 bands+4 indices are again to 
# much data for a laptop, so we choosed to go for just 5: Three bands + NDVI + NBR


import ee

ee.Initialize(project="gen-lang-client-0326473018")
print("GEE initialized\n")

STUDY_AREA = ee.Geometry.Rectangle([-150.5, 63.5, -149.0, 64.5])
STUDY_NAME = "interior_alaska_fairbanks"
FOLDER = "wildfire_data"


def export(image, description, scale=30):
    task = ee.batch.Export.image.toDrive(
        image=image, description=description, folder=FOLDER,
        region=STUDY_AREA, scale=scale, maxPixels=1e13, fileFormat="GeoTIFF",
    )
    task.start()
    print(f"  -> Started: {description} (scale={scale}m)")


print("Exporting terrain (ALOS DEM - works for Alaska)...")
dem = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2").select("DSM").mosaic().clip(STUDY_AREA)
terrain = ee.Terrain.products(dem).select(["elevation", "slope", "aspect"])
export(terrain, f"terrain_{STUDY_NAME}", scale=30)

print("\nExporting Sentinel-2 (30m, core bands)...")


def cloud_mask(image):
    scl = image.select("SCL")
    return image.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))


def add_idx(image):
    return image.addBands([
        image.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        image.normalizedDifference(["B8", "B12"]).rename("NBR"),
    ])


bands = ["B4", "B8", "B12", "NDVI", "NBR"]  # Red, NIR, SWIR2, NDVI, NBR

s2_pre = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(STUDY_AREA).filterDate("2022-05-01", "2022-06-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .map(cloud_mask).map(add_idx).select(bands).median().clip(STUDY_AREA)
)
export(s2_pre, f"s2_pre_fire_{STUDY_NAME}_2022", scale=30)

s2_post = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(STUDY_AREA).filterDate("2022-08-01", "2022-09-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .map(cloud_mask).map(add_idx).select(bands).median().clip(STUDY_AREA)
)
export(s2_post, f"s2_post_fire_{STUDY_NAME}_2022", scale=30)

print(f"\n3 exports submitted. Check: https://code.earthengine.google.com/tasks")