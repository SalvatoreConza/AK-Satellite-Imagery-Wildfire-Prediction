# this script got me problem for different reasons, just look
# to fix.exports2.py


import ee

PROJECT = "gen-lang-client-0326473018"

ee.Initialize(project=PROJECT)
print("GEE initialized\n")

# cancel all READY tasks (duplicates) ──
print("Cancelling duplicate READY tasks...")
tasks = ee.batch.Task.list()
cancelled = 0
for t in tasks:
    if t.status()["state"] == "READY":
        t.cancel()
        cancelled += 1
print(f"  Cancelled {cancelled} READY tasks\n")

# config
STUDY_BOUNDS = [-150.5, 63.5, -149.0, 64.5]
STUDY_AREA = ee.Geometry.Rectangle(STUDY_BOUNDS)
STUDY_NAME = "interior_alaska_fairbanks"
FIRE_YEAR = 2022
FOLDER = "wildfire_data"


def export(image, description, scale=30):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=FOLDER,
        region=STUDY_AREA,
        scale=scale,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"  -> Export started: {description} (scale={scale}m)")
    return task


# re-export Sentinel-2 at 20m (was failing at 10m) ──
print("Re-exporting Sentinel-2 at 20m resolution...")


def cloud_mask_s2(image):
    scl = image.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask)


def add_indices(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return image.addBands([ndvi, nbr, ndmi])


bands = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]

# pre-fire S2
s2_pre = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(STUDY_AREA)
    .filterDate("2022-05-01", "2022-06-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .map(cloud_mask_s2)
    .map(add_indices)
    .select(bands)
    .median()
    .clip(STUDY_AREA)
)
export(s2_pre, f"s2_pre_fire_{STUDY_NAME}_{FIRE_YEAR}", scale=20)

# post-fire S2
s2_post = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(STUDY_AREA)
    .filterDate("2022-08-01", "2022-09-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .map(cloud_mask_s2)
    .map(add_indices)
    .select(bands)
    .median()
    .clip(STUDY_AREA)
)
export(s2_post, f"s2_post_fire_{STUDY_NAME}_{FIRE_YEAR}", scale=20)

# re-export terrain 
print("\nRe-exporting terrain...")
dem = ee.Image("USGS/SRTMGL1_003")
terrain = ee.Terrain.products(dem).select(["elevation", "slope", "aspect"]).clip(STUDY_AREA)
export(terrain, f"terrain_{STUDY_NAME}", scale=30)

# re-export land cover 
print("\nRe-exporting land cover...")
# use MODIS land cover (global coverage, works for Alaska)
landcover = (
    ee.ImageCollection("MODIS/061/MCD12Q1")
    .filterDate("2022-01-01", "2022-12-31")
    .first()
    .select("LC_Type1")
    .clip(STUDY_AREA)
)
export(landcover, f"landcover_{STUDY_NAME}", scale=500)

print(f"\n{'='*60}")
print("4 new export tasks submitted!")
print("Check progress: https://code.earthengine.google.com/tasks")
print(f"Files will appear in Google Drive -> {FOLDER}/")
print(f"{'='*60}")
