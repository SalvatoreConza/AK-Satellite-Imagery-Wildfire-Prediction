"""
preprocessing.py
=================
Preprocessing pipeline for wildfire prediction data.

Takes raw GeoTIFF exports from GEE and ERA5 NetCDF files,
and produces model-ready tiles with aligned features.

Pipeline steps:
1. Load and align satellite imagery (Sentinel-2 + Sentinel-1)
2. Compute derived features (vegetation indices, SAR ratios)
3. Load and resample ERA5 weather data to match spatial grid
4. Tile the study area into fixed-size patches
5. Generate labels from fire perimeter data
6. Save as numpy arrays / PyTorch tensors for training
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import xarray as xr
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import json


# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────

TILE_SIZE = 64          # pixels per tile (64x64 at 10m = 640m x 640m)
TARGET_CRS = "EPSG:3338"  # Alaska Albers Equal Area projection
TARGET_RES = 10         # meters per pixel
OVERLAP = 0             # tile overlap in pixels (0 for no overlap)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TILES_DIR = Path("data/tiles")


# ─────────────────────────────────────────────────
# RASTER UTILITIES
# ─────────────────────────────────────────────────

def load_raster(filepath, target_crs=TARGET_CRS, target_res=TARGET_RES):
    """
    Load a GeoTIFF file and reproject to target CRS/resolution.
    
    Returns:
        data: np.ndarray of shape (bands, height, width)
        profile: rasterio profile dict
    """
    with rasterio.open(filepath) as src:
        # Calculate transform for target CRS/resolution
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height,
            *src.bounds, resolution=target_res
        )
        
        profile = src.profile.copy()
        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })
        
        data = np.zeros((src.count, height, width), dtype=np.float32)
        
        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=data[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
    
    return data, profile


def normalize_band(band, method="minmax", clip_range=None):
    """
    Normalize a single band.
    
    Args:
        band: 2D numpy array
        method: 'minmax' (0-1), 'standard' (zero mean, unit variance), or 'clip'
        clip_range: tuple (min, max) for clipping before normalization
    """
    if clip_range:
        band = np.clip(band, clip_range[0], clip_range[1])
    
    mask = np.isfinite(band) & (band != 0)
    
    if method == "minmax":
        vmin, vmax = np.nanpercentile(band[mask], [2, 98]) if mask.any() else (0, 1)
        if vmax - vmin < 1e-8:
            return np.zeros_like(band)
        return (band - vmin) / (vmax - vmin)
    
    elif method == "standard":
        mean = np.nanmean(band[mask]) if mask.any() else 0
        std = np.nanstd(band[mask]) if mask.any() else 1
        if std < 1e-8:
            return np.zeros_like(band)
        return (band - mean) / std
    
    return band


# ─────────────────────────────────────────────────
# SENTINEL-2 PREPROCESSING
# ─────────────────────────────────────────────────

def preprocess_sentinel2(filepath):
    """
    Load and preprocess Sentinel-2 composite.
    
    Expected bands from our GEE export (14 bands):
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NBR, NDMI, BSI
    
    Returns:
        dict with band names as keys, 2D arrays as values
    """
    band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
                  "B11", "B12", "NDVI", "NBR", "NDMI", "BSI"]
    
    data, profile = load_raster(filepath)
    
    print(f"  Sentinel-2: loaded {data.shape[0]} bands, {data.shape[1]}x{data.shape[2]} pixels")
    
    result = {}
    for i, name in enumerate(band_names):
        if i < data.shape[0]:
            if name in ["NDVI", "NBR", "NDMI", "BSI"]:
                # Indices are already in [-1, 1] range
                result[name] = np.clip(data[i], -1, 1)
            else:
                # Reflectance bands: S2 SR values are scaled by 10000
                result[name] = normalize_band(data[i] / 10000.0, method="minmax")
    
    return result, profile


# ─────────────────────────────────────────────────
# SENTINEL-1 SAR PREPROCESSING
# ─────────────────────────────────────────────────

def preprocess_sentinel1(filepath):
    """
    Load and preprocess Sentinel-1 SAR composite.
    
    Expected bands: VV, VH, VH_VV_ratio (all in dB)
    
    SAR values in GEE are in dB (typically -25 to 0 for VV, -30 to -5 for VH).
    We normalize to [0, 1] range.
    """
    band_names = ["VV", "VH", "VH_VV_ratio"]
    
    data, profile = load_raster(filepath)
    
    print(f"  Sentinel-1: loaded {data.shape[0]} bands, {data.shape[1]}x{data.shape[2]} pixels")
    
    result = {}
    for i, name in enumerate(band_names):
        if i < data.shape[0]:
            # SAR dB values: clip to typical range and normalize
            if name == "VV":
                result[name] = normalize_band(data[i], clip_range=(-25, 0))
            elif name == "VH":
                result[name] = normalize_band(data[i], clip_range=(-30, -5))
            else:
                result[name] = normalize_band(data[i], clip_range=(-15, 0))
    
    return result, profile


# ─────────────────────────────────────────────────
# ERA5 WEATHER DATA PREPROCESSING
# ─────────────────────────────────────────────────

def preprocess_era5(era5_dir, target_bounds, target_shape, months=[5, 6, 7, 8]):
    """
    Load ERA5 NetCDF files and create time-series weather features.
    
    Resamples coarse ERA5 grid (~31km) to match satellite image grid.
    
    Args:
        era5_dir: directory containing ERA5 NetCDF files
        target_bounds: (west, south, east, north) in target CRS
        target_shape: (height, width) of target raster
        months: list of months to include
    
    Returns:
        weather_ts: dict of {variable_name: np.array of shape (timesteps, H, W)}
        timestamps: list of datetime objects
    """
    from scipy.interpolate import RegularGridInterpolator
    
    all_data = {}
    timestamps = []
    
    era5_vars = {
        "t2m": "temperature_2m",
        "d2m": "dewpoint_2m", 
        "u10": "wind_u_10m",
        "v10": "wind_v_10m",
        "tp": "precipitation",
        "swvl1": "soil_moisture",
    }
    
    for month in months:
        filepath = os.path.join(era5_dir, f"era5_*_{month:02d}.nc")
        import glob
        files = glob.glob(filepath)
        if not files:
            print(f"  Warning: No ERA5 file found for month {month}")
            continue
        
        ds = xr.open_dataset(files[0])
        
        for era5_name, friendly_name in era5_vars.items():
            if era5_name in ds:
                var_data = ds[era5_name].values  # shape: (time, lat, lon)
                
                if friendly_name not in all_data:
                    all_data[friendly_name] = []
                all_data[friendly_name].append(var_data)
        
        if "time" in ds.dims:
            timestamps.extend(pd.to_datetime(ds.time.values).tolist())
        
        ds.close()
    
    # Concatenate all months
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)
    
    # Compute derived weather features
    if "temperature_2m" in all_data and "dewpoint_2m" in all_data:
        # Relative humidity from temperature and dewpoint
        t = all_data["temperature_2m"] - 273.15  # K to °C
        td = all_data["dewpoint_2m"] - 273.15
        all_data["relative_humidity"] = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    
    if "wind_u_10m" in all_data and "wind_v_10m" in all_data:
        # Wind speed from u and v components
        all_data["wind_speed"] = np.sqrt(
            all_data["wind_u_10m"]**2 + all_data["wind_v_10m"]**2
        )
    
    print(f"  ERA5: loaded {len(timestamps)} timesteps, {len(all_data)} variables")
    
    return all_data, timestamps


def compute_weather_statistics(weather_data, window_days=30):
    """
    Compute rolling statistics from weather time-series.
    
    For each variable, compute:
    - Mean over window
    - Max over window  
    - Variance over window
    
    These become spatial features that can be tiled alongside satellite data.
    """
    stats = {}
    
    # Assuming 6-hourly data: 4 observations per day
    window_size = window_days * 4
    
    for var_name, data in weather_data.items():
        if var_name in ["wind_u_10m", "wind_v_10m", "dewpoint_2m"]:
            continue  # Skip intermediate variables
        
        n_times = data.shape[0]
        
        if n_times < window_size:
            # Use all available data
            stats[f"{var_name}_mean"] = np.nanmean(data, axis=0)
            stats[f"{var_name}_max"] = np.nanmax(data, axis=0)
            stats[f"{var_name}_std"] = np.nanstd(data, axis=0)
        else:
            # Use last `window_size` timesteps
            window = data[-window_size:]
            stats[f"{var_name}_mean"] = np.nanmean(window, axis=0)
            stats[f"{var_name}_max"] = np.nanmax(window, axis=0)
            stats[f"{var_name}_std"] = np.nanstd(window, axis=0)
    
    return stats


# ─────────────────────────────────────────────────
# TILING: Cut study area into model-ready patches
# ─────────────────────────────────────────────────

def create_tiles(feature_stack, label_raster, tile_size=TILE_SIZE, overlap=OVERLAP,
                 min_valid_fraction=0.8, min_positive_fraction=0.001):
    """
    Tile the study area into fixed-size patches for model training.
    
    Args:
        feature_stack: np.array of shape (channels, H, W)
        label_raster: np.array of shape (H, W), binary (0/1)
        tile_size: int, pixels per tile side
        overlap: int, pixel overlap between tiles
        min_valid_fraction: float, minimum fraction of non-NaN pixels
        min_positive_fraction: float, minimum fraction of positive pixels to keep tile
    
    Returns:
        tiles_X: list of np.arrays, shape (channels, tile_size, tile_size)
        tiles_y: list of int labels (0=no fire, 1=fire)
        tile_coords: list of (row_start, col_start) tuples
    """
    _, H, W = feature_stack.shape
    step = tile_size - overlap
    
    tiles_X = []
    tiles_y = []
    tile_coords = []
    
    n_total = 0
    n_kept = 0
    
    for i in range(0, H - tile_size + 1, step):
        for j in range(0, W - tile_size + 1, step):
            n_total += 1
            
            # Extract tile
            tile_features = feature_stack[:, i:i+tile_size, j:j+tile_size]
            tile_label = label_raster[i:i+tile_size, j:j+tile_size]
            
            # Check valid data fraction
            valid_mask = np.all(np.isfinite(tile_features), axis=0)
            valid_fraction = valid_mask.mean()
            
            if valid_fraction < min_valid_fraction:
                continue
            
            # Fill NaN with 0
            tile_features = np.nan_to_num(tile_features, 0.0)
            
            # Determine tile label
            # A tile is "fire" if more than min_positive_fraction of pixels burned
            fire_fraction = (tile_label > 0).mean()
            tile_class = 1 if fire_fraction > min_positive_fraction else 0
            
            tiles_X.append(tile_features.astype(np.float32))
            tiles_y.append(tile_class)
            tile_coords.append((i, j))
            n_kept += 1
    
    print(f"  Tiling: {n_kept}/{n_total} tiles kept ({tile_size}x{tile_size})")
    print(f"    Fire tiles: {sum(tiles_y)}, No-fire tiles: {len(tiles_y) - sum(tiles_y)}")
    
    return tiles_X, tiles_y, tile_coords


# ─────────────────────────────────────────────────
# SPATIAL TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────

def spatial_train_test_split(tile_coords, tiles_X, tiles_y, test_fraction=0.2, seed=42):
    """
    Split tiles into train/test sets using spatial blocking.
    
    Unlike random splitting, this prevents spatial autocorrelation
    from leaking information between train and test sets.
    
    Strategy: Divide the study area into spatial blocks (5x5 grid)
    and assign entire blocks to train or test.
    """
    np.random.seed(seed)
    
    coords = np.array(tile_coords)
    
    # Create spatial blocks
    n_blocks = 5
    row_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max() + 1, n_blocks + 1)
    col_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max() + 1, n_blocks + 1)
    
    # Assign each tile to a block
    block_ids = np.zeros(len(coords), dtype=int)
    for idx, (r, c) in enumerate(coords):
        row_block = np.searchsorted(row_edges[1:], r)
        col_block = np.searchsorted(col_edges[1:], c)
        block_ids[idx] = row_block * n_blocks + col_block
    
    unique_blocks = np.unique(block_ids)
    np.random.shuffle(unique_blocks)
    
    n_test_blocks = max(1, int(len(unique_blocks) * test_fraction))
    test_blocks = set(unique_blocks[:n_test_blocks])
    
    train_idx = [i for i, b in enumerate(block_ids) if b not in test_blocks]
    test_idx = [i for i, b in enumerate(block_ids) if b in test_blocks]
    
    X_train = [tiles_X[i] for i in train_idx]
    y_train = [tiles_y[i] for i in train_idx]
    X_test = [tiles_X[i] for i in test_idx]
    y_test = [tiles_y[i] for i in test_idx]
    
    print(f"  Spatial split: {len(X_train)} train, {len(X_test)} test tiles")
    print(f"    Train fire rate: {sum(y_train)/max(len(y_train),1):.3f}")
    print(f"    Test fire rate:  {sum(y_test)/max(len(y_test),1):.3f}")
    
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────

def run_preprocessing_pipeline(raw_dir="data/raw", output_dir="data/tiles"):
    """
    Run the complete preprocessing pipeline.
    
    Expects these files in raw_dir (from GEE export):
    - s2_pre_fire_*.tif
    - s1_pre_fire_*.tif
    - fire_labels_*.tif
    - terrain_*.tif
    - landcover_*.tif
    - era5/ directory with NetCDF files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Find files
    raw = Path(raw_dir)
    s2_files = list(raw.glob("s2_pre_fire_*.tif"))
    s1_files = list(raw.glob("s1_pre_fire_*.tif"))
    label_files = list(raw.glob("fire_labels_*.tif"))
    terrain_files = list(raw.glob("terrain_*.tif"))
    
    if not s2_files:
        print("ERROR: No Sentinel-2 files found. Run data_acquisition.py first.")
        return
    
    # 1. Load Sentinel-2
    print("\n[1/5] Loading Sentinel-2...")
    s2_data, s2_profile = preprocess_sentinel2(str(s2_files[0]))
    
    # 2. Load Sentinel-1
    print("\n[2/5] Loading Sentinel-1...")
    if s1_files:
        s1_data, _ = preprocess_sentinel1(str(s1_files[0]))
    else:
        print("  Warning: No Sentinel-1 data found, skipping SAR features")
        s1_data = {}
    
    # 3. Load terrain
    print("\n[3/5] Loading terrain...")
    if terrain_files:
        terrain_raw, _ = load_raster(str(terrain_files[0]))
        terrain_data = {
            "elevation": normalize_band(terrain_raw[0]),
            "slope": normalize_band(terrain_raw[1]),
            "aspect": terrain_raw[2] / 360.0 if terrain_raw.shape[0] > 2 else None,
        }
    else:
        terrain_data = {}
    
    # 4. Stack all features
    print("\n[4/5] Stacking features...")
    all_bands = {}
    all_bands.update(s2_data)
    all_bands.update(s1_data)
    all_bands.update({k: v for k, v in terrain_data.items() if v is not None})
    
    band_names = list(all_bands.keys())
    
    # Ensure all bands have the same shape
    ref_shape = list(all_bands.values())[0].shape
    feature_stack = np.stack([
        np.resize(all_bands[name], ref_shape) if all_bands[name].shape != ref_shape 
        else all_bands[name]
        for name in band_names
    ], axis=0)
    
    print(f"  Feature stack: {feature_stack.shape[0]} channels, {feature_stack.shape[1]}x{feature_stack.shape[2]}")
    print(f"  Bands: {band_names}")
    
    # 5. Load labels and create tiles
    print("\n[5/5] Creating tiles...")
    if label_files:
        label_data, _ = load_raster(str(label_files[0]))
        label_raster = label_data[0]
        
        # Resize to match feature stack if needed
        if label_raster.shape != ref_shape:
            from skimage.transform import resize
            label_raster = resize(label_raster, ref_shape, order=0, preserve_range=True)
    else:
        print("  Warning: No label files found, creating dummy labels")
        label_raster = np.zeros(ref_shape, dtype=np.float32)
    
    tiles_X, tiles_y, tile_coords = create_tiles(feature_stack, label_raster)
    
    # Spatial train/test split
    X_train, y_train, X_test, y_test = spatial_train_test_split(
        tile_coords, tiles_X, tiles_y
    )
    
    # Save
    output = Path(output_dir)
    np.save(output / "X_train.npy", np.stack(X_train))
    np.save(output / "y_train.npy", np.array(y_train))
    np.save(output / "X_test.npy", np.stack(X_test))
    np.save(output / "y_test.npy", np.array(y_test))
    
    # Save metadata
    metadata = {
        "band_names": band_names,
        "tile_size": TILE_SIZE,
        "target_crs": TARGET_CRS,
        "target_resolution_m": TARGET_RES,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_channels": feature_stack.shape[0],
        "fire_rate_train": sum(y_train) / max(len(y_train), 1),
        "fire_rate_test": sum(y_test) / max(len(y_test), 1),
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Preprocessing complete!")
    print(f"  Saved to: {output_dir}")
    print(f"  Train: {metadata['n_train']} tiles, Test: {metadata['n_test']} tiles")
    print(f"  Channels: {metadata['n_channels']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_preprocessing_pipeline()
