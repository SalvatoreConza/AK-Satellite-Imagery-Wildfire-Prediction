"""
preprocess_real_data.py
========================
Preprocess the real satellite data downloaded from GEE.
Works with what's available: S1 SAR, dNBR, fire labels.
S2 optical, terrain, and landcover will be added when ready.

Usage: python src/preprocess_real_data.py
"""

import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import re
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
TILES_DIR = Path("data/tiles")
TILES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:4326"  # Keep in lat/lon to match all sources
TILE_SIZE = 64
TARGET_RES = 0.0003  # ~30m in degrees at 64°N latitude


def find_tif_groups():
    """Find and group available TIF files by type."""
    files = list(RAW_DIR.glob("*.tif"))
    
    groups = defaultdict(list)
    for f in files:
        name = f.stem.lower()
        if "s1_pre_fire" in name:
            groups["s1_pre"].append(f)
        elif "s1_post_fire" in name:
            groups["s1_post"].append(f)
        elif "s2_pre_fire" in name:
            groups["s2_pre"].append(f)
        elif "s2_post_fire" in name:
            groups["s2_post"].append(f)
        elif "dnbr" in name:
            groups["dnbr"].append(f)
        elif "fire_labels" in name:
            groups["labels"].append(f)
        elif "terrain" in name:
            groups["terrain"].append(f)
        elif "landcover" in name:
            groups["landcover"].append(f)
    
    return groups


def merge_tiles(file_list):
    """Merge multiple GEE tile exports into a single raster."""
    if len(file_list) == 1:
        return rasterio.open(file_list[0])
    
    datasets = [rasterio.open(f) for f in sorted(file_list)]
    merged, transform = merge(datasets)
    
    # Create an in-memory profile
    profile = datasets[0].profile.copy()
    profile.update({
        "height": merged.shape[1],
        "width": merged.shape[2],
        "transform": transform,
        "count": merged.shape[0],
    })
    
    for ds in datasets:
        ds.close()
    
    return merged, profile


def load_and_resample(file_list, target_shape, target_transform, target_crs):
    """Load raster(s), merge if needed, and resample to target grid."""
    if len(file_list) == 1:
        src = rasterio.open(file_list[0])
        src_data = src.read()
        src_transform = src.transform
        src_crs = src.crs
        n_bands = src.count
        src.close()
    else:
        # Merge tiles
        datasets = [rasterio.open(f) for f in sorted(file_list)]
        src_data, src_transform = merge(datasets)
        src_crs = datasets[0].crs
        n_bands = src_data.shape[0]
        for ds in datasets:
            ds.close()
    
    H, W = target_shape
    dst_data = np.zeros((n_bands, H, W), dtype=np.float32)
    
    for i in range(n_bands):
        reproject(
            source=src_data[i],
            destination=dst_data[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    
    return dst_data


def normalize(band, clip_pct=(2, 98)):
    """Normalize a band to [0, 1] using percentile clipping."""
    valid = band[np.isfinite(band) & (band != 0)]
    if len(valid) == 0:
        return np.zeros_like(band)
    vmin, vmax = np.percentile(valid, clip_pct)
    if vmax - vmin < 1e-8:
        return np.zeros_like(band)
    return np.clip((band - vmin) / (vmax - vmin), 0, 1)


def create_tiles(features, labels, tile_size=TILE_SIZE, min_valid=0.7):
    """Tile features and labels into model-ready patches."""
    n_channels, H, W = features.shape
    
    tiles_X = []
    tiles_y = []
    coords = []
    
    for i in range(0, H - tile_size + 1, tile_size):
        for j in range(0, W - tile_size + 1, tile_size):
            tile_feat = features[:, i:i+tile_size, j:j+tile_size]
            tile_label = labels[i:i+tile_size, j:j+tile_size]
            
            # Check enough valid data
            valid_frac = np.mean(np.all(np.isfinite(tile_feat), axis=0))
            if valid_frac < min_valid:
                continue
            
            # Replace NaN with 0
            tile_feat = np.nan_to_num(tile_feat, 0.0).astype(np.float32)
            
            # Label: fire if >1% of pixels burned
            fire_frac = np.mean(tile_label > 0)
            label = 1 if fire_frac > 0.01 else 0
            
            tiles_X.append(tile_feat)
            tiles_y.append(label)
            coords.append((i, j))
    
    return tiles_X, tiles_y, coords


def spatial_split(tiles_X, tiles_y, coords, test_frac=0.2, n_blocks=5):
    """Spatial block split to prevent data leakage."""
    np.random.seed(42)
    coords_arr = np.array(coords)
    
    row_edges = np.linspace(coords_arr[:, 0].min(), coords_arr[:, 0].max() + 1, n_blocks + 1)
    col_edges = np.linspace(coords_arr[:, 1].min(), coords_arr[:, 1].max() + 1, n_blocks + 1)
    
    block_ids = np.zeros(len(coords), dtype=int)
    for idx, (r, c) in enumerate(coords):
        rb = min(np.searchsorted(row_edges[1:], r), n_blocks - 1)
        cb = min(np.searchsorted(col_edges[1:], c), n_blocks - 1)
        block_ids[idx] = rb * n_blocks + cb
    
    unique = np.unique(block_ids)
    np.random.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    test_blocks = set(unique[:n_test])
    
    train_idx = [i for i, b in enumerate(block_ids) if b not in test_blocks]
    test_idx = [i for i, b in enumerate(block_ids) if b in test_blocks]
    
    return train_idx, test_idx


def main():
    print("=" * 60)
    print("PREPROCESSING REAL SATELLITE DATA")
    print("=" * 60)
    
    groups = find_tif_groups()
    
    print("\nAvailable data:")
    for gtype, files in groups.items():
        total_mb = sum(f.stat().st_size / 1e6 for f in files)
        print(f"  {gtype:12s}: {len(files)} file(s), {total_mb:.0f} MB total")
    
    if "labels" not in groups:
        print("\nERROR: No fire_labels file found!")
        return
    
    # ── Step 1: Determine target grid from fire labels (smallest file) ──
    print("\n[1/4] Setting up target grid from fire labels...")
    with rasterio.open(groups["labels"][0]) as src:
        target_crs = src.crs
        target_transform = src.transform
        target_shape = (src.height, src.width)
        labels_data = src.read(1).astype(np.float32)
    
    print(f"  Grid: {target_shape[1]}x{target_shape[0]} pixels")
    print(f"  CRS: {target_crs}")
    print(f"  Burned pixels: {np.sum(labels_data > 0)} / {labels_data.size} "
          f"({100*np.mean(labels_data > 0):.2f}%)")
    
    # ── Step 2: Load and align all features ──
    print("\n[2/4] Loading and resampling features...")
    all_bands = []
    band_names = []
    
    # dNBR
    if "dnbr" in groups:
        print("  Loading dNBR...")
        dnbr = load_and_resample(groups["dnbr"], target_shape, target_transform, target_crs)
        all_bands.append(normalize(dnbr[0]))
        band_names.append("dNBR")
        print(f"    dNBR range: [{np.nanmin(dnbr):.3f}, {np.nanmax(dnbr):.3f}]")
    
    # S1 SAR pre-fire
    if "s1_pre" in groups:
        print("  Loading S1 pre-fire SAR (this may take a minute for large files)...")
        s1_pre = load_and_resample(groups["s1_pre"], target_shape, target_transform, target_crs)
        for i, name in enumerate(["VV_pre", "VH_pre", "VH_VV_pre"]):
            if i < s1_pre.shape[0]:
                all_bands.append(normalize(s1_pre[i]))
                band_names.append(name)
        print(f"    S1 pre: {s1_pre.shape[0]} bands loaded")
    
    # S1 SAR post-fire
    if "s1_post" in groups:
        print("  Loading S1 post-fire SAR...")
        s1_post = load_and_resample(groups["s1_post"], target_shape, target_transform, target_crs)
        for i, name in enumerate(["VV_post", "VH_post", "VH_VV_post"]):
            if i < s1_post.shape[0]:
                all_bands.append(normalize(s1_post[i]))
                band_names.append(name)
        print(f"    S1 post: {s1_post.shape[0]} bands loaded")
    
    # S2 optical pre-fire (if available)
    if "s2_pre" in groups:
        print("  Loading S2 pre-fire optical...")
        s2_pre = load_and_resample(groups["s2_pre"], target_shape, target_transform, target_crs)
        s2_names = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]
        for i in range(min(s2_pre.shape[0], len(s2_names))):
            all_bands.append(normalize(s2_pre[i]))
            band_names.append(f"{s2_names[i]}_pre")
        print(f"    S2 pre: {s2_pre.shape[0]} bands loaded")
    
    # Terrain (if available)
    if "terrain" in groups:
        print("  Loading terrain...")
        terrain = load_and_resample(groups["terrain"], target_shape, target_transform, target_crs)
        for i, name in enumerate(["elevation", "slope", "aspect"]):
            if i < terrain.shape[0]:
                all_bands.append(normalize(terrain[i]))
                band_names.append(name)
    
    # Stack all features
    feature_stack = np.stack(all_bands, axis=0)
    print(f"\n  Feature stack: {feature_stack.shape[0]} channels, "
          f"{feature_stack.shape[1]}x{feature_stack.shape[2]} pixels")
    print(f"  Bands: {band_names}")
    
    # ── Step 3: Create tiles ──
    print("\n[3/4] Creating tiles...")
    tiles_X, tiles_y, coords = create_tiles(feature_stack, labels_data)
    
    print(f"  Total tiles: {len(tiles_y)}")
    print(f"  Fire tiles: {sum(tiles_y)} ({100*sum(tiles_y)/max(len(tiles_y),1):.1f}%)")
    print(f"  No-fire tiles: {len(tiles_y) - sum(tiles_y)}")
    
    if len(tiles_y) == 0:
        print("ERROR: No valid tiles created! Check your data.")
        return
    
    # ── Step 4: Split and save ──
    print("\n[4/4] Spatial split and saving...")
    train_idx, test_idx = spatial_split(tiles_X, tiles_y, coords)
    
    X_train = np.stack([tiles_X[i] for i in train_idx])
    y_train = np.array([tiles_y[i] for i in train_idx])
    X_test = np.stack([tiles_X[i] for i in test_idx])
    y_test = np.array([tiles_y[i] for i in test_idx])
    
    print(f"  Train: {len(y_train)} tiles (fire: {sum(y_train)})")
    print(f"  Test:  {len(y_test)} tiles (fire: {sum(y_test)})")
    
    np.save(TILES_DIR / "X_train.npy", X_train)
    np.save(TILES_DIR / "y_train.npy", y_train)
    np.save(TILES_DIR / "X_test.npy", X_test)
    np.save(TILES_DIR / "y_test.npy", y_test)
    
    metadata = {
        "band_names": band_names,
        "n_channels": len(band_names),
        "tile_size": TILE_SIZE,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fire_rate_train": float(y_train.mean()),
        "fire_rate_test": float(y_test.mean()),
        "target_crs": str(target_crs),
    }
    with open(TILES_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DONE! Tiles saved to {TILES_DIR}/")
    print(f"  {metadata['n_channels']} channels: {band_names}")
    print(f"  {metadata['n_train']} train + {metadata['n_test']} test tiles")
    print(f"\nNext: python src/model.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()