# how to run: python demo_pipeline.py


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# virtual alaska creation made by a grid 50x50 with some vegetations values
# temperature and SAR based on common sense (more warmer and more vegetation in the south than in the north) 
def generate_synthetic_alaska_data(n_locations=500, grid_size=50):

    print("=" * 60)
    print("STEP 1: Generating synthetic Alaska wildfire data")
    print("=" * 60)
    
    # create spatial grid (simulating tiles across interior Alaska)
    lats = np.linspace(63.0, 65.0, grid_size)
    lons = np.linspace(-152.0, -148.0, grid_size)
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    
    n_tiles = grid_size * grid_size
    
    # Sentinel2 features 
    # NDVI: higher in boreal forest (south), lower in tundra (north)
    base_ndvi = 0.6 - 0.15 * (lat_grid.flatten() - 63.0) / 2.0
    ndvi = base_ndvi + np.random.normal(0, 0.1, n_tiles)
    ndvi = np.clip(ndvi, -0.1, 0.9)
    
    # NBR (Normalized Burn Ratio): correlates with vegetation health
    nbr = 0.8 * ndvi + np.random.normal(0, 0.05, n_tiles)
    
    # NDMI (Moisture Index): lower in drier areas
    ndmi = 0.5 * ndvi + 0.3 + np.random.normal(0, 0.08, n_tiles)
    
    # BSI (Bare Soil Index): higher where less vegetation
    bsi = -0.6 * ndvi + 0.2 + np.random.normal(0, 0.05, n_tiles)
    
    # Sentinel1 SAR features 
    # VV backscatter (soil moisture proxy): -25 to 0 dB
    vv = -15 + 5 * ndvi + np.random.normal(0, 2, n_tiles)
    
    # VH backscatter (vegetation structure): -30 to -5 dB
    vh = -20 + 7 * ndvi + np.random.normal(0, 2, n_tiles)
    
    # VH/VV ratio (vegetation density indicator)
    vh_vv = vh - vv + np.random.normal(0, 0.5, n_tiles)
    
    # terrain features 
    # elevation: varies across grid
    elevation = 200 + 300 * np.sin(lon_grid.flatten() * 2) + 100 * np.sin(lat_grid.flatten() * 3)
    elevation += np.random.normal(0, 50, n_tiles)
    
    slope = np.abs(np.gradient(elevation.reshape(grid_size, grid_size))[0].flatten()) * 10
    slope += np.random.normal(0, 2, n_tiles)
    slope = np.clip(slope, 0, 45)
    
    aspect = np.random.uniform(0, 360, n_tiles)
    
    # ERA5 Weather features (monthly averages for pre-fire period) ──
    # Temperature (°C): warmer south, seasonal variation
    temp_mean = 18 + 3 * np.random.randn(n_tiles) - 2 * (lat_grid.flatten() - 63.0)
    temp_max = temp_mean + 5 + 2 * np.random.randn(n_tiles)
    
    # Relative humidity (%): lower = drier = more fire risk
    humidity_mean = 55 + 15 * np.random.randn(n_tiles) + 10 * (ndmi - 0.5)
    humidity_mean = np.clip(humidity_mean, 15, 95)
    
    # wind speed (m/s)
    wind_mean = 3 + 2 * np.random.randn(n_tiles)
    wind_max = wind_mean + 4 + 2 * np.random.randn(n_tiles)
    wind_mean = np.clip(wind_mean, 0.5, 15)
    wind_max = np.clip(wind_max, wind_mean, 25)
    
    # days since last rain
    days_no_rain = np.random.exponential(8, n_tiles)
    days_no_rain = np.clip(days_no_rain, 0, 60)
    
    # soil moisture (volumetric, 0-1)
    soil_moisture = 0.3 + 0.2 * (humidity_mean / 100) + np.random.normal(0, 0.05, n_tiles)
    soil_moisture = np.clip(soil_moisture, 0.05, 0.6)
    
    # precipitation last 30 days (mm)
    precip_30d = 30 + 20 * np.random.randn(n_tiles) + 15 * (humidity_mean / 100)
    precip_30d = np.clip(precip_30d, 0, 150)
    
    # lightning density (ignition source) 
    lightning_density = np.random.exponential(0.3, n_tiles)
    
    # generate fire labels (target variable) 
    # fire probability model (mimics real fire drivers)
    fire_logit = (
        -3.0                                # base (fires are rare)
        + 2.0 * (ndvi - 0.3)                # more vegetation = more fuel
        - 2.5 * (soil_moisture - 0.3)       # drier soil = more risk
        + 1.5 * (temp_mean - 15) / 10       # hotter = more risk
        - 1.0 * (humidity_mean - 50) / 30   # drier air = more risk
        + 0.8 * (wind_max - 5) / 5          # windier = more spread
        + 1.2 * (days_no_rain - 10) / 15    # longer dry spell = more risk
        + 1.5 * (lightning_density - 0.3)   # more lightning = more ignition
        - 0.5 * (elevation - 300) / 300     # lower elevation = more risk
        + np.random.normal(0, 0.5, n_tiles) # noise
    )
    
    fire_prob = 1 / (1 + np.exp(-fire_logit))
    fire_label = (np.random.uniform(0, 1, n_tiles) < fire_prob).astype(int)
    
    # create 3-class labels: 0=No Risk, 1=Moderate, 2=High Risk
    risk_class = np.zeros(n_tiles, dtype=int)
    risk_class[fire_prob > 0.3] = 1  # Moderate
    risk_class[fire_prob > 0.6] = 2  # High
    
    #  assemble feature matrix 
    feature_names = [
        # Sentinel2 (optical)
        "NDVI", "NBR", "NDMI", "BSI",
        # Sentinel1 (SAR)
        "VV_backscatter", "VH_backscatter", "VH_VV_ratio",
        # terrain
        "elevation", "slope", "aspect_sin", "aspect_cos",
        # weather
        "temp_mean", "temp_max", "humidity_mean",
        "wind_mean", "wind_max", "days_no_rain",
        "soil_moisture", "precip_30d",
        # ignition
        "lightning_density",
    ]
    
    X = np.column_stack([
        ndvi, nbr, ndmi, bsi,
        vv, vh, vh_vv,
        elevation, slope, np.sin(np.radians(aspect)), np.cos(np.radians(aspect)),
        temp_mean, temp_max, humidity_mean,
        wind_mean, wind_max, days_no_rain,
        soil_moisture, precip_30d,
        lightning_density,
    ])
    
    coords = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
    
    print(f"  Generated {n_tiles} tiles with {len(feature_names)} features")
    print(f"  Fire rate: {fire_label.mean():.1%}")
    print(f"  Risk distribution: No={np.sum(risk_class==0)}, "
          f"Moderate={np.sum(risk_class==1)}, High={np.sum(risk_class==2)}")
    
    return X, fire_label, risk_class, fire_prob, coords, feature_names, lat_grid, lon_grid


# this function help us to define a way to spatially split the data, if we randomically
# split them in fact a train tile might be next to a test tile, and is no good because the 
# model would cheat by memorizing the location, Since neighboring tiles usually have the 
# same weather and vegetation. For this reason we cut the map into large blocks and we hide them 
# for testing.

def spatial_block_split(X, y, coords, test_fraction=0.2, n_blocks=5):
    """
    Split data using spatial blocks to prevent autocorrelation leakage.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Spatial block train/test split")
    print("=" * 60)
    
    lat_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max() + 0.01, n_blocks + 1)
    lon_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max() + 0.01, n_blocks + 1)
    
    block_ids = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        lat_block = np.searchsorted(lat_edges[1:], coords[i, 0])
        lon_block = np.searchsorted(lon_edges[1:], coords[i, 1])
        block_ids[i] = lat_block * n_blocks + lon_block
    
    unique_blocks = np.unique(block_ids)
    np.random.shuffle(unique_blocks)
    
    n_test = max(1, int(len(unique_blocks) * test_fraction))
    test_blocks = set(unique_blocks[:n_test])
    
    test_mask = np.array([b in test_blocks for b in block_ids])
    train_mask = ~test_mask
    
    print(f"  {n_blocks}x{n_blocks} spatial blocks → {len(unique_blocks)} total blocks")
    print(f"  Test blocks: {n_test}, Train blocks: {len(unique_blocks) - n_test}")
    print(f"  Train: {train_mask.sum()} samples (fire rate: {y[train_mask].mean():.1%})")
    print(f"  Test:  {test_mask.sum()} samples (fire rate: {y[test_mask].mean():.1%})")
    
    return (X[train_mask], y[train_mask], coords[train_mask],
            X[test_mask], y[test_mask], coords[test_mask])


# train of three different models, random forest, gradient boosting and MLP
def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names):
    """Train multiple models and compare performance."""
    print("\n" + "=" * 60)
    print("STEP 3: Training models")
    print("=" * 60)
    
    # normalize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, random_state=42,
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        if "MLP" in name:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # metrics
        report = classification_report(y_test, y_pred, target_names=["No Fire", "Fire"],
                                       output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "auc_roc": auc,
            "avg_precision": ap,
            "report": report,
            "confusion_matrix": cm,
        }
        
        print(f"    AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
        print(f"    Fire class — Precision: {report['Fire']['precision']:.3f}, "
              f"Recall: {report['Fire']['recall']:.3f}, F1: {report['Fire']['f1-score']:.3f}")
    
    return results, scaler

# this will help us to understand from which features the random forest have learned
# the most, we expect features like temperature or moisture, different features may 
# indicate that the random three is learning noise 
def analyze_feature_importance(results, feature_names):
    """Analyze and plot feature importance from Random Forest."""
    print("\n" + "=" * 60)
    print("STEP 4: Feature importance analysis")
    print("=" * 60)
    
    rf = results["Random Forest"]["model"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n  Top 10 most important features:")
    for rank, idx in enumerate(indices[:10], 1):
        print(f"    {rank:2d}. {feature_names[idx]:25s} — importance: {importances[idx]:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(15, len(feature_names))
    top_idx = indices[:top_n]
    
    colors = []
    for idx in top_idx:
        name = feature_names[idx]
        if name in ["NDVI", "NBR", "NDMI", "BSI"]:
            colors.append("#2ecc71")  # green for optical
        elif name.startswith("V"):
            colors.append("#3498db")  # blue for SAR
        elif name in ["elevation", "slope", "aspect_sin", "aspect_cos"]:
            colors.append("#95a5a6")  # gray for terrain
        elif name == "lightning_density":
            colors.append("#e74c3c")  # red for ignition
        else:
            colors.append("#f39c12")  # orange for weather
    
    bars = ax.barh(range(top_n), importances[top_idx], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Wildfire Prediction — Feature Importance (Random Forest)")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Optical (Sentinel-2)"),
        Patch(facecolor="#3498db", label="SAR (Sentinel-1)"),
        Patch(facecolor="#f39c12", label="Weather (ERA5)"),
        Patch(facecolor="#95a5a6", label="Terrain"),
        Patch(facecolor="#e74c3c", label="Ignition"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/feature_importance.png")


# we then measures how good the model is at separating fire from no fire (ROC curve)
# and then if the model give us false alarms (confusion matrics)
def plot_evaluation(results, y_test):
    """Generate evaluation plots: ROC curves, PR curves, confusion matrices."""
    print("\n" + "=" * 60)
    print("STEP 5: Generating evaluation plots")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {"Random Forest": "#2ecc71", "Gradient Boosting": "#3498db", "MLP (Neural Net)": "#e74c3c"}
    
    # ROC curve 
    ax = axes[0]
    for name, res in results.items():
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc_roc']:.3f})", color=colors[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # precision-recall curve 
    ax = axes[1]
    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(recall, precision, label=f"{name} (AP={res['avg_precision']:.3f})", 
                color=colors[name], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Fire Class)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # confusion matrix (best model) 
    ax = axes[2]
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    cm = results[best_name]["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Fire", "Fire"])
    ax.set_yticklabels(["No Fire", "Fire"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({best_name})")
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/model_evaluation.png")


# comparision of three maps sides by sides: the groudn thruth, prediction and risk levels
def plot_risk_map(fire_prob, risk_class, coords_test, y_test, results, 
                  lat_grid, lon_grid, all_coords, all_fire_prob):
    """Generate spatial risk map showing predictions vs ground truth."""
    print("\n" + "=" * 60)
    print("STEP 6: Generating wildfire risk map")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    grid_size = lat_grid.shape[0]
    
    # Panel 1: Ground truth fire probability 
    ax = axes[0]
    prob_grid = all_fire_prob.reshape(grid_size, grid_size)
    im1 = ax.pcolormesh(lon_grid, lat_grid, prob_grid,
                        cmap="YlOrRd", vmin=0, vmax=1, shading="auto")
    ax.set_title("Ground Truth: Fire Probability")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im1, ax=ax, label="P(fire)")
    
    # Panel 2: Model predictions on test set 
    ax = axes[1]
    # Start with a gray background
    bg = np.full((grid_size, grid_size), np.nan)
    ax.pcolormesh(lon_grid, lat_grid, bg, cmap="Greys", vmin=0, vmax=1, shading="auto", alpha=0.3)
    
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    y_prob_test = results[best_name]["y_prob"]
    
    scatter = ax.scatter(coords_test[:, 1], coords_test[:, 0], c=y_prob_test,
                        cmap="YlOrRd", vmin=0, vmax=1, s=15, edgecolors="none")
    ax.set_title(f"Model Prediction: {best_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(scatter, ax=ax, label="Predicted P(fire)")
    
    # Panel 3: Risk classification map 
    ax = axes[2]
    # predict risk classes using thresholds
    pred_risk = np.zeros_like(y_prob_test, dtype=int)
    pred_risk[y_prob_test > 0.3] = 1
    pred_risk[y_prob_test > 0.6] = 2
    
    cmap = mcolors.ListedColormap(["#27ae60", "#f39c12", "#e74c3c"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    scatter = ax.scatter(coords_test[:, 1], coords_test[:, 0], c=pred_risk,
                        cmap=cmap, norm=norm, s=15, edgecolors="none")
    ax.set_title("Risk Classification Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["No Risk", "Moderate", "High Risk"])
    
    for ax in axes:
        ax.set_xlim(lon_grid.min(), lon_grid.max())
        ax.set_ylim(lat_grid.min(), lat_grid.max())
    
    plt.suptitle("Alaska Interior — Wildfire Risk Prediction", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "risk_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/risk_map.png")


# summary report

def generate_summary(results, feature_names):
    """Generate a text summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY: Model Comparison")
    print("=" * 60)
    
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "Model": name,
            "AUC-ROC": f"{res['auc_roc']:.4f}",
            "Avg Precision": f"{res['avg_precision']:.4f}",
            "Fire Precision": f"{res['report']['Fire']['precision']:.3f}",
            "Fire Recall": f"{res['report']['Fire']['recall']:.3f}",
            "Fire F1": f"{res['report']['Fire']['f1-score']:.3f}",
            "Accuracy": f"{res['report']['accuracy']:.3f}",
        })
    
    df = pd.DataFrame(summary_rows)
    print("\n" + df.to_string(index=False))
    
    best = max(results, key=lambda k: results[k]["auc_roc"])
    print(f"\n  ★ Best model: {best} (AUC-ROC = {results[best]['auc_roc']:.4f})")
    
    # Save summary
    df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR}/model_comparison.csv")
    
    return df


# MAIN PIPELINE
def main():
    print("\n" + "█" * 60)
    print("  ALASKA WILDFIRE PREDICTION — DEMO PIPELINE")
    print("█" * 60 + "\n")
    
    # 1. generate data
    X, y_binary, y_risk, fire_prob, coords, feature_names, lat_grid, lon_grid = \
        generate_synthetic_alaska_data(grid_size=50)
    
    # 2. spatial split
    X_train, y_train, coords_train, X_test, y_test, coords_test = \
        spatial_block_split(X, y_binary, coords)
    
    # 3. train models
    results, scaler = train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names)
    
    # 4. feature importance
    analyze_feature_importance(results, feature_names)
    
    # 5. evaluation plots
    plot_evaluation(results, y_test)
    
    # 6. risk map
    plot_risk_map(fire_prob, y_risk, coords_test, y_test, results,
                  lat_grid, lon_grid, coords, fire_prob)
    
    # 7. summary
    generate_summary(results, feature_names)
    
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE — All outputs saved to ./outputs/")
    print("█" * 60)
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()
