# quality control inspectore for the weater data, we verified the correct 
# units of each variables and eventually empty files 

"""
explore_era5.py
================
Quick exploration of the ERA5 weather data you downloaded.
Reads from the extracted per-month folders.

Usage: python src/explore_era5.py
"""

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

ERA5_DIR = Path("data/raw/era5")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# we look inside a folder of the chosed month and dowload all the nc files and we
# merge them, this will save us from dowloading separate files for different variables or weeks 
def load_month(month_dir):
    """Load both NC files from an extracted month folder and merge them."""
    nc_files = list(Path(month_dir).glob("*.nc"))
    datasets = []
    for f in nc_files:
        try:
            ds = xr.open_dataset(f, engine="netcdf4")
            datasets.append(ds)
        except Exception as e:
            print(f"  Warning: could not open {f.name}: {e}")
    if datasets:
        return xr.merge(datasets)
    return None


def explore_era5():
    print("=" * 60)
    print("EXPLORING ERA5 WEATHER DATA — Alaska Fire Season 2022")
    print("=" * 60)

    # find extracted month folders (04, 05, 06, 07, 08, 09)
    month_dirs = sorted([d for d in ERA5_DIR.iterdir() if d.is_dir() and d.name.isdigit()])

    if not month_dirs:
        print("ERROR: No extracted month folders found.")
        print("Run extract_era5.py first!")
        return

    print(f"\nFound {len(month_dirs)} month folders:")
    for d in month_dirs:
        files = list(d.glob("*.nc"))
        print(f"  {d.name}/  ({len(files)} files: {', '.join(f.name for f in files)})")

    # load first month to inspect structure
    print(f"\n--- Inspecting month {month_dirs[0].name} ---")
    ds = load_month(month_dirs[0])
    if ds is None:
        print("ERROR: Could not load any data.")
        return

    print(f"\nVariables:")
    for var in ds.data_vars:
        shape = ds[var].shape
        dims = ds[var].dims
        print(f"  {var:30s}  shape={shape}  dims={dims}")

    print(f"\nCoordinates:")
    for coord in ds.coords:
        vals = ds[coord].values
        if np.ndim(vals) > 0 and np.size(vals) > 1:
            try:
                print(f"  {coord:20s}  {len(vals)} values, range: {vals.min()} to {vals.max()}")
            except Exception:
                print(f"  {coord:20s}  {len(vals)} values")
        else:
            print(f"  {coord:20s}  {vals}")

    # figure out the time dimension name
    time_dim = None
    for candidate in ["valid_time", "time", "forecast_reference_time"]:
        if candidate in ds.dims:
            time_dim = candidate
            break
    if time_dim is None:
        for dim in ds.dims:
            if ds[dim].size > 10:
                time_dim = dim
                break

    print(f"\n  Time dimension: '{time_dim}' ({ds[time_dim].size} steps)")
    ds.close()

    # load all months
    print("\n--- Loading full fire season (Apr-Sep 2022) ---")
    all_months = []
    for d in month_dirs:
        m = load_month(d)
        if m is not None:
            all_months.append(m)
            print(f"  Loaded {d.name}: {m[time_dim].size} timesteps")

    ds_all = xr.concat(all_months, dim=time_dim)
    print(f"  Combined: {ds_all[time_dim].size} total timesteps")

    # we must identify the variables, the ERA5 names for them are quite criptic
    print(f"\n  All variables: {list(ds_all.data_vars)}")
    var_map = {}
    for var in ds_all.data_vars:
        vl = var.lower()
        if "t2m" in vl or var == "t2m":
            var_map["temperature"] = var
        elif "d2m" in vl or var == "d2m":
            var_map["dewpoint"] = var
        elif "u10" in vl or var == "u10":
            var_map["wind_u"] = var
        elif "v10" in vl or var == "v10":
            var_map["wind_v"] = var
        elif "tp" == vl or "precip" in vl or var == "tp":
            var_map["precipitation"] = var
        elif "swvl" in vl or var == "swvl1":
            var_map["soil_moisture"] = var

    print(f"\n  Detected variables:")
    for friendly, era5_name in var_map.items():
        print(f"    {friendly:20s} -> {era5_name}")

    # get spatial dimension names
    spatial_dims = [d for d in ds_all.dims if d != time_dim]
    print(f"  Spatial dims: {spatial_dims}")

    # plot 1: Weather timeline 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ERA5 Weather — Interior Alaska Fire Season 2022", fontsize=14, fontweight="bold")

    if "temperature" in var_map:
        ax = axes[0, 0]
        temp = ds_all[var_map["temperature"]]
        if float(temp.mean()) > 100:
            temp = temp - 273.15
        temp_mean = temp.mean(dim=spatial_dims)
        temp_series = temp_mean.to_series()
        ax.plot(temp_series.index, temp_series.values, color="#e74c3c", linewidth=0.5, alpha=0.7)
        rolling = temp_series.rolling(window=28, center=True).mean()
        ax.plot(rolling.index, rolling.values, color="#c0392b", linewidth=2, label="7-day avg")
        ax.set_ylabel("Temperature (C)")
        ax.set_title("2m Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="blue", linestyle="--", alpha=0.3)

    if "wind_u" in var_map and "wind_v" in var_map:
        ax = axes[0, 1]
        u = ds_all[var_map["wind_u"]]
        v = ds_all[var_map["wind_v"]]
        ws = np.sqrt(u**2 + v**2)
        ws_mean = ws.mean(dim=spatial_dims)
        ws_series = ws_mean.to_series()
        ax.plot(ws_series.index, ws_series.values, color="#3498db", linewidth=0.5, alpha=0.7)
        rolling = ws_series.rolling(window=28, center=True).mean()
        ax.plot(rolling.index, rolling.values, color="#2980b9", linewidth=2, label="7-day avg")
        ax.set_ylabel("Wind Speed (m/s)")
        ax.set_title("10m Wind Speed")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if "temperature" in var_map and "dewpoint" in var_map:
        ax = axes[1, 0]
        t = ds_all[var_map["temperature"]]
        td = ds_all[var_map["dewpoint"]]
        if float(t.mean()) > 100:
            t = t - 273.15
            td = td - 273.15
        rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
        rh_mean = rh.mean(dim=spatial_dims)
        rh_series = rh_mean.to_series()
        ax.plot(rh_series.index, rh_series.values, color="#27ae60", linewidth=0.5, alpha=0.7)
        rolling = rh_series.rolling(window=28, center=True).mean()
        ax.plot(rolling.index, rolling.values, color="#229954", linewidth=2, label="7-day avg")
        ax.axhline(y=30, color="red", linestyle="--", alpha=0.5, label="High fire risk (<30%)")
        ax.set_ylabel("Relative Humidity (%)")
        ax.set_title("Relative Humidity (derived)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if "soil_moisture" in var_map:
        ax = axes[1, 1]
        sm = ds_all[var_map["soil_moisture"]]
        sm_mean = sm.mean(dim=spatial_dims)
        sm_series = sm_mean.to_series()
        ax.plot(sm_series.index, sm_series.values, color="#8e44ad", linewidth=0.5, alpha=0.7)
        rolling = sm_series.rolling(window=28, center=True).mean()
        ax.plot(rolling.index, rolling.values, color="#6c3483", linewidth=2, label="7-day avg")
        ax.set_ylabel("Soil Water (m3/m3)")
        ax.set_title("Soil Moisture (top layer)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "era5_weather_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: outputs/era5_weather_timeline.png")

    # plot 2: Spatial maps for July 
    jul_dir = ERA5_DIR / "07"
    if jul_dir.exists():
        ds_jul = load_month(jul_dir)
        if ds_jul is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle("ERA5 Spatial Maps — July 2022 (Peak Fire Season)", fontsize=14, fontweight="bold")

            if "temperature" in var_map:
                ax = axes[0]
                temp_jul = ds_jul[var_map["temperature"]].mean(dim=time_dim)
                if float(temp_jul.mean()) > 100:
                    temp_jul = temp_jul - 273.15
                temp_jul.plot(ax=ax, cmap="RdYlBu_r", add_colorbar=True)
                ax.set_title("Mean Temperature (C)")

            if "wind_u" in var_map and "wind_v" in var_map:
                ax = axes[1]
                u_jul = ds_jul[var_map["wind_u"]].mean(dim=time_dim)
                v_jul = ds_jul[var_map["wind_v"]].mean(dim=time_dim)
                ws_jul = np.sqrt(u_jul**2 + v_jul**2)
                ws_jul.plot(ax=ax, cmap="YlOrRd", add_colorbar=True)
                ax.set_title("Mean Wind Speed (m/s)")

            if "soil_moisture" in var_map:
                ax = axes[2]
                sm_jul = ds_jul[var_map["soil_moisture"]].mean(dim=time_dim)
                sm_jul.plot(ax=ax, cmap="YlGnBu", add_colorbar=True)
                ax.set_title("Mean Soil Moisture (m3/m3)")

            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "era5_spatial_july2022.png", dpi=150, bbox_inches="tight")
            plt.close()
            ds_jul.close()
            print(f"  Saved: outputs/era5_spatial_july2022.png")

    # summary stats 
    print(f"\n--- Fire Season Weather Summary ---")
    if "temperature" in var_map:
        t_all = ds_all[var_map["temperature"]].values.flatten()
        if np.nanmean(t_all) > 100:
            t_all = t_all - 273.15
        print(f"  Temperature:    mean={np.nanmean(t_all):.1f}C, max={np.nanmax(t_all):.1f}C, min={np.nanmin(t_all):.1f}C")

    if "wind_u" in var_map and "wind_v" in var_map:
        u_all = ds_all[var_map["wind_u"]].values.flatten()
        v_all = ds_all[var_map["wind_v"]].values.flatten()
        ws_all = np.sqrt(u_all**2 + v_all**2)
        print(f"  Wind speed:     mean={np.nanmean(ws_all):.1f} m/s, max={np.nanmax(ws_all):.1f} m/s")

    if "soil_moisture" in var_map:
        sm_all = ds_all[var_map["soil_moisture"]].values.flatten()
        print(f"  Soil moisture:  mean={np.nanmean(sm_all):.3f} m3/m3, min={np.nanmin(sm_all):.3f}")

    for m in all_months:
        m.close()

    print(f"\n{'='*60}")
    print("DONE! Check outputs/ for weather visualizations")
    print(f"{'='*60}")


if __name__ == "__main__":
    explore_era5()