"""Extract ERA5 zip files into per-month folders."""
import zipfile, glob, os

for f in sorted(glob.glob("data/raw/era5/era5_*.nc")):
    month = f.split("_")[-1].replace(".nc", "")
    out = f"data/raw/era5/{month}"
    os.makedirs(out, exist_ok=True)
    zipfile.ZipFile(f).extractall(out)
    print(f"Extracted {f} -> {out}/")
    for name in os.listdir(out):
        size = os.path.getsize(os.path.join(out, name))
        print(f"  {name}  ({size/1024:.0f} KB)")

print("\nDone! Now run: python src/explore_era5.py")
