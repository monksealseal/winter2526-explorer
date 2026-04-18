"""D3: backfill precip in data/cube_winter.nc with CPC 2025 + 2026.

Before this script, the cube's ``precip`` variable covered only
2026-01-01 through 2026-03-31 (90 days, with the first 61 days of the
Nov 1 2025 - Mar 31 2026 window NaN-filled). After this script, the
same variable spans the full 151-day window, sourced from:

- precip.2025.nc (inside zip 001): CPC Global PRCP V1.0, 2025-01-01..2025-12-31
- precip.2026.nc (inside zip 003): CPC Global PRCP V1.0, 2026-01-01..2026-04-13

Both files are global, 0.5 deg gauge-analysis (Chen et al. 2008,
Xie et al. 2007). We preserve the existing preprocess.py pattern
(nearest-neighbor interp to ERA5 0.25 deg CONUS, clip to >= 0 mm/day).

All other cube variables (t2m, t2m_anom, z500, z500_anom) are left
untouched.

Usage (from repo root):
    python scripts/d3_backfill_precip.py
"""
from __future__ import annotations
import shutil
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
NEW  = ROOT / "New Downloaded Files"
CUBE = ROOT / "data" / "cube_winter.nc"

LAT_MIN, LAT_MAX = 22.5, 50.0
LON_MIN, LON_MAX = -125.0, -67.0
WIN_START, WIN_END = "2025-11-01", "2026-03-31"

ZIP_001 = NEW / "Gen Circ_ Group 2 Subgroup 2-20260418T201632Z-3-001.zip"
ZIP_003 = NEW / "Gen Circ_ Group 2 Subgroup 2-20260418T201632Z-3-003.zip"
MEMBER_2025 = "Gen Circ_ Group 2 Subgroup 2/Misc/precip.2025.nc"
MEMBER_2026 = "Gen Circ_ Group 2 Subgroup 2/Misc/precip.2026.nc"


def extract(zip_path: Path, member: str, tmp_dir: Path) -> Path:
    dst = tmp_dir / Path(member).name
    if not dst.exists():
        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out, length=32 * 1024 * 1024)
    return dst


def remap_to_conus(p: xr.DataArray, ref_lat: np.ndarray, ref_lon: np.ndarray) -> xr.DataArray:
    """Match preprocess.py's existing pattern exactly: subset in 0-360,
    shift to -180..180, rename, nearest-neighbor to the ERA5 grid, clip >=0."""
    p_us = p.sel(lat=slice(LAT_MAX, LAT_MIN),
                 lon=slice(LON_MIN % 360, LON_MAX % 360))
    p_us = p_us.assign_coords(lon=(((p_us.lon + 180) % 360) - 180)).sortby("lon")
    p_us = p_us.rename({"lat": "latitude", "lon": "longitude"})
    return (p_us.interp(latitude=ref_lat, longitude=ref_lon, method="nearest")
               .clip(min=0))


def main():
    tmp = Path.home() / "_d3_tmp"
    tmp.mkdir(exist_ok=True)

    print("[1/5] Loading existing cube to preserve non-precip vars ...")
    with xr.open_dataset(CUBE) as ds:
        cube = ds.load()
    ref_lat = cube.latitude.values
    ref_lon = cube.longitude.values
    target_time = cube.time.values  # 151 daily timestamps

    print("[2/5] Extracting precip.2025.nc + precip.2026.nc ...")
    p25_path = extract(ZIP_001, MEMBER_2025, tmp)
    p26_path = extract(ZIP_003, MEMBER_2026, tmp)

    print("[3/5] Opening, concatenating, slicing to window ...")
    with xr.open_dataset(p25_path, engine="netcdf4") as ds25, \
         xr.open_dataset(p26_path, engine="netcdf4") as ds26:
        p25 = ds25["precip"].load()
        p26 = ds26["precip"].load()
    # Time ranges should not overlap (2025 ends 2025-12-31, 2026 starts 2026-01-01).
    p_all = xr.concat([p25, p26], dim="time").sortby("time")
    p_all = p_all.sel(time=slice(WIN_START, WIN_END))
    if p_all.sizes["time"] != 151:
        raise RuntimeError(f"Expected 151 days, got {p_all.sizes['time']}")

    print("[4/5] Regridding to ERA5 CONUS 0.25 deg ...")
    p_cube = remap_to_conus(p_all, ref_lat, ref_lon)
    # Reindex time to exactly match the cube's time coord (same 151 days,
    # same normalization). reindex is safer than assuming identical dtype.
    p_cube = p_cube.reindex(time=target_time)
    # Sanity: NaN count should be 0 for all 151 days within CONUS.
    nan_days = int(np.isnan(p_cube.mean(dim=("latitude", "longitude"))).sum())
    print(f"   valid days = {151 - nan_days}/151,  NaN days = {nan_days}")

    print("[5/5] Writing cube_winter.nc with refreshed precip ...")
    p_cube.name = "precip"
    p_cube.attrs = {
        "units": "mm/day",
        "long_name": "Daily total of precipitation",
        "source": "NOAA CPC Global PRCP V1.0 (2025 + 2026 RT)",
        "regridding": "nearest-neighbor from 0.5 deg to ERA5 0.25 deg CONUS, clipped >= 0",
        "references": "Chen et al. 2008, Xie et al. 2007",
    }
    cube_new = cube.drop_vars("precip").assign(precip=p_cube)
    cube_new.attrs["description"] = (
        "Winter 2025-2026 Explorer cube. "
        "T2m: ERA5 daily 0Z/12Z mean, climatology 2016-2024. "
        "Z500: ERA5 daily 0Z/12Z mean, climatology 1994-2020 (coverage ends 2026-02-28). "
        "Precip: CPC Global PRCP V1.0 (2025-11-01 to 2026-03-31, D3 backfill)."
    )

    enc = {v: {"zlib": True, "complevel": 4} for v in cube_new.data_vars}
    tmp_cube = CUBE.with_suffix(".nc.new")
    cube_new.to_netcdf(tmp_cube, engine="netcdf4", encoding=enc)
    # Atomic replace
    CUBE.unlink()
    tmp_cube.rename(CUBE)
    print(f"   wrote {CUBE}  ({CUBE.stat().st_size/1024/1024:.1f} MB)")

    # Final verification
    with xr.open_dataset(CUBE) as check:
        pm = check.precip.mean(dim=("latitude", "longitude"))
        valid = int((~np.isnan(pm)).sum())
        print(f"   verify: time dim = {check.sizes['time']}, precip valid days = {valid}")
        print(f"   time range: {str(check.time.min().values)[:10]} .. {str(check.time.max().values)[:10]}")


if __name__ == "__main__":
    main()
