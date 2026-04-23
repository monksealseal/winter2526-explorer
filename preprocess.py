"""Regenerate cube_winter.nc, cube_climo_djf.nc, and index files from raw ERA5.

This is a committed reference copy of what produced the data/ folder. Not needed
for running the app, but essential when anyone wants to add variables or
reproduce the pipeline.

Usage (Colab, after mounting Drive):
    python preprocess.py --gencirc /content/drive/MyDrive/gencirc --out ./data
"""
from __future__ import annotations
import argparse, shutil, urllib.request, zipfile
from pathlib import Path
import numpy as np
import xarray as xr
from indices import fetch_mjo

LAT_MIN, LAT_MAX = 22.5, 50.0
LON_MIN, LON_MAX = -125.0, -67.0
WIN_START, WIN_END = "2025-11-01", "2026-03-31"
CLIMO_T2M = ("2016-01-01", "2024-12-31")
CLIMO_Z   = ("1994-01-01", "2020-12-31")
G = 9.80665


def subset(da):
    return da.sel(latitude=slice(LAT_MAX, LAT_MIN),
                  longitude=slice(LON_MIN, LON_MAX))

def daily(da, t="valid_time"):
    return da.resample({t: "1D"}).mean().rename({t: "time"})

def doy_mean(da, s, e):
    return da.sel(time=slice(s, e)).groupby("time.dayofyear").mean("time")

def djf_mean(da, s, e):
    c = da.sel(time=slice(s, e))
    return c.where(c["time.month"].isin([12, 1, 2]), drop=True).mean("time")

def strip_aux(da):
    return da.drop_vars(["number", "expver", "dayofyear"], errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gencirc", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gencirc = Path(args.gencirc)
    out = Path(args.out)
    indices_dir = out / "indices"
    out.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] T2m ...")
    t2m = daily(subset(xr.open_dataset(
        gencirc / "ERA5_2mT_2016_2026_hourly_0Z_12Z-003_",
        engine="netcdf4", chunks={"valid_time": 365})["t2m"])) - 273.15
    t2m.name, t2m.attrs["units"] = "t2m", "degC"
    t2m_climo = doy_mean(t2m, *CLIMO_T2M)
    t2m_w = t2m.sel(time=slice(WIN_START, WIN_END)).load()
    t2m_anom = (t2m_w.groupby("time.dayofyear") - t2m_climo).load()

    print("[2/4] Z500 ...")
    big_zip = gencirc / "Gen Circ_ Group 2 Subgroup 2-20260417T222352Z-3-005.zip"
    z500_tmp = Path("/tmp/z500_1994_2026_FIXED.nc")
    if not z500_tmp.exists():
        with zipfile.ZipFile(big_zip) as zf, \
             zf.open("Gen Circ_ Group 2 Subgroup 2/ERA5/z500_1994_2026_FIXED.nc") as src, \
             open(z500_tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
    z = subset(xr.open_dataset(z500_tmp, engine="netcdf4", chunks={"valid_time": 365})["z"]
               .squeeze("pressure_level", drop=True))
    z = daily(z) / G
    z.name, z.attrs["units"] = "z500", "m"
    z_climo = doy_mean(z, *CLIMO_Z)
    z_w = z.sel(time=slice(WIN_START, WIN_END)).load()
    z_anom = (z_w.groupby("time.dayofyear") - z_climo).load()

    print("[3/4] Precip ...")
    precip_tmp = Path("/tmp/precip.2026.nc")
    if not precip_tmp.exists():
        with zipfile.ZipFile(big_zip) as zf, \
             zf.open("Gen Circ_ Group 2 Subgroup 2/Misc/precip.2026.nc") as src, \
             open(precip_tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
    p = xr.open_dataset(precip_tmp, engine="netcdf4")["precip"]
    p_us = p.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN % 360, LON_MAX % 360))
    p_us = p_us.assign_coords(lon=(((p_us.lon + 180) % 360) - 180)).sortby("lon")
    p_us = p_us.rename({"lat": "latitude", "lon": "longitude"})
    p_r = p_us.interp(latitude=t2m_w.latitude, longitude=t2m_w.longitude,
                      method="nearest").clip(min=0)
    p_r.name, p_r.attrs["units"] = "precip", "mm/day"
    p_w = p_r.sel(time=slice(WIN_START, WIN_END)).load()

    print("[4/4] Writing ...")
    cube = xr.Dataset({"t2m": strip_aux(t2m_w), "t2m_anom": strip_aux(t2m_anom),
                       "z500": strip_aux(z_w), "z500_anom": strip_aux(z_anom),
                       "precip": strip_aux(p_w)})
    cube.attrs["description"] = (
        f"Winter 2025-2026 Explorer cube. "
        f"T2m clim {CLIMO_T2M[0]}..{CLIMO_T2M[1]}. "
        f"Z500 clim {CLIMO_Z[0]}..{CLIMO_Z[1]}. "
        "Precip: CPC Global V1.0.")
    cube.to_netcdf(out / "cube_winter.nc", engine="h5netcdf",
                   encoding={v: {"zlib": True, "complevel": 4} for v in cube.data_vars})

    climo_djf = xr.Dataset({
        "t2m_djf_mean":  strip_aux(djf_mean(t2m, *CLIMO_T2M)).astype("float32"),
        "z500_djf_mean": strip_aux(djf_mean(z, *CLIMO_Z)).astype("float32"),
    })
    climo_djf.to_netcdf(out / "cube_climo_djf.nc", engine="h5netcdf",
                        encoding={v: {"zlib": True, "complevel": 4} for v in climo_djf.data_vars})

    members = {
        "ao.csv":  "Gen Circ_ Group 2 Subgroup 2/AO_QBO_NAO_Indexes/norm.daily.ao.index.b500101.current.csv",
        "nao.csv": "Gen Circ_ Group 2 Subgroup 2/AO_QBO_NAO_Indexes/norm.daily.nao.index.b500101.current.csv",
        "pna.txt": "Gen Circ_ Group 2 Subgroup 2/AO_QBO_NAO_Indexes/norm.daily.pna.index.b500101.current.ascii",
        "qbo.csv": "Gen Circ_ Group 2 Subgroup 2/AO_QBO_NAO_Indexes/qbo.csv",
    }
    with zipfile.ZipFile(big_zip) as zf:
        for name, m in members.items():
            with zf.open(m) as src, open(indices_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)

    try:
        with urllib.request.urlopen(
            "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=30) as r:
            (indices_dir / "oni.txt").write_bytes(r.read())
    except Exception as e:
        print(f"  ONI fetch failed: {e}")

    result = fetch_mjo(indices_dir)
    if "error" in result:
        print(f"  MJO fetch failed: {result['error']}")
    else:
        print(f"  MJO: {result['source']} -> {result['filename']}")
    print("Done.")


if __name__ == "__main__":
    main()
