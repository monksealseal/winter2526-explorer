"""Teleconnection index parsers and fetchers."""
from __future__ import annotations
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd


def to_monthly(series: pd.Series, min_days: int = 15) -> pd.Series:
    """Aggregate a daily-cadence index series to month-start means.

    Months with fewer than ``min_days`` valid observations become NaN so that
    downstream correlations aren't biased by partial months (common at the
    edges of the analysis window, e.g. December 2025 precip).
    """
    if series.empty:
        return pd.Series(dtype=float)
    monthly = series.resample("MS").mean()
    counts = series.resample("MS").count()
    return monthly.where(counts >= min_days)


def parse_romi(path) -> pd.DataFrame:
    """Parse the NOAA PSL ROMI (Real-time OLR-based MJO Index) file.

    File layout (whitespace-delimited, no header):
        year  month  day  flag  ROMI1  ROMI2  amplitude

    The 4th column is a method/QC flag (currently 0) and is ignored.
    ``amplitude`` is provided directly; ``phase`` (1-8) is derived from
    (ROMI1, ROMI2) using the Wheeler & Hendon (2004) octant convention
    so downstream code can treat ROMI and RMM interchangeably.

    Returns a DataFrame indexed by date with columns
    ``[RMM1, RMM2, phase, amplitude]``.

    Reference: Kiladis, G. N., J. Dias, K. H. Straub, M. C. Wheeler,
    S. N. Tulich, K. Kikuchi, K. M. Weickmann, and M. J. Ventrice, 2014:
    A comparison of OLR and circulation-based indices for tracking the MJO.
    Mon. Wea. Rev., 142, 1697-1715.
    """
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["year", "month", "day", "flag", "RMM1", "RMM2", "amplitude"],
        comment="#",
    )
    df["Date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce")
    df = (df.dropna(subset=["Date"])
            .set_index("Date").sort_index())
    # Derive phase 1-8 from (RMM1, RMM2) octant.
    # atan2 returns in (-π, π]; shift so phase 1 starts at -π (SW corner)
    # and increments counter-clockwise, matching Wheeler & Hendon (2004) Fig 7.
    angle = np.arctan2(df["RMM2"].to_numpy(), df["RMM1"].to_numpy())
    phase = np.floor((angle + np.pi) / (np.pi / 4.0)).astype(int) + 1
    df["phase"] = np.clip(phase, 1, 8)
    return df[["RMM1", "RMM2", "phase", "amplitude"]]


def parse_daily_ao_nao_pna(path) -> pd.Series:
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["Year", "Month", "Day", "Value"],
                     na_values=["-9.9", "-99.9", "*"])
    df["Date"] = pd.to_datetime(
        dict(year=df.Year, month=df.Month, day=df.Day), errors="coerce")
    return (df.dropna(subset=["Date"])
              .set_index("Date")["Value"]
              .astype(float).sort_index())


def parse_qbo(path) -> pd.Series:
    df = pd.read_csv(path, skiprows=1, header=None, names=["Date", "Value"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return (df.dropna().set_index("Date")["Value"].sort_index())


def parse_oni(path) -> pd.Series:
    season_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    seas, yr, _tot, anom = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
                    if seas in season_month:
                        rows.append((pd.Timestamp(year=yr, month=season_month[seas], day=15), anom))
                except ValueError:
                    continue
    return pd.Series(dict(rows)).sort_index()


def parse_mjo_rmm(path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                rmm1, rmm2 = float(parts[3]), float(parts[4])
                phase = int(parts[5]); amp = float(parts[6])
                if not (1 <= mo <= 12 and 1 <= dy <= 31):
                    continue
                rows.append((pd.Timestamp(year=yr, month=mo, day=dy),
                             rmm1, rmm2, phase, amp))
            except (ValueError, TypeError):
                continue
    return (pd.DataFrame(rows, columns=["Date", "RMM1", "RMM2", "phase", "amplitude"])
              .set_index("Date").sort_index())


MJO_FALLBACK_URLS = [
    "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
    "https://psl.noaa.gov/mjo/mjoindex/data/rmm.74toRealtime.txt",
]

def fetch_mjo(out_path, timeout=30):
    for url in MJO_FALLBACK_URLS:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (gencirc-app)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                Path(out_path).write_bytes(r.read())
            return url
        except Exception as e:
            print(f"  MJO fetch {url} failed: {type(e).__name__}: {e}")
    return None


def load_all_indices(indices_dir) -> dict:
    indices_dir = Path(indices_dir)
    out = {}
    parsers = [
        ("ao",  "ao.csv",  parse_daily_ao_nao_pna),
        ("nao", "nao.csv", parse_daily_ao_nao_pna),
        ("pna", "pna.txt", parse_daily_ao_nao_pna),
        ("qbo", "qbo.csv", parse_qbo),
        ("oni", "oni.txt", parse_oni),
    ]
    for key, fname, parser in parsers:
        p = indices_dir / fname
        if p.exists():
            try:
                out[key] = parser(p)
            except Exception as e:
                print(f"Failed to parse {fname}: {e}")
    # MJO: prefer ROMI (NOAA PSL, real-time OLR-based, Kiladis et al. 2014)
    # over the BoM RMM file (Wheeler & Hendon 2004) because the canonical
    # BoM real-time URL stalled updating in Feb 2024. Both parsers return
    # the same column schema so the rest of the app is agnostic.
    romi_path = indices_dir / "romi.txt"
    rmm_path = indices_dir / "mjo_rmm.txt"
    if romi_path.exists():
        try:
            out["mjo"] = parse_romi(romi_path)
            out["mjo_source"] = "ROMI (Kiladis et al. 2014; NOAA PSL)"
        except Exception as e:
            print(f"Failed to parse romi.txt: {e}")
    elif rmm_path.exists():
        try:
            out["mjo"] = parse_mjo_rmm(rmm_path)
            out["mjo_source"] = "RMM (Wheeler & Hendon 2004; BoM)"
        except Exception as e:
            print(f"Failed to parse mjo_rmm.txt: {e}")
    return out
