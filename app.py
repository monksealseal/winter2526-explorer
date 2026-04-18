"""Winter 2025-2026 Explorer — Streamlit app.

Tabs:
  1. This Winter   — monthly mean maps (T2m anom, Z500 anom, precip)
  2. Indices       — teleconnection time series with SE-US T2m overlay
                     + correlation table with bootstrap CI and references
  3. Composites    — Welch's-t composite difference + correlation maps
                     with lag slider and proper significance
  4. Methods & Data — methods, formulas, data provenance, references

All figures on this page are rendered with cartopy (PlateCarree) at
ERA5 0.25° native resolution. Correlation CIs use the moving-block
bootstrap; composite-difference significance uses a per-cell Welch's
t-test. See tab 4 for methods details.

Every control reads/writes st.query_params so any view is URL-shareable.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scistats
import streamlit as st
import xarray as xr

# Silence cartopy's one-shot Natural Earth download notice on cold start.
warnings.filterwarnings("ignore", message="Downloading")

from indices import load_all_indices, to_monthly
from plots import make_map
from stats import (
    effective_n,
    block_bootstrap_corr,
    welch_t_composite,
    corr_map_t_significance,
)

DATA_DIR = Path(__file__).parent / "data"
CUBE_PATH = DATA_DIR / "cube_winter.nc"
CLIMO_PATH = DATA_DIR / "cube_climo_djf.nc"
INDICES_DIR = DATA_DIR / "indices"

SE_US_BOX   = dict(lat_min=25, lat_max=37, lon_min=-92, lon_max=-75)
FLORIDA_BOX = dict(lat_min=24, lat_max=31, lon_min=-87, lon_max=-80)

VAR_META = {
    "t2m":       {"label": "2 m temperature (°C)",      "cmap": "RdBu_r", "vmin": -30,  "vmax": 30},
    "t2m_anom":  {"label": "2 m T anomaly (°C)",        "cmap": "RdBu_r", "vmin": -15,  "vmax": 15},
    "z500":      {"label": "500 mb height (m)",          "cmap": "Viridis","vmin": 5000, "vmax": 5900},
    "z500_anom": {"label": "500 mb height anomaly (m)",  "cmap": "RdBu_r", "vmin": -250, "vmax": 250},
    "precip":    {"label": "Precipitation (mm/day)",     "cmap": "Blues",  "vmin": 0,    "vmax": 20},
}

INDEX_META = {
    "ao":      {"label": "AO",           "cadence": "daily",   "unit": "σ"},
    "nao":     {"label": "NAO",          "cadence": "daily",   "unit": "σ"},
    "pna":     {"label": "PNA",          "cadence": "daily",   "unit": "σ"},
    "qbo":     {"label": "QBO",          "cadence": "daily",   "unit": "m/s"},
    "oni":     {"label": "ONI (ENSO)",   "cadence": "monthly", "unit": "°C"},
    "mjo_amp": {"label": "MJO amplitude","cadence": "daily",   "unit": "σ"},
}

REFERENCE_R = {
    "ao_t2m":  {"r": 0.561,  "source": "Abby, slide 38",           "method": "monthly"},
    "nao_t2m": {"r": 0.107,  "source": "Abby, slide 46",           "method": "monthly"},
    "pna_t2m": {"r": -0.113, "source": "Abby, slide 47 (approx.)", "method": "monthly"},
}

# ----- Provenance and methodology metadata (displayed in Methods tab and
# in inline figure captions). These constants are the single source of
# truth for period/source strings used elsewhere in the app. -----

T2M_CLIMO_BASE  = "2016-2024 ERA5 daily mean"
Z500_CLIMO_BASE = "1994-2020 ERA5 daily mean"
PRECIP_SOURCE   = "NOAA CPC Global PRCP V1.0 (regridded to ERA5 0.25°)"

BOOTSTRAP_N     = 1000
BOOTSTRAP_ALPHA = 0.05

PROVENANCE = [
    {"variable": "t2m / t2m_anom", "source": "ERA5 (Copernicus C3S) 2-m temperature, hourly → daily mean from 00/12 UTC",
     "period": "2025-11-01 → 2026-03-31", "climatology": T2M_CLIMO_BASE,
     "resolution": "0.25° global", "doi_ref": "Hersbach et al. 2020, QJRMS 146, 1999-2049"},
    {"variable": "z500 / z500_anom", "source": "ERA5 500 hPa geopotential height",
     "period": "2025-11-01 → 2026-02-28", "climatology": Z500_CLIMO_BASE,
     "resolution": "0.25° global", "doi_ref": "Hersbach et al. 2020, QJRMS 146, 1999-2049"},
    {"variable": "precip", "source": PRECIP_SOURCE,
     "period": "2026-01-01 → 2026-03-31", "climatology": "— (not anomalized here)",
     "resolution": "0.5° native → 0.25° interpolated", "doi_ref": "Xie et al. 2007, J. Hydromet. 8, 607-626"},
    {"variable": "AO, NAO, PNA", "source": "NOAA CPC daily teleconnection indices",
     "period": "daily, 1950-present", "climatology": "CPC normalisation",
     "resolution": "—", "doi_ref": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"},
    {"variable": "QBO", "source": "NOAA CPC QBO 30 hPa zonal-mean u-wind",
     "period": "monthly", "climatology": "—",
     "resolution": "—", "doi_ref": "Naujokat 1986, J. Atmos. Sci. 43, 1873-1877"},
    {"variable": "ONI (ENSO)", "source": "NOAA CPC Oceanic Niño Index (3-month SST anomaly, Niño 3.4)",
     "period": "3-month running mean", "climatology": "1991-2020 centered 30-year",
     "resolution": "—", "doi_ref": "Huang et al. 2017, J. Climate 30, 8179-8205 (ERSSTv5)"},
    {"variable": "MJO ROMI (primary)",
     "source": "NOAA PSL Real-time OLR-based MJO Index (romi.cpcolr.1x.txt)",
     "period": "daily, 1991-present", "climatology": "CPC OLR-derived",
     "resolution": "—",
     "doi_ref": "Kiladis et al. 2014, Mon. Wea. Rev. 142, 1697-1715"},
    {"variable": "MJO RMM (archived fallback)",
     "source": "BoM rmm.74toRealtime.txt (upstream feed stalled Feb 2024)",
     "period": "daily, 1974-02-2024",
     "climatology": "—", "resolution": "—",
     "doi_ref": "Wheeler & Hendon 2004, Mon. Wea. Rev. 132, 1917-1932"},
]

REFERENCES = [
    ("Bretherton et al. (1999)", "The effective number of spatial degrees of freedom of a time-varying field", "J. Climate 12, 1990-2009"),
    ("Künsch (1989)", "The jackknife and the bootstrap for general stationary observations", "Ann. Statist. 17, 1217-1241"),
    ("Wilks (2011)", "Statistical Methods in the Atmospheric Sciences, 3rd ed.", "Academic Press"),
    ("Welch (1947)", "The generalization of Student's problem when several different population variances are involved", "Biometrika 34, 28-35"),
    ("von Storch & Zwiers (1999)", "Statistical Analysis in Climate Research", "Cambridge Univ. Press"),
    ("Hersbach et al. (2020)", "The ERA5 global reanalysis", "QJRMS 146, 1999-2049"),
    ("Xie et al. (2007)", "A gauge-based analysis of daily precipitation over East Asia", "J. Hydromet. 8, 607-626"),
    ("Wheeler & Hendon (2004)", "An all-season real-time multivariate MJO index", "Mon. Wea. Rev. 132, 1917-1932"),
    ("Kiladis et al. (2014)", "A comparison of OLR and circulation-based indices for tracking the MJO", "Mon. Wea. Rev. 142, 1697-1715"),
    ("Thompson & Wallace (1998)", "The Arctic Oscillation signature in the wintertime geopotential height and temperature fields", "GRL 25, 1297-1300"),
    ("Hurrell (1995)", "Decadal trends in the North Atlantic Oscillation", "Science 269, 676-679"),
    ("Wallace & Gutzler (1981)", "Teleconnections in the geopotential height field during the NH winter", "Mon. Wea. Rev. 109, 784-812"),
]

@st.cache_resource
def load_cube():   return xr.open_dataset(CUBE_PATH)
@st.cache_resource
def load_climo():  return xr.open_dataset(CLIMO_PATH) if CLIMO_PATH.exists() else None
@st.cache_data
def load_indices_cached(): return load_all_indices(INDICES_DIR)

def qp_get(key, default, cast=str):
    v = st.query_params.get(key)
    if v is None: return default
    try: return cast(v)
    except (ValueError, TypeError): return default

def qp_set(**kw):
    for k, v in kw.items():
        if v is None:
            if k in st.query_params: del st.query_params[k]
        else:
            st.query_params[k] = str(v)

def box_mean(da, box):
    sel = da.sel(latitude=slice(box["lat_max"], box["lat_min"]),
                 longitude=slice(box["lon_min"], box["lon_max"]))
    w = np.cos(np.deg2rad(sel.latitude))
    return sel.weighted(w).mean(dim=["latitude", "longitude"])

@st.cache_data(show_spinner=False)
def cached_bootstrap_corr(x: tuple, y: tuple, n_boot: int = BOOTSTRAP_N,
                          block_len: int | None = None) -> dict:
    """Cached wrapper around stats.block_bootstrap_corr.

    Streamlit's cache requires hashable inputs, so x and y are passed as
    tuples (not ndarrays). Results are identical to calling
    ``block_bootstrap_corr(x, y, ...)`` directly.
    """
    return block_bootstrap_corr(np.asarray(x, dtype=float),
                                np.asarray(y, dtype=float),
                                n_boot=n_boot, block_len=block_len)


def fmt_ci(result: dict) -> str:
    """Format a bootstrap CI result as ``r = +0.56 [+0.31, +0.74]``."""
    if not np.isfinite(result.get("r", np.nan)):
        return "—"
    return (f"{result['r']:+.3f} "
            f"[{result['ci_lo']:+.3f}, {result['ci_hi']:+.3f}]")

def align_index_to_cube(series, cube_time):
    idx_daily = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time])
    if len(series) == 0:
        return pd.Series(np.nan, index=idx_daily)
    cadence_monthly = (series.index[1] - series.index[0]).days > 20 if len(series) > 1 else False
    if cadence_monthly:
        rng = pd.date_range(series.index.min(), idx_daily.max() + pd.Timedelta(days=32), freq="D")
        return series.reindex(rng, method="ffill").reindex(idx_daily)
    return series.reindex(idx_daily)

def correlation_map(field: np.ndarray, idx_vals: np.ndarray) -> tuple[np.ndarray, int]:
    """Grid-point Pearson correlation between ``field(t, lat, lon)`` and
    ``idx_vals(t)``. NaN-safe per cell.

    Returns ``(r_map, n_used)`` where n_used is the number of days with a
    finite index value (so the same n applies to every cell up to the
    cell's own NaN days).
    """
    mask_t = ~np.isnan(idx_vals)
    f = field[mask_t]; x = idx_vals[mask_t]
    T, H, W = f.shape
    f_flat = f.reshape(T, H * W)
    r_flat = np.full(H * W, np.nan)
    for j in range(H * W):
        fj = f_flat[:, j]
        v = ~np.isnan(fj)
        if v.sum() < 10: continue
        xv = x[v]; fv = fj[v]
        if np.std(fv) < 1e-10 or np.std(xv) < 1e-10: continue
        r_flat[j] = np.corrcoef(fv, xv)[0, 1]
    return r_flat.reshape(H, W), int(mask_t.sum())

def get_series(indices, key):
    if key == "mjo_amp":
        return indices["mjo"]["amplitude"] if "mjo" in indices else pd.Series(dtype=float)
    return indices.get(key, pd.Series(dtype=float))

st.set_page_config(page_title="Winter 2025-2026 Explorer", page_icon="❄️", layout="wide")

try:
    cube = load_cube()
    t_max = float(np.abs(cube.t2m_anom).max())
    z_max = float(np.abs(cube.z500_anom).max())
    assert t_max < 50, f"T2m anomaly range ({t_max:.1f}) exceeds ±50 °C — likely unit bug"
    assert z_max < 1000, f"Z500 anomaly range ({z_max:.1f}) exceeds ±1000 m — likely unit bug"
except AssertionError as e:
    st.error(f"Cube sanity check failed: {e}. Re-run preprocess.py.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load cube: {e}")
    st.stop()

climo = load_climo()
indices = load_indices_cached()
available_indices = [k for k in INDEX_META if (k in indices) or (k == "mjo_amp" and "mjo" in indices)]

st.title("❄️ Winter 2025-2026 Explorer")
st.caption(
    "ERA5 reanalysis + NOAA CPC teleconnection indices over CONUS · "
    f"Nov 1 2025 → Mar 31 2026 · "
    f"{len(available_indices)}/{len(INDEX_META)} indices loaded"
)

MJO_LOADED = "mjo" in indices

tab_about, tab_rc, tab1, tab2, tab3, tab4 = st.tabs([
    "📜 About & authorship",
    "🧭 Research compass",
    "This Winter",
    "Indices",
    "Composites & Correlations",
    "Methods & Data",
])

# ==================================================================
# Tab RC: Research Compass — purpose-built panels for Tori's proposal
# ==================================================================
with tab_about:
    qp_set(tab="about")
    st.header("📜 About this app — authorship, methods, and AI assistance")

    st.markdown("""
This Streamlit application was built collaboratively by a human and an AI
assistant. Because academic norms for AI-assisted research software are
still being established (as of April 2026), this page documents the
division of labor transparently so readers, reviewers, and instructors
can weigh the evidence produced here appropriately.

**Human author.** Eduardo Siman (University of Miami RSMAS, MPO-551
General Circulation, Spring 2026). Contributed: the initial repository
and data-preprocessing pipeline, strategic direction and prompts,
research context (Group 2 Subgroup A's proposal, group-lead Tori's
questions), quality standards, manual data fetches where the AI sandbox
was firewalled, and browser-based testing with bug reports. Eduardo
**did not write any of the Python code added during the AI-assisted
development sessions**.

**AI assistant.** Claude 4.7 (model ID `claude-opus-4-7`), operated via
Anthropic's Claude Code CLI harness. Contributed: all application code
added during these sessions (in `app.py`, `indices.py`, `stats.py`, and
`plots.py`), selection of statistical methods and their primary-literature
references, figure design, inline documentation text, scientific
interpretation of the numeric outputs, and bug fixes.

The rest of this page gives a step-by-step chronology, data-provenance
detail, a methods list with citations, and a How-to-cite block.
""")

    st.markdown("## Tools used")
    st.markdown("""
- **Claude 4.7** — Anthropic's large language model. Specific model ID:
  `claude-opus-4-7` (the "Opus" tier of Claude 4.7, used with the 1M-token
  context window variant). Provider: Anthropic, via the Claude Code CLI.
- **Claude Code** — Anthropic's command-line coding harness. Runs the
  model inside a sandboxed Linux environment with tool access (file I/O,
  Bash, Python execution, web search). The sandbox blocks direct HTTP
  access to most scientific data hosts (BoM, NOAA PSL, CPC, IRI), which
  is the reason the human had to fetch ROMI and index files manually
  from a local PowerShell session.
- **Git / GitHub** — source control; the full commit history of this app
  is a faithful record of who changed what and when. Every commit
  authored by the AI is tagged `Author: Claude <noreply@anthropic.com>`;
  every commit authored by the human is tagged `Author: Eduardo Siman
  <esiman@msn.com>`. See the repo's commit log for verification.
- **Streamlit Community Cloud** — hosts the live deployment. Auto-deploys
  from the `main` branch of
  [`monksealseal/winter2526-explorer`](https://github.com/monksealseal/winter2526-explorer).
- **Prompting medium** — the human communicated with the model
  exclusively through natural-language prompts typed into Claude Code.
  No custom fine-tuning or prompt templates were used beyond Claude
  Code's default system prompt.
""")

    st.markdown("## Chronology, step by step")

    with st.expander("**Day 0 — before any AI session (Eduardo Siman, solo)**", expanded=False):
        st.markdown("""
All of the following existed in the repository *before* Claude was
invoked, and all of it was written by Eduardo Siman (with credit to the
standard scientific-Python ecosystem — `xarray`, `numpy`, `pandas`,
`scipy`, `streamlit`, `plotly` — that the code depends on):

- **Repository scaffold.** `app.py`, `indices.py`, `preprocess.py`,
  `requirements.txt`, `README.md`, `.streamlit/config.toml`, `.gitignore`.
- **Data-preprocessing pipeline** (`preprocess.py`). Downloaded ERA5
  reanalysis fields (2-m temperature, 500 mb geopotential height), CPC
  Global PRCP V1.0 precipitation, plus NOAA CPC teleconnection indices
  (AO, NAO, PNA, QBO, ONI). Regridded precipitation to the ERA5 0.25°
  grid; computed daily anomalies against variable-specific climatology
  base periods (ERA5 T2m 2016-2024, 9 yr; Z500 1994-2020, 27 yr).
- **Data cubes.** `data/cube_winter.nc` (42.7 MB, 151 days × 111 lat ×
  233 lon covering 22.5-50 °N / -125 to -67 °E), `data/cube_climo_djf.nc`
  (0.2 MB climatology bundle).
- **Teleconnection index files.** `data/indices/{ao.csv, nao.csv,
  pna.txt, qbo.csv, oni.txt}` with their respective parsers in
  `indices.py`.
- **First version of `app.py`.** Four tabs ("This Winter", "Indices",
  "Composites & Correlations", "Dataset Inspector") with daily plots,
  a correlation/composite panel using parametric-p significance only,
  and a data-caveats tab. Plotly heatmaps, no cartopy. ~600 lines.

Git blame for every file dated before the AI sessions shows Eduardo as
the sole author. The initial commit is
[`62a5817`](https://github.com/monksealseal/winter2526-explorer/commit/62a5817)
(*"Initial Winter 25-26 Explorer"*), authored Apr 18 2026 03:18 UTC.
""")

    with st.expander("**Session 1, Phase 1 — monthly aggregations (same day)**", expanded=False):
        st.markdown("""
**Prompt by Eduardo** (paraphrased for brevity; full prompt is in session
transcript): *"Enhance the app with Phase 1 improvements: add monthly
index aggregations to Tab 2 with a Daily/Monthly toggle; add Z500
anomaly to the Tab 1 variable selector; enhance negative-phase
composite documentation in Tab 3."*

**What Claude 4.7 did:**

1. Added `to_monthly()` helper to `indices.py` — resamples daily-cadence
   indices (AO, NAO, PNA, QBO) to month-start means with a configurable
   ≥15-valid-days gate.
2. Added a Daily / Monthly radio to Tab 2's left column. In Monthly mode,
   indices render as bar charts and the correlation table recomputes on
   monthly-mean index × monthly-mean SE-US T2m anomaly — matching the
   cadence behind the group's reference r-values. View state is
   URL-shareable via `&view=monthly`.
3. Added an interpretive caption in Tab 1 that appears when `z500_anom`
   is selected (red = ridge/high, blue = trough/low; PNA-positive =
   Alaska ridge + eastern-US trough).
4. Expanded Tab 3's threshold-radio help text and composite-map caption
   so the (positive − negative) direction and negative-phase reading
   are explicit.

Landed on `main` as commit [`4fb19f4`](https://github.com/monksealseal/winter2526-explorer/commit/4fb19f4).
Two files changed (`app.py`, `indices.py`); 137 insertions, 37 deletions.
""")

    with st.expander("**Session 1, Phase 2.1 — publication-quality retrofit + Research compass**",
                     expanded=False):
        st.markdown("""
**Prompts by Eduardo** (paraphrased):
1. *"What else can we build that would help the group with their
   research?"*
2. After Claude listed options: *"Whatever we do it has to be
   publication quality. It has to be something that Prof. Emily Becker
   would look at and be able to understand. It has to be grounded in
   established scientific methods and it needs to explain what it is
   doing."*
3. Eduardo then provided group-lead **Tori's project questionnaire**
   (her research questions, methods, and a-priori hypothesis about
   MJO phase 7-8 → eastern-US cold). Eduardo: *"Let's focus on Tori's
   questions — she is the group lead."*

**What Claude 4.7 did** (Eduardo wrote no code during this phase):

*New modules:*

- **`stats.py`** — four helpers: `effective_n()` (lag-1-AR adjustment,
  Bretherton et al. 1999), `block_bootstrap_corr()` (moving-block
  bootstrap for correlation CIs, Künsch 1989 / Wilks 2011), plus a
  `welch_t_composite()` and `corr_map_t_significance()` pair for
  pixel-wise significance with Welch-Satterthwaite degrees of freedom.
- **`plots.py`** — a single `make_map()` factory. PlateCarree cartopy
  axes, Natural Earth 50 m coastlines / borders / states, diverging
  colormap centered on zero for anomalies, journal-style caption
  block, optional contour overlay, stippling, and analysis-box
  highlights.
- **`requirements.txt`** and **`packages.txt`** — cartopy, matplotlib,
  and the apt dependencies for Streamlit Cloud's build.

*Tab retrofits:*

- **Tab 1 (This Winter)** — cartopy map with a caption that names
  the climatology base period and flags coverage caveats.
- **Tab 3 (Composites & Correlations)** — composite difference and
  correlation maps redrawn with cartopy, Welch's-t stippling on
  composites, `n_eff`-adjusted t-test on correlations, and a new
  bootstrap 95 % CI on the SE-US box-time-series correlation.
  Absolute-phase composites get Z500 contour overlays. An
  "About this analysis" expander lists formulas and references inline.
- **Tab 4** — renamed *Dataset Inspector* → *Methods & Data*. Full
  methods section, data-provenance table, known-limitations section,
  and a references bibliography.

*New tab:*

- **🧭 Research compass** — purpose-built panels for each of Tori's
  five primary questions (Q1 cold-event detection, Q2 MJO phase × lag
  heatmap, Q3 MJO phase 7-8 lagged Z500 composite, Q4 MJO × ENSO
  conditional 2×2, Q5 OLS multiple regression of FL T2m on AO+NAO+PNA+ONI).
  Deferred questions Q6 (La Niña analog search), Q7 (Rossby wave train
  Hovmöller), and Q8 (250 mb jet) are enumerated with the specific
  data-scope each would require.

Landed on `main` as two commits:
[`fa77041`](https://github.com/monksealseal/winter2526-explorer/commit/fa77041)
(main feature) and
[`42c03ec`](https://github.com/monksealseal/winter2526-explorer/commit/42c03ec)
(a NameError bug fix Eduardo caught by loading the deployed app and
reporting the traceback; Claude fixed it within the same session).
Net diff for Phase 2.1: ~1200 insertions, ~150 deletions across 5 files.
""")

    with st.expander("**Session 1, Phase 2.2 — MJO data sourcing (human did the fetching)**",
                     expanded=False):
        st.markdown("""
The Research-compass panels Q2, Q3, Q4 depend on a daily MJO-phase
index. The Claude Code sandbox blocks outbound HTTP to the canonical
scientific hosts (bom.gov.au, psl.noaa.gov, cpc.ncep.noaa.gov,
iridl.ldeo.columbia.edu) — they all return `403 Host not in allowlist`.
Claude therefore could not download MJO data itself.

**What Eduardo did** (from a local PowerShell session on his machine,
where those hosts are reachable):

1. Ran an initial fetch of the BoM Wheeler-Hendon RMM file
   (`rmm.74toRealtime.txt`). The file downloaded successfully but
   parser-side checks revealed it stalled updating in February 2024 —
   the canonical BoM feed had effectively been abandoned.
2. After several diagnostic round-trips with Claude (trying NOAA CPC
   and IRI mirror URLs, all dead), Eduardo located the working source:
   NOAA PSL's real-time OLR-based MJO index
   `https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt`, current
   through 2026-04-16.
3. Downloaded ROMI with a one-line `urllib.request` script in PowerShell
   and committed it as [`462582b`](https://github.com/monksealseal/winter2526-explorer/commit/462582b)
   (`Add ROMI MJO index data (1991 - 2026-04-16)`, 12 890 daily records,
   644 KB).

**What Claude 4.7 did** in response:

- Added `parse_romi()` to `indices.py` that reads the PSL format
  (`year month day flag ROMI1 ROMI2 amplitude`) and derives phase
  1-8 from `atan2(ROMI2, ROMI1)` using the Wheeler & Hendon (2004)
  octant convention.
- Updated `load_all_indices()` to prefer ROMI over the stale RMM file
  when both are present and to tag the active source via
  `indices["mjo_source"]`.
- Added Kiladis et al. (2014) to the bibliography and a new ROMI row
  to the Methods & Data provenance table.
- Updated the Research-compass MJO status banner to name the active
  source and flag staleness if the last date is more than 30 days old.

Landed as commits
[`d67e099`](https://github.com/monksealseal/winter2526-explorer/commit/d67e099)
(Claude's code changes) and
[`462582b`](https://github.com/monksealseal/winter2526-explorer/commit/462582b)
(Eduardo's data commit). The data commit is the *only* commit on this
repo authored by `Eduardo Siman <esiman@msn.com>` during the AI-assisted
sessions — every other commit is authored by Claude.
""")

with tab_rc:
    qp_set(tab="compass")
    import matplotlib.pyplot as _plt

    st.header("🧭 Research compass — Tori's project questions")
    st.caption(
        "Each card below is anchored to a specific question from the group's "
        "research proposal. Figures and statistics are pre-configured with defaults "
        "that answer that question directly. Open **Composites & Correlations** for "
        "unconstrained exploration with the same methods."
    )

    # ---- MJO data status banner ----
    if not MJO_LOADED:
        st.warning(
            "**MJO data not loaded.** Questions Q2, Q3, Q4 depend on a daily "
            "RMM1/RMM2 (or ROMI) phase index. Preferred source is the NOAA PSL "
            "ROMI file (Kiladis et al. 2014, real-time OLR-based), since the BoM "
            "Wheeler-Hendon RMM feed stalled updating in Feb 2024.\n\n"
            "Run the fetch locally and commit the file:\n\n"
            "```powershell\n"
            "python -c \"import urllib.request; from pathlib import Path; "
            "req=urllib.request.Request('https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt', "
            "headers={'User-Agent':'Mozilla/5.0'}); "
            "Path('data/indices/romi.txt').write_bytes(urllib.request.urlopen(req,timeout=30).read())\"\n"
            "git add data/indices/romi.txt\n"
            "git commit -m 'Add ROMI MJO index' && git push\n"
            "```"
        )
    else:
        _mjo = indices["mjo"]
        _src = indices.get("mjo_source", "MJO index")
        _winter_days = int(((_mjo.index >= cube.time.min().values) &
                            (_mjo.index <= cube.time.max().values)).sum())
        _last = _mjo.index.max().date()
        _stale = (pd.Timestamp.utcnow().tz_convert(None) - _mjo.index.max()).days > 30
        if _stale:
            st.warning(
                f"⚠ MJO loaded but **stale** — source: {_src}. Last date: "
                f"{_last} ({(pd.Timestamp.utcnow().tz_convert(None) - _mjo.index.max()).days} days "
                f"behind today). {_winter_days} days cover the winter window "
                f"({str(cube.time.min().values)[:10]} → {str(cube.time.max().values)[:10]}). "
                f"Consider switching to a current source (see Methods & Data)."
            )
        else:
            st.success(
                f"✓ MJO loaded — source: {_src}. {len(_mjo):,} total days, "
                f"last date {_last}, {_winter_days} days cover the winter window "
                f"({str(cube.time.min().values)[:10]} → {str(cube.time.max().values)[:10]})."
            )

    # Shared daily series used across the panels
    _cube_time = cube.time.values
    _dt_idx = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in _cube_time])
    fl_t2m_daily = (box_mean(cube.t2m_anom, FLORIDA_BOX)
                    .to_series().reindex(_dt_idx))
    se_t2m_daily = (box_mean(cube.t2m_anom, SE_US_BOX)
                    .to_series().reindex(_dt_idx))

    # ===== Q1: What were the coldest weeks over Florida? =====
    st.markdown("---")
    st.markdown("### Q1 · What were the coldest weeks over Florida?")
    st.caption(
        "*Method: contiguous runs of ≥N days with Florida-box mean T2m anomaly "
        "below a user threshold. Identifies the specific periods deserving focused "
        "dynamical diagnosis. "
        "Florida box: "
        f"{FLORIDA_BOX['lat_min']}-{FLORIDA_BOX['lat_max']}°N, "
        f"{abs(FLORIDA_BOX['lon_max'])}-{abs(FLORIDA_BOX['lon_min'])}°W.*"
    )
    q1c1, q1c2 = st.columns([1, 3])
    with q1c1:
        q1_thresh = st.slider("Threshold (°C anom)", -8.0, 0.0,
                              qp_get("q1_thr", -2.0, float), step=0.5, key="q1_thresh",
                              help="FL-box mean T2m anomaly must be below this value.")
        q1_min_days = st.slider("Min duration (days)", 1, 10,
                                qp_get("q1_min", 3, int), key="q1_min_days")
        qp_set(q1_thr=q1_thresh, q1_min=q1_min_days)
    # Event detection
    meets = (fl_t2m_daily < q1_thresh).fillna(False).values
    events = []
    i = 0
    while i < len(meets):
        if meets[i]:
            j = i
            while j < len(meets) and meets[j]:
                j += 1
            run_idx = _dt_idx[i:j]
            if len(run_idx) >= q1_min_days:
                period = fl_t2m_daily.loc[run_idx[0]:run_idx[-1]]
                events.append({
                    "start": run_idx[0].date(),
                    "end": run_idx[-1].date(),
                    "duration (d)": len(run_idx),
                    "peak date": period.idxmin().date(),
                    "min T2m anom (°C)": f"{period.min():+.2f}",
                    "mean T2m anom (°C)": f"{period.mean():+.2f}",
                })
            i = j
        else:
            i += 1
    with q1c2:
        fig_q1 = go.Figure()
        fig_q1.add_trace(go.Scatter(x=fl_t2m_daily.index, y=fl_t2m_daily.values,
                                     mode="lines",
                                     line=dict(color="#333", width=1.2),
                                     name="FL T2m anom"))
        fig_q1.add_hline(y=q1_thresh, line=dict(color="steelblue", dash="dash", width=1),
                         annotation_text=f"threshold = {q1_thresh}°C",
                         annotation_position="top left")
        for e in events:
            fig_q1.add_vrect(x0=e["start"], x1=e["end"],
                             fillcolor="steelblue", opacity=0.18, line_width=0)
        fig_q1.update_layout(
            title="Florida-box T2m anomaly with cold events shaded",
            height=300, margin=dict(l=60, r=20, t=50, b=40),
            yaxis_title="T2m anom (°C)", xaxis_title="")
        st.plotly_chart(fig_q1, use_container_width=True)
    if events:
        st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True)
        st.markdown(
            f"**Interpretation:** {len(events)} cold event(s) detected over Florida "
            f"this winter (threshold = {q1_thresh}°C, ≥{q1_min_days} days). "
            "The peak dates in this table are the natural anchor points for the "
            "MJO-phase, Z500-anomaly, and wave-train diagnostics below."
        )
    else:
        st.info(f"No cold events with threshold {q1_thresh}°C and ≥{q1_min_days} days.")

    # ===== Q5: Were seasonal teleconnections sufficient? =====
    st.markdown("---")
    st.markdown("### Q5 · Were seasonal teleconnections sufficient to explain Florida cold anomalies?")
    st.caption(
        "*Method: ordinary-least-squares multiple regression of daily Florida T2m "
        "anomaly on AO, NAO, PNA, and ONI. If R² is low, these four modes leave "
        "a lot of variance unexplained — consistent with sub-seasonal (MJO) or "
        "other drivers. The residual is the 'what's left over' time series, which "
        "should align with MJO phase if Tori's hypothesis holds.*"
    )
    X_cols = {}
    for k in ("ao", "nao", "pna", "oni"):
        if k in indices:
            X_cols[k] = align_index_to_cube(indices[k], _cube_time).values
    if len(X_cols) < 2:
        st.warning("Need at least 2 teleconnection indices loaded for this panel.")
    else:
        y_series = fl_t2m_daily.copy()
        X_df = pd.DataFrame(X_cols, index=_dt_idx)
        data = pd.concat([y_series.rename("fl_t2m"), X_df], axis=1).dropna()
        if len(data) < 30:
            st.warning(f"Insufficient overlapping daily data (n = {len(data)}).")
        else:
            y_arr = data["fl_t2m"].values
            X_arr = np.column_stack([np.ones(len(data)), data[list(X_cols.keys())].values])
            beta, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
            yhat = X_arr @ beta
            residual = y_arr - yhat
            n_obs, k_reg = len(y_arr), X_arr.shape[1]
            ss_res = float(np.sum(residual ** 2))
            ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
            r_sq = 1.0 - ss_res / ss_tot
            adj_r_sq = 1.0 - (1.0 - r_sq) * (n_obs - 1) / (n_obs - k_reg)
            sigma2 = ss_res / (n_obs - k_reg)
            try:
                cov = sigma2 * np.linalg.inv(X_arr.T @ X_arr)
                se_coef = np.sqrt(np.diag(cov))
                t_coef = beta / se_coef
                p_coef = 2.0 * (1.0 - scistats.t.cdf(np.abs(t_coef), df=n_obs - k_reg))
            except np.linalg.LinAlgError:
                se_coef = np.full_like(beta, np.nan)
                t_coef = np.full_like(beta, np.nan)
                p_coef = np.full_like(beta, np.nan)

            names = ["intercept"] + list(X_cols.keys())
            reg_df = pd.DataFrame({
                "regressor": names,
                "β": [f"{b:+.3f}" for b in beta],
                "SE": [f"{s:.3f}" for s in se_coef],
                "t": [f"{t:+.2f}" for t in t_coef],
                "p (OLS)": [f"{p:.3g}" for p in p_coef],
            })
            q5c1, q5c2 = st.columns([1, 2])
            with q5c1:
                st.metric("R² (variance explained)", f"{r_sq*100:.1f}%")
                st.metric("Adjusted R²", f"{adj_r_sq*100:.1f}%")
                st.metric("Sample size n (days)", f"{n_obs}")
                st.dataframe(reg_df, hide_index=True, use_container_width=True)
            with q5c2:
                fig_q5 = go.Figure()
                fig_q5.add_trace(go.Scatter(x=data.index, y=y_arr, mode="lines",
                                             name="observed FL T2m anom",
                                             line=dict(color="black", width=1.2)))
                fig_q5.add_trace(go.Scatter(x=data.index, y=yhat, mode="lines",
                                             name=f"explained by {'+'.join(X_cols)}",
                                             line=dict(color="#d95f02", width=1.2)))
                fig_q5.add_trace(go.Scatter(x=data.index, y=residual, mode="lines",
                                             name="residual (unexplained)",
                                             line=dict(color="steelblue", width=1, dash="dot")))
                fig_q5.add_hline(y=0, line=dict(color="#999", width=0.4))
                fig_q5.update_layout(
                    title="FL T2m anomaly · observed vs. teleconnection-explained vs. residual",
                    height=360, margin=dict(l=60, r=20, t=50, b=40),
                    yaxis_title="T2m anom (°C)", legend=dict(orientation="h"))
                st.plotly_chart(fig_q5, use_container_width=True)

            st.markdown(
                f"**Interpretation.** The four seasonal teleconnections "
                f"({', '.join(k.upper() for k in X_cols)}) jointly explain "
                f"**{r_sq*100:.1f}%** of daily FL T2m anomaly variance this winter "
                f"(adjusted R² = {adj_r_sq*100:.1f}%). The remaining "
                f"**{(1-r_sq)*100:.1f}%** is the residual time series plotted in blue "
                f"— the part that seasonal modes *cannot* account for. If Tori's "
                f"hypothesis holds, this residual should show structure around the "
                f"cold-event windows identified in Q1 and should correlate with "
                f"MJO phase at 5-15-day leads."
            )
            st.caption(
                "Caveat: OLS standard errors assume independent residuals, which is "
                "false for autocorrelated daily data — the OLS p-values above are "
                "anti-conservative. Treat significance as indicative; for "
                "publication use Newey-West (HAC) SEs or a block bootstrap."
            )

    # ===== Q2: Did MJO phases modulate Florida T2m? =====
    st.markdown("---")
    st.markdown("### Q2 · Did MJO phases modulate Florida T2m anomalies at 5-15 day leads?")
    st.caption(
        "*Method: 8-phase composite of FL T2m anomaly at lags 0, 5, 10, 15 days. "
        "Only days with MJO amplitude ≥ 1 are counted (phase is noisy below that). "
        "Tori's a-priori hypothesis: phases 7-8 (enhanced convection over the "
        "Western Hemisphere/Africa) should favour eastern-US cold at +10 days.*"
    )
    if not MJO_LOADED:
        st.info("*MJO RMM data required — see banner at top of this tab.*")
    else:
        mjo = indices["mjo"]  # DataFrame: RMM1, RMM2, phase, amplitude
        lags = [0, 5, 10, 15]
        phases = list(range(1, 9))
        grid = np.full((len(lags), len(phases)), np.nan)
        counts = np.zeros_like(grid, dtype=int)
        for li, lag in enumerate(lags):
            # index at t-lag predicts FL T2m at t, so shift MJO forward by `lag`
            mjo_lagged = mjo.reindex(_dt_idx, method="nearest", tolerance=pd.Timedelta(days=1)).shift(lag)
            active = mjo_lagged["amplitude"] >= 1.0
            for pj, ph in enumerate(phases):
                mask = active & (mjo_lagged["phase"] == ph)
                mask_arr = mask.fillna(False).values
                if mask_arr.sum() >= 3:
                    grid[li, pj] = float(np.nanmean(fl_t2m_daily.values[mask_arr]))
                    counts[li, pj] = int(mask_arr.sum())
        fig_q2 = go.Figure(data=go.Heatmap(
            z=grid, x=[f"Phase {p}" for p in phases],
            y=[f"lag +{lag}d" for lag in lags],
            colorscale="RdBu_r", zmid=0,
            hovertemplate="%{y} · %{x}<br>mean T2m anom = %{z:+.2f}°C<extra></extra>",
            colorbar=dict(title="FL T2m anom (°C)"),
            text=[[f"n={counts[li,pj]}" for pj in range(8)] for li in range(len(lags))],
            texttemplate="%{text}", textfont=dict(size=9, color="black"),
        ))
        fig_q2.update_layout(
            title="Florida T2m anomaly composite · by MJO phase × lag (amp ≥ 1 only)",
            height=320, margin=dict(l=80, r=20, t=50, b=40))
        st.plotly_chart(fig_q2, use_container_width=True)
        st.caption(
            f"Each cell: mean FL T2m anomaly on days when the MJO was in that phase "
            f"with amplitude ≥ 1, at the stated lag (positive lag = MJO leads). "
            f"n shown in-cell. With only {cube.sizes['time']} winter days, any phase × "
            f"lag bin with n < 5-8 should be read cautiously. "
            f"Source: {indices.get('mjo_source', 'MJO index')}. "
            f"Phase convention: Wheeler & Hendon (2004) octants; ROMI phases "
            f"derived from atan2(ROMI2, ROMI1) (Kiladis et al. 2014)."
        )

    # ===== Q3: Did MJO phases 7-8 enhance cold-air outbreaks? =====
    st.markdown("---")
    st.markdown("### Q3 · Did MJO phases 7-8 enhance eastern-US troughing / cold outbreaks?")
    st.caption(
        "*Method: lagged Z500-anomaly composite over CONUS on days when MJO was "
        "in phase 7 or 8 (amplitude ≥ 1). Expect (a priori): an eastern-US trough "
        "developing at +5 to +15 days through a Rossby wave train forced by "
        "enhanced WH/African convection.*"
    )
    if not MJO_LOADED:
        st.info("*MJO RMM data required — see banner at top of this tab.*")
    else:
        mjo = indices["mjo"]
        lats, lons = cube.latitude.values, cube.longitude.values
        z500_anom = cube.z500_anom.values  # (T, H, W)
        cols = st.columns(4)
        for li, lag in enumerate([0, 5, 10, 15]):
            mjo_lagged = mjo.reindex(_dt_idx, method="nearest",
                                     tolerance=pd.Timedelta(days=1)).shift(lag)
            mask = ((mjo_lagged["phase"].isin([7, 8])) &
                    (mjo_lagged["amplitude"] >= 1.0)).fillna(False).values
            n_days = int(mask.sum())
            with cols[li]:
                if n_days < 3:
                    st.info(f"lag +{lag}d · n={n_days} (insufficient)")
                else:
                    comp = np.nanmean(z500_anom[mask], axis=0)
                    fig_q3 = make_map(
                        lats, lons, comp,
                        cmap="RdBu_r", center_on_zero=True,
                        title=f"lag +{lag}d (n={n_days})",
                        subtitle=f"MJO phase 7-8, amp ≥ 1",
                        caption="", units="Z500 anom (m)",
                        figsize=(4.5, 2.8),
                        highlight_boxes=[{**FLORIDA_BOX, "label": "FL"}],
                        contour_levels=np.arange(-300, 301, 60),
                    )
                    st.pyplot(fig_q3, use_container_width=True)
                    _plt.close(fig_q3)
        st.caption(
            "A ridge over the Gulf of Alaska with a downstream eastern-US trough "
            "at +5 to +15 days is the canonical phase-7/8 signature (Johnson et al. "
            "2014, Mon. Wea. Rev. 142, 1556-1577). Climatology baseline: "
            f"{Z500_CLIMO_BASE}."
        )

    # ===== Q4: MJO × ENSO joint conditional =====
    st.markdown("---")
    st.markdown("### Q4 · Constructive interference between MJO and ENSO?")
    st.caption(
        "*Method: 2×2 conditional composite of CONUS T2m anomaly at +10 day lag. "
        "Rows split MJO phase (1-2 vs 7-8 at amplitude ≥ 1); columns split ENSO "
        "state (ONI ≤ -0.5 = La Niña-ish vs ONI > -0.5 = neutral/warm). "
        "Tests whether the MJO effect depends on the ENSO background — the "
        "'constructive interference' framing in Tori's proposal.*"
    )
    if not MJO_LOADED:
        st.info("*MJO RMM data required — see banner at top of this tab.*")
    elif "oni" not in indices:
        st.warning("*ONI index not loaded; cannot condition on ENSO state.*")
    else:
        mjo = indices["mjo"]
        lats, lons = cube.latitude.values, cube.longitude.values
        t2m_anom_arr = cube.t2m_anom.values
        lag_q4 = 10
        mjo_lagged = mjo.reindex(_dt_idx, method="nearest",
                                 tolerance=pd.Timedelta(days=1)).shift(lag_q4)
        oni_aligned = pd.Series(
            align_index_to_cube(indices["oni"], _cube_time).values,
            index=_dt_idx)
        amp_ok = mjo_lagged["amplitude"] >= 1.0
        mjo_groups = {
            "MJO phase 1-2 (enhanced Pacific conv.)": mjo_lagged["phase"].isin([1, 2]) & amp_ok,
            "MJO phase 7-8 (enhanced WH/Africa conv.)": mjo_lagged["phase"].isin([7, 8]) & amp_ok,
        }
        enso_groups = {
            "Weak La Niña (ONI ≤ -0.5)": oni_aligned <= -0.5,
            "Neutral / warm (ONI > -0.5)": oni_aligned > -0.5,
        }
        rows_q4 = list(mjo_groups.keys())
        cols_q4 = list(enso_groups.keys())
        grid_cols = st.columns(len(cols_q4))
        for ci, (enso_label, enso_mask) in enumerate(enso_groups.items()):
            with grid_cols[ci]:
                st.markdown(f"**{enso_label}**")
                for mjo_label, mjo_mask in mjo_groups.items():
                    joint = (mjo_mask & enso_mask).fillna(False).values
                    n_days = int(joint.sum())
                    if n_days < 3:
                        st.info(f"{mjo_label} · n={n_days} (insufficient)")
                        continue
                    comp = np.nanmean(t2m_anom_arr[joint], axis=0)
                    fig_q4 = make_map(
                        lats, lons, comp,
                        cmap="RdBu_r", center_on_zero=True,
                        title=f"{mjo_label}",
                        subtitle=f"+{lag_q4}d lag · n={n_days} days",
                        caption="", units="T2m anom (°C)",
                        figsize=(5.0, 3.0),
                        highlight_boxes=[{**FLORIDA_BOX, "label": "FL"}],
                    )
                    st.pyplot(fig_q4, use_container_width=True)
                    _plt.close(fig_q4)
        st.caption(
            "Compare the phase 7-8 row across the two ENSO columns: stronger cold "
            "over Florida under weak La Niña than under neutral would indicate "
            "constructive interference. With only ~150 winter days split four ways, "
            "any single cell is drawn from tens of days at most — treat differences "
            "between the four panels as suggestive, not conclusive. Climatology "
            f"baseline: {T2M_CLIMO_BASE}."
        )

    # ===== Deferred questions =====
    st.markdown("---")
    with st.expander("Deferred questions — need additional data"):
        st.markdown(f"""
- **Q6. How did 2025-26 compare to previous La Niña winters?**
  Needs daily ERA5 T2m and Z500 for 2016-17, 2017-18, 2020-21, 2021-22, 2022-23
  (i.e. the La Niña members of the 2016-2024 climatology period). Current
  cube only covers 2025-11 → 2026-03.
  *Scope:* extend `preprocess.py` to write a `cube_lanina_winters.nc` bundle;
  estimated +50-80 MB. Analysis then = monthly cosine-similarity analog search
  (Van den Dool 1994; Lorenz 1969) of this winter's Z500 / T2m anomalies against
  each prior La Niña month.

- **Q7. Rossby wave-train mechanism ("inspired by Kathy").**
  Needs Z500 across the full Northern Hemisphere, not just CONUS (−125 to −67°E).
  Current cube is CONUS-bounded.
  *Scope:* extend `preprocess.py` to regrid hemispheric Z500; estimated
  +30-50 MB. Analysis = longitude-time Hovmöller of Z500 anomaly averaged over
  30-60°N, plus (optionally) Takaya-Nakamura (2001) wave-activity flux vectors
  overlaid on composite Z500 maps.

- **Q8. 250-mb jet-stream configuration.**
  Needs `u250` (and ideally `v250`) added to `preprocess.py`.
  *Scope:* +2 variables on current grid; estimated +15 MB. Analysis = jet-axis
  diagnostic (latitude of 250 mb u-wind max by longitude) and variance maps.
""")

with tab1:
    months = {"Dec 2025": "2025-12", "Jan 2026": "2026-01", "Feb 2026": "2026-02", "Mar 2026": "2026-03"}
    c1, c2 = st.columns([1, 3])
    with c1:
        month_default = qp_get("month", "Dec 2025", str)
        month_default = month_default if month_default in months else "Dec 2025"
        month_label = st.radio("Month", list(months.keys()),
                               index=list(months.keys()).index(month_default), key="w_month")
        v_opts = ["t2m_anom", "z500_anom", "precip"]
        v_default = qp_get("field", "t2m_anom", str)
        v_default = v_default if v_default in v_opts else "t2m_anom"
        field_t1 = st.selectbox("Variable", v_opts,
                                index=v_opts.index(v_default),
                                format_func=lambda k: VAR_META[k]["label"], key="w_field")
    qp_set(tab="winter", month=month_label, field=field_t1)

    ms = cube.sel(time=months[month_label])[field_t1]
    monthly_mean = ms.mean(dim="time")
    valid_days = int((~ms.mean(dim=("latitude", "longitude")).isnull()).sum())
    meta = VAR_META[field_t1]

    is_anom = field_t1.endswith("_anom")
    climo_note = {
        "t2m_anom": T2M_CLIMO_BASE, "z500_anom": Z500_CLIMO_BASE,
    }.get(field_t1, "—")
    source_note = PRECIP_SOURCE if field_t1 == "precip" else "ERA5 (Hersbach et al. 2020)"
    subtitle_parts = [source_note, f"{valid_days} valid day{'s' if valid_days != 1 else ''}"]
    if is_anom:
        subtitle_parts.insert(1, f"climatology: {climo_note}")
    subtitle = " · ".join(subtitle_parts)
    caption = (f"Fig. {month_label} — {meta['label']}. "
               f"Diverging colormap centered on zero for anomaly fields. "
               f"Dashed box marks the SE-US analysis region "
               f"({SE_US_BOX['lat_min']}-{SE_US_BOX['lat_max']}°N, "
               f"{abs(SE_US_BOX['lon_max'])}-{abs(SE_US_BOX['lon_min'])}°W).")
    if field_t1 == "t2m_anom":
        caption += " Limitation: 9-year (2016-2024) climatology may be warm-biased vs. the 1991-2020 WMO normal."
    if field_t1 == "precip" and month_label == "Dec 2025":
        caption += " CPC precipitation data begins 2026-01-01 — December 2025 rendered as NaN."
    if field_t1 == "z500_anom" and month_label == "Mar 2026":
        caption += " ERA5 Z500 coverage ends 2026-02-28 — March 2026 rendered as NaN."

    with c2:
        if valid_days == 0:
            st.warning(f"No valid days in {month_label} for {field_t1} — see Methods & Data for coverage.")
        else:
            fig = make_map(
                monthly_mean.latitude.values, monthly_mean.longitude.values,
                monthly_mean.values,
                cmap=meta["cmap"],
                vmin=meta["vmin"], vmax=meta["vmax"],
                center_on_zero=is_anom,
                title=f"{meta['label']} · {month_label} mean",
                subtitle=subtitle,
                caption=caption,
                units=meta["label"],
                highlight_boxes=[{**SE_US_BOX, "label": "SE-US"}],
                contour_levels=np.arange(-300, 301, 60) if field_t1 == "z500_anom" else None,
            )
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as _plt
            _plt.close(fig)

    if valid_days and valid_days < 25:
        st.warning(f"⚠️ Only {valid_days} valid days in {month_label} for {field_t1}. "
                   "See Methods & Data for coverage.")

    if field_t1 == "z500_anom":
        st.caption(
            "**Reading Z500 anomaly maps:** red = positive anomaly (ridging / high pressure), "
            "blue = negative anomaly (troughing / low pressure). Contours drawn every 60 m. "
            "A PNA-positive pattern shows an Alaska ridge with an eastern-US trough "
            "(Wallace & Gutzler 1981)."
        )

    c_a, c_b = st.columns(2)
    c_a.metric("SE-US box mean", f"{float(box_mean(monthly_mean, SE_US_BOX)):.2f}")
    c_b.metric("Florida box mean", f"{float(box_mean(monthly_mean, FLORIDA_BOX)):.2f}")

    with st.expander("About this analysis"):
        st.markdown(f"""
**What:** Grid-point monthly mean of **{meta['label']}** over CONUS for
**{month_label}**, computed as the simple arithmetic mean across the
{valid_days} daily snapshots in the month.

**Anomaly definition (for `*_anom` variables):** daily value minus the
same-calendar-day mean from the {climo_note if is_anom else '— (not applicable)'}
baseline, averaged over the month.

**Projection:** PlateCarree (equirectangular), native ERA5 0.25° grid.
Black contours on Z500 are every 60 m.

**Why this plot:** it's the standard "what happened this month" figure —
directly comparable to CPC monthly climate summaries and to the composite
panels in Tab 3. Use the Composites & Correlations tab to test attribution.

**Caveats:** the 2016-2024 T2m climatology is shorter than the WMO-standard
30-year baseline; early/late-season anomalies may be mildly warm-biased.
""")

with tab2:
    c_l, c_r = st.columns([1, 3])
    with c_l:
        view_default = qp_get("view", "daily", str)
        view_default = view_default if view_default in ("daily", "monthly") else "daily"
        view_mode = st.radio(
            "View", ["Daily", "Monthly"],
            index=0 if view_default == "daily" else 1, key="idx_view",
            help="Monthly aggregates indices and SE-US T2m anomaly to month-start means "
                 "(≥15 valid days required). Matches the methodology behind Abby's reference "
                 "r-values (slides 38/46/47)."
        )
        is_monthly = view_mode == "Monthly"

        default_picks = [p for p in qp_get("idx", "ao,nao", str).split(",") if p in available_indices]
        if not default_picks:
            default_picks = available_indices[:2]
        picks = st.multiselect("Indices to show", available_indices, default=default_picks,
                               format_func=lambda k: INDEX_META[k]["label"], key="idx_picks")
        overlay = st.checkbox("Overlay SE-US T2m anomaly",
                              value=qp_get("overlay", "1", str) == "1", key="overlay")
    qp_set(tab="indices",
           view="monthly" if is_monthly else "daily",
           idx=",".join(picks) if picks else None,
           overlay="1" if overlay else "0")

    if not picks:
        c_r.info("Select at least one index to display.")
    else:
        cube_time = cube.time.values
        t2m_se_daily = box_mean(cube.t2m_anom, SE_US_BOX).to_series()
        lo_str, hi_str = str(cube_time.min())[:10], str(cube_time.max())[:10]
        t2m_se_monthly = to_monthly(t2m_se_daily)
        t2m_overlay = t2m_se_monthly if is_monthly else t2m_se_daily

        fig = make_subplots(rows=len(picks), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=[INDEX_META[p]["label"] for p in picks])
        for i, pick in enumerate(picks, 1):
            s = get_series(indices, pick)
            if s.empty: continue
            s_win = s.loc[lo_str:hi_str]
            if is_monthly and INDEX_META[pick]["cadence"] == "daily":
                s_win = to_monthly(s_win)
            s_win = s_win.dropna()
            if s_win.empty:
                continue

            if is_monthly:
                colors = ["crimson" if v >= 0 else "royalblue" for v in s_win.values]
                fig.add_trace(go.Bar(
                    x=s_win.index, y=s_win.values, marker_color=colors,
                    name=INDEX_META[pick]["label"], showlegend=False,
                    hovertemplate="%{x|%b %Y}<br>%{y:+.3f}<extra></extra>"
                ), row=i, col=1)
            else:
                pos, neg = s_win.where(s_win >= 0), s_win.where(s_win < 0)
                fig.add_trace(go.Scatter(x=s_win.index, y=pos.values, mode="lines",
                                         line=dict(color="crimson", width=1.5), name="≥ 0",
                                         legendgroup="pos", showlegend=(i == 1)), row=i, col=1)
                fig.add_trace(go.Scatter(x=s_win.index, y=neg.values, mode="lines",
                                         line=dict(color="royalblue", width=1.5), name="< 0",
                                         legendgroup="neg", showlegend=(i == 1)), row=i, col=1)
            fig.add_hline(y=0, line=dict(color="black", width=0.4, dash="dash"), row=i, col=1)
            fig.update_yaxes(title_text=INDEX_META[pick]["unit"], row=i, col=1)

        if overlay and not t2m_overlay.dropna().empty:
            t_plot = t2m_overlay.dropna()
            fig.add_trace(go.Scatter(
                x=t_plot.index, y=t_plot.values,
                mode="lines+markers" if is_monthly else "lines",
                line=dict(color="black", width=1.2, dash="dot"),
                name="SE-US T2m anom", showlegend=True), row=1, col=1)

        fig.update_layout(height=max(400, 180 * len(picks)), hovermode="x unified",
                         margin=dict(l=60, r=40, t=60, b=40))
        with c_r:
            st.plotly_chart(fig, use_container_width=True)

        cadence_label = "Monthly" if is_monthly else "Daily"
        r_col = f"r (this app, {cadence_label.lower()})"
        st.markdown(f"### {cadence_label} correlations with SE-US T2m anomaly")
        rows = []
        skipped = []
        for pick in picks:
            s = get_series(indices, pick)
            if s.empty: continue
            ref = REFERENCE_R.get(f"{pick}_t2m")
            ref_r = f"{ref['r']:+.3f}" if ref else "—"
            ref_src = ref["source"] if ref else "—"

            if is_monthly:
                if INDEX_META[pick]["cadence"] == "daily":
                    s_m = to_monthly(s.loc[lo_str:hi_str])
                else:
                    s_m = s.loc[lo_str:hi_str].resample("MS").mean()
                common = pd.concat([s_m, t2m_se_monthly], axis=1,
                                   keys=["idx", "t2m"]).dropna()
                n = len(common)
                if n < 3:
                    skipped.append(f"{INDEX_META[pick]['label']} (only {n} month{'s' if n != 1 else ''} of overlap)")
                    continue
                r = common["idx"].corr(common["t2m"])
                n_eff = n
            else:
                s_a = align_index_to_cube(s, cube_time)
                t_a = t2m_se_daily.reindex(
                    pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time]))
                common = pd.concat([s_a, t_a], axis=1, keys=["idx", "t2m"]).dropna()
                if len(common) < 10:
                    skipped.append(f"{INDEX_META[pick]['label']} (only {len(common)} overlapping days)")
                    continue
                r = common["idx"].corr(common["t2m"])
                n = len(common)
                n_eff = effective_n(common["idx"].values) if INDEX_META[pick]["cadence"] == "daily" else n

            if abs(1 - r**2) > 1e-10 and n_eff > 2:
                t_s = r * np.sqrt(n_eff - 2) / np.sqrt(1 - r**2)
                p = 2 * (1 - scistats.t.cdf(abs(t_s), df=n_eff - 2))
            else:
                p = np.nan
            rows.append({
                "Index": INDEX_META[pick]["label"],
                r_col: f"{r:+.3f}",
                "n": n, "n_eff": n_eff,
                "p (n_eff adj)": f"{p:.3f}" if not np.isnan(p) else "—",
                "Group ref r": ref_r,
                "Group ref source": ref_src,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if is_monthly:
                st.caption(
                    "**Monthly correlations** use month-start means (≥15 valid days per month). "
                    "This matches the cadence behind Abby's reference r-values, so magnitudes should be "
                    "comparable — expect larger |r| than the daily view because weather noise averages out. "
                    "Caveat: only ~4 winter months of overlap means n is tiny and p-values are weak."
                )
            else:
                st.caption(
                    "**Daily correlations.** Abby's reference r=0.561 is labeled *monthly*; daily values are "
                    "noisier so a smaller r is expected. Switch the view above to **Monthly** to reconcile "
                    "with the group's methodology before citing."
                )
        if skipped:
            st.caption("Skipped (insufficient overlap): " + "; ".join(skipped))

with tab3:
    c_ctrl, c_maps = st.columns([1, 3])
    with c_ctrl:
        idx_default = qp_get("cidx", "ao", str)
        idx_default = idx_default if idx_default in available_indices else (available_indices[0] if available_indices else "ao")
        idx_pick = st.selectbox("Index", available_indices,
                                index=available_indices.index(idx_default) if idx_default in available_indices else 0,
                                format_func=lambda k: INDEX_META[k]["label"], key="c_idx")

        f_opts = ["t2m_anom", "precip", "z500_anom", "t2m", "z500"]
        f_default = qp_get("cfield", "t2m_anom", str)
        f_default = f_default if f_default in f_opts else "t2m_anom"
        field_pick = st.selectbox("Field", f_opts, index=f_opts.index(f_default),
                                  format_func=lambda k: VAR_META[k]["label"], key="c_field")

        lag = st.slider("Lag (days: index leads field)", -30, 30,
                       qp_get("lag", 0, int), key="c_lag",
                       help="Positive: index at t-lag predicts field at t")

        th_default = qp_get("thmode", "sign", str)
        th_mode = st.radio("Threshold",
            ["Sign (> 0 vs < 0)", "±σ standardized"],
            index=0 if th_default == "sign" else 1, key="c_thmode",
            help="Composite difference = (positive index days) − (negative index days). "
                 "Sign: split on 0. ±σ: require |index| > threshold·σ to enter each group — "
                 "stricter but smaller samples. To read **negative-phase** impacts "
                 "(e.g. AO<0 cold outbreaks), look for **blue** in the difference map: field is "
                 "lower during positive index, i.e. **higher during negative index**.")
        if th_mode.startswith("±"):
            threshold = st.slider("σ threshold", 0.25, 2.5,
                                 qp_get("th", 1.0, float), step=0.25, key="c_th")
        else:
            threshold = 0.0

        show_sig = st.checkbox("Significance stippling (p<0.05, n_eff adj.)",
                               value=qp_get("sig", "0", str) == "1", key="c_sig")

    qp_set(tab="composites", cidx=idx_pick, cfield=field_pick, lag=lag,
           thmode="sigma" if th_mode.startswith("±") else "sign",
           th=(threshold if th_mode.startswith("±") else None),
           sig="1" if show_sig else "0")

    series = get_series(indices, idx_pick)
    if series.empty:
        c_maps.warning(f"'{INDEX_META[idx_pick]['label']}' is not available. Check data/indices/.")
    else:
        cube_time = cube.time.values
        field = cube[field_pick].values

        idx_aligned = align_index_to_cube(series, cube_time)
        if lag != 0:
            idx_aligned = idx_aligned.shift(lag)
        idx_vals = idx_aligned.values
        valid = ~np.isnan(idx_vals)

        if th_mode.startswith("±"):
            full_std = float(np.nanstd(series.dropna().values))
            mask_pos = valid & (idx_vals >  threshold * full_std)
            mask_neg = valid & (idx_vals < -threshold * full_std)
            th_label = f"±{threshold:.2f}σ"
        else:
            mask_pos = valid & (idx_vals > 0)
            mask_neg = valid & (idx_vals < 0)
            th_label = "sign"

        comp = welch_t_composite(field, mask_pos, mask_neg, alpha=BOOTSTRAP_ALPHA)
        r_map, n_corr = correlation_map(field, idx_vals)
        n_eff = effective_n(idx_vals[valid])
        r_sig_mask = corr_map_t_significance(r_map, n_eff, alpha=BOOTSTRAP_ALPHA)
        # critical |r| for the caption (two-sided t at n_eff-2 df)
        if n_eff > 2:
            _tc = scistats.t.ppf(1 - BOOTSTRAP_ALPHA/2, df=n_eff - 2)
            r_crit = float(_tc / np.sqrt(_tc**2 + n_eff - 2))
        else:
            r_crit = 1.0

        lats, lons = cube.latitude.values, cube.longitude.values
        meta = VAR_META[field_pick]
        idx_label = INDEX_META[idx_pick]["label"]
        field_label = meta["label"]
        climo_base = (T2M_CLIMO_BASE if field_pick == "t2m_anom" else
                      Z500_CLIMO_BASE if field_pick == "z500_anom" else
                      "— (raw field, not anomalized)" if field_pick in ("t2m", "z500", "precip") else "—")

        import matplotlib.pyplot as _plt
        with c_maps:
            a, b = st.columns(2)
            with a:
                composite_caption = (
                    f"Fig. Composite difference of {field_label} between days with "
                    f"{idx_label} > 0 and < 0 (threshold: {th_label}; lag {lag:+d}d). "
                    f"n(+) = {comp['n_pos']}, n(−) = {comp['n_neg']}. "
                    f"Stippling: two-sided Welch's t-test, α = {BOOTSTRAP_ALPHA} (von Storch & Zwiers 1999 §6). "
                    "Diverging colormap centered on zero."
                )
                if comp["n_pos"] < 5 or comp["n_neg"] < 5:
                    st.warning(f"Sample size is small (n+={comp['n_pos']}, n−={comp['n_neg']}). "
                               "Interpret with caution.")
                fig_d = make_map(
                    lats, lons, comp["diff"],
                    cmap="RdBu_r", center_on_zero=True,
                    title=f"Composite Δ {field_label}",
                    subtitle=f"{idx_label} (+) − (−) · lag {lag:+d}d · threshold {th_label}",
                    caption=composite_caption,
                    units=f"Δ {field_label}",
                    stipple_mask=comp["sig"] if show_sig else None,
                    highlight_boxes=[{**SE_US_BOX, "label": "SE-US"}],
                )
                st.pyplot(fig_d, use_container_width=True)
                _plt.close(fig_d)

            with b:
                corr_caption = (
                    f"Fig. Per-grid-cell Pearson r between daily {idx_label} and {field_label} "
                    f"(lag {lag:+d}d). n = {n_corr} days; effective sample size n_eff = {n_eff} after "
                    f"AR(1) adjustment (Bretherton et al. 1999). "
                    f"Stippling: two-sided t-test on r with df = n_eff − 2, α = {BOOTSTRAP_ALPHA} "
                    f"(|r| > {r_crit:.3f})."
                )
                fig_r = make_map(
                    lats, lons, r_map,
                    cmap="RdBu_r", vmin=-1, vmax=1,
                    title=f"Correlation r · {idx_label} vs {field_label}",
                    subtitle=f"lag {lag:+d}d · n={n_corr}, n_eff={n_eff} · |r| for α=0.05 ≈ {r_crit:.3f}",
                    caption=corr_caption,
                    units="r",
                    stipple_mask=r_sig_mask if show_sig else None,
                    highlight_boxes=[{**SE_US_BOX, "label": "SE-US"}],
                )
                st.pyplot(fig_r, use_container_width=True)
                _plt.close(fig_r)

        # ---- SE-US box summary with bootstrap CI ----
        se_r_da = xr.DataArray(r_map, dims=("latitude", "longitude"),
                               coords=dict(latitude=lats, longitude=lons))
        se_mean_r = float(box_mean(se_r_da, SE_US_BOX))

        se_field_box = box_mean(cube[field_pick], SE_US_BOX).to_series()
        idx_series = pd.Series(idx_aligned.values,
                               index=pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time]))
        paired = pd.concat([idx_series, se_field_box], axis=1,
                           keys=["idx", "field"]).dropna()
        if len(paired) >= 10:
            bs = cached_bootstrap_corr(tuple(paired["idx"].values),
                                       tuple(paired["field"].values),
                                       n_boot=BOOTSTRAP_N)
            se_r_str = fmt_ci(bs)
            se_n_str = f"n = {bs['n']}, block length = {bs['block_len']}d"
        else:
            se_r_str = "insufficient overlap"
            se_n_str = f"n = {len(paired)}"

        mm1, mm2 = st.columns(2)
        mm1.metric("SE-US mean r (map average)", f"{se_mean_r:+.3f}",
                   help="Cosine-weighted mean of the correlation map over the SE-US box.")
        mm2.metric("SE-US box r (time-series bootstrap)", se_r_str,
                   help=f"Pearson r between {idx_label} and SE-US box-mean {field_label}. "
                        f"95% CI from moving-block bootstrap, B = {BOOTSTRAP_N}. {se_n_str}.")

        with st.expander("Absolute-phase composites"):
            st.caption(
                f"Mean {field_label} on days in the positive vs negative phase of {idx_label}. "
                "Use the negative-phase panel to read the pattern directly "
                "(e.g. for AO<0 cold-air outbreaks) without mentally inverting the difference map. "
                f"Climatology baseline: {climo_base}."
            )
            aa, bb = st.columns(2)
            abs_meta = VAR_META[field_pick]
            is_anom_abs = field_pick.endswith("_anom")
            for panel, data, lbl, n in [
                (aa, comp["mean_pos"], f"positive {idx_label}", comp["n_pos"]),
                (bb, comp["mean_neg"], f"negative {idx_label}", comp["n_neg"]),
            ]:
                with panel:
                    if np.isfinite(data).any():
                        fig_a = make_map(
                            lats, lons, data,
                            cmap=abs_meta["cmap"],
                            vmin=abs_meta["vmin"], vmax=abs_meta["vmax"],
                            center_on_zero=is_anom_abs,
                            title=f"Mean on {lbl} ({n} days)",
                            subtitle=f"lag {lag:+d}d · threshold {th_label}",
                            caption=f"Simple time-mean of {field_label}. No significance test.",
                            units=field_label,
                            highlight_boxes=[{**SE_US_BOX, "label": "SE-US"}],
                            contour_levels=np.arange(-300, 301, 60) if field_pick == "z500_anom" else None,
                        )
                        st.pyplot(fig_a, use_container_width=True)
                        _plt.close(fig_a)

        with st.expander("About this analysis"):
            st.markdown(f"""
**Composite difference map.** For each day, the index value at time
``t − lag`` is compared to zero (or ±{threshold if th_mode.startswith('±') else 0:.2f}σ).
Days in the positive and negative groups are averaged separately for every grid cell,
and the difference (positive − negative) is tested with a two-sided **Welch's t-test**
(Welch 1947; von Storch & Zwiers 1999 §6). Stippling marks cells where the
difference is significant at α = {BOOTSTRAP_ALPHA}. Welch's test does not require
equal variances between the two composite groups.

**Correlation map.** Cell-wise Pearson correlation between the daily index and the
daily field. Significance is a two-sided t-test on r with df = ``n_eff − 2``, where
``n_eff = n (1 − r1) / (1 + r1)`` uses the lag-1 autocorrelation of the index to
adjust for serial dependence (Bretherton et al. 1999). The critical |r| for this
n_eff is stated in the figure subtitle.

**SE-US box time-series correlation** uses a **moving-block bootstrap** (Künsch 1989;
Wilks 2011 §5.3.5), B = {BOOTSTRAP_N}, block length ≈ n / n_eff. 95% CI is the
2.5/97.5 percentiles of the resampling distribution. This is more honest than a
parametric p-value on autocorrelated daily data.

**Limitations.** With only 151 winter days, n_eff is typically 15-25 for daily
teleconnection indices — significance tests are weak and should be read alongside
the CI on the SE-US summary. Composite differences at tails of the distribution
(±2σ) are doubly weak because group sizes shrink.
""")

with tab4:
    qp_set(tab="methods")
    st.header("Methods & Data")
    st.caption(
        "Everything shown in the other tabs is computed with the methods "
        "documented here. Figures produced by this app can be cited with the "
        "provenance and references listed below."
    )

    # ---- 1. Methods ----
    st.markdown("## 1. Methods")
    st.markdown("""
**Anomalies.** For every variable, the daily anomaly is the daily value
minus the same-calendar-day mean of the climatology base period
(variable-specific, listed in §3).

**Correlation at a point / box.** Pearson correlation ``r`` between a daily
teleconnection index and the area-weighted (cos φ) mean of the field over
the named box. 95% confidence interval from the **moving-block bootstrap**
(Künsch 1989; Wilks 2011 §5.3.5): B = 1000 resamples of contiguous blocks
of length ``n / n_eff``, where ``n_eff = n (1 − r1) / (1 + r1)`` with ``r1``
the lag-1 autocorrelation of the index (Bretherton et al. 1999).

**Correlation map.** Per grid cell, Pearson r between the daily index and
the daily field; cells with fewer than 10 finite pairs are masked. Stippling
marks cells where a two-sided t-test on r exceeds α = 0.05 using df =
``n_eff − 2``. This is a pointwise test; we do not apply a global
field-significance correction (Wilks 2016, *BAMS* 97, 2263-2273) and the
effective density of significant cells will be overstated under the null.

**Composite difference.** Days are split by the sign (or ±σ magnitude) of
the lagged index. The positive and negative groups are averaged separately
at each grid cell and differenced. Per-cell significance from a two-sided
**Welch's unequal-variance t-test** (Welch 1947; von Storch & Zwiers 1999
§6) with the Welch-Satterthwaite degrees of freedom.

**Absolute-phase composites.** Simple time-mean of the field over each
phase group; no significance test shown.

**Multiple regression.** Ordinary least squares ``y = Xβ + ε`` where ``y``
is the daily Florida-box T2m anomaly and ``X`` is (intercept, AO, NAO, PNA,
ONI) on the same day. Reported: β, OLS SE, t statistic, two-sided p-value
(anti-conservative under residual autocorrelation — flagged in the panel).

**Event detection.** Contiguous runs of days for which the Florida-box
T2m anomaly drops below a user-set threshold, with a minimum-duration
filter. Event peak = date of minimum anomaly within the run.

**Projection.** All maps render with cartopy in PlateCarree (Equidistant
Cylindrical) on the native ERA5 0.25° grid. Coastline, country, and
state polygons from Natural Earth (1:50 m).
""")

    # ---- 2. Data provenance ----
    st.markdown("## 2. Data provenance")
    st.dataframe(pd.DataFrame(PROVENANCE), hide_index=True, use_container_width=True)

    # ---- 3. Cube extent and current state ----
    st.markdown("## 3. Current cube state")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Extent**")
        st.markdown(f"- Time: `{str(cube.time.min().values)[:10]}` → `{str(cube.time.max().values)[:10]}` ({cube.sizes['time']} days)")
        st.markdown(f"- Lat: {float(cube.latitude.min()):.2f}° to {float(cube.latitude.max()):.2f}° ({cube.sizes['latitude']} pts)")
        st.markdown(f"- Lon: {float(cube.longitude.min()):.2f}° to {float(cube.longitude.max()):.2f}° ({cube.sizes['longitude']} pts)")
        st.markdown("- Grid: ~0.25° (ERA5 native)")
    with c2:
        st.markdown("**Variables (missingness)**")
        for var in cube.data_vars:
            n_nan = int(cube[var].isnull().sum()); total = cube[var].size
            st.markdown(f"- `{var}` — {VAR_META.get(var, {}).get('label', var)}  *({100 * n_nan / total:.1f}% NaN)*")

    # ---- 4. Indices loaded ----
    st.markdown("## 4. Indices loaded")
    idx_rows = []
    for k in INDEX_META:
        base = "mjo" if k == "mjo_amp" else k
        if base in indices:
            s = indices[base]
            idx_rows.append({"Index": INDEX_META[k]["label"],
                             "Range": f"{s.index.min().date()} → {s.index.max().date()}",
                             "N": len(s), "Cadence": INDEX_META[k]["cadence"]})
        else:
            idx_rows.append({"Index": INDEX_META[k]["label"], "Range": "NOT LOADED",
                             "N": 0, "Cadence": INDEX_META[k]["cadence"]})
    st.dataframe(pd.DataFrame(idx_rows), hide_index=True, use_container_width=True)

    if "mjo" not in indices:
        st.info(
            "**How to enable MJO RMM data.** "
            "This sandbox blocks outbound HTTP to BoM/NOAA; do the fetch on a "
            "machine with internet access and commit the result:\n\n"
            "```bash\n"
            "python -c \"from indices import fetch_mjo; from pathlib import Path; "
            "print(fetch_mjo(Path('data/indices/mjo_rmm.txt')))\"\n"
            "git add data/indices/mjo_rmm.txt\n"
            "git commit -m 'Add MJO RMM data for Tori analyses'\n"
            "git push\n"
            "```\n"
            "After the next Streamlit Cloud redeploy, Q2–Q4 in the Research "
            "Compass tab will activate."
        )

    # ---- 5. Known limitations ----
    st.markdown("## 5. Known limitations")
    st.markdown(f"""
- **T2m climatology = {T2M_CLIMO_BASE}** — 9 years, shorter than the WMO-standard
  30-year normal. Anomalies may be warm-biased vs. 1991-2020 because recent years
  are warmer than the mid-climate baseline.
- **Z500 climatology = {Z500_CLIMO_BASE}** — close to WMO-standard (27 years).
- **Precipitation coverage: 2026-01-01 onward.** December 2025 precip is absent;
  December maps render NaN. Data source: {PRECIP_SOURCE}.
- **Z500 coverage ends 2026-02-28.** March 2026 Z500 panels render NaN.
- **Daily means** from 00/12 UTC snapshots only — no overnight minima; slight
  (<0.5 °C) warm bias over land during the 12 UTC pass.
- **n is small.** Only 151 winter days total; with lag-1 autocorrelation the
  effective sample size is typically 15-30 for daily teleconnection indices.
  Parametric p-values below ~0.05 should be treated as indicative, not decisive.
- **No field-significance correction.** Stippling on the maps flags pointwise
  significance; neighbouring grid cells are correlated and the *expected*
  number of "significant" cells under the null is >5%. Use Wilks (2016) FDR
  control if results are cited.
- **Does not replace expert judgment.** Confirm findings with Prof. Becker /
  Prof. Nolan before presenting or citing.
""")

    # ---- 6. Reference r-values from the group's deck ----
    st.markdown("## 6. Reference r-values from the group's deck")
    st.dataframe(pd.DataFrame([
        {"Index": k.split("_")[0].upper(), "Reference r": f"{v['r']:+.3f}",
         "Source": v["source"], "Method": v["method"]}
        for k, v in REFERENCE_R.items()
    ]), hide_index=True, use_container_width=True)

    # ---- 7. References ----
    st.markdown("## 7. References")
    for author, title, journal in REFERENCES:
        st.markdown(f"- **{author}** — {title}. *{journal}*.")

    with st.expander("Raw xarray cube repr"):
        st.code(repr(cube), language="python")
