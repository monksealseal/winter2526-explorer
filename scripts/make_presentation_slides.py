"""Generate two publication-quality PNG slides for the group presentation.

Outputs (16:9, 13.33 x 7.5 in @ 300 DPI = 4000 x 2250 px, suitable for
dropping directly into Google Slides):

  slides/slide1_fl_events.png
      Florida-box T2m anomaly time series (Nov 2025 - Mar 2026) with the
      six detected cold-air-outbreak windows shaded and an event table
      below. Pure observational: no statistical modelling.

  slides/slide2_regression.png
      OLS multiple regression of daily FL T2m anomaly on AO, NAO, PNA,
      ONI. Three-panel layout: observed/fitted/residual time series,
      large R-squared badge, coefficient bar chart with 95% CI and
      p-values.

Run from the repository root:

  python scripts/make_presentation_slides.py

The script is fully deterministic given fixed data and has no RNG
dependence; rerun after any data update (e.g. new ERA5 day) and the
PNGs will reflect current state.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats as scistats

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indices import load_all_indices  # noqa: E402
from stats import welch_t_composite  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SLIDES = REPO / "slides"
SLIDES.mkdir(exist_ok=True)

FLORIDA_BOX = dict(lat_min=24, lat_max=31, lon_min=-87, lon_max=-80)
SE_US_BOX = dict(lat_min=25, lat_max=37, lon_min=-92, lon_max=-75)
COLD_THRESH = -2.0  # degrees C; same default as the Research-compass Q1 panel
MIN_DURATION = 3    # days

T2M_CLIMO_BASE = "2016-2024 ERA5"
Z500_CLIMO_BASE = "1994-2020 ERA5"


def _add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.7, edgecolor="black", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.6, edgecolor="black", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   linewidth=0.3, edgecolor="#666", zorder=3)


def _box_rect(ax, box, label, crs):
    import matplotlib.patches as mpatches
    rect = mpatches.Rectangle(
        (box["lon_min"], box["lat_min"]),
        box["lon_max"] - box["lon_min"],
        box["lat_max"] - box["lat_min"],
        linewidth=1.2, edgecolor="#111", facecolor="none",
        linestyle="--", transform=crs, zorder=6,
    )
    ax.add_patch(rect)
    ax.text(box["lon_min"] + 0.4, box["lat_max"] - 0.6, label,
            fontsize=7, color="#111", transform=crs,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            zorder=7)


def box_mean(da, box):
    sel = da.sel(
        latitude=slice(box["lat_max"], box["lat_min"]),
        longitude=slice(box["lon_min"], box["lon_max"]),
    )
    w = np.cos(np.deg2rad(sel.latitude))
    return sel.weighted(w).mean(dim=["latitude", "longitude"])


def align_index_to_cube(series, cube_time):
    idx_daily = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time])
    if len(series) == 0:
        return pd.Series(np.nan, index=idx_daily)
    if len(series) > 1:
        cadence_monthly = (series.index[1] - series.index[0]).days > 20
    else:
        cadence_monthly = False
    if cadence_monthly:
        rng = pd.date_range(series.index.min(),
                            idx_daily.max() + pd.Timedelta(days=32), freq="D")
        return series.reindex(rng, method="ffill").reindex(idx_daily)
    return series.reindex(idx_daily)


def detect_cold_events(series, threshold, min_duration):
    meets = (series < threshold).fillna(False).values
    events = []
    i = 0
    n = len(meets)
    while i < n:
        if meets[i]:
            j = i
            while j < n and meets[j]:
                j += 1
            dates = series.index[i:j]
            if len(dates) >= min_duration:
                period = series.loc[dates[0]:dates[-1]]
                events.append({
                    "start": dates[0],
                    "end": dates[-1],
                    "duration": len(dates),
                    "peak_date": period.idxmin(),
                    "min": float(period.min()),
                    "mean": float(period.mean()),
                })
            i = j
        else:
            i += 1
    return events


# ---------------- Matplotlib defaults (presentation-scaled) ----------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
})


def make_slide1():
    cube = xr.open_dataset(DATA / "cube_winter.nc")
    cube_time = cube.time.values
    dt_idx = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time])
    fl = box_mean(cube.t2m_anom, FLORIDA_BOX).to_series().reindex(dt_idx)
    events = detect_cold_events(fl, COLD_THRESH, MIN_DURATION)

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.4, 1.6], hspace=0.55,
                           left=0.06, right=0.97, top=0.90, bottom=0.11)
    ax = fig.add_subplot(gs[0])

    for k, e in enumerate(events, 1):
        ax.axvspan(e["start"], e["end"] + pd.Timedelta(days=1),
                   alpha=0.22, color="#4a90d9", zorder=0, linewidth=0)

    ax.plot(fl.index, fl.values, color="#111", linewidth=1.4,
            label="Florida-box mean T2m anomaly")
    ax.axhline(COLD_THRESH, color="#1f77b4", linestyle="--", linewidth=1,
               label=f"cold-event threshold: {COLD_THRESH:.0f} °C")
    ax.axhline(0, color="gray", linewidth=0.4)

    ymin, ymax = fl.min(), fl.max()
    ylabel_y = ymax + 0.12 * (ymax - ymin)
    for k, e in enumerate(events, 1):
        midpoint = e["start"] + (e["end"] - e["start"]) / 2
        ax.annotate(
            f"#{k}",
            xy=(midpoint, ylabel_y),
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#1f3e6e",
        )
    ax.set_ylim(ymin - 0.25 * (ymax - ymin),
                ymax + 0.32 * (ymax - ymin))

    ax.set_title(
        "Florida cold anomalies · winter 2025-26 "
        f"(six events ≥ {MIN_DURATION} days below {COLD_THRESH:.0f} °C)",
        loc="left",
    )
    ax.set_ylabel("2-m temperature anomaly (°C)")
    ax.set_xlabel("")
    ax.legend(loc="lower left", ncol=2)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_xlim(dt_idx.min(), dt_idx.max())

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    header = ["Event", "Start", "End", "Days",
              "Peak date", "Min °C", "Mean °C"]
    body = []
    for k, e in enumerate(events, 1):
        body.append([
            f"#{k}",
            e["start"].strftime("%Y-%m-%d"),
            e["end"].strftime("%Y-%m-%d"),
            f"{e['duration']}",
            e["peak_date"].strftime("%Y-%m-%d"),
            f"{e['min']:+.1f}",
            f"{e['mean']:+.1f}",
        ])
    total_days = sum(e["duration"] for e in events)
    body.append([
        "Σ", "—", "—", f"{total_days}",
        "—", "—", "—",
    ])
    tbl = ax2.table(cellText=body, colLabels=header,
                    cellLoc="center", loc="center",
                    colColours=["#e8eef7"] * len(header),
                    colWidths=[0.07, 0.16, 0.16, 0.09,
                               0.16, 0.10, 0.10])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
        if r == len(body):  # total row
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f5f5f5")

    total_pct = 100 * total_days / len(fl)
    cap = (
        f"Florida cold events, winter 2025-26 (Nov 1 – Mar 31, {len(fl)} days). "
        f"Detection rule: contiguous runs of ≥ {MIN_DURATION} days with the "
        f"cos-latitude-weighted Florida-box (24–31 °N, 87–80 °W) mean T2m "
        f"anomaly below {COLD_THRESH:.0f} °C. "
        f"Six events totalling {total_days} days = "
        f"{total_pct:.0f} % of the winter season. "
        f"Peak intensity: {min(e['min'] for e in events):+.1f} °C on "
        f"{min(events, key=lambda e: e['min'])['peak_date'].strftime('%-d %b %Y')}. "
        f"Data: ERA5 daily means (Hersbach et al. 2020), anomalies vs "
        f"{T2M_CLIMO_BASE} daily climatology."
    )
    fig.text(0.06, 0.015, cap, fontsize=9, color="#444",
             ha="left", va="bottom", wrap=True)

    out = SLIDES / "slide1_fl_events.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out, events, len(fl)


def make_slide2():
    cube = xr.open_dataset(DATA / "cube_winter.nc")
    cube_time = cube.time.values
    dt_idx = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time])
    fl = box_mean(cube.t2m_anom, FLORIDA_BOX).to_series().reindex(dt_idx)
    indices = load_all_indices(DATA / "indices")

    X_cols = {}
    for k in ("ao", "nao", "pna", "oni"):
        if k in indices:
            X_cols[k] = align_index_to_cube(indices[k], cube_time).values
    X_df = pd.DataFrame(X_cols, index=dt_idx)
    data = pd.concat([fl.rename("fl_t2m"), X_df], axis=1).dropna()
    y = data["fl_t2m"].values
    X = np.column_stack([np.ones(len(data)), data[list(X_cols.keys())].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, k = len(y), X.shape[1]
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot
    adj_r_sq = 1.0 - (1.0 - r_sq) * (n - 1) / (n - k)
    sigma2 = ss_res / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / se
    p_val = 2.0 * (1.0 - scistats.t.cdf(np.abs(t_stat), df=n - k))

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[2.2, 1.8], width_ratios=[2.4, 1.0],
        hspace=0.50, wspace=0.25,
        left=0.06, right=0.97, top=0.90, bottom=0.11,
    )

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.plot(data.index, y, color="#111", linewidth=1.4,
               label="Observed FL T2m anom.")
    ax_ts.plot(data.index, yhat, color="#d95f02", linewidth=1.3,
               label=f"Fitted by {' + '.join(X_cols)}")
    ax_ts.plot(data.index, resid, color="#4a90d9", linewidth=1.0,
               linestyle=":", label="Residual")
    ax_ts.axhline(0, color="gray", linewidth=0.4)
    ax_ts.set_ylabel("T2m anomaly (°C)")
    ax_ts.set_title(
        "FL T2m anomaly — observed, OLS-fitted, residual",
        loc="left",
    )
    ax_ts.legend(loc="lower right", ncol=3, fontsize=10)
    ax_ts.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax_ts.set_xlim(data.index.min(), data.index.max())

    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_r2.axis("off")
    ax_r2.text(0.5, 0.90, "Variance explained", ha="center",
               fontsize=13, color="#555", transform=ax_r2.transAxes)
    ax_r2.text(0.5, 0.68, f"R² = {r_sq:.3f}", ha="center",
               fontsize=32, fontweight="bold", color="#d95f02",
               transform=ax_r2.transAxes)
    ax_r2.text(0.5, 0.52, f"adjusted R² = {adj_r_sq:.3f}", ha="center",
               fontsize=13, color="#444", transform=ax_r2.transAxes)
    ax_r2.text(0.5, 0.40, f"n = {n} daily obs.", ha="center",
               fontsize=12, color="#666", transform=ax_r2.transAxes)
    ax_r2.text(0.5, 0.18,
               f"{(1 - r_sq) * 100:.0f} % of daily variance\n"
               f"UNEXPLAINED by these four modes",
               ha="center", fontsize=12, color="#111",
               fontweight="bold", transform=ax_r2.transAxes)
    ax_r2.add_patch(plt.Rectangle(
        (0.02, 0.05), 0.96, 0.92, transform=ax_r2.transAxes,
        facecolor="#fff6ec", edgecolor="#d95f02",
        linewidth=1.3,
    ))

    # Standardised effect sizes: expected change in FL T2m (°C) for a 1-SD
    # move in each predictor. Makes AO/NAO/PNA (standardised σ indices)
    # directly comparable with ONI (raw °C SST anomaly) because we rescale
    # each coefficient by the in-sample SD of that predictor.
    x_sd = data[list(X_cols.keys())].std().values
    effect = beta[1:] * x_sd
    effect_se = se[1:] * x_sd
    effect_ci = effect_se * float(scistats.t.ppf(0.975, df=n - k))
    ax_b = fig.add_subplot(gs[1, :])
    names = [n.upper() for n in X_cols]
    colors = ["#4a90d9" if c < 0 else "#d95f02" for c in effect]
    xs = np.arange(len(names))
    ax_b.bar(xs, effect, yerr=effect_ci, capsize=6,
             color=colors, alpha=0.85,
             edgecolor="black", linewidth=0.6)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(names, fontsize=12)
    for i, (c, p, sd_i) in enumerate(zip(effect, p_val[1:], x_sd)):
        offset = 8 if c >= 0 else -28
        ax_b.annotate(
            f"{c:+.2f} °C per 1 SD\n"
            f"(1 SD = {sd_i:.2f} of predictor)\n"
            f"p = {p:.3f}",
            xy=(i, c), xytext=(0, offset),
            textcoords="offset points", ha="center",
            fontsize=10,
            color="#111",
        )
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_ylabel("Δ FL T2m anomaly  (°C per 1 SD of predictor)")
    ax_b.set_title(
        "Standardised effect sizes, with 95 % OLS confidence intervals",
        loc="left",
    )
    ax_b.grid(axis="y", linewidth=0.3, alpha=0.5)
    y_max_abs = np.max(np.abs(effect) + effect_ci)
    ax_b.set_ylim(-y_max_abs * 1.55, y_max_abs * 1.55)

    cap = (
        "OLS multiple regression: "
        "FL_T2m_anom(t) = β₀ + β_AO·AO(t) + β_NAO·NAO(t) + β_PNA·PNA(t) "
        "+ β_ONI·ONI(t) + ε(t). "
        "Daily AO/NAO/PNA from NOAA CPC; ONI is monthly, forward-filled to "
        "daily, so its β is effectively a DJF→JFM→FMA offset. "
        "OLS p-values assume i.i.d. residuals and are anti-conservative under "
        "the real positive autocorrelation of daily T2m (lag-1 ≈ 0.6); "
        "cite as indicative. Climatology base: "
        f"{T2M_CLIMO_BASE}."
    )
    fig.text(0.06, 0.015, cap, fontsize=9, color="#444",
             ha="left", va="bottom", wrap=True)

    out = SLIDES / "slide2_regression.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out, {
        "r_sq": r_sq, "adj_r_sq": adj_r_sq, "n": n, "k": k,
        "names": names, "beta": beta[1:], "se": se[1:], "p": p_val[1:],
        "effect": effect, "effect_ci": effect_ci,
    }


def make_slide3():
    """Monthly Z500 anomaly circulation (Dec 2025, Jan 2026, Feb 2026)."""
    cube = xr.open_dataset(DATA / "cube_winter.nc")
    lats, lons = cube.latitude.values, cube.longitude.values
    months = [("Dec 2025", "2025-12"),
              ("Jan 2026", "2026-01"),
              ("Feb 2026", "2026-02")]

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.12,
                           left=0.035, right=0.97, top=0.83, bottom=0.18)
    proj = ccrs.PlateCarree()

    # Consistent color range across all three panels so they are comparable.
    z_data = [cube.z500_anom.sel(time=slug).mean(dim="time").values
              for _, slug in months]
    vmax = max(float(np.nanmax(np.abs(d))) for d in z_data) if z_data else 200
    vmax = max(50.0, min(vmax, 220.0))
    vmax = round(vmax / 20) * 20

    for i, ((label, slug), data) in enumerate(zip(months, z_data)):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        ax.set_extent([float(np.min(lons)), float(np.max(lons)),
                       float(np.min(lats)), float(np.max(lats))], crs=proj)
        im = ax.pcolormesh(
            lons, lats, data,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            shading="auto", transform=proj, rasterized=True,
        )
        cs = ax.contour(
            lons, lats, data,
            levels=np.arange(-300, 301, 60),
            colors="black", linewidths=0.5, transform=proj, zorder=4,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
        _add_map_features(ax)
        _box_rect(ax, FLORIDA_BOX, "FL", proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25,
                          color="gray", alpha=0.5, linestyle=":")
        gl.top_labels = False; gl.right_labels = False
        if i > 0:
            gl.left_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.set_title(label, fontsize=13, fontweight="bold", loc="left")
    # Shared colorbar below the row of maps.
    cbar_ax = fig.add_axes([0.15, 0.11, 0.70, 0.022])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cb.set_label("500 hPa geopotential height anomaly (m) "
                 f"vs {Z500_CLIMO_BASE} climatology",
                 fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.suptitle(
        "Monthly 500 hPa height anomalies over CONUS, winter 2025-26",
        fontsize=15, fontweight="bold", x=0.035, y=0.945, ha="left",
    )
    fig.text(
        0.035, 0.905,
        "Red = ridge (high pressure), blue = trough (low pressure). "
        "Black contours every 60 m.",
        fontsize=10, style="italic", color="#444", ha="left",
    )
    cap = (
        "Monthly means of ERA5 500 hPa geopotential height anomalies. Anomaly "
        f"defined relative to {Z500_CLIMO_BASE} daily climatology. "
        "Cos-latitude-weighted where averaged. Dashed rectangle marks the "
        "Florida analysis box (24–31 °N, 87–80 °W). March 2026 not shown: "
        "Z500 coverage ends 28 Feb 2026 in this dataset."
    )
    fig.text(0.035, 0.015, cap, fontsize=9, color="#444",
             ha="left", va="bottom", wrap=True)
    out = SLIDES / "slide3_z500_monthly.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out, {"vmax_m": vmax}


def make_slide4():
    """AO(+) − AO(−) composite of daily T2m anomaly with Welch-t stippling."""
    cube = xr.open_dataset(DATA / "cube_winter.nc")
    indices = load_all_indices(DATA / "indices")
    cube_time = cube.time.values
    lats, lons = cube.latitude.values, cube.longitude.values

    if "ao" not in indices:
        raise RuntimeError("AO index not loaded; cannot make slide 4.")
    ao = align_index_to_cube(indices["ao"], cube_time)
    ao_vals = ao.values
    valid = ~np.isnan(ao_vals)
    mask_pos = valid & (ao_vals > 0)
    mask_neg = valid & (ao_vals < 0)
    field = cube.t2m_anom.values
    comp = welch_t_composite(field, mask_pos, mask_neg, alpha=0.05)

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = gridspec.GridSpec(
        1, 1, left=0.08, right=0.92, top=0.82, bottom=0.15,
    )
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    ax.set_extent([float(np.min(lons)), float(np.max(lons)),
                   float(np.min(lats)), float(np.max(lats))], crs=proj)
    diff = comp["diff"]
    vmax = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 4.0
    vmax = round(vmax + 0.49)
    im = ax.pcolormesh(
        lons, lats, diff,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        shading="auto", transform=proj, rasterized=True,
    )
    _add_map_features(ax)
    _box_rect(ax, FLORIDA_BOX, "FL", proj)
    _box_rect(ax, SE_US_BOX, "SE-US", proj)

    # Welch-t stippling at every 3rd cell where significant at alpha=0.05.
    sig = np.asarray(comp["sig"], dtype=bool)
    if sig.any():
        lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
        sub = np.zeros_like(sig)
        sub[::3, ::3] = sig[::3, ::3]
        ax.scatter(
            lon_g[sub], lat_g[sub],
            s=2.2, c="black", marker=".", linewidths=0,
            alpha=0.75, transform=proj, zorder=5,
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.5, linestyle=":")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 9}; gl.ylabel_style = {"size": 9}

    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, shrink=0.85)
    cbar.set_label("Δ 2-m temperature anomaly (°C)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "AO negative days are systematically colder over the eastern U.S.",
        fontsize=15, fontweight="bold", x=0.08, y=0.94, ha="left",
    )
    fig.text(
        0.08, 0.895,
        f"Composite difference of T2m anomaly  ·  (AO > 0) − (AO < 0)  ·  "
        f"n(+) = {comp['n_pos']}, n(−) = {comp['n_neg']} days  ·  "
        f"stippling: two-sided Welch's t-test, α = 0.05",
        fontsize=10, style="italic", color="#444", ha="left",
    )
    cap = (
        "Per-grid-cell composite difference between daily T2m anomalies on "
        "days of positive vs negative AO phase during winter 2025–26. Blue "
        "shading means the eastern U.S. is colder on AO-negative days, the "
        "canonical cold-outbreak signature (Thompson & Wallace 1998). "
        "Stippling marks cells where the two-sided Welch unequal-variance "
        "t-test is significant at α = 0.05. No field-significance correction "
        "(Wilks 2016) applied; with ~151 days split evenly, this is a single-"
        "winter diagnostic, not a climatological estimate."
    )
    fig.text(0.08, 0.015, cap, fontsize=9, color="#444",
             ha="left", va="bottom", wrap=True)
    out = SLIDES / "slide4_ao_composite.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out, {"n_pos": comp["n_pos"], "n_neg": comp["n_neg"],
                 "sig_frac": float(sig.mean())}


def make_slide5():
    """MJO phase × lag heatmap for Florida T2m anomaly."""
    cube = xr.open_dataset(DATA / "cube_winter.nc")
    indices = load_all_indices(DATA / "indices")
    if "mjo" not in indices:
        raise RuntimeError("MJO not loaded; cannot make slide 5.")
    cube_time = cube.time.values
    dt_idx = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time])
    fl = box_mean(cube.t2m_anom, FLORIDA_BOX).to_series().reindex(dt_idx)
    mjo = indices["mjo"].reindex(
        dt_idx, method="nearest", tolerance=pd.Timedelta(days=1))
    lags = [0, 5, 10, 15]
    phases = list(range(1, 9))
    grid = np.full((len(lags), len(phases)), np.nan)
    counts = np.zeros_like(grid, dtype=int)
    for li, lag in enumerate(lags):
        mjo_lagged = mjo.shift(lag)
        active = mjo_lagged["amplitude"] >= 1.0
        for pj, ph in enumerate(phases):
            mask = (active & (mjo_lagged["phase"] == ph)).fillna(False).values
            if mask.sum() >= 3:
                grid[li, pj] = float(np.nanmean(fl.values[mask]))
                counts[li, pj] = int(mask.sum())

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = gridspec.GridSpec(
        1, 1, left=0.08, right=0.94, top=0.82, bottom=0.18,
    )
    ax = fig.add_subplot(gs[0, 0])
    vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 3.0
    vmax = max(1.5, min(round(vmax + 0.5), 6.0))
    im = ax.imshow(
        grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        aspect="auto", origin="upper",
    )
    ax.set_xticks(np.arange(len(phases)))
    ax.set_xticklabels([f"P{p}" for p in phases], fontsize=12)
    ax.set_yticks(np.arange(len(lags)))
    ax.set_yticklabels([f"lag +{l}d" for l in lags], fontsize=12)
    ax.set_xlabel("MJO phase (Wheeler & Hendon 2004 octants)", fontsize=12)

    # Cell text: mean and n.
    for li in range(len(lags)):
        for pj in range(len(phases)):
            n = counts[li, pj]
            v = grid[li, pj]
            if np.isfinite(v):
                ax.text(pj, li - 0.18, f"{v:+.1f} °C",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="black" if abs(v) < vmax * 0.6 else "white")
                ax.text(pj, li + 0.22, f"n = {n}",
                        ha="center", va="center",
                        fontsize=9,
                        color="#444" if abs(v) < vmax * 0.6 else "#eee")
            else:
                ax.text(pj, li, f"n < 3",
                        ha="center", va="center",
                        fontsize=10, color="#999", style="italic")

    # Highlight phases 7-8 (Tori's a-priori hypothesis).
    import matplotlib.patches as mpatches
    rect = mpatches.Rectangle(
        (5.5, -0.5), 2, len(lags),
        fill=False, edgecolor="#111", linewidth=1.8, zorder=10,
    )
    ax.add_patch(rect)
    ax.text(6.5, -0.85, "Tori's a-priori\nhypothesis (P7-8)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#111")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, shrink=0.9)
    cbar.set_label("Mean FL T2m anomaly on matching days (°C)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Florida T2m anomaly composited by MJO phase × lag",
        fontsize=15, fontweight="bold", x=0.08, y=0.95, ha="left",
    )
    fig.text(
        0.08, 0.905,
        "Each cell: mean FL-box T2m anomaly on days when the MJO was in "
        "that phase (amplitude ≥ 1) that many days *earlier*. "
        "n in small text = number of matching days.",
        fontsize=10, style="italic", color="#444", ha="left",
    )
    cap = (
        "ROMI (Kiladis et al. 2014; NOAA PSL) phase derived from "
        "atan2(ROMI2, ROMI1) octant mapping. Amplitude threshold ≥ 1 σ. "
        f"Total winter window: {len(dt_idx)} days. With 8 phases × 4 lags, "
        "individual cells contain 10–30 days at most; bins with n < 3 are "
        "labelled as such. No multiple-testing correction across the 32 "
        "cells — read this as exploratory diagnostic, not a formal test."
    )
    fig.text(0.08, 0.015, cap, fontsize=9, color="#444",
             ha="left", va="bottom", wrap=True)
    out = SLIDES / "slide5_mjo_phase_lag.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out, {"grid": grid, "counts": counts,
                 "lags": lags, "phases": phases}


def make_slide6():
    """MJO phase 7-8 lagged Z500 anomaly composite (4 panels: lags 0/5/10/15)."""
    cube = xr.open_dataset(DATA / "cube_winter.nc")
    indices = load_all_indices(DATA / "indices")
    if "mjo" not in indices:
        raise RuntimeError("MJO not loaded; cannot make slide 6.")
    cube_time = cube.time.values
    dt_idx = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time])
    lats, lons = cube.latitude.values, cube.longitude.values
    z500_anom = cube.z500_anom.values
    mjo = indices["mjo"].reindex(
        dt_idx, method="nearest", tolerance=pd.Timedelta(days=1))
    lags = [0, 5, 10, 15]

    # Compute composites first to share a color scale.
    panels = []
    for lag in lags:
        mjo_lagged = mjo.shift(lag)
        mask = ((mjo_lagged["phase"].isin([7, 8])) &
                (mjo_lagged["amplitude"] >= 1.0)).fillna(False).values
        n = int(mask.sum())
        if n >= 3:
            comp = np.nanmean(z500_anom[mask], axis=0)
        else:
            comp = np.full(z500_anom.shape[1:], np.nan)
        panels.append({"lag": lag, "n": n, "comp": comp})

    finite_vals = [p["comp"] for p in panels if np.isfinite(p["comp"]).any()]
    if finite_vals:
        vmax = max(float(np.nanmax(np.abs(c))) for c in finite_vals)
        vmax = max(50.0, min(round(vmax + 9, -1), 220.0))
    else:
        vmax = 120.0

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = gridspec.GridSpec(
        2, 2, left=0.04, right=0.96, top=0.84, bottom=0.14,
        hspace=0.22, wspace=0.08,
    )
    proj = ccrs.PlateCarree()
    im = None
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs[r, c], projection=proj)
        ax.set_extent([float(np.min(lons)), float(np.max(lons)),
                       float(np.min(lats)), float(np.max(lats))], crs=proj)
        if np.isfinite(panel["comp"]).any():
            im = ax.pcolormesh(
                lons, lats, panel["comp"],
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                shading="auto", transform=proj, rasterized=True,
            )
            cs = ax.contour(
                lons, lats, panel["comp"],
                levels=np.arange(-300, 301, 40),
                colors="black", linewidths=0.45,
                transform=proj, zorder=4,
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        else:
            ax.text(0.5, 0.5,
                    f"n = {panel['n']}\ninsufficient sample",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#888")
        _add_map_features(ax)
        _box_rect(ax, FLORIDA_BOX, "FL", proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25,
                          color="gray", alpha=0.5, linestyle=":")
        gl.top_labels = False; gl.right_labels = False
        if c == 1:
            gl.left_labels = False
        if r == 0:
            gl.bottom_labels = False
        gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}
        ax.set_title(f"lag +{panel['lag']} d   (n = {panel['n']} days)",
                     fontsize=12, fontweight="bold", loc="left")

    if im is not None:
        cbar_ax = fig.add_axes([0.20, 0.08, 0.60, 0.018])
        cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cb.set_label("Z500 anomaly composite on phase-7/8 days (m)",
                     fontsize=10)
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Does MJO phase 7-8 pre-load a trough over the eastern U.S.?",
        fontsize=15, fontweight="bold", x=0.04, y=0.945, ha="left",
    )
    fig.text(
        0.04, 0.905,
        "Lag composite of Z500 anomaly on days when MJO was in phases 7–8 "
        "(amp ≥ 1) that many days *earlier*. "
        "Ridge (red) upstream and trough (blue) downstream would be the "
        "classical Rossby-wave-train signature.",
        fontsize=10, style="italic", color="#444", ha="left",
    )
    cap = (
        "ROMI phase from atan2(ROMI2, ROMI1). Composite means of ERA5 daily "
        "500 hPa geopotential height anomalies (vs "
        f"{Z500_CLIMO_BASE} climatology). "
        "Only phase-7/8 days with amplitude ≥ 1 σ contribute; each panel's n "
        "is printed in its title. Contours every 40 m. Reference: "
        "Johnson et al. 2014, Mon. Wea. Rev. 142, 1556-1577."
    )
    fig.text(0.04, 0.015, cap, fontsize=9, color="#444",
             ha="left", va="bottom", wrap=True)
    out = SLIDES / "slide6_mjo_z500_lag.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out, {"panels": [(p["lag"], p["n"]) for p in panels],
                 "vmax_m": vmax}


if __name__ == "__main__":
    s1, events, n_winter = make_slide1()
    print(f"wrote {s1}  |  {len(events)} cold events over {n_winter} winter days")
    s2, reg = make_slide2()
    print(f"wrote {s2}  |  R² = {reg['r_sq']:.3f}, adj = {reg['adj_r_sq']:.3f}, n = {reg['n']}")
    for name, b, s, p in zip(reg["names"], reg["beta"], reg["se"], reg["p"]):
        print(f"    {name:>4s}  β = {b:+.3f}  SE = {s:.3f}  p = {p:.4f}")
    s3, info3 = make_slide3()
    print(f"wrote {s3}  |  Z500 panels shared vmax = ±{info3['vmax_m']:.0f} m")
    s4, info4 = make_slide4()
    print(f"wrote {s4}  |  AO+ = {info4['n_pos']}d, AO- = {info4['n_neg']}d, sig fraction = {info4['sig_frac']:.1%}")
    s5, info5 = make_slide5()
    print(f"wrote {s5}  |  phase x lag grid filled (n>=3 cells):")
    for li, lag in enumerate(info5["lags"]):
        row = "    "
        for pj, ph in enumerate(info5["phases"]):
            v = info5["grid"][li, pj]
            n = info5["counts"][li, pj]
            if np.isfinite(v):
                row += f" P{ph}:{v:+.1f}({n})"
            else:
                row += f" P{ph}:--"
        print(f"  lag+{lag:02d}d: {row}")
    s6, info6 = make_slide6()
    print(f"wrote {s6}  |  phase-7/8 lag panels (lag, n): {info6['panels']}")
