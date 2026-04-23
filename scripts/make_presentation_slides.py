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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indices import load_all_indices  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SLIDES = REPO / "slides"
SLIDES.mkdir(exist_ok=True)

FLORIDA_BOX = dict(lat_min=24, lat_max=31, lon_min=-87, lon_max=-80)
COLD_THRESH = -2.0  # degrees C; same default as the Research-compass Q1 panel
MIN_DURATION = 3    # days

T2M_CLIMO_BASE = "2016-2024 ERA5"


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


if __name__ == "__main__":
    s1, events, n_winter = make_slide1()
    print(f"wrote {s1}  |  {len(events)} cold events over {n_winter} winter days")
    s2, reg = make_slide2()
    print(f"wrote {s2}  |  R² = {reg['r_sq']:.3f}, adj = {reg['adj_r_sq']:.3f}, n = {reg['n']}")
    for name, b, s, p in zip(reg["names"], reg["beta"], reg["se"], reg["p"]):
        print(f"    {name:>4s}  β = {b:+.3f}  SE = {s:.3f}  p = {p:.4f}")
