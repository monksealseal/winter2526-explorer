"""Winter 2025-2026 Explorer — Streamlit app.

Tabs:
  1. This Winter   — monthly mean maps (T2m anom, Z500 anom, precip)
  2. Indices       — teleconnection time series with SE-US T2m overlay
                     + correlation table with known-reference r-values
  3. Composites    — composite difference + correlation maps with lag slider
  4. Dataset       — cube schema, scientific caveats, reference comparisons

Every control reads/writes st.query_params so any view is URL-shareable.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st
import xarray as xr

from indices import load_all_indices, to_monthly

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

def effective_n(arr):
    s = np.asarray(arr); s = s[~np.isnan(s)]
    if len(s) < 10: return max(len(s), 2)
    r1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    r1 = max(-0.99, min(0.99, r1))
    return max(2, int(len(s) * (1 - r1) / (1 + r1)))

def align_index_to_cube(series, cube_time):
    idx_daily = pd.DatetimeIndex([pd.Timestamp(t).normalize() for t in cube_time])
    if len(series) == 0:
        return pd.Series(np.nan, index=idx_daily)
    cadence_monthly = (series.index[1] - series.index[0]).days > 20 if len(series) > 1 else False
    if cadence_monthly:
        rng = pd.date_range(series.index.min(), idx_daily.max() + pd.Timedelta(days=32), freq="D")
        return series.reindex(rng, method="ffill").reindex(idx_daily)
    return series.reindex(idx_daily)

def correlation_map(field, index_vals):
    mask_t = ~np.isnan(index_vals)
    f = field[mask_t]; x = index_vals[mask_t]
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

def composite_stats(field, mask_pos, mask_neg):
    n_pos = int(mask_pos.sum()); n_neg = int(mask_neg.sum())
    shape_2d = field.shape[1:]
    mean_pos = np.nanmean(field[mask_pos], axis=0) if n_pos > 0 else np.full(shape_2d, np.nan)
    mean_neg = np.nanmean(field[mask_neg], axis=0) if n_neg > 0 else np.full(shape_2d, np.nan)
    diff = mean_pos - mean_neg
    if n_pos > 1 and n_neg > 1:
        var_pos = np.nanvar(field[mask_pos], axis=0, ddof=1)
        var_neg = np.nanvar(field[mask_neg], axis=0, ddof=1)
        se = np.sqrt(var_pos / n_pos + var_neg / n_neg)
        t_stat = np.where(se > 0, diff / se, np.nan)
    else:
        t_stat = np.full(shape_2d, np.nan)
    return dict(mean_pos=mean_pos, mean_neg=mean_neg, diff=diff,
                n_pos=n_pos, n_neg=n_neg, t_stat=t_stat)

def add_sig_stipples(fig, mask_2d, lats, lons, every=3):
    if not mask_2d.any(): return
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    stipple = mask_2d & ((np.arange(mask_2d.shape[0])[:, None] % every == 0) &
                         (np.arange(mask_2d.shape[1])[None, :] % every == 0))
    if stipple.any():
        fig.add_trace(go.Scatter(
            x=lon_g[stipple], y=lat_g[stipple], mode="markers",
            marker=dict(size=3, color="black", symbol="circle-open"),
            showlegend=False, hoverinfo="skip"))

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

tab1, tab2, tab3, tab4 = st.tabs(["This Winter", "Indices", "Composites & Correlations", "Dataset Inspector"])

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

    fig = go.Figure(go.Heatmap(
        z=monthly_mean.values, x=monthly_mean.longitude.values, y=monthly_mean.latitude.values,
        colorscale=meta["cmap"], zmin=meta["vmin"], zmax=meta["vmax"],
        colorbar=dict(title=meta["label"]),
        hovertemplate="lat %{y:.2f}<br>lon %{x:.2f}<br>%{z:.2f}<extra></extra>"))
    fig.update_layout(
        title=f"{meta['label']} · {month_label} mean ({valid_days} valid days)",
        xaxis_title="Longitude (°)", yaxis_title="Latitude (°)",
        yaxis=dict(scaleanchor="x", scaleratio=1.2),
        height=540, margin=dict(l=60, r=40, t=60, b=50))
    with c2:
        st.plotly_chart(fig, use_container_width=True)

    if valid_days < 25:
        st.warning(f"⚠️ Only {valid_days} valid days in {month_label} for {field_t1}. "
                   "See Dataset Inspector for coverage.")

    if field_t1 == "z500_anom":
        st.caption(
            "**Reading Z500 anomaly maps:** red = positive anomaly (ridging/high pressure), "
            "blue = negative anomaly (troughing/low pressure). A PNA-positive pattern shows an "
            "Alaska ridge with an eastern-US trough."
        )

    c_a, c_b = st.columns(2)
    c_a.metric("SE-US box mean", f"{float(box_mean(monthly_mean, SE_US_BOX)):.2f}")
    c_b.metric("Florida box mean", f"{float(box_mean(monthly_mean, FLORIDA_BOX)):.2f}")

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
                p = 2 * (1 - stats.t.cdf(abs(t_s), df=n_eff - 2))
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

        comp = composite_stats(field, mask_pos, mask_neg)
        r_map, n_corr = correlation_map(field, idx_vals)
        n_eff = effective_n(idx_vals[valid])
        r_crit = (stats.t.ppf(0.975, df=n_eff - 2) /
                  np.sqrt(stats.t.ppf(0.975, df=n_eff - 2)**2 + n_eff - 2)) if n_eff > 2 else 1.0

        lats, lons = cube.latitude.values, cube.longitude.values
        meta = VAR_META[field_pick]

        with c_maps:
            a, b = st.columns(2)
            with a:
                st.markdown(f"**Composite difference** — (positive {INDEX_META[idx_pick]['label']} days) − (negative {INDEX_META[idx_pick]['label']} days)")
                st.caption(
                    f"threshold: {th_label} · n(+) = {comp['n_pos']} · n(−) = {comp['n_neg']} · lag = {lag:+d}d · "
                    "red = field higher on positive-index days; blue = field higher on negative-index days."
                )
                if comp["n_pos"] < 5 or comp["n_neg"] < 5:
                    st.warning(f"Sample size too small (n+={comp['n_pos']}, n-={comp['n_neg']}).")
                dmax = float(np.nanmax(np.abs(comp["diff"]))) if np.isfinite(comp["diff"]).any() else 1.0
                fig_d = go.Figure(go.Heatmap(
                    z=comp["diff"], x=lons, y=lats,
                    colorscale="RdBu_r", zmin=-dmax, zmax=dmax,
                    colorbar=dict(title=f"Δ {meta['label']}"),
                    hovertemplate="lat %{y:.2f}<br>lon %{x:.2f}<br>Δ %{z:.2f}<extra></extra>"))
                if show_sig and comp["n_pos"] > 2 and comp["n_neg"] > 2:
                    df_w = min(comp["n_pos"], comp["n_neg"]) - 1
                    t_c = stats.t.ppf(0.975, df=df_w)
                    add_sig_stipples(fig_d, np.abs(comp["t_stat"]) > t_c, lats, lons)
                fig_d.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1.2),
                                    height=450, margin=dict(l=40, r=40, t=10, b=40))
                st.plotly_chart(fig_d, use_container_width=True)

            with b:
                st.markdown(f"**Correlation map** (daily r, {INDEX_META[idx_pick]['label']} vs {meta['label']})")
                st.caption(f"n = {n_corr}d · n_eff = {n_eff} · |r| for p<0.05 ≈ {r_crit:.3f}")
                fig_r = go.Figure(go.Heatmap(
                    z=r_map, x=lons, y=lats,
                    colorscale="RdBu_r", zmin=-1, zmax=1,
                    colorbar=dict(title="r"),
                    hovertemplate="lat %{y:.2f}<br>lon %{x:.2f}<br>r %{z:.3f}<extra></extra>"))
                if show_sig:
                    add_sig_stipples(fig_r, np.abs(r_map) > r_crit, lats, lons)
                fig_r.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1.2),
                                    height=450, margin=dict(l=40, r=40, t=10, b=40))
                st.plotly_chart(fig_r, use_container_width=True)

        se_r_da = xr.DataArray(r_map, dims=("latitude", "longitude"),
                               coords=dict(latitude=lats, longitude=lons))
        st.metric("SE-US mean correlation", f"{float(box_mean(se_r_da, SE_US_BOX)):+.3f}")

        with st.expander("Absolute-phase composites (matches Juliette's notebook style)"):
            st.caption(
                f"Mean {VAR_META[field_pick]['label']} on days when "
                f"{INDEX_META[idx_pick]['label']} is in its **positive** vs **negative** phase. "
                "Use the negative-phase panel to read the pattern directly (e.g. for AO<0 "
                "cold-air outbreaks), without having to invert the difference map."
            )
            pos_lbl = f"positive {INDEX_META[idx_pick]['label']} ({comp['n_pos']} days)"
            neg_lbl = f"negative {INDEX_META[idx_pick]['label']} ({comp['n_neg']} days)"
            aa, bb = st.columns(2)
            abs_meta = VAR_META[field_pick]
            for panel, data, lbl in [(aa, comp["mean_pos"], pos_lbl),
                                      (bb, comp["mean_neg"], neg_lbl)]:
                with panel:
                    if np.isfinite(data).any():
                        vv = max(float(np.nanmax(np.abs(data))), 1e-6)
                        fig_a = go.Figure(go.Heatmap(
                            z=data, x=lons, y=lats,
                            colorscale=abs_meta["cmap"],
                            zmin=abs_meta["vmin"] if abs_meta["vmin"] is not None else -vv,
                            zmax=abs_meta["vmax"] if abs_meta["vmax"] is not None else vv,
                            colorbar=dict(title=abs_meta["label"]),
                            hovertemplate="lat %{y:.2f}<br>lon %{x:.2f}<br>%{z:.2f}<extra></extra>"))
                        fig_a.update_layout(title=f"Mean on {lbl}",
                            yaxis=dict(scaleanchor="x", scaleratio=1.2),
                            height=400, margin=dict(l=40, r=40, t=50, b=40))
                        st.plotly_chart(fig_a, use_container_width=True)

with tab4:
    qp_set(tab="dataset")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Extent")
        st.markdown(f"- **Time**: `{str(cube.time.min().values)[:10]}` → `{str(cube.time.max().values)[:10]}`")
        st.markdown(f"- **Days**: {cube.sizes['time']}")
        st.markdown(f"- **Lat**: {float(cube.latitude.min()):.2f}° to {float(cube.latitude.max()):.2f}° ({cube.sizes['latitude']} pts)")
        st.markdown(f"- **Lon**: {float(cube.longitude.min()):.2f}° to {float(cube.longitude.max()):.2f}° ({cube.sizes['longitude']} pts)")
        st.markdown(f"- **Grid**: ~0.25° (ERA5 native)")
    with c2:
        st.markdown("### Variables")
        for var in cube.data_vars:
            n_nan = int(cube[var].isnull().sum()); total = cube[var].size
            st.markdown(f"- `{var}` — {VAR_META.get(var, {}).get('label', var)}  *({100 * n_nan / total:.1f}% NaN)*")

    st.markdown("---")
    st.markdown("## 🚨 Scientific caveats — read before citing")
    st.markdown("""
- **T2m climatology = 2016-2024** (9 years). Source ERA5 T2m file starts in 2016, so we can't use WMO-standard 1991-2020. Anomalies may be cool-biased (2016-2024 warmer than 1991-2020).
- **Z500 climatology = 1994-2020** (27 years). Close to WMO-standard.
- **Precipitation coverage: Jan 1 – Mar 31 2026 only.** December 2025 precip missing. Source: CPC Global PRCP V1.0 regridded to ERA5 0.25°.
- **Z500 coverage ends Feb 28 2026.** March 2026 Z500 views show NaN.
- **Daily means** from 0Z + 12Z snapshots only, slight midday bias (~+0.5°C over land).
- **DJF** in this app = Dec+Jan+Feb. Some slides use Dec 1 – Mar 1.
- **Composite threshold default `> 0` / `< 0`** to match Juliette's notebook. ±σ option available.
- **Sample size small:** DJF ~90 days, effective N ~15-20 with lag-1 autocorrelation.
- **Does not replace expert judgment.** Confirm findings with Prof. Nolan before presenting.
""")

    st.markdown("---")
    st.markdown("## Reference r-values from the deck")
    st.dataframe(pd.DataFrame([
        {"Index": k.split("_")[0].upper(), "Reference r": f"{v['r']:+.3f}",
         "Source": v["source"], "Method": v["method"]}
        for k, v in REFERENCE_R.items()
    ]), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("## Indices loaded")
    rows = []
    for k in INDEX_META:
        base = "mjo" if k == "mjo_amp" else k
        if base in indices:
            s = indices[base]
            rows.append({"Index": INDEX_META[k]["label"],
                        "Range": f"{s.index.min().date()} → {s.index.max().date()}",
                        "N": len(s)})
        else:
            rows.append({"Index": INDEX_META[k]["label"], "Range": "NOT LOADED", "N": 0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("Raw cube repr (xarray)"):
        st.code(repr(cube), language="python")
