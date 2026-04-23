"""Cartopy-based publication figure factory for the Winter 2025-2026 Explorer.

All map-making code goes through ``make_map``. It returns a matplotlib
Figure that Streamlit renders via ``st.pyplot``. The figure is laid out
as a journal figure:

- PlateCarree projection at the native ERA5 0.25° grid resolution.
- Coastlines, state, and national borders from Natural Earth
  (via cartopy.feature).
- Gridlines with degree labels on the bottom/left only.
- Diverging colormap centered on zero for anomaly fields.
- Colorbar with units string.
- Standalone title + italic subtitle above the axes and a caption string
  below the axes describing the method, data source, and n.
- Optional significance stippling (bool mask) and contour overlay.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def make_map(
    lats: np.ndarray,
    lons: np.ndarray,
    field: np.ndarray,
    *,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    center_on_zero: bool = False,
    title: str = "",
    subtitle: str = "",
    caption: str = "",
    units: str = "",
    stipple_mask: np.ndarray | None = None,
    contour_levels: Sequence[float] | None = None,
    highlight_boxes: Sequence[dict] | None = None,
    figsize: tuple[float, float] = (8.5, 4.8),
):
    """Render a publication-quality CONUS map and return the Figure.

    Parameters
    ----------
    lats, lons : 1-D arrays
        Grid coordinates in degrees. Latitude is assumed descending or
        ascending; ``pcolormesh`` handles both.
    field : 2-D array, shape ``(len(lats), len(lons))``
        NaN cells are rendered transparent.
    cmap : str
        Matplotlib colormap name. Diverging maps (RdBu_r) require
        ``center_on_zero=True`` to be symmetric.
    vmin, vmax : float, optional
        Colormap limits. Ignored when ``center_on_zero=True``.
    center_on_zero : bool
        If True, set limits symmetric about zero using
        ``max(|vmin|, |vmax|, max(|field|))``.
    title, subtitle, caption : str
        Text blocks above (title+subtitle) and below (caption) the axes.
    units : str
        Colorbar label.
    stipple_mask : 2-D bool array, optional
        Marker stippling drawn every third grid point where True.
    contour_levels : sequence, optional
        Black contour lines overlaid on the pcolormesh.
    highlight_boxes : list of dict, optional
        Each ``{"lat_min", "lat_max", "lon_min", "lon_max", "label"}``
        draws a rectangle for a named analysis region (e.g. SE-US).
    figsize : (W, H) inches
        At 300 DPI these are journal single-column proportions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=proj))

    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    if center_on_zero:
        # If the caller passes vmin/vmax, honor them strictly (saturate any
        # data outside the range) so multi-panel figures share an identical
        # color scale. Only auto-scale to the data when neither bound is
        # given.
        if vmin is not None or vmax is not None:
            candidate = max(abs(vmin) if vmin is not None else 0.0,
                            abs(vmax) if vmax is not None else 0.0)
        elif np.isfinite(field).any():
            candidate = float(np.nanmax(np.abs(field)))
        else:
            candidate = 1.0
        vmin, vmax = -candidate, candidate

    im = ax.pcolormesh(
        lons, lats, field,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", transform=proj,
        rasterized=True,
    )

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="none", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="black", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, edgecolor="black", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="#666", zorder=3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    if contour_levels is not None and np.isfinite(field).any():
        cs = ax.contour(
            lons, lats, field,
            levels=contour_levels, colors="black",
            linewidths=0.5, transform=proj, zorder=4,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%g")

    if stipple_mask is not None and np.asarray(stipple_mask).any():
        lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
        stride = 3
        sub = np.zeros_like(stipple_mask, dtype=bool)
        sub[::stride, ::stride] = stipple_mask[::stride, ::stride]
        ax.scatter(
            lon_g[sub], lat_g[sub],
            s=2.5, c="black", marker=".", linewidths=0,
            alpha=0.7, transform=proj, zorder=5,
        )

    if highlight_boxes:
        for box in highlight_boxes:
            rect = mpatches.Rectangle(
                (box["lon_min"], box["lat_min"]),
                box["lon_max"] - box["lon_min"],
                box["lat_max"] - box["lat_min"],
                linewidth=1.2, edgecolor="#111", facecolor="none",
                linestyle="--", transform=proj, zorder=6,
            )
            ax.add_patch(rect)
            if box.get("label"):
                ax.text(
                    box["lon_min"] + 0.4, box["lat_max"] - 0.6, box["label"],
                    fontsize=7, color="#111", transform=proj,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                    zorder=7,
                )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    if units:
        cbar.set_label(units, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=12)
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes,
            fontsize=8, style="italic", color="#444",
        )
    if caption:
        fig.text(
            0.02, 0.00, caption, fontsize=7, color="#333",
            ha="left", va="bottom", wrap=True,
        )
        fig.subplots_adjust(bottom=0.14)

    fig.tight_layout()
    return fig
