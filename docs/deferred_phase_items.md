# Deferred phase items (parking lot)

Last updated: 2026-04-18 (end of Session 2, before Phase 5 / Explore tab).

This file records data-integration items that were scoped, agreed, and
then deferred so that Phase 5 (the 🔬 Explore tab) could land first.
Each entry has everything the next session needs to resume without
re-negotiating scope.

## Why parked

Phase 2 of Session 2 pivoted from "integrate raw data" to "build
exploration tools" after the team raised the research question *"rule
out causes for anomalies — we want to properly understand what causes
what"*. The user chose to invest Session 2's remaining budget in the
Explore tab (custom composite builder + partial attribution + sample-
size traffic lights) and to resume the D-queue next session.

D3 and D6 completed inside Session 2. The items below remain.

---

## D4a — T2m climatology rebase to WMO 1991-2020 (dual base)

**Status.** Deferred. Data is on disk (New Downloaded Files items #1
and #2 in the Session 2 inventory report). All scope decisions agreed.

**Goal.** Replace the current 9-year (2016-2024) T2m climatology with
the WMO-standard 30-year (1991-2020) mean, as an additional anomaly
variable `t2m_anom_1991_2020` alongside the existing `t2m_anom`.
Existing `t2m_anom` is preserved as `t2m_anom_2016_2024` so every
existing number (Q1-Q5, Tab 1, Tab 3) remains exactly reproducible.

**Data sources** (in `New Downloaded Files/`):
- `ERA5_2mT_1991_2015_hourly_0Z_12Z-004_` (5.2 GB, NH subset 20-70 °N,
  180 °W - 60 °E, daily 0Z/12Z, 1991-01-01 to 2015-12-31).
- `ERA5_2mT_2016_2026_hourly_0Z_12Z-005_` (2.1 GB, same spatial,
  2016-01-01 to 2026-04-10).

**Pipeline** (to be written as `scripts/d4a_rebase_t2m_climo.py`):
1. Concatenate both files along `valid_time`.
2. Subset to CONUS 22.5-50 °N, -125 to -67 °E to match cube grid.
3. Convert K → °C, daily-mean from 0Z/12Z.
4. Compute day-of-year means over 1991-01-01 → 2020-12-31.
5. Compute `t2m_anom_1991_2020` for the Nov 2025 - Mar 2026 window.
6. Write cube with both `t2m_anom_2016_2024` (= old `t2m_anom`) and
   `t2m_anom_1991_2020`.

**UI change.** Sidebar radio `1991-2020 (WMO)` (default) and
`2016-2024 (legacy)`. URL-shareable via `?climo=`.

**Size estimate.** +3-5 MB compressed.

---

## D2 — Q7 Rossby-sector Z500 cube

**Status.** Deferred. Data on disk.

**Goal.** Enable Q7 ("Rossby wave train Hovmöller") via a separate
cube `data/cube_z500_hemi.nc`: 20-70 °N, 180 °W - 60 °E, daily mean
for the winter window, with anomalies against a 1994-2020
climatology. Labelled "sector", not "circumglobal".

**Data sources.**
- `daily_z500_10012025_03312026.nc` (zip-006) for the observed winter.
- `z500_1994_2026_FIXED.nc` (zip-003) for the climatology.

**Pipeline** (`scripts/d2_hemi_z500_cube.py`):
1. Load Z500 (m), divide by G = 9.80665.
2. Compute day-of-year climatology from 1994-2020 over the sector.
3. Anomaly = observed − climo(doy).
4. Write `cube_z500_hemi.nc` (≈151 × 201 × 961).

**Size estimate.** 40-80 MB compressed. Separate file.

**UI change.** A new panel in the Research Compass answering Q7:
Hovmöller (time × longitude) of Z500 anom averaged 30-60 °N, winter
window on y-axis. Title explicitly "Pacific-Atlantic sector".

---

## D1 — Q8 250 mb jet cube (u250, v250, z250)

**Status.** Deferred. Data on disk, **Jan-Feb 2026 only**.

**Goal.** Enable Q8 ("250 mb jet") with daily u250, v250, z250 for
the active MJO phase-7/8 events this winter.

**Data source.** `era5_daily_JF_geopotential.nc` (zip-001), 7
pressure levels, 2026-01-01 → 2026-02-28, 15-80 °N / 180 °W - 0 °E.

**Pipeline** (`scripts/d1_jet_cube.py`):
1. Select `pressure_level == 250`.
2. Daily-mean from 0Z/12Z.
3. Keep full 15-80 °N / 180 °W - 0 °E.
4. Write `cube_jet.nc` (≈59 × 261 × 721).

**Size estimate.** 35-70 MB compressed.

**Caveats.** Jan-Feb 2026 only, not the full winter. No u/v
climatology available in folder, so jet anomaly comparisons are
deferred (would need historical ERA5 u/v at 250 hPa).

---

## D4b — ERA5 precip climatology + `precip_anom` (with validation gate)

**Status.** Deferred. Data on disk.

**Goal.** Add a `precip_anom` variable to `cube_winter.nc` defined
against a 1991-2020 ERA5-based precip climatology.

**Data sources.**
- `ERA5_total_prc_1991_2015_hourly_0Z_12Z-002_ (1)` (2.8 GB, 1991-2015).
- `ERA5_total_prc_2016_2026_hourly_0Z_12Z` (zip-001, 2016-2026).

**Validation gate — REQUIRED BEFORE COMMIT.** ERA5 daily mm/day vs
CPC daily mm/day over 2025-01-01 → 2026-04-10 overlap (≈15 months),
area-weighted over CONUS. Must pass BOTH:
- `R² ≥ 0.85`
- `|bias| ≤ 10 %`

If either fails, reject D4b — report numbers; do not add
`precip_anom` to the cube. The gate window is short (15 months) only
because the folder lacks CPC historical data for 1991-2020; a
stronger validation would require a separate CPC download.

**Pipeline** (`scripts/d4b_era5_precip_climo.py`):
1. Concatenate ERA5 precip files along time.
2. Convert ERA5 `tp` accumulated metres → daily mm/day.
3. Regrid to ERA5 0.25° CONUS.
4. Compute validation R² and bias on 2025-2026 overlap.
5. **If gate passes:** 1991-2020 doy climo; compute `precip_anom`;
   add to cube.
6. **If gate fails:** write skipped report; cube unchanged.

---

## D5 — Multi-level monthly Z500 climatology panel

**Status.** Deferred. Low priority (niche).

**Data sources.**
- `era5_z500_monthlyMean_1991-2020_CONUS.nc` (zip-006).
- `era5_z500_monthlyMean_winter25-26_CONUS.nc` (zip-003).

**Scope.** Add a vertical cross-section panel showing monthly Z at
13 pressure levels (1000 - 50 hPa) over CONUS against the 1991-2020
base. Only build if Tori asks for vertical structure.

---

## Meta-context for next session

- **Branch.** `claude/enhance-explorer-app-phase1-eJ9kP`; fast-forward
  merges to `main` after smoke tests pass.
- **Smoke test.** `python scripts/smoke_apptest.py` must exit with 0
  captured exceptions, 0 `st.error` calls, 1 title rendered.
- **Author tag.** Every Claude commit: `Author: Claude
  <noreply@anthropic.com>` (set via env vars for the commit).
- **Quality bar.** Cartopy PlateCarree with coastlines; moving-block
  bootstrap CIs; primary-literature references in every "About this
  analysis" expander; provenance-table row per dataset; chronology
  expander in About tab per phase.
- **Raw inputs.** `New Downloaded Files/` in the repo root, ~14 GB
  total, gitignored. All raw inputs for D1-D5 live there. If that
  folder has been cleared on the user's disk, re-fetching is out of
  scope for Claude (the sandbox is firewalled from CPC/PSL/ECMWF);
  Claude must pause and ask the user to re-download.

**Recommended next-session order** (risk-ascending as agreed):
D4a → D2 → D1 → D4b (with gate) → D5.
