# Winter 2025-2026 Explorer

Interactive app for the UM RSMAS General Circulation term project. Explore teleconnection indices, monthly anomaly maps, and composite/correlation analyses for Winter 2025-2026 over CONUS.

## Data sources (all real)

- ERA5 2m temperature (2016–2026, 0.25°) — ECMWF
- ERA5 500 mb geopotential (1994–Feb 2026, 0.25°) — ECMWF
- CPC Global Precipitation V1.0 (Jan–Mar 2026) — NOAA CPC
- Daily AO / NAO / PNA / QBO — NOAA CPC
- ONI (ENSO) — NOAA CPC
- MJO RMM — BOM (manual; see below)

## Deploy to Streamlit Cloud

1. Create a new GitHub repo: `winter2526-explorer`
2. Copy the entire contents of this folder into it (all files + `data/`)
3. `git add . && git commit -m "Initial deploy" && git push`
4. Go to https://share.streamlit.io → sign in with GitHub → **New app**
   - Repository: `<your-username>/winter2526-explorer`
   - Branch: `main`
   - Main file: `app.py`
5. Click **Deploy**. First build ~2 minutes.
6. Share the URL in the group Slack and the project tracker.

## Manual MJO fetch

BOM blocks many cloud IPs (HTTP 403). If `data/indices/mjo_rmm.txt` is missing:

1. In a browser, open: http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
2. Save as plain text: `data/indices/mjo_rmm.txt`
3. Commit and push.

Without the file, MJO simply does not appear in the dropdowns. Everything else works.

## Shareable URLs

Every control writes to the URL. Example:
`https://<your-app>.streamlit.app/?tab=composites&cidx=ao&cfield=t2m_anom&lag=10&sig=1`

Paste in Slack → click → exact same view loads.

## Scientific caveats

See the **Dataset Inspector** tab. Key ones:
- T2m climatology is 2016-2024 (9 years), not WMO-standard 30-year
- Z500 climatology is 1994-2020 (27 years), close to WMO-standard
- Precipitation covers Jan-Mar 2026 only
- Z500 ends Feb 28, 2026
- Daily means from 0Z+12Z only (slight midday bias)
- Small effective N (~15-20) due to autocorrelation; use n_eff-adjusted p-values
- Confirm findings with Prof. Nolan before citing in the presentation

## Adding variables

1. Edit `preprocess.py` to read the new field
2. Run: `python preprocess.py --gencirc /content/drive/MyDrive/gencirc --out ./data`
3. Add an entry in `VAR_META` in `app.py`
4. `git add . && git commit && git push` — Streamlit Cloud redeploys in ~2 min
