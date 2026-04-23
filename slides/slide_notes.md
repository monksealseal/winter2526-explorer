# Presentation slides — speaker notes & copy-paste text

Two slides, sized 13.33 × 7.5 in at 300 DPI so they fill a standard
Google Slides (16:9) page without scaling or blurring. Drop the PNG into
the slide, then paste the title/bullets from here into the slide's text
boxes and the speaker-notes block into the Speaker Notes pane (*View →
Show Speaker Notes* in Google Slides).

---

## Slide 1 — `slide1_fl_events.png`

**Title (top of slide):**

> Florida experienced six distinct cold outbreaks this winter

**Sub-title or kicker (one line):**

> 27 days below −2 °C anomaly · deepest: −10.8 °C on 1 Feb 2026

**Bullet list (three, for the body of the slide):**

- Event detection: contiguous runs of ≥ 3 days with Florida-box mean
  (24–31 °N, 87–80 °W, cos-latitude weighted) T2m anomaly below
  −2 °C.
- Six events totalling 27 days — about 18 % of the 151-day winter.
- Event #4 (1–9 Feb 2026, 9 days, min −10.8 °C) is the deepest and
  longest; it is the natural anchor for the dynamical diagnostics on
  the next slide.

**Speaker notes (paste into Speaker Notes pane):**

> I'm showing the box-average 2-metre temperature anomaly for
> Florida — 24 to 31 north, 87 to 80 west — plotted daily from
> November 1st through March 31st. The horizontal dashed line is the
> threshold we used to identify cold outbreaks: a daily anomaly below
> minus 2 degrees Celsius. The shaded regions are the six contiguous
> runs of at least three days that met that threshold, numbered in
> chronological order.
>
> Two things I want you to take away from this slide. First, this
> winter was not a single cold spell — it was six distinct events, with
> warm recovery between them. That pattern argues against a purely
> seasonal-mean explanation. Second, event number four — the first
> nine days of February — is exceptional: nine consecutive days of
> −5 to −11 °C anomalies, peaking at −10.8 °C on 1 February. That is
> the single most anomalous week of the winter and it will anchor our
> MJO and Z500 composites on the following slides.
>
> Methodologically: ERA5 daily means, anomalies calculated against a
> 2016–2024 climatology, cos-latitude-weighted box average. No
> smoothing, no statistical model — this is just a threshold-and-run
> detector applied to observational data, so there's nothing
> statistical to disagree about.

**Caption text already printed on the PNG (for your reference):**

> Florida cold events, winter 2025–26 (Nov 1 – Mar 31, 151 days).
> Detection rule: contiguous runs of ≥ 3 days with the
> cos-latitude-weighted Florida-box (24–31 °N, 87–80 °W) mean T2m
> anomaly below −2 °C. Six events totalling 27 days = 18 % of the
> winter season. Peak intensity: −10.8 °C on 1 Feb 2026. Data: ERA5
> daily means (Hersbach et al. 2020), anomalies vs 2016–2024 ERA5
> daily climatology.

---

## Slide 2 — `slide2_regression.png`

**Title:**

> Seasonal teleconnections explain only 15 % of day-to-day Florida cold
> variability

**Sub-title or kicker:**

> R² = 0.15 for AO + NAO + PNA + ONI → 85 % must come from elsewhere
> (MJO, stratospheric coupling, synoptic chaos)

**Bullet list (three for the body):**

- OLS multiple regression of daily FL T2m anomaly on AO, NAO, PNA, and
  ONI over the winter window (n = 120 daily observations with complete
  data on all four indices).
- Joint R² = 0.154, adjusted R² = 0.124 — four seasonal modes *together*
  account for only about one-sixth of day-to-day Florida variance.
- Per-standard-deviation effect sizes: only PNA is statistically
  distinguishable from zero (−0.90 °C per 1 SD, p = 0.013). AO, NAO,
  and ONI coefficients have confidence intervals that cross zero.

**Speaker notes (paste into Speaker Notes pane):**

> The top-left panel shows three time series overlaid: the observed
> Florida T2m anomaly in black, the OLS fit — which is what AO, NAO,
> PNA, and ONI can explain combined — in orange, and the residual
> (observed minus fit) as the dotted blue line. You can see by eye that
> the orange fit is smooth and misses almost every sharp cold
> excursion: the big February outbreak we highlighted on the previous
> slide is almost entirely in the residual, not in the fit.
>
> The top-right box is the headline number. R squared equals 0.154.
> Adjusted R squared — which penalizes for the four regressors — is
> 0.124. In words: over 120 daily observations this winter, the four
> seasonal teleconnection modes combined explain about 15 percent of
> day-to-day Florida temperature variance. 85 percent is unexplained
> by these modes. That 85 percent is what sub-seasonal mechanisms —
> principally the MJO in Tori's proposal — have to account for.
>
> The bottom panel shows the coefficient for each mode, rescaled so
> they are directly comparable. The units are degrees Celsius change
> in Florida T2m per one-standard-deviation move in each index,
> measured on this winter's own data. AO is plus 0.31, NAO is minus
> 0.18, PNA is minus 0.90, ONI is minus 0.65. Of those four, only PNA
> is statistically distinguishable from zero at the 5 percent level,
> with p equals 0.013. The other three have confidence intervals that
> cross zero — we cannot reject zero effect from daily AO, NAO, or ONI
> on Florida temperature this winter.
>
> Two important caveats that the professor will appreciate upfront.
> First, the OLS p-values assume independent residuals. Daily
> temperatures are autocorrelated at roughly lag-1 ≈ 0.6, so these
> p-values are anti-conservative. For a published paper we would use
> HAC (Newey-West) standard errors or a block bootstrap; the PNA
> significance would likely survive either, but the margin would
> shrink. Second, ONI is a monthly index forward-filled to daily,
> so its effective variability this winter is only 0.13 degrees
> Celsius. It's essentially a seasonal offset, not a day-to-day
> covariate. Its apparently-large raw coefficient in the unstandardized
> regression was misleading for exactly this reason, which is why we
> show the standardized bar chart instead.

**Caption text already printed on the PNG:**

> OLS multiple regression:
> FL_T2m_anom(t) = β₀ + β_AO·AO(t) + β_NAO·NAO(t) + β_PNA·PNA(t)
>                 + β_ONI·ONI(t) + ε(t). Daily AO/NAO/PNA from NOAA
> CPC; ONI is monthly, forward-filled to daily, so its β is
> effectively a DJF→JFM→FMA offset. OLS p-values assume i.i.d.
> residuals and are anti-conservative under the real positive
> autocorrelation of daily T2m (lag-1 ≈ 0.6); cite as indicative.
> Climatology base: 2016–2024 ERA5.

---

## Scientific self-check before presenting

Points the professor is most likely to press on, with the short honest
answer:

- **"Why only 9 years of climatology?"** — ERA5 daily-mean file for
  this domain started in 2016; 1991–2020 WMO-standard climatology
  wasn't available to us. Anomalies may be biased warm by ~0.2–0.5 °C
  (2016–2024 warmer than 1991–2020). Stated in Methods & Data tab.
- **"Are your p-values valid?"** — No, strictly. OLS p's assume
  independence; daily data is autocorrelated (lag-1 ≈ 0.6, effective
  n ≈ 50, not 120). We flag this in the caption. The qualitative
  story (R² ≈ 0.15, PNA the only standout) is robust to HAC-correction.
- **"Why not use a stationary climate-model AMIP ensemble for
  baseline?"** — Out of scope for a term project. The ERA5-against-
  ERA5-climatology approach is adequate for diagnosing which mode best
  tracks observed variability within this winter.
- **"Where is the MJO analysis?"** — In the Research compass tab of
  the live app (Q2/Q3/Q4). The two slides shown here are the
  *motivation* (this slide) and the *observational target* (Slide 1)
  for the MJO diagnostics; the composites themselves are in the next
  slides of the deck.

---

## File locations and regeneration

- Scripts: `scripts/make_presentation_slides.py`
- PNGs:    `slides/slide1_fl_events.png`, `slides/slide2_regression.png`
- Notes:   `slides/slide_notes.md` (this file)

To regenerate after a data update (new ERA5 day, new ROMI day, etc.),
simply re-run:

```
python scripts/make_presentation_slides.py
```

Both PNGs and this notes file should be version-controlled alongside
the code in `main`, so any downstream viewer can reproduce the figures
from this repository alone.
