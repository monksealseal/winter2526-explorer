# Speaking script — Winter 2025-26 Florida Cold-Wave Presentation

**Total time:** ≈ 12 min spoken + 5-8 min Q&A.
**Seven slides** at roughly 1.5 min each; spend extra on Slides 2, 5, and 7.
**Delivery tip:** pause for 2 seconds after every bolded number. Let
the audience absorb it. Don't bury the punch lines.

---

## Opening (before Slide 1, ~15 seconds)

> "Good [morning/afternoon]. Our group is Group 2 Subgroup A, and our
> project looks at the anomalous winter we just experienced over the
> Southeastern U.S. My focus today is a specific question: when Florida
> had its cold outbreaks this winter, **what drove them**? Seasonal
> teleconnections, sub-seasonal MJO activity, or something else?
> Seven slides. I'll show you what we observed, what the standard
> seasonal analysis says, and where our own hypothesis ended up being
> partially wrong — honestly."

---

## Slide 1 — Florida cold events

> "This slide is pure observation, no statistical model. The black line
> is the daily Florida-box T2m anomaly: 24 to 31 north, 87 to 80 west,
> cosine-latitude weighted, anomalies against 2016-2024 ERA5
> climatology."
>
> "The dashed horizontal line at **minus 2 degrees** is our cold-event
> detection threshold. Any contiguous run of three or more days below
> that line is an event. You can see six of them this winter — numbered
> 1 through 6 at the top of the plot, details in the table below."
>
> "Two things I want you to take away. First: this was a **sequence of
> distinct cold episodes**, not one continuous cold season. You can see
> warm recovery between every pair. Second: event number 4, the first
> nine days of February, is the outlier — nine days, minimum
> **minus 10.8 degrees Celsius** on 1 February. That event alone is
> the primary dynamical target for the rest of the talk."
>
> "Total across all six events: **27 days below threshold**, about
> 18 percent of the 151-day winter."
>
> [Transition] "So we have a pattern of repeated cold outbreaks. The
> rest of the talk asks: what mechanism drives them?"

**Pause, then click to next slide.**

---

## Slide 2 — Seasonal teleconnections explain only 15%

> "The first thing any climate scientist would try is to regress
> Florida temperature on the standard seasonal teleconnection indices.
> That's this slide."
>
> "Top left: observed Florida T2m anomaly in black; the OLS fit from AO
> plus NAO plus PNA plus ONI in orange; and the residual — observed
> minus fit — as the dotted blue line. By eye you can see the orange
> fit is smooth and **misses every sharp cold excursion**. The February
> outbreak is almost entirely in the residual."
>
> "Top right is the headline number. **R squared equals 0.15.** Adjusted
> R squared is 0.124. With 120 days of complete data, the four seasonal
> modes together explain 15 percent of day-to-day Florida temperature
> variance. **85 percent is unexplained.**"
>
> "Bottom panel: standardised effect sizes — degrees Celsius change in
> FL T2m for a one-SD move in each predictor. We standardise because
> AO, NAO, and PNA are already in sigma units, but ONI is in degrees
> Celsius with an in-sample SD of only 0.13 this winter. If we don't
> standardise, ONI's raw coefficient looks huge but it's a units
> artifact."
>
> "Only **PNA is individually significant** at alpha equals 0.05 —
> beta equals minus 0.9 degrees Celsius per 1 SD of PNA, p equals
> 0.013. Negative PNA associates with Florida cold, consistent with
> the canonical teleconnection literature."
>
> "Two honest caveats. OLS p-values assume independent residuals;
> daily temperatures have lag-1 autocorrelation around 0.6, so these
> p-values are anti-conservative. And ONI is forward-filled monthly to
> daily, so its coefficient is effectively a seasonal offset, not a
> day-to-day covariate."
>
> "Bottom line: **85 percent of daily variance needs a non-seasonal
> explanation**. That's the puzzle for the rest of the talk."
>
> [Transition] "Before MJO, let's look at what the circulation itself
> was doing."

---

## Slide 3 — Monthly Z500 circulation: AO-negative pattern

> "Three panels: 500-millibar geopotential height anomalies for
> December, January, and February. March is not shown because Z500
> coverage ends 28 February 2026 in our dataset."
>
> "Red is ridge — positive anomaly, high pressure. Blue is trough —
> negative anomaly, low pressure. Black contours every 60 meters."
>
> "The signature is **consistent across all three months**: a ridge
> over the western and central U.S., a trough over the eastern U.S.
> This is the canonical Thompson-Wallace 1998 **AO-negative** pattern —
> or PNA-negative if you prefer that convention."
>
> "The mechanism is straightforward: when the eastern trough amplifies,
> the jet stream buckles southward and polar air from Canada advects
> down the eastern seaboard into the Southeast. The pattern
> intensifies from December through February, which matches exactly
> when the big cold events clustered on Slide 1."
>
> [Transition] "So the persistent circulation was AO-negative. Is that
> quantitatively correlated with FL T2m?"

---

## Slide 4 — AO composite: systematic eastern-US cold on AO-negative days

> "Composite difference map: mean T2m anomaly on AO-positive days
> minus mean T2m anomaly on AO-negative days. 35 AO-positive days
> versus 85 AO-negative days — **strongly AO-negative winter.**"
>
> "Blue — negative values — means the field is colder on AO-negative
> days relative to AO-positive days. Over the eastern U.S. and Florida
> we see broad blue shading of roughly minus 2 to minus 4 degrees."
>
> "The stippling marks cells where a two-sided Welch's unequal-
> variance t-test is significant at alpha equals 0.05. **27 percent of
> CONUS grid cells pass this threshold** — a substantial marginal
> signal for a single winter."
>
> "Two caveats. No field-significance correction is applied, so the
> stippled fraction slightly overstates the true discovery rate.
> And AO didn't emerge individually significant in Slide 2's
> regression — that's because AO and NAO share variance. Slide 4 shows
> the MARGINAL AO signal; Slide 2 showed AO's UNIQUE contribution
> after controlling for the others. Classical multicollinearity story."
>
> [Transition] "Circumstantial evidence for an AO-negative winter is
> solid. Now the sub-seasonal MJO hypothesis."

---

## Slide 5 — MJO phase × lag heatmap: hypothesis partially falsified

> "This slide is the central empirical test of Tori's original
> hypothesis. Her proposal said: MJO phases 7-8 — enhanced convection
> over the Western Hemisphere and Africa — should drive Florida cold
> at **+5 to +15 day lead** via a Rossby-wave-train pathway. Phases
> 7-8 are the black-bordered cells on the right of the heatmap."
>
> "Each cell: mean Florida T2m anomaly on days when the MJO was in
> that phase that many days earlier. Amplitude threshold: 1 sigma.
> n is printed in small text — how many matching days contributed."
>
> "Here's the finding. **The cold signal is in phases 1-2, not 7-8.**"
>
> [Pause, point at phase 2 column]
>
> "Phase 2 at lag +5 days is minus 2.0 degrees on 30 matching days.
> Phase 2 at lag +10 days is **minus 2.3 degrees on 25 days**. Phase 1
> at lag +5 days is minus 2.1 on 16 days. Those are the strongest
> cold signals in the whole grid."
>
> "Phases 7-8? Neutral to slightly warm — plus 0.2 to plus 1.5 on
> 22-28 days per cell. **Not the predicted direction.**"
>
> "Three interpretations to flag."
>
> "**First**: the macro hypothesis — MJO modulates FL cold at
> sub-seasonal lead — is supported. Phases 1-2 are active-MJO states
> and they produce a clear signal."
>
> "**Second**: phases 1-2 correspond to enhanced convection over
> Africa and the Indian Ocean — a different Rossby-wave excitation
> geometry than the phase-7/8 Pacific pathway in Johnson et al. 2014.
> So our result isn't inconsistent with MJO mechanism — it's
> inconsistent with one SPECIFIC published pathway."
>
> "**Third** caveat: 32 cells without multiple-testing correction. A
> couple of these are certainly false discoveries — phase 6 at lag
> +5 d shows plus 3.1 on only 7 days, almost certainly noise."
>
> [Transition] "If phase 7-8 doesn't drive FL cold, then the Z500
> composite for phase-7/8 days shouldn't show the classic wave train
> either. Slide 6."

---

## Slide 6 — Phase-7/8 Z500 lag composite: negative result

> "Lagged Z500 anomaly composite for the 28 days per lag when the MJO
> was in phase 7 or 8. Lag 0 at top left, through lag +15 at bottom
> right. This slide was generated BEFORE we saw Slide 5, so it tests
> the original hypothesis on its own terms."
>
> "Reading the panels: at lag 0 there's a ridge over Alaska-western
> U.S. and weakly negative anomalies over the Gulf — the kind of
> pattern we'd expect if the hypothesis held. But at lag +5, +10, and
> +15, the pattern becomes DIFFUSE. The sharp eastern-U.S. trough that
> a wave-train mechanism would predict is **weak or absent**."
>
> "Consistent with Slide 5: phase 7-8 is not systematically pre-loading
> a cold-outbreak circulation this winter."
>
> "For follow-up work, the equivalent plot for phase 1-2 would be the
> natural next step. We **did not generate it**, deliberately. The
> phase-1-2 finding was generated by this winter's data; testing it on
> that same data is textbook multiple-testing inflation. The honest
> path is to treat phase 1-2 as a **hypothesis generated by this data,
> not confirmed by it**. Replication against 2020-21, 2021-22, and
> 2022-23 ROMI data is the right test."
>
> [Transition] "Summary slide."

---

## Slide 7 — Five-point summary

> "Five points, five numbers."
>
> "**One**: six distinct Florida cold outbreaks, 27 total days below
> minus 2 degrees, peak minus 10.8 on 1 February."
>
> "**Two**: seasonal teleconnections combined explain only **15 percent**
> of daily FL T2m variance. 85 percent unexplained. Only PNA is
> individually significant."
>
> "**Three**: the circulation was **AO-negative**. The composite
> difference shows eastern-U.S. cold significant over 27 percent of
> CONUS."
>
> "**Four**: **hypothesis update**. The MJO phase window that drives
> FL cold this winter is phases 1-2 at +5 to +10 day lead, NOT phases
> 7-8 as originally hypothesised. The mechanism type — MJO-forced
> sub-seasonal teleconnection — survives; the specific phase pathway
> differs."
>
> "**Five**: bottom line — a predominantly AO-negative winter with
> sub-seasonal MJO modulation at phases 1-2. Single seasonal indices
> don't tell the story; **the story is sub-seasonal plus interaction.**"
>
> "Five numbers to remember: R-squared equals **0.15**, **85 percent**
> unexplained, **minus 10.8 degrees** on 1 February, MJO phases
> **1-2 at +10 days**, **27 percent** of CONUS significantly cold on
> AO-negative days."
>
> "Happy to take questions."

---

## Anticipated questions — answers you can lean on

**Q: "Your climatology is only 9 years. Isn't that too short?"**

> "Yes, it's shorter than WMO-standard 30-year normals. ERA5 daily
> file for this regridded CONUS domain starts in 2016. Anomalies may
> be warm-biased by about 0.2 to 0.5 degrees relative to 1991-2020
> because 2016-2024 is warmer than 1991-2020. We state this in the
> Methods & Data tab of the Streamlit app. It does not affect the
> *qualitative* story — R squared, composite differences, phase
> lag patterns — because those compare within the same climatology."

**Q: "Your OLS p-values assume independence. Daily data are
autocorrelated — are the significances real?"**

> "Honest answer: they're anti-conservative. Lag-1 autocorrelation of
> daily FL T2m is about 0.6, so the effective sample size is closer to
> 50 than 120. For publication we'd use Newey-West HAC standard errors
> or a moving-block bootstrap. The PNA p-value at 0.013 would most
> likely survive either correction — the margin would just shrink.
> The 15 percent R squared is robust."

**Q: "Slide 4 says AO-negative is cold, but Slide 2 says AO isn't
significant. Which is it?"**

> "Both. Slide 4 shows the MARGINAL AO signal — how cold the eastern
> U.S. is on AO-negative days. Slide 2 shows the UNIQUE contribution
> of AO after controlling for NAO, PNA, ONI in a multivariate
> regression. AO and NAO share about half their variance — once NAO
> is in the model, AO's unique contribution is small, which
> is why its p-value is 0.50 in the regression but the composite
> still shows a clear pattern. Classical multicollinearity. We don't
> compute VIFs in the app; we should for the final write-up."

**Q: "Your Slide 5 result contradicts Johnson et al. 2014. Are you
sure your MJO phase labels are correct?"**

> "Good question. We verified the octant mapping: atan2 of ROMI2 and
> ROMI1, mapped to eight phases via floor((angle + pi) / (pi/4)) + 1,
> where angle = -pi gives phase 1 — the southwest corner of ROMI
> space — matching Wheeler-Hendon 2004 Figure 7. That test is in the
> app's 'About & authorship' tab. Phase conventions are easy to flip
> by accident, though, so an independent sanity check on phase labels
> against a published recent-winter analysis is the right audit before
> we publish this. That's on our list."

**Q: "This is one winter. How do you generalise?"**

> "We don't — not from this data alone. This is a single-winter
> diagnostic, not a climatology. To generalise, we'd replicate the
> phase-1-2 finding in other recent La-Niña-ish winters with strong
> MJO — 2020-21, 2021-22 — and pool. If the phase-1-2 signature
> survives replication, it becomes a publishable mechanism. If it
> doesn't, we wrote a careful null result for this specific winter
> and we move on. Either outcome is useful."

**Q: "Why didn't you use an atmospheric model / AMIP ensemble?"**

> "Out of scope for a term project. The ERA5-against-ERA5-climatology
> approach is adequate for diagnosing which observed modes best track
> observed FL T2m variability within a single winter. An AMIP
> ensemble would be the right tool for causal attribution; we're
> doing diagnostic fingerprinting."

**Q: "What about stratospheric coupling? Was there a sudden
stratospheric warming?"**

> "We don't have stratospheric fields — 50 or 10 millibar Z — in our
> data cube. A polar-cap geopotential height anomaly at 10 millibars
> is the standard SSW diagnostic and would be the natural next
> addition. The persistent AO-negative this winter is itself
> suggestive of weak-vortex / downward-propagation conditions, but we
> can't test it with the current data."

---

## Timing cue sheet (printable)

| Time (min) | Slide | Key line to deliver |
|------------|-------|---------------------|
| 0.0–0.2    | Opening  | "when Florida had cold outbreaks, what drove them?" |
| 0.2–1.7    | 1 FL events  | "six distinct events, 27 days, peak −10.8 °C" |
| 1.7–4.2    | 2 Regression | "R² = 0.15. 85 percent unexplained." |
| 4.2–5.7    | 3 Z500 monthly | "ridge west, trough east — AO-negative" |
| 5.7–7.2    | 4 AO composite | "27 percent of CONUS significantly cold on AO-negative" |
| 7.2–9.7    | 5 MJO phase × lag | "phases 1-2, not 7-8 — hypothesis partially falsified" |
| 9.7–11.0   | 6 MJO Z500 lag | "consistent negative result for phase 7-8" |
| 11.0–12.0  | 7 Summary | "five numbers to remember" |
| 12.0–20.0  | Q&A | — |

---

## Closing (after Q&A, if time)

> "A one-sentence summary: the winter was **AO-negative with
> sub-seasonal MJO modulation at phases 1-2** — not phases 7-8 as we
> first thought — and our next step is to test the phase-1-2 finding
> against independent winters. Thank you."

---

## Files referenced

- Seven slide PNGs in `slides/slide1..slide7*.png`.
- Per-slide speaker notes for Slides 1-2 in `slides/slide_notes.md`.
- This script — `slides/speaking_script.md`.
- Live interactive app — [winter2526-explorer.streamlit.app](https://winter2526-explorer.streamlit.app/).
- Repository — [monksealseal/winter2526-explorer](https://github.com/monksealseal/winter2526-explorer).
