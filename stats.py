"""Statistical helpers for the Winter 2025-2026 Explorer.

Implements the methods that back the "publication-quality" figures:

- ``effective_n``: lag-1-autocorrelation-adjusted sample size
  (Bretherton et al. 1999, *J. Climate* 12, 1990-2009; Bayley & Hammersley 1946).
- ``block_bootstrap_corr``: moving-block bootstrap 95% CI on Pearson r
  (Künsch 1989, *Ann. Stat.* 17, 1217-1241; Wilks 2011 §5.3.5).
- ``welch_t_composite``: Welch's unequal-variance t-test for composite
  differences at every grid cell, returning both the t statistic and a
  two-sided significance mask at α=0.05 (von Storch & Zwiers 1999 §6).
- ``corr_map_t_significance``: parametric t-test on Pearson r with the
  autocorrelation-adjusted effective df.

All functions are NaN-safe and return native Python / numpy types so they
can be cached with ``st.cache_data``.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as _stats


def effective_n(arr) -> int:
    """Effective sample size under lag-1 AR(1) autocorrelation.

    ``n_eff = n * (1 - r1) / (1 + r1)`` where ``r1`` is the lag-1
    autocorrelation of the series. Applied to a correlation or mean
    before converting to a p-value so that daily autocorrelated data
    don't inflate statistical significance.

    References
    ----------
    Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M.,
    Bladé, I. (1999), "The effective number of spatial degrees of
    freedom of a time-varying field", *J. Climate* 12, 1990-2009.
    """
    s = np.asarray(arr, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 10:
        return max(len(s), 2)
    r1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    r1 = float(np.clip(r1, -0.99, 0.99))
    return max(2, int(len(s) * (1.0 - r1) / (1.0 + r1)))


def auto_block_length(n: int, n_eff: int) -> int:
    """Suggested moving-block length ``≈ n / n_eff`` capped at ``n/3``.

    Heuristic: the block should span the autocorrelation time so that
    resampled blocks are approximately independent, while leaving at
    least three blocks so the sample has variety.
    """
    if n_eff < 1:
        return 1
    return int(max(1, min(n // 3 if n >= 9 else 1, np.ceil(n / n_eff))))


def block_bootstrap_corr(x, y, *, n_boot: int = 1000,
                         block_len: int | None = None,
                         seed: int = 42) -> dict:
    """Moving-block bootstrap 95% CI for Pearson correlation.

    Resamples contiguous blocks of length ``block_len`` from the paired
    series, recomputes r, and reports the 2.5/97.5 percentiles of the
    resampling distribution.

    Parameters
    ----------
    x, y : array-like
        Two 1-D series of equal length. NaN-paired drops.
    n_boot : int
        Number of bootstrap resamples (default 1000).
    block_len : int, optional
        Block length in samples. Defaults to ``auto_block_length``.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys ``r, ci_lo, ci_hi, n, block_len, n_boot``.

    References
    ----------
    Künsch, H. R. (1989), "The jackknife and the bootstrap for general
    stationary observations", *Ann. Stat.* 17, 1217-1241.
    Wilks, D. S. (2011), *Statistical Methods in the Atmospheric
    Sciences*, 3rd ed., §5.3.5.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    null = dict(r=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                n=n, block_len=0, n_boot=0)
    if n < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return null
    if block_len is None:
        block_len = auto_block_length(n, effective_n(x))
    block_len = max(1, min(block_len, n))
    r_obs = float(np.corrcoef(x, y)[0, 1])
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len + 1
    rs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        xb, yb = x[idx], y[idx]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            rs[b] = np.nan
        else:
            rs[b] = np.corrcoef(xb, yb)[0, 1]
    lo, hi = np.nanpercentile(rs, [2.5, 97.5])
    return dict(r=r_obs, ci_lo=float(lo), ci_hi=float(hi),
                n=n, block_len=int(block_len), n_boot=int(n_boot))


def welch_t_composite(field, mask_pos, mask_neg, *, alpha: float = 0.05) -> dict:
    """Per-grid-cell Welch's t-test on a composite difference.

    For each cell, tests H0: mean(field[pos days]) = mean(field[neg days])
    against a two-sided alternative, allowing unequal variances
    (Welch-Satterthwaite degrees of freedom).

    Parameters
    ----------
    field : ndarray, shape (T, ...)
        Time-leading 3-D+ array.
    mask_pos, mask_neg : 1-D bool arrays, length T
        Membership in each composite.
    alpha : float
        Significance level for the returned mask.

    Returns
    -------
    dict with ``diff, t, sig, n_pos, n_neg``.

    References
    ----------
    Welch, B. L. (1947), Biometrika 34, 28-35.
    von Storch, H. & Zwiers, F. W. (1999), *Statistical Analysis in
    Climate Research*, §6.
    """
    n_pos = int(mask_pos.sum())
    n_neg = int(mask_neg.sum())
    spatial = field.shape[1:]
    if n_pos < 2 or n_neg < 2:
        nan_a = np.full(spatial, np.nan)
        return dict(diff=nan_a, t=nan_a, sig=np.zeros(spatial, dtype=bool),
                    n_pos=n_pos, n_neg=n_neg)
    mean_p = np.nanmean(field[mask_pos], axis=0)
    mean_n = np.nanmean(field[mask_neg], axis=0)
    var_p = np.nanvar(field[mask_pos], axis=0, ddof=1)
    var_n = np.nanvar(field[mask_neg], axis=0, ddof=1)
    se = np.sqrt(var_p / n_pos + var_n / n_neg)
    diff = mean_p - mean_n
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(se > 0, diff / se, np.nan)
        num = (var_p / n_pos + var_n / n_neg) ** 2
        den = (var_p ** 2) / ((n_pos ** 2) * (n_pos - 1)) + \
              (var_n ** 2) / ((n_neg ** 2) * (n_neg - 1))
        df = np.where(den > 0, num / den, np.nan)
    t_crit = _stats.t.ppf(1.0 - alpha / 2.0, df=np.nan_to_num(df, nan=1.0))
    sig = np.where(np.isfinite(t_stat) & np.isfinite(df),
                   np.abs(t_stat) > t_crit, False)
    return dict(diff=diff, t=t_stat, sig=sig, n_pos=n_pos, n_neg=n_neg)


def corr_map_t_significance(r_map, n_eff: int, alpha: float = 0.05) -> np.ndarray:
    """Two-sided t-test on a correlation map with effective sample size.

    Converts each Pearson r to ``t = r * sqrt((n_eff - 2) / (1 - r^2))``
    and returns a boolean mask where ``|t|`` exceeds the critical value
    for df = ``n_eff - 2``.

    References
    ----------
    Wilks (2011) §5.2; Bretherton et al. (1999) for n_eff.
    """
    if n_eff <= 2:
        return np.zeros_like(r_map, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_crit = float(_stats.t.ppf(1.0 - alpha / 2.0, df=n_eff - 2))
        r_crit = t_crit / np.sqrt(t_crit ** 2 + n_eff - 2)
    return np.isfinite(r_map) & (np.abs(r_map) > r_crit)
