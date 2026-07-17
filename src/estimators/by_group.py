"""
Per-operator breakdown: run the propagation / tip / skip / delay-budget metrics
group-by-group, so operators can be compared ("who is best at the delay game").

Joins the identity->group map from find_groups.py onto the slot panel's `leader`
(a node identity), then aggregates each operator's leader slots. The sealing-
lateness slope is the per-operator version of the Channel-C headline: within that
operator's own 4-slot windows, does sealing a block later earn more tips?
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.estimators.inference import cluster_robust_fit


def attach_groups(panel: pd.DataFrame, id2group_path: str) -> pd.DataFrame:
    with open(id2group_path, encoding="utf-8") as f:
        m = json.load(f)
    out = panel.copy()
    out["group"] = out["leader"].map(m).fillna("(unmapped)")
    return out


def _seal_slope(d: pd.DataFrame, min_n: int = 50) -> tuple[float, float, int]:
    """Within-window tips~seal_lateness slope (SOL per +100ms) for one operator."""
    dd = d[d["produced"] & d["seal_lateness_ms"].notna()].copy()
    dd = dd[dd.groupby("window_id")["seal_lateness_ms"].transform("count") >= 2]
    if len(dd) < min_n:
        return (float("nan"), float("nan"), len(dd))
    for c in ("tips_sol", "seal_lateness_ms"):
        dd[c + "_w"] = dd[c] - dd.groupby("window_id")[c].transform("mean")
    X = dd["seal_lateness_ms_w"].to_numpy().reshape(-1, 1)
    y = dd["tips_sol_w"].to_numpy()
    clusters = dd["leader"].to_numpy()
    fit = cluster_robust_fit(X, y, clusters, names=["seal"])
    # With <2 leader-clusters the CR1 sandwich is degenerate (se→0, t→inf), so
    # the t is meaningless for a single-node operator — report the slope only.
    t = fit.t_stat(0) if fit.n_clusters >= 2 else float("nan")
    return (float(fit.coef[0]) * 100.0, t, fit.n)


def group_table(panel: pd.DataFrame, top: int = 20, min_slots: int = 200) -> pd.DataFrame:
    """One row per operator: presence, propagation, tips, skips, delay-budget slope."""
    rows = []
    for grp, d in panel.groupby("group"):
        n_sched = len(d)
        n_prod = int(d["produced"].sum())
        if n_prod < min_slots:
            continue
        prod = d[d["produced"]]
        slope, t, n = _seal_slope(d)
        rows.append({
            "group": grp,
            "n_val": d["leader"].nunique(),
            "n_prod": n_prod,
            "skip_rate": (n_sched - n_prod) / n_sched,
            "dead": int(d["dead"].sum()),
            "shred_ms_med": prod["shred_ms"].median(),
            "tips_sol_per_slot": prod["tips_sol"].mean(),
            "seal_slope_100ms": slope,
            "seal_t": t,
            "seal_n": n,
        })
    tbl = pd.DataFrame(rows).sort_values("n_prod", ascending=False)
    return tbl.head(top).reset_index(drop=True)


def counterfactual_by_group(
    panel: pd.DataFrame,
    tip_slope_100ms: float,
    fee_slope_100ms: float,
    span_h: float,
    speedup: float = 6.0,
    sol_price_usd: float = 150.0,
    top: int = 20,
    min_slots: int = 200,
) -> pd.DataFrame:
    """Per-operator value of `speedup`x faster transmission, by time period.

    The delay-budget mechanism, ported from the Ethereum study, across BOTH
    revenue channels that respond to it:
      * faster transit saves Δ = transit x (1 - 1/speedup) milliseconds
      * the leader spends Δ as extra sealing delay at UNCHANGED skip risk
      * value = (dRev/dms) x Δ per leader slot, summed over
          Channel C  (Jito tips)  with pooled slope `tip_slope_100ms`
          Channel C' (leader fees) with pooled slope `fee_slope_100ms`
        each a POOLED network price of a millisecond, applied to the operator's
        OWN transit and slot rate.

    MODELED COUNTERFACTUAL on preliminary data: transmission is not actually 6x
    faster, the pooled slopes are still noisy, no deadline/headroom cap is applied
    (would lower these), and it assumes the leader re-tunes sealing to spend the
    budget. Order-of-magnitude, not a quote.
    """
    dtip = max(0.0, tip_slope_100ms / 100.0)  # SOL/ms
    dfee = max(0.0, fee_slope_100ms / 100.0)
    save_frac = 1.0 - 1.0 / speedup
    rows = []
    for grp, d in panel.groupby("group"):
        prod = d[d["produced"]]
        n_prod = len(prod)
        if n_prod < min_slots:
            continue
        transit_ms = float(prod["shred_ms"].median())
        delta_ms = transit_ms * save_frac
        slots_per_year = n_prod / span_h * 24 * 365
        up_tips = dtip * delta_ms * slots_per_year   # SOL/yr, Channel C
        up_fees = dfee * delta_ms * slots_per_year   # SOL/yr, Channel C'
        up_total = up_tips + up_fees
        tips_now = float(prod["tips_sol"].mean()) * slots_per_year
        fees_now = (float(prod["fee_sol"].mean()) if "fee_sol" in prod else 0.0) * slots_per_year
        rev_now = tips_now + fees_now
        rows.append({
            "group": grp,
            "n_val": d["leader"].nunique(),
            "transit_ms": transit_ms,
            "rev_now_sol_yr": rev_now,
            "up_tips_sol_yr": up_tips,
            "up_fees_sol_yr": up_fees,
            "uplift_sol_yr": up_total,
            "uplift_pct": (up_total / rev_now * 100) if rev_now else float("nan"),
            "usd_day": up_total * sol_price_usd / 365,
            "usd_week": up_total * sol_price_usd / 52,
            "usd_month": up_total * sol_price_usd / 12,
            "usd_year": up_total * sol_price_usd,
        })
    return pd.DataFrame(rows).sort_values("uplift_sol_yr", ascending=False).head(top).reset_index(drop=True)
