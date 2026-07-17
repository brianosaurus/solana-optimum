"""
Channel C on Solana: does a leader capture more Jito tips when it seals later?

This is the heir to the Ethereum study's Channel C ("faster transport buys delay
budget; delay buys MEV"). The identifying idea is WITHIN-LEADER-WINDOW variation.

Why within-window
-----------------
A naive regression of tips on lateness is confounded: busy slots have both more
tips AND later sealing, so the cross-sectional slope is mostly congestion, not a
causal delay-budget effect (exactly the omitted-variable trap the Ethereum
STUDY_GUIDE warns about). We remove it by comparing slots of the SAME leader in
the SAME 4-slot window: leader quality, stake, connectivity and geography are all
held fixed, and network load is near-fixed across the ~1.6s window. What still
varies across positions 0..3 is how much of Jito's continuous bundle auction the
leader had seen by the time it sealed — the delay budget itself.

Two specifications, reported side by side:
  (a) POSITION design  : tips ~ pos_in_window, absorbing window fixed effects.
      Each +1 position = +400ms more auction exposure, leader held fixed.
      A positive slope is the clean signature of the hypothesis.
  (b) CROSS-SECTION     : tips ~ shred_ms with no FE. The confounded comparison,
      shown deliberately so the FE-vs-no-FE gap makes the confound visible.

Inference: cluster by LEADER (a leader's slots share its infrastructure and its
searcher connections), reusing the repo's CR1 sandwich + wild bootstrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.estimators.inference import (
    cluster_robust_fit,
    cluster_robust_fit_2way,
    wild_cluster_bootstrap,
)

SLOT_MS = 400.0  # one position step ~ one 400ms slot of extra auction exposure


@dataclass(frozen=True)
class ChannelCResult:
    spec: str
    slope: float          # SOL per unit of the running variable
    se: float
    t: float
    p_boot: float | None
    n: int
    n_clusters: int
    unit: str

    def __str__(self) -> str:
        p = f"{self.p_boot:.4f}" if self.p_boot is not None else "n/a"
        return (f"[{self.spec}] slope = {self.slope:+.6f} SOL/{self.unit} "
                f"(se {self.se:.6f}, t {self.t:+.2f}, wild-p {p}) "
                f"n={self.n} G_leaders={self.n_clusters}")


def _within(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    """Absorb `by` fixed effects by group-demeaning (the within transform)."""
    g = df.groupby(by)
    out = df.copy()
    for c in cols:
        out[c + "_w"] = df[c] - g[c].transform("mean")
    return out


def position_design(
    panel: pd.DataFrame, bootstrap_reps: int = 0, seed: int = 0
) -> ChannelCResult:
    """Spec (a): tips ~ pos_in_window with window fixed effects, leader-clustered.

    Restricted to produced slots inside windows where >=2 positions are observed
    with a single leader — the cells that actually carry within-window contrast.
    """
    d = panel[panel["produced"] & panel["leader"].notna()].copy()
    # keep windows with within-window variation and one leader
    grp = d.groupby("window_id")
    d = d[grp["pos_in_window"].transform("nunique") >= 2]
    d = d[d.groupby("window_id")["leader"].transform("nunique") == 1]
    if d.empty:
        raise SystemExit("no within-window contrast yet — collect more data")

    d = _within(d, ["tips_sol", "pos_in_window", "cu"], by="window_id")
    # design: [demeaned pos, demeaned cu]  (no intercept — absorbed by the FE)
    X = np.column_stack([d["pos_in_window_w"].to_numpy(), d["cu_w"].to_numpy()])
    y = d["tips_sol_w"].to_numpy()
    clusters = d["leader"].to_numpy()
    names = ["pos_in_window", "cu"]

    fit = cluster_robust_fit(X, y, clusters, names=names)
    p_boot = None
    if bootstrap_reps > 0 and fit.n_clusters > 1:
        _, p_boot, _ = wild_cluster_bootstrap(X, y, clusters, test_idx=0,
                                              reps=bootstrap_reps, seed=seed)
    return ChannelCResult("position+windowFE", float(fit.coef[0]), float(fit.se[0]),
                          fit.t_stat(0), p_boot, fit.n, fit.n_clusters, "position(+400ms)")


def sealing_design(
    panel: pd.DataFrame, outcome: str = "tips_sol", bootstrap_reps: int = 0, seed: int = 0
) -> ChannelCResult:
    """Sharper spec: `outcome` ~ seal_lateness_ms with window FE, leader-clustered.

    `seal_lateness_ms` is the block's completion delay beyond what its window
    position predicts — the leader's discretionary sealing delay, holding leader
    and window fixed. The most direct within-window test of "sealing later earns
    more revenue": the coefficient is SOL of `outcome` per extra ms of sealing
    delay, reported per +100ms. Run it on `tips_sol` (Channel C, Jito) or
    `fee_sol` (Channel C', priority+base fees).
    """
    d = panel[panel["produced"] & panel["leader"].notna()].dropna(
        subset=["seal_lateness_ms", outcome]).copy()
    grp = d.groupby("window_id")
    d = d[grp["seal_lateness_ms"].transform("count") >= 2]
    d = d[d.groupby("window_id")["leader"].transform("nunique") == 1]
    if d.empty:
        raise SystemExit("no within-window sealing contrast yet — collect more data")

    d = _within(d, [outcome, "seal_lateness_ms", "cu"], by="window_id")
    X = np.column_stack([d["seal_lateness_ms_w"].to_numpy(), d["cu_w"].to_numpy()])
    y = d[outcome + "_w"].to_numpy()
    clusters = d["leader"].to_numpy()

    fit = cluster_robust_fit(X, y, clusters, names=["seal_lateness_ms", "cu"])
    p_boot = None
    if bootstrap_reps > 0 and fit.n_clusters > 1:
        _, p_boot, _ = wild_cluster_bootstrap(X, y, clusters, test_idx=0,
                                              reps=bootstrap_reps, seed=seed)
    # report per +100ms of sealing delay
    return ChannelCResult(f"{outcome} seal+windowFE", float(fit.coef[0]) * 100.0,
                          float(fit.se[0]) * 100.0, fit.t_stat(0), p_boot,
                          fit.n, fit.n_clusters, "+100ms sealing")


def sealing_two_way(panel: pd.DataFrame, outcome: str = "tips_sol") -> dict | None:
    """The same within-window sealing design, clustered TWO ways: leader x hour.

    Leader-only clustering assumes independence across leaders, which congestion
    violates — a busy minute hits everyone at once. This re-runs the identical
    design with the Cameron-Gelbach-Miller two-way sandwich and reports how far
    the standard errors move, so the claim can be checked rather than asserted.

    Hour buckets come from the slot number (9000 slots ~ 1 hour at 400ms), so no
    clock is involved. Note the honest limit: a short panel yields few hour
    clusters, and CGM leans on BOTH dimensions having enough of them.
    """
    d = panel[panel["produced"] & panel["leader"].notna()].dropna(
        subset=["seal_lateness_ms", outcome]).copy()
    grp = d.groupby("window_id")
    d = d[grp["seal_lateness_ms"].transform("count") >= 2]
    d = d[d.groupby("window_id")["leader"].transform("nunique") == 1]
    if d.empty:
        return None

    d = _within(d, [outcome, "seal_lateness_ms", "cu"], by="window_id")
    X = np.column_stack([d["seal_lateness_ms_w"].to_numpy(), d["cu_w"].to_numpy()])
    y = d[outcome + "_w"].to_numpy()
    leader = d["leader"].to_numpy()
    hour = (d["slot"] // 9000).astype(str).to_numpy()

    one = cluster_robust_fit(X, y, leader, names=["seal", "cu"])
    two = cluster_robust_fit_2way(X, y, leader, hour, names=["seal", "cu"])
    t = two.t_stat(0)
    p = math.erfc(abs(t) / math.sqrt(2))  # two-sided normal
    return {
        "slope": float(two.coef[0]) * 100.0,
        "t": float(t),
        "p": float(p),
        "se_inflation": float(two.se[0] / one.se[0]) if one.se[0] else None,
        "n_hours": int(len(set(hour))),
        "n_leaders": int(one.n_clusters),
    }


def cross_section(panel: pd.DataFrame) -> ChannelCResult:
    """Spec (b): the confounded tips ~ shred_ms, no FE. Shown for contrast."""
    d = panel[panel["produced"] & panel["leader"].notna()].dropna(subset=["shred_ms"]).copy()
    X = np.column_stack([np.ones(len(d)), d["shred_ms"].to_numpy(), d["cu"].to_numpy()])
    y = d["tips_sol"].to_numpy()
    fit = cluster_robust_fit(X, y, d["leader"].to_numpy(),
                             names=["intercept", "shred_ms", "cu"])
    return ChannelCResult("cross-section (confounded)", float(fit.coef[1]) * SLOT_MS,
                          float(fit.se[1]) * SLOT_MS, fit.t_stat(1), None,
                          fit.n, fit.n_clusters, "position(+400ms)")


def position_means(panel: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: mean tips by position, within-window-demeaned and raw.

    The demeaned column is the honest one — it strips the level differences
    between windows and shows the pure position gradient.
    """
    d = panel[panel["produced"] & panel["leader"].notna()].copy()
    d = _within(d, ["tips_sol"], by="window_id")
    return (d.groupby("pos_in_window")
            .agg(n=("tips_sol", "size"),
                 tips_raw=("tips_sol", "mean"),
                 tips_within=("tips_sol_w", "mean"))
            .reset_index())
