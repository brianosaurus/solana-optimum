"""
Channel B on Solana: the skip/fork hazard as a function of propagation time.

This is the cost side that caps Channel C. A leader that seals too late risks its
block not propagating before the network moves on — it is then abandoned as a dead
fork (SLOT_DEAD) or skipped, and the leader loses everything (priority fees + tips).
That risk is exactly what stops a rational leader from spending unlimited delay
budget, and it is why B and C are mutually-exclusive uses of the same milliseconds.

We measure the hazard the Ethereum study's way — from SEEN blocks. A SLOT_DEAD
slot DID receive shreds and form a bank before dying, so it has its own
FIRST_SHRED/COMPLETED timing; the hazard P(dead | propagation) is therefore
estimable without survivorship bias. Slots skipped outright (no block at all) are
a separate missed-proposal outcome with no propagation time and are reported as a
rate only.

With only a few hours of data, dead slots are rare — so this reports counts
honestly and flags when it is underpowered rather than manufacturing a curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hazard_by_bin(panel: pd.DataFrame, edges=(0, 200, 300, 350, 400, 600, 1200)) -> pd.DataFrame:
    """P(SLOT_DEAD | shred_ms) over produced slots, binned by propagation time."""
    d = panel[panel["produced"]].dropna(subset=["shred_ms"]).copy()
    d["bin"] = pd.cut(d["shred_ms"], bins=list(edges))
    out = (d.groupby("bin", observed=True)
           .agg(n=("dead", "size"), n_dead=("dead", "sum"))
           .reset_index())
    out["hazard"] = out["n_dead"] / out["n"]
    return out


def summary(panel: pd.DataFrame) -> dict:
    """Overall skip/dead accounting, with a power flag."""
    prod = panel["produced"]
    n_prod = int(prod.sum())
    n_dead = int(panel.loc[prod, "dead"].sum())
    # a leader-scheduled slot we saw NO block for (present in panel, not produced,
    # not dead) is skipped outright
    n_skip = int((~prod & ~panel["dead"] & panel["leader"].notna()).sum())
    n_sched = int(panel["leader"].notna().sum())
    return {
        "produced": n_prod,
        "dead_forks": n_dead,
        "skipped": n_skip,
        "scheduled_slots": n_sched,
        "dead_rate": n_dead / n_prod if n_prod else float("nan"),
        "skip_rate": n_skip / n_sched if n_sched else float("nan"),
        "underpowered": n_dead < 30,  # too few events to fit a hazard curve
    }
