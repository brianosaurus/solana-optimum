#!/usr/bin/env python3
"""
Per-operator break detection: who changed their block propagation, and when.

Discriminating adoption from noise
----------------------------------
A cluster-specific transit step is a necessary signature of send-side mump2p
adoption — but client releases, peering upgrades and Xatu sentry churn also
move transit. Three discriminators:

  1. STAGGERING. Adoption happens per operator on different dates. A break that
     hits every cluster within a few days is a common shock (we already know
     Oct 1 is one), not adoption.
  2. DIRECTION + MAGNITUDE. Adoption should be a sustained DOWN step in the
     operator's own-block transit, not a blip (30-day guard on each side).
  3. TIMELINE FIT. Candidate dates should fall in the announced deployment arc
     (Jul-Sep 2025) to be attributable to mump2p at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def best_break(s: pd.Series, guard: int = 21):
    """Least-squares single break on a daily series (same as the event study)."""
    s = s.dropna()
    if len(s) < 2 * guard + 10:
        return None, np.nan, np.nan, 0.0
    x = s.to_numpy()
    sse0 = float(((x - x.mean()) ** 2).sum())
    best = (None, np.nan, np.nan, 0.0)
    for i in range(guard, len(x) - guard):
        a, b = x[:i], x[i:]
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        gain = 1 - sse / sse0 if sse0 > 0 else 0.0
        if gain > best[3]:
            best = (s.index[i], float(a.mean()), float(b.mean()), gain)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("data/adoption_scan.parquet"))
    ap.add_argument("--top", type=int, default=14, help="operator clusters to test")
    args = ap.parse_args()

    df = pd.read_parquet(args.panel)
    df["date"] = pd.to_datetime(df["date"])

    ops = df.groupby("operator").agg(
        blocks=("slot", "size"),
        idx_lo=("proposer_index", "min"),
        idx_hi=("proposer_index", "max"),
    ).sort_values("blocks", ascending=False)

    print("=" * 96)
    print("PER-OPERATOR ADOPTION SCAN — own-block transit, daily median per fee-recipient cluster")
    print("=" * 96)
    print(f"slot-rows {len(df):,} | window {df.date.min().date()} .. {df.date.max().date()} "
          f"| clusters tested: top {args.top} of {len(ops)}")

    print(f"\n{'operator (fee recipient)':>46} {'blocks':>7} {'index range':>19} "
          f"{'break':>11} {'before':>7} {'after':>7} {'Δ%':>7} {'r²':>6}")

    rows = []
    for op in ops.head(args.top).index:
        d = df[df.operator == op]
        daily = d.groupby("date")["transit_ms"].median()
        when, before, after, gain = best_break(daily)
        delta = (after - before) / before if before else np.nan
        rows.append({"op": op, "when": when, "delta": delta, "gain": gain})
        label = op[:44]
        print(f"{label:>46} {ops.loc[op,'blocks']:>7,} "
              f"{ops.loc[op,'idx_lo']:>8,}-{ops.loc[op,'idx_hi']:<9,} "
              f"{str(when.date()) if when is not None else 'none':>11} "
              f"{before:>7.0f} {after:>7.0f} {delta:>+6.1%} {gain:>6.3f}")

    r = pd.DataFrame(rows).dropna(subset=["when"])
    print("\n--- DISCRIMINATION ---")
    # Staggering test: spread of break dates across clusters.
    dates = pd.to_datetime(r["when"])
    span = (dates.max() - dates.min()).days
    within3d = (dates - dates.median()).abs().dt.days.le(3).mean()
    print(f"break-date span across clusters : {span} days")
    print(f"clusters breaking within ±3d of the median date: {within3d:.0%}")
    if within3d >= 0.7:
        print("  => breaks are SYNCHRONISED -> common shock (client/fork/sentry), NOT adoption")
    else:
        print("  => breaks are STAGGERED -> operator-specific changes exist; check timeline fit")

    dep = r[(r.when >= "2025-07-01") & (r.when <= "2025-09-30") & (r.delta < -0.15)]
    print(f"\nclusters with a >15% DOWN step inside the Jul-Sep deployment arc: {len(dep)}")
    for _, x in dep.iterrows():
        print(f"  {x['op'][:44]}  {x['when'].date()}  {x['delta']:+.1%}  r2={x['gain']:.3f}")
    if len(dep) == 0:
        print("  none — no operator shows an adoption-shaped propagation change in the window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
