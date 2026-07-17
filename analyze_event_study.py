#!/usr/bin/env python3
"""
Break analysis for the mump2p deployment event study (hoodi vs sepolia).

Method
------
We do not know the exact deployment date — Optimum said "summer 2025" and first
published Hoodi results on 2025-09-23. So:

1. SCAN: for every candidate break date, fit segmented means to hoodi's daily
   transit and take the date that maximises the fit improvement (classic single
   -break least squares). Report the break and the before/after levels.
2. CONTROL: run the identical scan on sepolia. A "break" that appears on both
   networks is client releases / fork upgrades / Xatu sentry changes — not
   mump2p, which only Hoodi has.
3. DIFFERENCE: hoodi-minus-sepolia daily series kills everything common to both
   (shared client stack, shared forks, shared sentry infrastructure). A mump2p
   effect must survive in the difference.
4. GUARDRAIL: n_sentries composition. A transit shift coinciding with a sentry
   count jump is an observation artifact until proven otherwise.

Outputs the table, the break candidates, and a figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["transit_med_ms", "arrival_med_ms", "arrival_p90_ms", "orphan_rate"]

# Optimum's public timeline, for annotation only — the scan is free to disagree.
ANNOUNCED = pd.Timestamp("2025-06-24")
FIRST_RESULTS = pd.Timestamp("2025-09-23")


def best_break(s: pd.Series) -> tuple[pd.Timestamp, float, float, float]:
    """Single least-squares break: date minimising SSE of a two-segment mean fit.

    Returns (break_date, mean_before, mean_after, r2_gain) where r2_gain is the
    share of variance explained by allowing the break at all. Small r2_gain =
    no meaningful break anywhere.
    """
    s = s.dropna()
    x = s.to_numpy()
    n = len(x)
    sse0 = float(((x - x.mean()) ** 2).sum())
    best = (None, np.nan, np.nan, 0.0)
    # Require 30 days on each side so a two-day blip cannot masquerade as a break.
    for i in range(30, n - 30):
        a, b = x[:i], x[i:]
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        gain = 1 - sse / sse0 if sse0 > 0 else 0.0
        if gain > best[3]:
            best = (s.index[i], float(a.mean()), float(b.mean()), gain)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("data/event_study.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.panel)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    print("=" * 78)
    print("MUMP2P DEPLOYMENT EVENT STUDY — hoodi (treated) vs sepolia (control)")
    print("=" * 78)
    print(f"window: {df.index.min().date()} .. {df.index.max().date()}")
    for net in ("hoodi", "sepolia"):
        d = df[df.network == net]
        print(f"  {net:>8}: {len(d)} days, sentries {d.n_sentries.min()}-{d.n_sentries.max()}")

    print(f"\nannounced {ANNOUNCED.date()} | first Hoodi results {FIRST_RESULTS.date()}")

    for metric in METRICS:
        print(f"\n--- {metric} ---")
        piv = df.pivot_table(index=df.index, columns="network", values=metric)
        for net in ("hoodi", "sepolia"):
            when, before, after, gain = best_break(piv[net])
            direction = "DOWN" if after < before else "UP"
            print(f"  {net:>8}: break {when.date() if when else 'none'}  "
                  f"{before:8.2f} -> {after:8.2f} ({direction}, "
                  f"{(after - before) / before:+.1%})   r2_gain={gain:.3f}")
        # The difference series: what survives after removing common shocks.
        diff = (piv["hoodi"] - piv["sepolia"]).dropna()
        when, before, after, gain = best_break(diff)
        print(f"  {'H-S diff':>8}: break {when.date() if when else 'none'}  "
              f"{before:8.2f} -> {after:8.2f}   r2_gain={gain:.3f}")

    # Monthly summary for eyeballing.
    print("\n--- monthly means (transit_med_ms) ---")
    m = (df.groupby([pd.Grouper(freq="MS"), "network"])["transit_med_ms"]
           .mean().unstack())
    m["hoodi - sepolia"] = m["hoodi"] - m["sepolia"]
    print(m.round(1).to_string())

    # Sentry composition guardrail.
    print("\n--- monthly sentry counts (composition check) ---")
    sc = (df.groupby([pd.Grouper(freq="MS"), "network"])["n_sentries"]
            .mean().unstack())
    print(sc.round(1).to_string())

    # Figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        piv = df.pivot_table(index=df.index, columns="network",
                             values="transit_med_ms")
        for net, c in (("hoodi", "#d95f0e"), ("sepolia", "#2c7fb8")):
            axes[0].plot(piv.index, piv[net].rolling(7).median(), label=net, color=c)
        axes[0].axvline(ANNOUNCED, ls="--", c="#666")
        axes[0].axvline(FIRST_RESULTS, ls=":", c="#666")
        axes[0].set_ylabel("transit median (ms, 7d rolling)")
        axes[0].legend()
        axes[0].set_title("mump2p deployment window: announced (--) to first results (:)")

        diff = (piv["hoodi"] - piv["sepolia"]).rolling(7).median()
        axes[1].plot(diff.index, diff, color="#333")
        axes[1].axvline(ANNOUNCED, ls="--", c="#666")
        axes[1].axvline(FIRST_RESULTS, ls=":", c="#666")
        axes[1].axhline(0, c="#ccc")
        axes[1].set_ylabel("hoodi − sepolia (ms)")

        out = Path("data/figures/event_study.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        print(f"\nfigure -> {out}")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
