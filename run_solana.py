#!/usr/bin/env python3
"""
Solana port runner: build the panel and run the Channel-C and Channel-B designs
against whatever the frankfurt collectors have gathered so far.

    SENTRY_DATA_DIR=data SOLANA_RPC_URL=... python run_solana.py

Re-runnable at any time; every day the collectors add data the estimates sharpen.
This is a first-look harness, not the final study — read the printed caveats.
"""

from __future__ import annotations

import os

import pandas as pd

import json
import os.path

from src.estimators import by_group, channel_c, skip_hazard
from src.sol_panel import build_panel


def main() -> None:
    dd = os.getenv("SENTRY_DATA_DIR", "data")
    rpc = os.getenv("SOLANA_RPC_URL", "")
    reps = int(os.getenv("BOOTSTRAP_REPS", "0"))  # 0 = skip bootstrap for speed
    if not rpc:
        raise SystemExit("set SOLANA_RPC_URL (needed for the leader schedule)")

    print("building panel ...", flush=True)
    panel = build_panel(dd, rpc)
    try:
        panel.to_parquet(os.path.join(dd, "panel.parquet"))
    except Exception:  # pyarrow not installed on the box — pickle is dependency-free
        panel.to_pickle(os.path.join(dd, "panel.pkl"))

    span = panel["slot"].max() - panel["slot"].min()
    print(f"\n=== PANEL ===  {len(panel):,} slots, "
          f"span {span:,} slots (~{span * 0.4 / 3600:.1f}h), "
          f"{panel['leader'].nunique()} leaders")
    print(f"produced {int(panel['produced'].sum()):,} | "
          f"with tips {int((panel['tips_sol'] > 0).sum()):,} | "
          f"shred_ms median {panel['shred_ms'].median():.0f}")

    # ---- Channel C -------------------------------------------------------
    print("\n=== CHANNEL C: do later positions in a leader's window earn more tips? ===")
    pm = channel_c.position_means(panel)
    print("mean tips (SOL) by position in the 4-slot window:")
    print(pm.to_string(index=False))
    print()
    def _seal(outcome):
        try:
            return channel_c.sealing_design(panel, outcome=outcome, bootstrap_reps=reps)
        except SystemExit:
            return None
    try:
        print(f"(coarse: position)  {channel_c.position_design(panel, bootstrap_reps=reps)}")
    except SystemExit as e:
        print(f"(coarse: position) not runnable yet: {e}")
    seal_tips = _seal("tips_sol")
    seal_fees = _seal("fee_sol") if "fee_sol" in panel else None
    if seal_tips:
        print(f"(sharp: tips sealing)  {seal_tips}")
    if seal_fees:
        print(f"(sharp: fee sealing )  {seal_fees}")
    print("  → within-window slopes hold leader + window fixed; positive supports "
          "the delay-budget hypothesis.")
    print(f"(confounded)              {channel_c.cross_section(panel)}")
    print("  → the confounded comparison; gap vs the within-window slopes IS the "
          "congestion bias.")

    # ---- Channel B -------------------------------------------------------
    print("\n=== CHANNEL B: skip / dead-fork hazard (the cost that caps C) ===")
    s = skip_hazard.summary(panel)
    print(f"produced {s['produced']:,} | dead forks {s['dead_forks']} "
          f"(rate {s['dead_rate']:.4%}) | skipped {s['skipped']} "
          f"(rate {s['skip_rate']:.4%})")
    if s["underpowered"]:
        print(f"  ⚠ only {s['dead_forks']} dead-fork events — hazard curve is "
              "underpowered; let data accumulate before trusting P(dead|transit).")
    else:
        print(skip_hazard.hazard_by_bin(panel).to_string(index=False))

    # ---- Per-operator breakdown + 6x counterfactual ----------------------
    id2g = os.path.join(dd, "identity_to_group.json")
    if not os.path.exists(id2g):
        print("\n(no identity_to_group.json — run find_groups.py to enable "
              "per-operator tables)")
        return

    speedup = float(os.getenv("SPEEDUP", "6"))
    sol_price = float(os.getenv("SOL_PRICE_USD", "150"))
    span_h = (panel["slot"].max() - panel["slot"].min()) * 0.4 / 3600
    gp = by_group.attach_groups(panel, id2g)

    print("\n=== PER-OPERATOR METRICS (top by leader slots observed) ===")
    gt = by_group.group_table(gp, top=15)
    show = gt.copy()
    show["skip_rate"] = (show["skip_rate"] * 100).round(3).astype(str) + "%"
    show["tips_sol_per_slot"] = show["tips_sol_per_slot"].round(5)
    show["shred_ms_med"] = show["shred_ms_med"].round(0)
    show["seal_slope_100ms"] = show["seal_slope_100ms"].round(5)
    show["seal_t"] = show["seal_t"].round(2)
    print(show[["group", "n_val", "n_prod", "skip_rate", "shred_ms_med",
                "tips_sol_per_slot", "seal_slope_100ms", "seal_t"]].to_string(index=False))

    tip_slope = seal_tips.slope if seal_tips else 0.0
    fee_slope = seal_fees.slope if seal_fees else 0.0
    print(f"\n=== {speedup:.0f}x-FASTER-TRANSMISSION COUNTERFACTUAL, per operator ===")
    print(f"  delay price: Jito tips {tip_slope:+.5f} + fees {fee_slope:+.5f} "
          f"SOL/100ms; SOL@${sol_price:.0f}")
    print("  ⚠ MODELED counterfactual on preliminary data — order-of-magnitude, "
          "not a quote; no headroom cap yet; pooled slopes still noisy.")
    cf = by_group.counterfactual_by_group(gp, tip_slope, fee_slope, span_h,
                                          speedup, sol_price, top=15)
    disp = cf.copy()
    for c in ("uplift_sol_yr", "up_tips_sol_yr", "up_fees_sol_yr"):
        disp[c] = disp[c].round(0)
    disp["uplift_pct"] = disp["uplift_pct"].round(1).astype(str) + "%"
    for c in ("usd_day", "usd_week", "usd_month", "usd_year"):
        disp[c] = disp[c].map(lambda v: f"${v:,.0f}")
    print(disp[["group", "n_val", "transit_ms", "up_tips_sol_yr", "up_fees_sol_yr",
                "uplift_pct", "usd_day", "usd_week", "usd_month", "usd_year"]].to_string(index=False))

    # machine-readable dump for the revenue graph
    def _stat(r):
        return {"slope": r.slope, "t": r.t, "p": r.p_boot, "n": r.n,
                "clusters": r.n_clusters} if r else None
    graph = {
        "speedup": speedup, "sol_price_usd": sol_price, "span_h": span_h,
        "tip_slope_100ms": tip_slope, "fee_slope_100ms": fee_slope,
        "stats": {"tips": _stat(seal_tips), "fees": _stat(seal_fees),
                  "two_way": {"tips": channel_c.sealing_two_way(panel, "tips_sol"),
                              "fees": channel_c.sealing_two_way(panel, "fee_sol")
                              if "fee_sol" in panel else None},
                  "n_slots": int(len(panel)), "n_leaders": int(panel["leader"].nunique()),
                  "bootstrap_reps": reps},
        "network": {
            "jito_tips_sol_day": float(panel["tips_sol"].sum() / span_h * 24),
            "fees_sol_day": float(panel["fee_sol"].sum() / span_h * 24) if "fee_sol" in panel else 0.0,
        },
        "operators": cf.to_dict("records"),
    }
    with open(os.path.join(dd, "revenue_graph.json"), "w") as f:
        json.dump(graph, f, indent=2)
    print(f"\nwrote {os.path.join(dd, 'revenue_graph.json')} for the revenue graph")


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
