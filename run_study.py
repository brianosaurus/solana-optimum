#!/usr/bin/env python3
"""
Block-propagation latency and validator performance on Ethereum mainnet.

Run:
    python run_study.py --start 2025-04-01 --end 2025-06-30

Pipeline
--------
1. STREAM the panel. For each day: download the Xatu parquet, aggregate it down
   to one row per slot, delete the raw file, move on. Peak disk stays ~2GB even
   though the full study touches ~140GB of raw data. Without streaming, a 90-day
   panel would not fit on the deployment host's disk at all.

2. ESTIMATE.
     a. Binned dose-response      -> the descriptive curve.
     b. Sharp RD at the 4s deadline -> the causal estimate.
     c. Placebos + density test   -> is the RD believable?

3. PRICE. Convert the effect into ETH/validator/year.

Everything is read-only. This program has no keys and sends no transactions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from src.estimators.rdd import density_test, rd_estimate
from src.counterfactual import (
    per_validator_table,
    private_total,
    profit_grid,
    run_counterfactual,
)
from src.mev import MevParams, measure_mev_curve
from src.operators import COMBINED_VALIDATORS, OPERATORS
from src.panel import build_slot_panel
from src.plots import dose_response_plot, rd_plot
from src.revenue import RevenueModel
from src.xatu import (
    ATTESTATION_DEADLINE_MS,
    T_CANON_BLOCK,
    T_COMMITTEE,
    T_ELAB_ATT,
    T_EVENTS_BLOCK,
    T_PROPOSER_DUTY,
    XatuPaths,
    connect,
    daterange,
)

# Tables we must pull per day. The attestation table also needs day D+1, because
# an attestation for a slot late on day D can be included on day D+1 — see the
# day-boundary discussion in src/panel.py.
TABLES = [
    T_CANON_BLOCK,
    T_EVENTS_BLOCK,
    T_PROPOSER_DUTY,
    T_ELAB_ATT,
    T_COMMITTEE,
]


def fetch_day(paths: XatuPaths, table: str, date: dt.date, raw_dir: Path) -> Path | None:
    """Download one table-day. Returns None if Xatu has no file for it.

    A missing day is normal (Xatu has gaps); it must not abort the study, but it
    MUST be reported — silently skipping days is how a panel quietly loses a
    month and nobody notices.
    """
    dest = raw_dir / f"{table}_{date.isoformat()}.parquet"
    if dest.exists():
        return dest

    url = paths.day(table, date)
    r = subprocess.run(
        ["curl", "-sS", "--fail", "--retry", "3", "-o", str(dest), url],
        capture_output=True,
    )
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return None
    return dest


def build_panel(cfg, paths: XatuPaths) -> pd.DataFrame:
    """Stream the slot panel day by day, keeping peak disk small."""
    con = connect(cfg)
    frames: list[pd.DataFrame] = []
    missing: list[dt.date] = []

    for date in daterange(cfg.start_date, cfg.end_date):
        nxt = date + dt.timedelta(days=1)

        # Pull day D for everything, plus day D+1 for attestations (lookahead).
        got = {t: fetch_day(paths, t, date, cfg.raw_dir) for t in TABLES}
        got_next = fetch_day(paths, T_ELAB_ATT, nxt, cfg.raw_dir)

        if any(v is None for v in got.values()) or got_next is None:
            missing.append(date)
            print(f"  {date}  SKIPPED (missing Xatu file)", file=sys.stderr)
            for p in list(got.values()) + [got_next]:
                if p:
                    p.unlink(missing_ok=True)
            continue

        build_slot_panel(con, paths, date, cfg.min_sentries, local_dir=cfg.raw_dir)
        day_df = con.execute("SELECT * FROM slot_panel").df()
        day_df["date"] = date
        frames.append(day_df)

        # Delete the raw files immediately. This is what keeps peak disk ~2GB.
        for p in got.values():
            p.unlink(missing_ok=True)
        got_next.unlink(missing_ok=True)

        print(f"  {date}  {len(day_df):>5} slots", file=sys.stderr)

    if missing:
        # Loud, not silent. A reader must be able to see exactly what was dropped.
        print(f"\n!! {len(missing)} day(s) missing from Xatu: {missing}", file=sys.stderr)

    if not frames:
        raise SystemExit("No data retrieved — check the date range and Xatu availability.")

    df = pd.concat(frames, ignore_index=True)
    validate_panel(df)
    return df


def validate_panel(df: pd.DataFrame) -> None:
    """Fail loudly on impossible values.

    These are not paranoia. An earlier version summed `len(validators)` across
    raw attestation rows, which double-counted validators whose attestation was
    included in more than one block (overlapping aggregates; competing forks).
    That produced attester counts ABOVE the committee size and hence a NEGATIVE
    missed-attestation rate — a number that is not merely wrong but impossible,
    and which sailed straight through into a results table.

    A rate outside [0, 1] is a logic error, not a data quirk. Crash on it.
    """
    problems: list[str] = []

    for col in ("correct_head_rate", "missed_attestation_rate"):
        s = df[col].dropna()
        if len(s) and (s.min() < -1e-9 or s.max() > 1 + 1e-9):
            problems.append(
                f"{col} outside [0,1]: min={s.min():.4f} max={s.max():.4f}"
            )

    over = df.dropna(subset=["n_attested", "committee_size"])
    n_over = int((over["n_attested"] > over["committee_size"] + 1).sum())
    if n_over:
        problems.append(
            f"{n_over} slots have more attesters than committee members "
            "(attestation de-duplication is broken)"
        )

    d = df["mean_inclusion_distance"].dropna()
    if len(d) and (d.min() < 1 - 1e-9):
        problems.append(
            f"inclusion distance below 1 ({d.min():.3f}) — an attestation cannot "
            "be included in a block before the slot it attests to"
        )

    if problems:
        raise SystemExit(
            "PANEL VALIDATION FAILED — refusing to report results:\n  - "
            + "\n  - ".join(problems)
        )


def dose_response(df: pd.DataFrame) -> pd.DataFrame:
    """The descriptive curve: correct-head rate by block arrival bucket."""
    edges = [0, 1000, 2000, 3000, 3500, 4000, 5000, 6000, 12000]
    labels = [
        "<1s", "1-2s", "2-3s", "3.0-3.5s", "3.5-4.0s",
        "4-5s (LATE)", "5-6s", ">6s",
    ]
    d = df.dropna(subset=["arrival_ms", "correct_head_rate"]).copy()
    d["bucket"] = pd.cut(d["arrival_ms"], bins=edges, labels=labels, right=False)

    out = (
        d.groupby("bucket", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "slots": len(g),
                    # Weight by committee size: a slot with 30k attesters is more
                    # informative than one with 8k. An unweighted mean-of-rates
                    # would over-weight thin slots.
                    "correct_head_pct": 100
                    * np.average(g["correct_head_rate"], weights=g["n_attested"]),
                    "mean_incl_dist": np.average(
                        g["mean_inclusion_distance"], weights=g["n_attested"]
                    ),
                    "missed_att_pct": 100 * g["missed_attestation_rate"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat)
    ap.add_argument("--end", type=dt.date.fromisoformat)
    ap.add_argument("--bandwidth", type=float, default=1500.0,
                    help="RD bandwidth in ms around the 4000ms cutoff")
    ap.add_argument("--reps", type=int, default=None,
                    help="wild cluster bootstrap replications")
    ap.add_argument("--panel", type=Path, default=None,
                    help="reuse a cached panel parquet instead of rebuilding")
    args = ap.parse_args()

    cfg = load_config()
    if args.start:
        object.__setattr__(cfg, "start_date", args.start)
    if args.end:
        object.__setattr__(cfg, "end_date", args.end)
    reps = args.reps if args.reps is not None else cfg.bootstrap_reps

    paths = XatuPaths(cfg.xatu_base_url)
    panel_path = args.panel or (cfg.cache_dir / "slot_panel.parquet")

    # ---------------- 1. Panel ----------------
    if args.panel and args.panel.exists():
        print(f"Reusing cached panel: {args.panel}", file=sys.stderr)
        df = pd.read_parquet(args.panel)
    else:
        print(
            f"Building panel {cfg.start_date} -> {cfg.end_date} "
            f"({cfg.n_days} days)…",
            file=sys.stderr,
        )
        df = build_panel(cfg, paths)
        df.to_parquet(panel_path)
        print(f"Panel cached -> {panel_path}", file=sys.stderr)

    print("\n" + "=" * 78)
    print("BLOCK PROPAGATION LATENCY AND VALIDATOR PERFORMANCE — ETHEREUM MAINNET")
    print("=" * 78)
    print(f"Window        : {df['date'].min()} .. {df['date'].max()}")
    print(f"Slots         : {len(df):,}")
    print(f"With timing   : {df['arrival_ms'].notna().sum():,}")
    print(f"Missed props  : {int(df['missed_proposal'].sum()):,} "
          f"({100 * df['missed_proposal'].mean():.3f}%)")

    # ---------------- 2a. Dose-response ----------------
    print("\n--- DOSE-RESPONSE: block arrival time vs attester outcomes ---")
    print(dose_response(df).to_string(index=False))

    # ---------------- 2b. Sharp RD at the deadline ----------------
    # Cluster by DAY. Slots within a day share network conditions, mempool
    # state, and blob demand; treating them as independent would understate the
    # SEs. Day is the coarsest sensible unit that still leaves many clusters.
    d = df.dropna(subset=["arrival_ms", "correct_head_rate"]).copy()
    clusters = pd.factorize(d["date"])[0]

    print(f"\n--- SHARP RD AT THE {ATTESTATION_DEADLINE_MS}ms ATTESTATION DEADLINE ---")
    print("Estimand: the causal effect of a block missing the deadline on the")
    print("share of (randomly assigned) attesters who vote for the correct head.\n")

    res = rd_estimate(
        d["arrival_ms"].to_numpy(),
        d["correct_head_rate"].to_numpy(),
        clusters,
        bandwidth=args.bandwidth,
        weights=d["n_attested"].to_numpy(),
        bootstrap_reps=reps,
        seed=cfg.seed,
    )
    print(f"  {res}")

    # Bandwidth robustness: the estimate must not hinge on one arbitrary window.
    print("\n  Bandwidth sensitivity:")
    for h in (500, 750, 1000, 1500, 2000):
        r = rd_estimate(
            d["arrival_ms"].to_numpy(),
            d["correct_head_rate"].to_numpy(),
            clusters,
            bandwidth=float(h),
            weights=d["n_attested"].to_numpy(),
        )
        print(f"    h={h:>5}ms   tau={r.tau:+.4f}  (se {r.se:.4f})")

    # ---------------- 2c. Validity checks ----------------
    print("\n--- RD VALIDITY ---")
    dens = density_test(d["arrival_ms"].to_numpy())
    print(f"  Density (McCrary-style) at cutoff: left={dens['left']:.0f} "
          f"right={dens['right']:.0f}  z={dens['z']:+.2f}")
    print("    (|z| > ~2 would suggest proposers manipulate arrival time around")
    print("     the deadline, which would invalidate the design.)")

    # Covariate placebos: blob count and block size must NOT jump at the cutoff.
    # If they do, the 'effect' might be theirs, not the deadline's.
    print("\n  Covariate balance (these should all be ~0):")
    for cov in ("blob_count", "tx_count", "payload_bytes"):
        if cov not in d or d[cov].isna().all():
            continue
        r = rd_estimate(
            d["arrival_ms"].to_numpy(),
            d[cov].astype(float).to_numpy(),
            clusters,
            bandwidth=args.bandwidth,
        )
        print(f"    {cov:>14}: tau={r.tau:+12.4f}  (t {r.t_stat:+.2f})")

    # Placebo cutoffs: run the same RD at fake thresholds where nothing should
    # happen. A "significant" jump at 2500ms would mean our estimator finds
    # discontinuities in noise.
    print("\n  Placebo cutoffs (no spec threshold here — should be ~0):")
    for fake in (2000, 2500, 3000, 5500):
        r = rd_estimate(
            d["arrival_ms"].to_numpy(),
            d["correct_head_rate"].to_numpy(),
            clusters,
            cutoff=float(fake),
            bandwidth=800.0,
            weights=d["n_attested"].to_numpy(),
        )
        print(f"    cutoff={fake:>5}ms   tau={r.tau:+.4f}  (t {r.t_stat:+.2f})")

    # ---------------- 3. Price it ----------------
    if cfg.network != "mainnet":
        # On a testnet every dollar figure is fiction: the ETH is free, there is
        # no MEV market (no relay bid tables), and validators bear no capital
        # cost. Printing USD against testnet physics would launder meaningless
        # numbers into a results table, so the entire pricing layer is gated.
        print("\n--- PRICING SKIPPED ---")
        print(f"  network is '{cfg.network}': testnet ETH has no price and no MEV")
        print("  market exists. Physics above is real; dollars would be fiction.")
        print("\n" + "=" * 78)
        return 0

    print("\n--- WHAT LATENCY COSTS ---")
    rev = RevenueModel()
    px = cfg.eth_price_usd
    print(f"  ETH price used : ${px:,.2f}  (set ETH_PRICE_USD in .env)")
    print(f"  Base reward    : {rev.base_reward_per_epoch_eth * 1e9:,.0f} Gwei/epoch")
    print(f"  Head component : {rev.head_reward_per_epoch_eth * 1e9:,.0f} Gwei/epoch "
          f"({14 / 54:.1%} of attestation income)")

    # Marginal slope of head-miss probability per ms, over the steep part of the
    # curve (3s -> 5s), which is where real operators actually sit. Weighted by
    # attesters so fat committees count more.
    steep = d[(d["arrival_ms"] >= 3000) & (d["arrival_ms"] <= 5000)]
    if len(steep) <= 100:
        print("\n  Too few slots in the 3-5s range to fit a slope.")
        print("\n" + "=" * 78)
        return 0

    slope = np.polyfit(
        steep["arrival_ms"], 1 - steep["correct_head_rate"], 1,
        w=steep["n_attested"],
    )[0]
    print(f"\n  Marginal head-miss probability per +1ms (3-5s): {slope:.3e}")

    # Two ways to price latency, reported side by side:
    #   MARGINAL — the slope: what the NEXT 100ms costs you.
    #   DISCRETE — the RD tau: what CROSSING the deadline costs you. This is the
    #              causal estimate, and it is much larger, because the deadline
    #              is a cliff rather than a ramp.
    scenarios = [
        ("+100ms slower (marginal)", rev.attester_cost_per_100ms(slope)),
        ("+250ms slower (marginal)", rev.attester_cost_per_100ms(slope) * 2.5),
        (
            "block crosses the 4s deadline (RD tau)",
            rev.attester_cost_of_head_misses(abs(res.tau)),
        ),
    ]

    for label, eth_yr in scenarios:
        print(f"\n  ### {label}")
        for fleet, n in (("1 validator", 1), ("10,000 validators", 10_000),
                         ("100,000 validators", 100_000)):
            p = rev.periods(eth_yr * n, px)
            print(f"    {fleet:>18} | "
                  f"day {p['day']['eth']:>10.6f} ETH  ${p['day']['usd']:>12,.2f}")
            print(f"    {'':>18} | "
                  f"week{p['week']['eth']:>10.6f} ETH  ${p['week']['usd']:>12,.2f}")
            print(f"    {'':>18} | "
                  f"mo  {p['month']['eth']:>10.6f} ETH  ${p['month']['usd']:>12,.2f}")
            print(f"    {'':>18} | "
                  f"YEAR{p['year']['eth']:>10.4f} ETH  ${p['year']['usd']:>12,.2f}")

    # The proposer side. A late block doesn't just cost attesters their head
    # vote — it can get the PROPOSER reorged, forfeiting the entire block value
    # (consensus reward + tips + MEV). This dwarfs the attester cost.
    reorg_rate = float(df["missed_proposal"].mean())
    print("\n  ### proposer-side: reorg / missed-proposal exposure")
    print(f"    observed missed-proposal rate: {reorg_rate:.4%}")
    for fleet, n in (("10,000 validators", 10_000), ("100,000 validators", 100_000)):
        props = rev.expected_proposals_per_year(n)
        eth_yr = rev.proposer_cost_of_reorgs(reorg_rate, props)
        p = rev.periods(eth_yr, px)
        print(f"    {fleet:>18} | {props:>8,.0f} blocks/yr | "
              f"YEAR {p['year']['eth']:>9.2f} ETH  ${p['year']['usd']:>12,.2f}")

    # ---------------- 3b. COUNTERFACTUAL: what does adopting Optimum earn? ----
    # Needs the MEV accrual curve. dV/dt is a slow-moving structural parameter,
    # so we measure it on a few days and cache it (each day is a ~1.8GB download).
    mev_path = cfg.cache_dir / "mev_params.json"
    if mev_path.exists():
        mp = MevParams.load(mev_path)
        print(f"\n  MEV params (cached from {len(mp.days)} day(s))")
    else:
        print("\n  Measuring MEV accrual curve (downloading bid traces)…",
              file=sys.stderr)
        sample = [cfg.start_date + dt.timedelta(days=i) for i in (0, 7, 14)]
        mp, curve = measure_mev_curve(
            cfg.xatu_base_url, sample, cfg.raw_dir,
            memory_limit=cfg.duckdb_memory_limit, threads=cfg.duckdb_threads,
        )
        mp.save(mev_path)
        curve.to_csv(cfg.cache_dir / "mev_curve.csv", index=False)

    print(f"  dV/dt              : {mp.dv_dt_eth_per_ms:.3e} ETH/ms "
          f"({mp.dv_dt_eth_per_ms * 100:.6f} ETH per +100ms of delay)")
    print(f"  mean block value   : {mp.mean_block_value_eth:.4f} ETH "
          f"(what a reorg destroys)")

    print("\n" + "=" * 78)
    print("COUNTERFACTUAL: PROFIT UPLIFT FROM ADOPTING A PROPAGATION ACCELERATOR")
    print("=" * 78)
    print("Transit time (p90 - min sentry spread) is the ONLY component a")
    print("propagation product can compress. Publication delay — most of arrival")
    print("time — is the proposer's own timing game and is untouched.\n")

    # Full channel x scenario detail for one representative fleet.
    cf = run_counterfactual(
        d, rev, n_validators=30_000,
        dv_dt_eth_per_ms=mp.dv_dt_eth_per_ms,
        mean_block_value_eth=mp.mean_block_value_eth,
        eth_price_usd=px,
    )
    print("### Channel detail — a 30,000-validator operator (≈ Everstake / P2P scale)\n")
    show = cf[["scenario", "channel", "eth_year",
               "usd_day", "usd_week", "usd_month", "usd_year"]]
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(show.to_string(index=False))
    print("\n  Notes:")
    for _, r in cf[cf.scenario == "Optimum stated (6x)"].iterrows():
        print(f"    {r['channel']}: {r['note']}")

    # ---- PER STAKED VALIDATOR: the natural, size-invariant unit ----
    # Every channel is linear in fleet size, so this multiplied by any fleet
    # gives that fleet's uplift. High precision, because at n=1 the figures are
    # genuinely tiny and rounding them to cents would hide the finding.
    print("\n### PER STAKED VALIDATOR (avg effective balance "
          f"{rev.effective_balance_eth} ETH, post-Pectra)")
    for sc_name in ("Optimum stated (6x)", "conservative (3x)", "skeptical (2x)"):
        pv = per_validator_table(
            d, rev, sc_name,
            dv_dt_eth_per_ms=mp.dv_dt_eth_per_ms,
            mean_block_value_eth=mp.mean_block_value_eth,
            eth_price_usd=px,
        )
        print(f"\n  -- {sc_name} --")
        print(f"  {'channel':>34} {'$/hour':>9} {'$/day':>9} {'$/week':>9} "
              f"{'$/month':>10} {'$/year':>10} {'ETH/yr':>12}")
        for _, r in pv.iterrows():
            print(f"  {r['channel']:>34} {r['usd_hour']:>9.5f} {r['usd_day']:>9.4f} "
                  f"{r['usd_week']:>9.3f} {r['usd_month']:>10.3f} "
                  f"{r['usd_year']:>10.2f} {r['eth_year']:>12.6f}")
        tot = pv[pv.channel.str.startswith("TOTAL")].iloc[0]
        print(f"     => per ETH staked: {tot['eth_year_per_eth_staked']:.8f} ETH/yr "
              f"(${tot['eth_year_per_eth_staked'] * px:.4f}/yr per ETH)")

    # ---- THE GRID: profit by channel x fleet size x period, in USD ----
    # Fleet sizes span from a single validator up to the largest operators, so a
    # reader can locate their own scale. Note that at small fleets these are
    # long-run averages over events that essentially never happen in any given
    # hour — see RevenueModel.periods().
    FLEETS = [1, 10, 100, 1_000, 10_000, 30_000, 100_000]
    for sc_name in ("Optimum stated (6x)", "conservative (3x)", "skeptical (2x)"):
        grid = profit_grid(
            d, rev, FLEETS, sc_name,
            dv_dt_eth_per_ms=mp.dv_dt_eth_per_ms,
            mean_block_value_eth=mp.mean_block_value_eth,
            eth_price_usd=px,
        )
        print(f"\n### PROFIT UPLIFT — {sc_name}   (USD, ETH @ ${px:,.2f})")
        print(f"  {'fleet':>8} {'channel':>34} "
              f"{'$/hour':>9} {'$/day':>10} {'$/week':>11} {'$/month':>12} {'$/year':>13}")
        for n in FLEETS:
            for _, r in grid[grid.fleet == n].iterrows():
                print(f"  {r['fleet']:>8,} {r['channel']:>34} "
                      f"{r['usd_hour']:>9,.2f} {r['usd_day']:>10,.2f} "
                      f"{r['usd_week']:>11,.2f} {r['usd_month']:>12,.2f} "
                      f"{r['usd_year']:>13,.2f}")
            print()

    # Per named operator. TOTAL = A + max(B, C), never A+B+C: channels B and C
    # are mutually exclusive uses of the same saved milliseconds (bank them as
    # earlier arrival, or spend them as later publication — not both).
    print("\n### PRIVATE UPLIFT BY OPERATOR — total = A + max(B, C)")
    for sc in ("Optimum stated (6x)", "conservative (3x)", "skeptical (2x)"):
        print(f"\n  -- {sc} --")
        print(f"  {'operator':>13} {'validators':>11} "
              f"{'$/day':>10} {'$/week':>11} {'$/month':>12} {'$/year':>13}")
        for op in list(OPERATORS) + [None]:
            n = COMBINED_VALIDATORS if op is None else op.validators
            name = "ALL SEVEN" if op is None else op.name
            c = run_counterfactual(
                d, rev, n_validators=n,
                dv_dt_eth_per_ms=mp.dv_dt_eth_per_ms,
                mean_block_value_eth=mp.mean_block_value_eth,
                eth_price_usd=px,
            )
            eth_yr = private_total(c, sc)
            p = rev.periods(eth_yr, px)
            print(f"  {name:>13} {n:>11,} "
                  f"{p['day']['usd']:>10,.0f} {p['week']['usd']:>11,.0f} "
                  f"{p['month']['usd']:>12,.0f} {p['year']['usd']:>13,.0f}")

    # ---------------- 4. Figures ----------------
    fig_dir = cfg.cache_dir / "figures"
    p1 = rd_plot(
        d, "correct_head_rate", res.tau, fig_dir / "rd_correct_head.png",
        ylabel="share of attesters voting the correct head",
    )
    p2 = dose_response_plot(d, fig_dir / "dose_response.png")
    if p1 or p2:
        print("\n--- FIGURES ---")
        for p in (p1, p2):
            if p:
                print(f"  {p}")
    else:
        print("\n  (matplotlib not installed — skipping figures)")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
