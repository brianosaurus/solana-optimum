#!/usr/bin/env python3
"""
Preliminary counterfactual, using ONLY the small Xatu tables.

Why this exists
---------------
The full study needs `canonical_beacon_elaborated_attestation` (~800MB/day) to
weight outcomes by how many validators were actually exposed to each block. That
is slow.

But the COUNTERFACTUAL — the profit uplift from a propagation speedup — barely
needs it. Its inputs are:

    arrival_ms, arrival_min_ms, prop_spread_ms   <- beacon_api_eth_v1_events_block (17MB/day)
    missed_proposal                              <- proposer_duty vs canonical_block (2MB/day)

That is ~20MB/day instead of ~1.5GB/day. So we can get a real, many-day estimate
in seconds rather than hours.

The one approximation
---------------------
We set `n_attested` to a constant. In the full study it weights each slot by its
actual committee size, so that a slot with 30k attesters counts more than one
with 8k. Committee sizes are near-uniform across slots by construction (the
validator set is partitioned evenly across the 32 slots of an epoch), so this
approximation is mild — it shifts Channel A by well under a percent.

Everything else is identical to the full pipeline: same src/counterfactual.py,
same src/revenue.py, same measured MEV curve.

    python preliminary.py --days 14
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import duckdb
import pandas as pd

from config import load_config
from src.counterfactual import per_validator_table, private_total, profit_grid, run_counterfactual
from src.mev import MevParams, measure_mev_curve
from src.operators import COMBINED_VALIDATORS, OPERATORS
from src.revenue import RevenueModel
from src.xatu import T_CANON_BLOCK, T_EVENTS_BLOCK, T_PROPOSER_DUTY, XatuPaths

SMALL_TABLES = [T_EVENTS_BLOCK, T_CANON_BLOCK, T_PROPOSER_DUTY]

# Committee size is near-uniform across slots, so a constant weight is a mild
# approximation. Its absolute value cancels out of every rate we compute.
NOMINAL_COMMITTEE = 27_500


def fetch(paths: XatuPaths, table: str, date: dt.date, raw: Path) -> Path | None:
    dest = raw / f"{table}_{date.isoformat()}.parquet"
    if dest.exists():
        return dest
    r = subprocess.run(
        ["curl", "-sS", "--fail", "--retry", "3", "-o", str(dest),
         paths.day(table, date)],
        capture_output=True,
    )
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return None
    return dest


def build_light_panel(cfg, paths: XatuPaths, dates: list[dt.date]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute("SET threads=2")

    frames = []
    for date in dates:
        got = {t: fetch(paths, t, date, cfg.raw_dir) for t in SMALL_TABLES}
        if any(v is None for v in got.values()):
            print(f"  {date}  SKIPPED (missing Xatu file)")
            continue

        df = con.execute(f"""
        WITH canon AS (
            SELECT slot, lower(hex(block_root)) AS root
            FROM read_parquet('{got[T_CANON_BLOCK]}')
        ),
        -- DATA QUALITY GATE, and it is not optional.
        --
        -- `propagation_slot_start_diff` has a monstrous right tail: some sentry
        -- sightings are tens of SECONDS after slot start. Those are not
        -- propagation — they are a sentry that was syncing, restarting, or
        -- clock-skewed, reporting a block it caught up on later. Left in, they
        -- dragged mean transit to 44,000 ms against a median of 986 ms, which
        -- would have silently corrupted every Channel C figure.
        --
        -- A block cannot meaningfully "propagate" for longer than the slot it
        -- belongs to, so we keep only sightings within one slot (12s).
        ev AS (
            SELECT e.slot,
                   lower(hex(e.block))          AS root,
                   e.propagation_slot_start_diff AS ms,
                   e.meta_client_name
            FROM read_parquet('{got[T_EVENTS_BLOCK]}') e
            WHERE e.propagation_slot_start_diff BETWEEN 0 AND 12000
        ),
        -- One row per SEEN block (canonical or not), with its arrival timing and
        -- whether it survived the fork choice.
        seen AS (
            SELECT v.slot,
                   v.root,
                   (c.root IS NOT NULL)                       AS is_canonical,
                   median(v.ms)::DOUBLE                       AS arrival_ms,
                   min(v.ms)::DOUBLE                          AS arrival_min_ms,
                   quantile_cont(v.ms, 0.9)::DOUBLE           AS arrival_p90_ms,
                   quantile_cont(v.ms, 0.9)::DOUBLE
                     - min(v.ms)::DOUBLE                      AS prop_spread_ms
            FROM ev v
            LEFT JOIN canon c ON c.slot = v.slot AND c.root = v.root
            GROUP BY v.slot, v.root, c.root
            HAVING count(DISTINCT v.meta_client_name) >= {cfg.min_sentries}
        )
        -- The panel is one row per SEEN block. `orphaned` — built, broadcast,
        -- and then beaten in the fork choice — is the propagation-caused loss
        -- that Channel B prices. That is a different event from a MISSED
        -- proposal (no block produced at all), which has no arrival time and
        -- which no networking product can fix.
        SELECT s.slot,
               s.arrival_ms,
               s.arrival_min_ms,
               s.prop_spread_ms,
               NOT s.is_canonical        AS orphaned,
               (d.slot IS NOT NULL AND c2.slot IS NULL) AS missed_proposal,
               {NOMINAL_COMMITTEE}       AS n_attested
        FROM seen s
        LEFT JOIN read_parquet('{got[T_PROPOSER_DUTY]}') d ON d.slot = s.slot
        LEFT JOIN canon c2 ON c2.slot = s.slot
        """).df()
        df["date"] = date
        frames.append(df)
        for p in got.values():
            p.unlink(missing_ok=True)
        print(f"  {date}  {len(df):>5} slots")

    if not frames:
        raise SystemExit("no data")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", type=dt.date.fromisoformat,
                    default=dt.date(2025, 6, 1))
    args = ap.parse_args()

    cfg = load_config()
    paths = XatuPaths(cfg.xatu_base_url)
    rev = RevenueModel()
    px = cfg.eth_price_usd

    dates = [args.start + dt.timedelta(days=i) for i in range(args.days)]
    print(f"Building light panel over {args.days} days…")
    df = build_light_panel(cfg, paths, dates)

    # MEV curve: dV/dt is a slow-moving structural parameter, so 1 day suffices.
    mev_path = cfg.cache_dir / "mev_params.json"
    if mev_path.exists():
        mp = MevParams.load(mev_path)
    else:
        print("\nMeasuring MEV accrual curve…")
        mp, curve = measure_mev_curve(
            cfg.xatu_base_url, [args.start], cfg.raw_dir,
            memory_limit=cfg.duckdb_memory_limit, threads=2,
        )
        mp.save(mev_path)
        curve.to_csv(cfg.cache_dir / "mev_curve.csv", index=False)

    d = df.dropna(subset=["arrival_ms", "prop_spread_ms"]).copy()

    print("\n" + "=" * 84)
    print("PRELIMINARY — PROFIT UPLIFT FROM A PROPAGATION ACCELERATOR (Ethereum mainnet)")
    print("=" * 84)
    print(f"Window        : {df.date.min()} .. {df.date.max()}  ({args.days} days)")
    print(f"Slots         : {len(df):,}   with timing: {len(d):,}")
    print(f"ETH price     : ${px:,.2f}")
    print(f"\nArrival decomposition (the whole basis of the counterfactual):")
    print(f"  publication (first sentry sighting) : median {d.arrival_min_ms.median():>7.0f} ms")
    print(f"  network arrival (median sentry)     : median {d.arrival_ms.median():>7.0f} ms")
    print(f"  TRANSIT (p90 - min)  <- compressible: median {d.prop_spread_ms.median():>7.0f} ms"
          f"   mean {d.prop_spread_ms.mean():.0f} ms")
    late = (d.arrival_ms > 4000).mean()
    print(f"  blocks past the 4s deadline         : {late:.3%}")
    print(f"  missed proposals                    : {df.missed_proposal.mean():.3%}")
    print(f"\nMEV curve: dV/dt = {mp.dv_dt_eth_per_ms:.3e} ETH/ms"
          f"   block value (plateau) = {mp.mean_block_value_eth:.4f} ETH")

    print("\n" + "-" * 84)
    print("PER STAKED VALIDATOR  (avg effective balance "
          f"{rev.effective_balance_eth} ETH, post-Pectra)")
    print("-" * 84)
    for sc in ("Optimum stated (6x)", "conservative (3x)", "skeptical (2x)"):
        pv = per_validator_table(d, rev, sc, mp.dv_dt_eth_per_ms,
                                 mp.mean_block_value_eth, px)
        print(f"\n  -- {sc} --")
        print(f"  {'channel':>36} {'$/hour':>9} {'$/day':>8} {'$/week':>8} "
              f"{'$/month':>9} {'$/year':>9} {'ETH/yr':>11}")
        for _, r in pv.iterrows():
            print(f"  {r['channel']:>36} {r['usd_hour']:>9.5f} {r['usd_day']:>8.4f} "
                  f"{r['usd_week']:>8.3f} {r['usd_month']:>9.3f} "
                  f"{r['usd_year']:>9.2f} {r['eth_year']:>11.6f}")
        t = pv[pv.channel.str.startswith("TOTAL")].iloc[0]
        print(f"  {'per ETH staked':>36} "
              f"{'':>9} {'':>8} {'':>8} {'':>9} "
              f"{t['eth_year_per_eth_staked'] * px:>9.4f} "
              f"{t['eth_year_per_eth_staked']:>11.8f}")

    print("\n" + "-" * 84)
    print("BY OPERATOR — total private uplift, A + max(B,C)")
    print("-" * 84)
    for sc in ("Optimum stated (6x)", "conservative (3x)", "skeptical (2x)"):
        print(f"\n  -- {sc} --")
        print(f"  {'operator':>13} {'validators':>11} {'$/hour':>9} {'$/day':>9} "
              f"{'$/week':>10} {'$/month':>11} {'$/year':>12}")
        for op in list(OPERATORS) + [None]:
            n = COMBINED_VALIDATORS if op is None else op.validators
            name = "ALL SEVEN" if op is None else op.name
            cf = run_counterfactual(d, rev, n, mp.dv_dt_eth_per_ms,
                                    mp.mean_block_value_eth, px)
            eth_yr = private_total(cf, sc)
            p = rev.periods(eth_yr, px)
            print(f"  {name:>13} {n:>11,} {p['hour']['usd']:>9.2f} "
                  f"{p['day']['usd']:>9.2f} {p['week']['usd']:>10,.0f} "
                  f"{p['month']['usd']:>11,.0f} {p['year']['usd']:>12,.0f}")

    print("\n" + "=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
