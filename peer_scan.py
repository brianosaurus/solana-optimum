#!/usr/bin/env python3
"""
Peer-level adoption hunt: did new gossip peers (Optimum gateways?) start winning
first-delivery races on Hoodi during the mump2p deployment window?

Why this is the last observable channel
---------------------------------------
Propagation aggregates showed no mump2p signature, and per-operator send-side
stepping showed zero staggering. But if mump2p gateways re-inject blocks into
gossipsub, they would appear in Xatu's libp2p layer as SENDING PEERS — new
peer identities that consistently deliver blocks to sentries first, faster
than organic mesh neighbours. That is a fingerprint no amount of shadow-mode
hides, PROVIDED the gateways peer (directly or nearly) with the sentries.

What the data supports, measured before building
------------------------------------------------
* `libp2p_gossipsub_beacon_block` records one row per (sentry, block): the
  FIRST accepted delivery and which peer sent it. Every row is a race win.
* Peer keys are stable day-to-day (65% adjacent-day overlap) but NOT across
  months (zero June<->Sept overlap): either salt rotation or replacement of the
  instrumented sentries (only ~4 report). So we track COHORTS by first-seen
  date, and flag "mass rebirth" days (>50% of keys new) as infrastructure
  events where cohort ages reset — those days cannot be read as adoption.

The adoption signature, precisely
---------------------------------
A cohort of peers first seen inside Jul-Sep 2025 that (a) persists for weeks,
(b) captures a materially growing share of first-delivery wins, and (c) wins
with lower propagation_slot_start_diff than incumbent peers. A salt rotation
fails (a)-(c) jointly because it replaces everyone at once; an organic new node
fails (b) because one mesh peer among many does not capture outsized share.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE = "https://data.ethpandaops.io/xatu/hoodi/databases/default"
T = "libp2p_gossipsub_beacon_block"


def fetch(d: dt.date, dest: Path) -> bool:
    if dest.exists():
        return True
    url = f"{BASE}/{T}/{d.year}/{d.month}/{d.day}.parquet"
    r = subprocess.run(["curl", "-sS", "--fail", "--retry", "3",
                        "-o", str(dest), url], capture_output=True)
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return False
    return True


def collect(start: dt.date, end: dt.date, raw: Path,
            con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Daily per-peer win stats: (date, peer, wins, median delivery ms)."""
    frames = []
    d = start
    while d <= end:
        p = raw / f"lp_{d}.parquet"
        if fetch(d, p):
            df = con.execute(f"""
              SELECT peer_id_unique_key AS peer,
                     count(*)                                   AS wins,
                     median(propagation_slot_start_diff)::DOUBLE AS med_ms,
                     count(DISTINCT meta_client_name)           AS sentries_won
              FROM read_parquet('{p}')
              WHERE propagation_slot_start_diff BETWEEN 0 AND 12000
              GROUP BY 1
            """).df()
            df["date"] = d
            frames.append(df)
            p.unlink(missing_ok=True)
        else:
            print(f"  {d}: MISSING", file=sys.stderr)
        if d.day == 1:
            print(f"  ...{d}", file=sys.stderr)
        d += dt.timedelta(days=1)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2025, 6, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 1, 31))
    ap.add_argument("--cache", type=Path, default=Path("data/peer_scan.parquet"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'; SET threads=2")

    if args.cache.exists():
        df = pd.read_parquet(args.cache)
        print(f"cache: {len(df):,} peer-days", file=sys.stderr)
    else:
        df = collect(args.start, args.end, args.raw, con)
        df.to_parquet(args.cache)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # First-seen date per peer key -> cohort. Ages are only meaningful between
    # mass-rebirth events, which we detect and annotate below.
    first_seen = df.groupby("peer")["date"].min().rename("born")
    df = df.join(first_seen, on="peer")
    df["age_d"] = (df["date"] - df["born"]).dt.days

    daily = df.groupby("date").agg(total_wins=("wins", "sum"),
                                   peers=("peer", "nunique"))
    born_per_day = df[df.age_d == 0].groupby("date")["peer"].nunique()
    daily["new_keys"] = born_per_day.reindex(daily.index).fillna(0)
    daily["new_key_frac"] = daily["new_keys"] / daily["peers"]

    # Mass-rebirth days: the salt/sentry-rotation fingerprint. Cohort ages are
    # not interpretable across these boundaries.
    rebirth = daily[daily["new_key_frac"] > 0.5].index
    print("\nMASS-REBIRTH DAYS (>50% of active keys new — infrastructure, not adoption):")
    for r in rebirth:
        print(f"  {r.date()}  ({daily.loc[r,'new_keys']:.0f}/{daily.loc[r,'peers']} keys new)")

    # Young-cohort win share: wins captured by peers aged 1-30 days (age 0
    # excluded so a rebirth day itself doesn't count; boundaries annotated).
    young = df[(df.age_d >= 1) & (df.age_d <= 30)].groupby("date")["wins"].sum()
    daily["young_share"] = (young.reindex(daily.index).fillna(0) / daily["total_wins"])

    # Speed edge: median delivery of young winners vs incumbents (age > 60d).
    yspeed = df[(df.age_d >= 1) & (df.age_d <= 30)].groupby("date")["med_ms"].median()
    ospeed = df[df.age_d > 60].groupby("date")["med_ms"].median()
    daily["young_med_ms"] = yspeed.reindex(daily.index)
    daily["incumbent_med_ms"] = ospeed.reindex(daily.index)

    print("\nMONTHLY: share of first-delivery wins captured by YOUNG peers (1-30d old)")
    m = daily.resample("MS").agg(young_share=("young_share", "mean"),
                                 peers=("peers", "mean"),
                                 young_ms=("young_med_ms", "median"),
                                 incumbent_ms=("incumbent_med_ms", "median"))
    m["speed_edge_ms"] = m["incumbent_ms"] - m["young_ms"]
    print(m.round({"young_share": 3, "peers": 0, "young_ms": 0,
                   "incumbent_ms": 0, "speed_edge_ms": 0}).to_string())

    # THE question: is there a persistent Jul-Sep cohort with outsized share?
    print("\nTOP individual peers by lifetime wins, with birth date and speed:")
    top = (df.groupby("peer")
             .agg(born=("born", "min"), days=("date", "nunique"),
                  wins=("wins", "sum"), med_ms=("med_ms", "median"))
             .sort_values("wins", ascending=False).head(12))
    share = top["wins"] / df["wins"].sum()
    for (peer, r), sh in zip(top.iterrows(), share):
        flag = " <-- born in deployment window" if pd.Timestamp("2025-07-01") <= r["born"] <= pd.Timestamp("2025-09-30") else ""
        print(f"  key …{str(peer)[-12:]}  born {r['born'].date()}  active {r['days']:>3}d  "
              f"wins {r['wins']:>7,}  med {r['med_ms']:>5.0f}ms{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
