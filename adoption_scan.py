#!/usr/bin/env python3
"""
Can we see WHO adopted mump2p on Hoodi, and WHEN? The per-operator scan.

The idea
--------
Network-level physics showed no mump2p signature — but a network median can
hide a minority of adopters. This scan raises the power by segmenting Hoodi's
proposers into OPERATORS and looking for per-operator step changes in how fast
their own blocks propagate.

The segmentation is free: Hoodi's validator set was deposited in per-operator
batches, and each operator sets its own execution_payload_fee_recipient. One
day of data shows clean contiguous ~50k-index blocks per recipient. A recipient
cluster proposing ~300 blocks/day gives a well-powered daily transit series per
operator.

If an operator switched its beacon nodes to publish via mump2p (and the overlay
re-injects into gossipsub, where our sentries listen), THEIR blocks' transit
steps down on THEIR adoption date while other clusters don't move — a staggered
adoption map, which is exactly what the original DiD needed.

What this CANNOT see, stated up front
-------------------------------------
* Shadow-mode adoption (blocks still published via vanilla gossipsub, mump2p
  observing in parallel) leaves gossip propagation untouched — invisible here
  by construction.
* Receive-side adoption (their nodes ingest faster but publish normally) does
  not change how their own blocks propagate — also invisible in this scan.
* A cluster-specific break is a NECESSARY signature of send-side adoption, not
  proof: an operator upgrading its peering or client at that date looks the
  same. Attribution needs the date to line up with mump2p's timeline and NOT
  with client releases (the control clusters give us that).
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

BASE = "https://data.ethpandaops.io/xatu/hoodi/databases/default"
MIN_SENTRIES = 3


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    r = subprocess.run(["curl", "-sS", "--fail", "--retry", "3",
                        "-o", str(dest), url], capture_output=True)
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2025, 6, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 1, 31))
    ap.add_argument("--out", type=Path, default=Path("data/adoption_scan.parquet"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=2")

    frames = []
    d = args.start
    while d <= args.end:
        ev = args.raw / f"as_ev_{d}.parquet"
        cb = args.raw / f"as_cb_{d}.parquet"
        ok = (fetch(f"{BASE}/beacon_api_eth_v1_events_block/{d.year}/{d.month}/{d.day}.parquet", ev)
              and fetch(f"{BASE}/canonical_beacon_block/{d.year}/{d.month}/{d.day}.parquet", cb))
        if ok:
            # One row per canonical slot: who proposed it (operator = fee
            # recipient), and how it propagated (per-slot sentry stats).
            df = con.execute(f"""
              WITH canon AS (
                SELECT slot, lower(hex(block_root)) AS root, proposer_index,
                       lower(hex(execution_payload_fee_recipient)) AS fee_hex
                FROM read_parquet('{cb}')
              ),
              ev AS (
                SELECT slot, lower(hex(block)) AS root,
                       propagation_slot_start_diff AS ms, meta_client_name
                FROM read_parquet('{ev}')
                WHERE propagation_slot_start_diff BETWEEN 0 AND 12000
              )
              SELECT c.slot, c.proposer_index, c.fee_hex,
                     median(v.ms)::DOUBLE                       AS arrival_ms,
                     min(v.ms)::DOUBLE                          AS publish_ms,
                     quantile_cont(v.ms, 0.9)::DOUBLE
                       - min(v.ms)::DOUBLE                      AS transit_ms
              FROM ev v
              JOIN canon c ON c.slot = v.slot AND c.root = v.root
              GROUP BY c.slot, c.proposer_index, c.fee_hex
              HAVING count(DISTINCT v.meta_client_name) >= {MIN_SENTRIES}
            """).df()
            df["date"] = d
            frames.append(df)
        else:
            print(f"  {d}: MISSING", file=sys.stderr)
        ev.unlink(missing_ok=True)
        cb.unlink(missing_ok=True)
        if d.day == 1:
            print(f"  ...{d} ({sum(len(f) for f in frames):,} slot-rows)", file=sys.stderr)
        d += dt.timedelta(days=1)

    out = pd.concat(frames, ignore_index=True)
    # fee_hex is hex-of-ASCII ("0x..."); decode to the readable address.
    out["operator"] = out["fee_hex"].map(
        lambda h: bytes.fromhex(h).decode("ascii", "replace") if isinstance(h, str) else "?")
    out.drop(columns=["fee_hex"]).to_parquet(args.out)
    print(f"\nwrote {len(out):,} slot-rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
