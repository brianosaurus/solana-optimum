#!/usr/bin/env python3
"""
The mump2p deployment event study — on the network where it actually ran.

Background
----------
The original goal of this project was a staggered DiD of mump2p adoption on
Ethereum MAINNET. That study is impossible: mump2p never ran on mainnet
(FINDINGS.md). But it DID run on the HOODI testnet — announced 2025-06-24,
"rolling out on Ethereum's Hoodi testnet this summer", with Optimum publishing
Hoodi results on 2025-09-23 and 2025-10-16. Hoodi is therefore the only public
network that has ever carried the treatment, and Xatu has its data.

Design
------
Interrupted time series with a never-treated control network:

    treated : hoodi    (mump2p deployed ~Jul-Sep 2025)
    control : sepolia  (no mump2p, same client software, same forks)

For every day on both networks we compute the propagation physics that a
load-bearing transport upgrade MUST move:

    arrival_med_ms     median (across slots) of per-slot median sentry arrival
    arrival_p90_ms     the slow tail of slots
    transit_med_ms     per-slot p90-minus-min sentry spread (network transit)
    orphan_rate        blocks seen but never canonical
    missed_rate        slots with no canonical block at all
    n_sentries         observers that day (composition check)

If mump2p was load-bearing on Hoodi, transit_med_ms should compress around the
deployment window on hoodi and NOT on sepolia. If it ran in shadow mode (their
own methodology post says blocks propagated "through both mump2p and Gossipsub
simultaneously"), nothing moves — and that null is itself the finding.

Honesty notes
-------------
* Attestation outcomes are NOT available for this window: hoodi's attestation
  tables in Xatu are empty stubs until ~2026. This is a propagation-physics
  study by necessity.
* Testnets are noisy: operators restart nodes, sentry sets change. We log
  n_sentries and flag composition shifts rather than pretending they don't
  happen. A transit change that coincides with a sentry-count jump is suspect.
* We do not know the exact deployment date — "summer 2025" brackets it. So we
  SCAN for a break rather than imposing one, and the placebo discipline is the
  control network, not a fake date.
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

BASE = "https://data.ethpandaops.io/xatu/{net}/databases/default"

# Networks: genesis derived FROM THE DATA (slot_start_date_time - slot*12), not
# from memory. Mainnet included so the same script can baseline against it.
NETWORKS = ("hoodi", "sepolia")

MIN_SENTRIES = 3  # testnets have fewer observers than mainnet's fleet


def day_url(net: str, table: str, d: dt.date) -> str:
    return f"{BASE.format(net=net)}/{table}/{d.year}/{d.month}/{d.day}.parquet"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    r = subprocess.run(["curl", "-sS", "--fail", "--retry", "3",
                        "-o", str(dest), url], capture_output=True)
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return False
    return True


def day_metrics(con: duckdb.DuckDBPyConnection, ev: Path, cb: Path) -> dict | None:
    """One day's propagation physics from events_block + canonical_block."""
    row = con.execute(f"""
      WITH canon AS (
        SELECT slot, lower(hex(block_root)) AS root
        FROM read_parquet('{cb}')
      ),
      ev AS (
        SELECT slot, lower(hex(block)) AS root,
               propagation_slot_start_diff AS ms, meta_client_name
        FROM read_parquet('{ev}')
        WHERE propagation_slot_start_diff BETWEEN 0 AND 12000
      ),
      seen AS (
        SELECT v.slot, v.root,
               (c.root IS NOT NULL)                        AS canonical,
               median(v.ms)::DOUBLE                        AS arr,
               quantile_cont(v.ms, 0.9)::DOUBLE
                 - min(v.ms)::DOUBLE                       AS transit
        FROM ev v
        LEFT JOIN canon c ON c.slot = v.slot AND c.root = v.root
        GROUP BY v.slot, v.root, c.root
        HAVING count(DISTINCT v.meta_client_name) >= {MIN_SENTRIES}
      )
      SELECT
        (SELECT count(*) FROM canon)                                  AS n_canonical,
        (SELECT max(slot) - min(slot) + 1 FROM canon)                 AS n_slots_span,
        (SELECT count(DISTINCT meta_client_name) FROM ev)             AS n_sentries,
        count(*) FILTER (WHERE canonical)                             AS n_timed,
        count(*) FILTER (WHERE NOT canonical)                         AS n_orphaned,
        median(arr)          FILTER (WHERE canonical)                 AS arrival_med_ms,
        quantile_cont(arr, 0.9) FILTER (WHERE canonical)              AS arrival_p90_ms,
        median(transit)      FILTER (WHERE canonical)                 AS transit_med_ms,
        avg(transit)         FILTER (WHERE canonical)                 AS transit_mean_ms
      FROM seen
    """).fetchone()

    if row is None or not row[0]:
        return None
    n_canon, span, n_sentries, n_timed, n_orph = row[0], row[1], row[2], row[3], row[4]
    return {
        "n_canonical": n_canon,
        # Every slot has a proposer duty by protocol; a slot with no canonical
        # block is a missed proposal. Span-based so we need no duty table.
        "missed_rate": 1.0 - n_canon / span if span else np.nan,
        "n_sentries": n_sentries,
        "orphan_rate": n_orph / (n_timed + n_orph) if (n_timed + n_orph) else np.nan,
        "arrival_med_ms": row[5],
        "arrival_p90_ms": row[6],
        "transit_med_ms": row[7],
        "transit_mean_ms": row[8],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat,
                    default=dt.date(2025, 6, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat,
                    default=dt.date(2026, 1, 31))
    ap.add_argument("--out", type=Path, default=Path("data/event_study.parquet"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=2")

    rows = []
    d = args.start
    while d <= args.end:
        for net in NETWORKS:
            ev = args.raw / f"es_{net}_ev_{d}.parquet"
            cb = args.raw / f"es_{net}_cb_{d}.parquet"
            ok = (fetch(day_url(net, "beacon_api_eth_v1_events_block", d), ev)
                  and fetch(day_url(net, "canonical_beacon_block", d), cb))
            if not ok:
                print(f"  {d} {net}: MISSING", file=sys.stderr)
                continue
            m = day_metrics(con, ev, cb)
            ev.unlink(missing_ok=True)
            cb.unlink(missing_ok=True)
            if m is None:
                print(f"  {d} {net}: empty", file=sys.stderr)
                continue
            m.update({"date": d, "network": net})
            rows.append(m)
        if d.day == 1:
            print(f"  ...{d} done ({len(rows)} rows)", file=sys.stderr)
        d += dt.timedelta(days=1)

    df = pd.DataFrame(rows)
    df.to_parquet(args.out)
    print(f"\nwrote {len(df)} network-days -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
