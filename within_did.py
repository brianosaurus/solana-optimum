#!/usr/bin/env python3
"""
Within-Hoodi DiD designs that need NO adopter labels.

The problem: we cannot classify validators as mump2p / not-mump2p, and a
two-network comparison (Hoodi vs Sepolia) has one unit per arm — an interrupted
time series, not a DiD. These designs put the second difference on things we CAN
observe inside Hoodi alone.

DESIGN A — characteristic-exposure DiD (the difficulty gradient)
----------------------------------------------------------------
Gossipsub transit scales with message size: more bytes, more chunks, more mesh
round-trips. An RLNC overlay is near-constant-time in size (parallel coded
chunks). Therefore, IF the overlay carries meaningful load after deployment,
the SIZE PENALTY — transit(Q4-size blocks) − transit(Q1-size blocks) — must
compress. Estimator:

    DiD = [penalty_post − penalty_pre]

with the pre-trend of the penalty as the falsification check, and the window
cut at 2025-09-28 to exclude the known common shock (the Fusaka-era client
rollout that also moved never-treated Sepolia — including it would attribute a
fork to the product).

DESIGN B — distributional DiD (mixture emergence)
-------------------------------------------------
If an unknown fraction alpha of proposers adopted at unknown dates, per-block
transit post-deployment is a MIXTURE: a fast overlay mode and a normal gossip
mode. Fit a 2-component Gaussian mixture on log-transit per month:

    adoption share  = weight of the fast component (if it emerges)
    effect on treated = gap between component means

Identification needs no labels — only that the treated mode is fast enough to
separate. Guardrails: a 2-component fit "improving" by a hair is noise; we
require the fast component to (a) EMERGE (weight rising from ~0 across the
deployment window), (b) persist, and (c) sit at overlay-like speeds. A stable
two-mode network (e.g. small-vs-big blocks) shows constant weights across all
months, including pre-deployment — that is the placebo.
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
MIN_SENTRIES = 3
DEPLOY_START = pd.Timestamp("2025-07-01")   # "rolling out this summer"
COMMON_SHOCK = pd.Timestamp("2025-09-28")   # exclude the Fusaka-era break


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    r = subprocess.run(["curl", "-sS", "--fail", "--retry", "3",
                        "-o", str(dest), url], capture_output=True)
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return False
    return True


def collect(start: dt.date, end: dt.date, raw: Path,
            con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-slot rows WITH block size and blob count (the exposure variables)."""
    frames = []
    d = start
    while d <= end:
        ev = raw / f"wd_ev_{d}.parquet"
        cb = raw / f"wd_cb_{d}.parquet"
        ok = (fetch(f"{BASE}/beacon_api_eth_v1_events_block/{d.year}/{d.month}/{d.day}.parquet", ev)
              and fetch(f"{BASE}/canonical_beacon_block/{d.year}/{d.month}/{d.day}.parquet", cb))
        if ok:
            df = con.execute(f"""
              WITH canon AS (
                SELECT slot, lower(hex(block_root)) AS root,
                       block_total_bytes AS bytes,
                       coalesce(execution_payload_blob_gas_used, 0) / 131072 AS blobs
                FROM read_parquet('{cb}')
              ),
              ev AS (
                SELECT slot, lower(hex(block)) AS root,
                       propagation_slot_start_diff AS ms, meta_client_name
                FROM read_parquet('{ev}')
                WHERE propagation_slot_start_diff BETWEEN 0 AND 12000
              )
              SELECT c.slot, c.bytes, c.blobs,
                     quantile_cont(v.ms, 0.9)::DOUBLE - min(v.ms)::DOUBLE AS transit_ms
              FROM ev v JOIN canon c ON c.slot = v.slot AND c.root = v.root
              GROUP BY c.slot, c.bytes, c.blobs
              HAVING count(DISTINCT v.meta_client_name) >= {MIN_SENTRIES}
            """).df()
            df["date"] = d
            frames.append(df)
        ev.unlink(missing_ok=True)
        cb.unlink(missing_ok=True)
        if d.day == 1:
            print(f"  ...{d}", file=sys.stderr)
        d += dt.timedelta(days=1)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Design B: 1-D two-component Gaussian EM on log-transit. numpy only, seeded.
# ---------------------------------------------------------------------------
def em_2gauss(x: np.ndarray, iters: int = 200, seed: int = 0):
    """Returns (w_fast, mu_fast_ms, mu_slow_ms, bic1, bic2)."""
    rng = np.random.default_rng(seed)
    x = np.log(np.clip(x, 20, None))
    n = len(x)

    # 1-component BIC baseline.
    mu0, sd0 = x.mean(), x.std() + 1e-9
    ll1 = float(np.sum(-0.5 * np.log(2 * np.pi * sd0**2) - (x - mu0)**2 / (2 * sd0**2)))
    bic1 = -2 * ll1 + 2 * np.log(n)

    # 2-component EM, initialised at the quartiles for determinism.
    mu = np.array([np.quantile(x, 0.2), np.quantile(x, 0.8)])
    sd = np.array([sd0, sd0])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        pdf = (w / (sd * np.sqrt(2 * np.pi))
               * np.exp(-(x[:, None] - mu)**2 / (2 * sd**2)))
        r = pdf / np.clip(pdf.sum(1, keepdims=True), 1e-300, None)
        nk = r.sum(0)
        w = nk / n
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu)**2).sum(0) / nk) + 1e-6
    ll2 = float(np.sum(np.log(np.clip(pdf.sum(1), 1e-300, None))))
    bic2 = -2 * ll2 + 5 * np.log(n)

    fast = int(np.argmin(mu))
    return (float(w[fast]), float(np.exp(mu[fast])), float(np.exp(mu[1 - fast])),
            bic1, bic2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2025, 6, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2025, 12, 31))
    ap.add_argument("--cache", type=Path, default=Path("data/within_did.parquet"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'; SET threads=2")

    if args.cache.exists():
        df = pd.read_parquet(args.cache)
    else:
        df = collect(args.start, args.end, args.raw, con)
        df.to_parquet(args.cache)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["transit_ms", "bytes"])
    df = df[df.transit_ms > 0]

    print("=" * 84)
    print("WITHIN-HOODI DiD — no adopter labels required")
    print("=" * 84)
    print(f"slots {len(df):,} | {df.date.min().date()} .. {df.date.max().date()}")

    # ---------------- DESIGN A: the difficulty gradient ----------------
    print("\nDESIGN A — size-penalty DiD (Q4-size transit minus Q1-size transit)")
    print("If an RLNC overlay carries load, big blocks lose their penalty.\n")
    print(f"{'month':>9} {'Q1 size kB':>10} {'Q4 size kB':>10} {'transit Q1':>10} "
          f"{'transit Q4':>10} {'PENALTY ms':>10}")
    rows = {}
    for m, g in df.groupby(df.date.dt.to_period("M")):
        q1, q3 = g.bytes.quantile(0.25), g.bytes.quantile(0.75)
        small, big = g[g.bytes <= q1], g[g.bytes >= q3]
        pen = big.transit_ms.median() - small.transit_ms.median()
        rows[str(m)] = pen
        print(f"{str(m):>9} {q1/1024:>10.0f} {q3/1024:>10.0f} "
              f"{small.transit_ms.median():>10.0f} {big.transit_ms.median():>10.0f} "
              f"{pen:>10.0f}")

    pre = df[df.date < DEPLOY_START]
    post = df[(df.date >= DEPLOY_START) & (df.date < COMMON_SHOCK)]

    def penalty(g):
        q1, q3 = g.bytes.quantile(0.25), g.bytes.quantile(0.75)
        return (g[g.bytes >= q3].transit_ms.median()
                - g[g.bytes <= q1].transit_ms.median())

    did = penalty(post) - penalty(pre)
    print(f"\n  penalty pre (Jun)              : {penalty(pre):>7.0f} ms")
    print(f"  penalty post (Jul-Sep, pre-shock): {penalty(post):>7.0f} ms")
    print(f"  DiD (post - pre)               : {did:>+7.0f} ms")
    print("  (negative & large => consistent with a size-insensitive transport"
          " carrying load; ~0 or positive => no overlay signature)")

    # ---------------- DESIGN B: mixture emergence ----------------
    print("\nDESIGN B — 2-component mixture on log(transit), monthly")
    print("Adoption without labels shows up as an EMERGING fast component.\n")
    print(f"{'month':>9} {'w_fast':>7} {'mu_fast ms':>10} {'mu_slow ms':>10} "
          f"{'2-comp preferred?':>18}")
    for m, g in df.groupby(df.date.dt.to_period("M")):
        if len(g) < 500:
            continue
        w, muf, mus, bic1, bic2 = em_2gauss(g.transit_ms.to_numpy())
        pref = "yes" if bic2 < bic1 - 10 else "no"
        print(f"{str(m):>9} {w:>7.2f} {muf:>10.0f} {mus:>10.0f} {pref:>18}")
    print("\n  Read: adoption = w_fast RISING from ~0 across Jul-Sep with mu_fast at"
          "\n  overlay-like speeds. Constant w_fast in ALL months (incl. June, before"
          "\n  deployment) = the network's ordinary two-mode structure, not adoption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
