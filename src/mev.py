"""
The MEV accrual curve V(t), measured from Xatu's relay bid traces.

V(t) is the block value a proposer captures by calling getHeader at t ms into its
slot — i.e. the running maximum bid across all MEV-Boost relays submitted by time
t. Its slope, dV/dt, is the payoff to waiting.

Why this matters more than the attestation channel
--------------------------------------------------
Measured on mainnet (2025-06-01), the mean best bid available to a proposer:

      t=0ms    0.027 ETH
    t=1000ms   0.033 ETH
    t=2000ms   0.037 ETH
    t=3000ms   0.045 ETH
    t=3500ms   0.051 ETH
    t=4000ms+  0.051 ETH   <- FLAT

Two facts fall out, and they drive the entire commercial argument:

1. Block value roughly DOUBLES across the slot. Waiting is enormously profitable.
   The only thing stopping a proposer waiting longer is the risk that its block
   fails to propagate before the 4s attestation deadline and gets reorged.

2. The curve PLATEAUS right after ~3.5s. Builders stop bidding on a block that
   cannot beat the deadline, because such a block is worthless.

Therefore a faster transport does not earn MEV by being fast. It earns MEV by
buying DELAY BUDGET: if transit is Δ ms quicker, the proposer can publish Δ ms
later and land at the same arrival time — same reorg risk, strictly better bid.

Implementation note (a real trap)
---------------------------------
`value` is a 32-byte LITTLE-endian uint256 in wei. Two consequences:
  * Decoding millions of bid rows in pandas OOMs a 15GB box.
  * A lexicographic max() over the raw little-endian blob is NOT the numeric max.

So we reverse the low 12 bytes to big-endian IN SQL (where a max() over a
fixed-width hex string IS the numeric max), aggregate to (slot x 250ms bucket)
first, and only decode the small result. 12 bytes covers 7.9e28 wei — far beyond
any block's value.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

T_BID_TRACE = "mev_relay_bid_trace"

# The window over which we fit dV/dt. Proposers running timing games publish
# roughly 1-3s into the slot, so this is the region that actually prices their
# marginal decision. Fitting over the whole 0-12s would be dominated by the
# post-deadline plateau and understate the slope.
FIT_LO_MS, FIT_HI_MS = 1000, 4000


@dataclass(frozen=True)
class MevParams:
    """Everything the counterfactual needs from the MEV side."""

    dv_dt_eth_per_ms: float  # slope of V(t): what a millisecond of delay is worth
    mean_block_value_eth: float  # plateau value: what a reorg destroys
    median_block_value_eth: float
    n_slots: int
    days: list[str]

    def save(self, p: Path) -> None:
        p.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(p: Path) -> "MevParams":
        return MevParams(**json.loads(p.read_text()))


def _be_hex_expr() -> str:
    """SQL that reverses the low 12 bytes of `value` from little- to big-endian."""
    return "||".join(f"substr(hex(value),{i},2)" for i in range(23, 0, -2))


def measure_mev_curve(
    base_url: str,
    dates: list[dt.date],
    raw_dir: Path,
    memory_limit: str = "3GB",
    threads: int = 2,
) -> tuple[MevParams, pd.DataFrame]:
    """Download bid traces for `dates` and estimate V(t) and dV/dt.

    Returns (params, curve). The curve is the V(t) table, useful for plotting.

    dV/dt is a slow-moving structural parameter — a few days is plenty. We do not
    need it for every day of the study window, and each day is a ~1.8GB download.
    """
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")

    frames = []
    used: list[str] = []
    for d in dates:
        url = f"{base_url}/{T_BID_TRACE}/{d.year}/{d.month}/{d.day}.parquet"
        dest = raw_dir / f"{T_BID_TRACE}_{d.isoformat()}.parquet"
        if not dest.exists():
            r = subprocess.run(
                ["curl", "-sS", "--fail", "--retry", "3", "-o", str(dest), url],
                capture_output=True,
            )
            if r.returncode != 0:
                dest.unlink(missing_ok=True)
                continue  # Xatu gap; skip this day rather than abort.

        g = con.execute(f"""
            WITH b AS (
                SELECT slot,
                       -- slot_start_date_time is UINT32; multiplying by 1000
                       -- overflows it. Cast BEFORE the multiply.
                       (timestamp_ms - slot_start_date_time::BIGINT * 1000) AS bid_ms,
                       {_be_hex_expr()} AS be_hex
                FROM read_parquet('{dest}')
                WHERE timestamp_ms IS NOT NULL
            )
            SELECT slot, (bid_ms // 250) * 250 AS t_bucket, max(be_hex) AS best_hex
            FROM b
            WHERE bid_ms BETWEEN 0 AND 12000
            GROUP BY slot, t_bucket
        """).df()
        frames.append(g)
        used.append(d.isoformat())
        dest.unlink(missing_ok=True)  # 1.8GB each — do not accumulate.

    if not frames:
        raise SystemExit("No MEV bid traces retrieved.")

    g = pd.concat(frames, ignore_index=True)
    g["eth"] = g["best_hex"].apply(lambda s: int(s, 16) / 1e18)

    # The best bid AVAILABLE by time t is a running max within each slot.
    g = g.sort_values(["slot", "t_bucket"])
    g["running_best"] = g.groupby("slot")["eth"].cummax()

    # Keep only slots that had a bid early. Otherwise slots that enter the sample
    # late would manufacture an upward slope out of nothing.
    early = set(g.loc[g.t_bucket <= 1000, "slot"].unique())
    g = g[g.slot.isin(early)]

    rows = []
    for t in sorted(g.t_bucket.unique()):
        best = g[g.t_bucket <= t].groupby("slot")["running_best"].max()
        rows.append({
            "t_ms": int(t),
            "slots": len(best),
            "mean_best_bid_eth": float(best.mean()),
            "median_best_bid_eth": float(best.median()),
        })
    curve = pd.DataFrame(rows)

    w = curve[(curve.t_ms >= FIT_LO_MS) & (curve.t_ms <= FIT_HI_MS)]
    slope = float(np.polyfit(w.t_ms, w.mean_best_bid_eth, 1)[0])

    # The plateau (t >= 6s) is the full value of the block — what a reorg destroys.
    plateau = curve[curve.t_ms >= 6000]
    params = MevParams(
        dv_dt_eth_per_ms=slope,
        mean_block_value_eth=float(plateau["mean_best_bid_eth"].iloc[-1]),
        median_block_value_eth=float(plateau["median_best_bid_eth"].iloc[-1]),
        n_slots=int(g.slot.nunique()),
        days=used,
    )
    return params, curve
