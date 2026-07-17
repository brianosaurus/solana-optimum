#!/usr/bin/env python3
"""
Extrapolating the Hoodi measurement onto mainnet — the honest transfer model.

The naive transfer ("6x on Hoodi => 6x on mainnet") is wrong by construction:
Hoodi's gossipsub baseline (~600-1,900ms transit across our study window) is
far slower than mainnet's (median-node transit ~331ms, p90 spread ~880ms). A
multiplier won against a slow baseline does not transfer to a fast one.

What an RLNC overlay physically delivers is not a multiplier but an ABSOLUTE
LATENCY FLOOR: coded chunks routed through dedicated gateways arrive in
near-constant time regardless of how deep the gossip mesh is. Optimum's own
Hoodi measurement is ~150ms gateway arrival. And because mump2p runs PARALLEL
to gossipsub (their methodology post), a node simply takes whichever arrives
first:

        transit_cf = min(transit_observed, FLOOR)

The same floor that is a "6.7x win" on Hoodi is a much smaller win on mainnet —
that asymmetry IS the extrapolation, and it cuts differently per channel:

  * Channel A (attester receive): driven by the MEDIAN node's transit (~331ms).
    min(331, 150) saves ~180ms. Implied multiplier ~2.2x, nowhere near 6x.
  * Channel C (MEV delay budget): driven by the P90 spread (~880ms) — a
    proposer must reach the BULK of attesters. min(880, 150) saves ~730ms.
    Implied multiplier ~4-6x. The overlay compresses the tail harder than the
    middle, which is exactly what a constant-time delivery layer should do.

Floor scenarios (stated, not assumed):
    150ms — the vendor's own Hoodi gateway figure, taken at face value
    250ms — geographic realism: mainnet spans more of the planet than Hoodi's
            testnet topology; speed-of-light + gateway hops cost something
    400ms — conservative: overlay under real mainnet load, imperfect peering

Carried caveat, in bold on everything downstream: the 150ms input is a
GATEWAY-level measurement from the vendor, captured during the window when
Hoodi's own gossipsub was at its worst (see FINDINGS), and our 8-month event
study found no consensus-visible effect on Hoodi at all. This extrapolation
prices the claim IF it holds at validator clients on mainnet — it is an upper
bound built from an unverified input, not a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import orphans
from src.counterfactual import _dose_response_fn, measured_head_miss_rate
from src.operators import COMBINED_VALIDATORS, OPERATORS
from src.revenue import RevenueModel
from src.xatu import ATTESTATION_DEADLINE_MS

FLOORS_MS = [150.0, 250.0, 400.0]
PX = 1805.50
DV_DT = 7.039e-06  # ETH per ms of delay, from relay bid traces


def channel_a_floor(d: pd.DataFrame, rev: RevenueModel, floor: float) -> float:
    """ETH/validator/yr: my node's transit floored; walk the measured
    dose-response, model-vs-model (baseline = same estimator at no-op)."""
    f = _dose_response_fn(d)
    w = d["n_attested"]
    transit = (d["arrival_ms"] - d["arrival_min_ms"]).clip(lower=0)

    base = float(np.average(f(d["arrival_min_ms"] + transit), weights=w))
    new = float(np.average(f(d["arrival_min_ms"] + np.minimum(transit, floor)),
                           weights=w))
    delta = max(0.0, base - new)
    return rev.attester_cost_of_head_misses(delta)


def channel_b_floor(d: pd.DataFrame, rev: RevenueModel, floor: float,
                    n_validators: int, block_value: float) -> float:
    """ETH/yr for a fleet: orphan hazard walked with floored transit."""
    transit = (d["arrival_ms"] - d["arrival_min_ms"]).clip(lower=0)
    base = float(orphans.hazard(d["arrival_min_ms"] + transit).mean())
    new = float(orphans.hazard(d["arrival_min_ms"] + np.minimum(transit, floor)).mean())
    avoided = max(0.0, base - new)
    return avoided * rev.expected_proposals_per_year(n_validators) * block_value


def channel_c_floor(d: pd.DataFrame, rev: RevenueModel, floor: float,
                    n_validators: int) -> float:
    """ETH/yr for a fleet: delay budget = p90 spread compressed to the floor,
    capped by headroom to the deadline (you cannot spend delay you don't have)."""
    spread = d["prop_spread_ms"].clip(lower=0)
    saved = (spread - floor).clip(lower=0)
    headroom = (ATTESTATION_DEADLINE_MS - d["arrival_ms"]).clip(lower=0)
    usable = float(np.minimum(saved, headroom).mean())
    return usable * DV_DT * rev.expected_proposals_per_year(n_validators)


def main() -> None:
    d = pd.read_parquet("data/slot_panel.parquet").dropna(
        subset=["arrival_ms", "arrival_min_ms", "prop_spread_ms",
                "correct_head_rate", "n_attested"])
    rev = RevenueModel()

    med_transit = float((d["arrival_ms"] - d["arrival_min_ms"]).clip(lower=0).median())
    p90_spread = float(d["prop_spread_ms"].median())

    print("HOODI -> MAINNET TRANSFER (absolute-floor model)")
    print(f"panel: {len(d):,} mainnet slots | median-node transit {med_transit:.0f}ms | "
          f"p90 spread {p90_spread:.0f}ms\n")

    print("implied multiplier of each floor, per network / quantity:")
    print(f"  {'floor':>7} {'Hoodi(~1000ms)':>15} {'mainnet median':>15} {'mainnet p90':>13}")
    for fl in FLOORS_MS:
        print(f"  {fl:>5.0f}ms {1000/fl:>14.1f}x {med_transit/fl:>14.1f}x "
              f"{p90_spread/fl:>12.1f}x")

    print("\nPER VALIDATOR / YEAR (USD @ $1,805.50):")
    print(f"  {'floor':>7} {'A attester':>11} {'C MEV':>9} {'TOTAL A+max(B,C)':>17}")
    pv = {}
    for fl in FLOORS_MS:
        a = channel_a_floor(d, rev, fl) * PX
        b = channel_b_floor(d, rev, fl, 1, 0.0513) * PX
        c = channel_c_floor(d, rev, fl, 1) * PX
        tot = a + max(b, c)
        pv[fl] = tot
        print(f"  {fl:>5.0f}ms {a:>11.2f} {c:>9.2f} {tot:>17.2f}")

    print("\ncomparison — the old multiplicative 6x model said: $35.10/validator/yr")

    print("\nBY OPERATOR, total private uplift $/yr (A + max(B,C); D excluded here):")
    hdr = f"  {'operator':>13} {'validators':>11}" + "".join(
        f" {'floor ' + str(int(f)) + 'ms':>13}" for f in FLOORS_MS)
    print(hdr)
    for op in list(OPERATORS) + [None]:
        n = COMBINED_VALIDATORS if op is None else op.validators
        name = "ALL SEVEN" if op is None else op.name
        cells = []
        for fl in FLOORS_MS:
            a = channel_a_floor(d, rev, fl) * n * PX
            b = channel_b_floor(d, rev, fl, n, 0.0513) * PX
            c = channel_c_floor(d, rev, fl, n) * PX
            cells.append(a + max(b, c))
        print(f"  {name:>13} {n:>11,}" + "".join(f" {v:>13,.0f}" for v in cells))


if __name__ == "__main__":
    main()
