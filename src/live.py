"""
Realtime slot tracker: price each Ethereum mainnet slot as it happens.

What this is
------------
The batch study (run_study.py) measures the historical relationship between
block propagation latency and validator revenue. This module does the same
arithmetic, but slot-by-slot, live, so a dashboard can show the counterfactual
uplift accruing in real time.

WHAT IS ACTUALLY BEING MEASURED — read before trusting any number here
----------------------------------------------------------------------
This is a MODELLED COUNTERFACTUAL, not realised profit. Specifically:

  * mump2p is NOT running on Ethereum mainnet (see FINDINGS.md). Nobody is
    capturing any of this money today. The ticker shows what an operator WOULD
    gain IF a propagation accelerator delivering the stated speedup were
    deployed.

  * ~93% of the modelled uplift is Channel C (MEV delay budget), which only pays
    if the operator ALSO re-tunes its timing games to spend the new safety
    margin. An operator that installs the software and changes no config earns
    almost none of it.

  * The accrual is an EXPECTED VALUE over observed slots, not a realised cashflow.
    A 30k-validator operator proposes ~3.4% of slots; we accrue
    (their share) x (per-block gain) on every slot rather than waiting for one of
    their blocks to actually come up. Over a day this converges; over a minute it
    is a smooth fiction standing in for a lumpy reality.

Two honest limits of the LIVE feed specifically
-----------------------------------------------
1. ONE VANTAGE POINT. We poll a single public beacon endpoint, so `arrival_ms`
   is "when the block became visible THERE", which folds in that provider's own
   propagation and our poll interval. Xatu's batch figure is a median across
   many geographically distributed sentries. The two are not identical, and the
   live one is noisier and slightly later.

2. TRANSIT IS NOT LIVE-MEASURABLE. Transit time (the only component a
   propagation product can compress) is derived from the SPREAD of sightings
   across sentries — p90 minus min. One vantage point has no spread. So transit
   is taken from the calibrated batch study, not observed live.

Everything downstream is the same model as the batch study: same reward weights,
same dV/dt, same A + max(B,C) rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from src import orphans
from src.revenue import EPOCHS_PER_YEAR, SLOTS_PER_EPOCH, RevenueModel
from src.xatu import ATTESTATION_DEADLINE_MS

# Mainnet beacon genesis (unix seconds).
GENESIS_UNIX = 1_606_824_023
SECONDS_PER_SLOT = 12


@dataclass(frozen=True)
class Calibration:
    """Structural parameters carried over from the batch study.

    These are NOT observable from a single live vantage point, so we import them
    from the 14/30-day Xatu panel and state that plainly.
    """

    # Transit time (p90 - min sentry spread), the compressible component.
    mean_transit_ms: float = 1002.0
    median_transit_ms: float = 880.0
    # Marginal value of a millisecond of publication delay, from relay bid traces.
    dv_dt_eth_per_ms: float = 7.039e-06
    # Plateau of V(t): what a reorg destroys.
    mean_block_value_eth: float = 0.0513

    # Reduction in the operator's OWN head-miss rate, by speedup. Measured on the
    # 214k-slot panel as (observed miss rate) - (miss rate of a node whose transit
    # is `k` times faster).
    #
    # This REPLACES the old `rescue_share` ("what share of LATE blocks get pulled
    # back under the deadline"), which was answering the wrong question. Arrival is
    # a distribution, not a point: 61% of head-vote misses happen on blocks whose
    # MEDIAN arrival was already ON TIME — they are tail nodes. The benefit of a
    # faster receive path is pulling YOUR node out of that tail on EVERY block, not
    # rescuing the rare late one. The old framing absurdly implied zero attestation
    # benefit at 2x and 3x.
    head_miss_delta: dict[str, float] = None  # set via the factories below

    # Rescue share is kept ONLY for Channel B (proposer reorgs), where the question
    # genuinely IS "does my block get back under the deadline".
    rescue_share: dict[str, float] = None

    @staticmethod
    def default() -> "Calibration":
        c = Calibration()
        object.__setattr__(c, "head_miss_delta",
                           {"2": 0.004825, "3": 0.006100, "6": 0.007031})
        object.__setattr__(c, "rescue_share", {"2": 0.015, "3": 0.141, "6": 0.491})
        return c

    @staticmethod
    def load(p: Path) -> "Calibration":
        if not p.exists():
            return Calibration.default()
        d = json.loads(p.read_text())
        c = Calibration(
            mean_transit_ms=d.get("mean_transit_ms", 1002.0),
            median_transit_ms=d.get("median_transit_ms", 880.0),
            dv_dt_eth_per_ms=d.get("dv_dt_eth_per_ms", 7.039e-06),
            mean_block_value_eth=d.get("mean_block_value_eth", 0.0513),
        )
        object.__setattr__(
            c, "head_miss_delta",
            d.get("head_miss_delta",
                  {"2": 0.004825, "3": 0.006100, "6": 0.007031}),
        )
        object.__setattr__(
            c, "rescue_share",
            d.get("rescue_share", {"2": 0.015, "3": 0.141, "6": 0.491}),
        )
        return c

    def save(self, p: Path) -> None:
        d = asdict(self)
        p.write_text(json.dumps(d, indent=2))


def slot_start_unix(slot: int) -> int:
    return GENESIS_UNIX + slot * SECONDS_PER_SLOT


def slot_of(unix_ts: float) -> int:
    return int((unix_ts - GENESIS_UNIX) // SECONDS_PER_SLOT)


@dataclass
class SlotGain:
    """Modelled counterfactual gain attributable to ONE observed slot."""

    slot: int
    arrival_ms: float
    late: bool
    speedup: float
    n_validators: int

    eth_a: float  # attester head votes (receive-side)
    eth_b: float  # proposer reorg avoided
    eth_c: float  # MEV delay budget

    @property
    def eth_total(self) -> float:
        # A + max(B, C): B and C are mutually exclusive uses of the same saved ms.
        return self.eth_a + max(self.eth_b, self.eth_c)


def price_slot(
    slot: int,
    arrival_ms: float,
    speedup: float,
    n_validators: int,
    rev: RevenueModel,
    cal: Calibration,
) -> SlotGain:
    """Price one observed slot for a hypothetical operator of `n_validators`.

    We accrue EXPECTED value on every slot rather than waiting for one of the
    operator's own blocks to come up. Over a day the two converge; the expected
    form just makes the ticker smooth instead of a step function that jumps once
    every few hours.
    """
    active = rev.total_active_balance_eth / rev.effective_balance_eth
    stake_share = n_validators / active  # also the share of slots they propose

    late = arrival_ms > ATTESTATION_DEADLINE_MS

    # ---- Channel A: receive-side attestation ----
    #
    # CORRECTED. The old version only fired when the block's MEDIAN arrival
    # exceeded 4000ms — i.e. it assumed a block arriving at 2,500ms costs the
    # operator nothing.
    #
    # That is false, and the batch panel proves it. Arrival is a DISTRIBUTION
    # across nodes. On a block that reaches the median node at 2,500ms, nodes in
    # the propagation tail still see it after the deadline and still lose the head
    # vote. Over 214k slots:
    #
    #     head misses on LATE blocks     : 0.368%  (39% of all misses)
    #     head misses on ON-TIME blocks  : 0.570%  (61% of all misses)  <-- ignored!
    #
    # So the benefit of a faster receive path is NOT "rescuing late blocks" — it
    # is pulling YOUR node out of the propagation tail on EVERY block. That is why
    # the old model absurdly reported zero attestation benefit at 2x and 3x.
    #
    # We now accrue the per-slot expected rescue directly: the operator's
    # validators attesting in this slot (n/32, since duties spread evenly across
    # the epoch) times the head reward times the reduction in miss probability.
    attesters_here = n_validators / SLOTS_PER_EPOCH
    miss_delta = cal.head_miss_delta.get(str(int(speedup)), 0.0)
    eth_a = attesters_here * rev.head_reward_per_epoch_eth * miss_delta

    # ---- Channel B: proposer reorg avoided ----
    #
    # Priced from the MEASURED orphan hazard (src/orphans.py: 188 orphans in
    # 214,712 seen blocks; a cliff at the deadline — 1.6% at 4.0-4.5s, 9.9% at
    # 4.5-5s, 28.5% at 5-6s). `avoided_fraction` is the share of proposals a
    # transit speedup rescues; it is small because orphaned blocks are mostly
    # late-PUBLISHED (median 4,290ms vs 1,831ms), which no transport fixes.
    # An earlier version used a guessed constant hazard gated on this slot being
    # late, which combined with a broken batch channel to report exactly $0 —
    # wrong in kind: small is the true answer, zero was a bug.
    #
    # Accrued as expected value per observed slot: the operator proposes
    # stake_share of slots, so per slot it banks stake_share x avoided x value.
    eth_b = stake_share * orphans.avoided_fraction(speedup) * cal.mean_block_value_eth

    # ---- Channel C: MEV delay budget ----
    # This is the one that pays, and unlike A and B it accrues on EVERY block the
    # operator proposes, not just the late ones. Saved transit = the delay budget
    # the proposer can now spend fishing for a better bid.
    saved_ms = cal.mean_transit_ms * (1.0 - 1.0 / speedup)
    # You cannot spend delay you don't have: cap at the headroom to the deadline.
    headroom_ms = max(0.0, ATTESTATION_DEADLINE_MS - arrival_ms)
    usable_ms = min(saved_ms, headroom_ms)
    eth_c = stake_share * usable_ms * cal.dv_dt_eth_per_ms

    return SlotGain(
        slot=slot, arrival_ms=arrival_ms, late=late, speedup=speedup,
        n_validators=n_validators, eth_a=eth_a, eth_b=eth_b, eth_c=eth_c,
    )
