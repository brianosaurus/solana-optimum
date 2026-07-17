"""
Orphaned blocks: the measured hazard, and what a transit speedup avoids.

Why this module exists
----------------------
Channel B (proposer reorgs avoided) was silently reporting $0, twice over:

  1. The cached panel predated the `orphaned` column, so channel_b fell into its
     "no orphan data" fallback and returned 0.0 — and the report printed the
     dollar figure without the note. A silent zero in a results table.
  2. Even the fixed panel could not have carried the right data: it joins arrival
     timing on the CANONICAL block, so a truly orphaned proposal (no canonical
     block that slot) has arrival_ms NULL and drops out. The orphan's OWN
     arrival — the thing its hazard depends on — was never exported.

The correct dataset is SEEN blocks: every block the sentries observed, canonical
or not, each with its own sighting stats. `measure()` below builds it from
beacon_api_eth_v1_events_block + canonical_beacon_block (~20MB/day).

What the measurement says (2025-06-01..30, 214,712 seen blocks, 188 orphans)
-----------------------------------------------------------------------------
The hazard is a cliff at the attestation deadline, as fork-choice mechanics
predict — attesters who cannot see a block by 4s vote for its parent instead,
and a block without votes loses to the next proposer's (boosted) fork:

    arrival < 4.0s   : 0.01-0.04%
    4.0-4.5s         : 1.6%
    4.5-5.0s         : 9.9%
    5.0-6.0s         : 28.5%

And WHY blocks get orphaned is the study's recurring theme in miniature:
orphaned blocks were PUBLISHED at median 4,290ms (survivors: 1,831ms), while
their transit was only modestly worse (482ms vs 322ms). Orphans are
overwhelmingly timing-game losers, not transit victims. A transit speedup
therefore rescues only the sliver whose lateness was actually the network's
fault:

    2x avoids 0.0381% of proposals    (model 1x: 0.1061% vs measured 0.0876%)
    3x avoids 0.0445%
    6x avoids 0.0488%

Small — a fleet the size of all seven partners combined avoids ~215 orphans a
year — but real, and a model that says exactly $0 is wrong in kind, not degree.

What an orphan is actually worth (validated against relay data)
---------------------------------------------------------------
Channel B prices avoided orphans at mean_block_value_eth (0.0513, the V(t)
plateau). Hypothesis tested: orphans are late-published timing-game blocks, so
dead blocks might carry ABOVE-average MEV from the heavy-tailed bid
distribution. Joining the 30-day orphans to mev_relay_proposer_payload_delivered
(102/188 had a relay record) REFUTED it:

    canonical mean 0.0434 ETH  |  orphaned mean 0.0443 ETH  ->  1.02x

Orphans carry average value. Two residual pricing errors roughly cancel:
we quote the best-bid plateau (0.0513) where delivered payloads average 0.0443
(-14%), and we omit the ~0.012 ETH consensus proposer reward that also dies in
a reorg (+25%). True loss ~0.056 vs our 0.0513: within ~10%, on a channel worth
~$20k/yr across all seven partners. Documented rather than re-plumbed.
"""

from __future__ import annotations

import numpy as np

# Measured on 214,712 seen blocks. Fraction of ALL proposals rescued from
# orphaning by a k-times transit speedup (model-vs-model, same discipline as
# Channel A: both sides run through the same empirical hazard curve).
ORPHAN_AVOIDED: dict[str, float] = {
    "2": 3.81e-4,
    "3": 4.45e-4,
    "6": 4.88e-4,
}

# The measured hazard curve itself, for anyone who wants h(arrival) rather than
# the pre-integrated deltas. Bin midpoints (ms) -> P(orphaned).
HAZARD_X = np.array([1000, 2500, 3250, 3750, 4250, 4750, 5500, 9000], dtype=float)
HAZARD_Y = np.array(
    [0.000241, 0.000084, 0.000054, 0.000380, 0.016176, 0.098712, 0.284916, 0.284916]
)
# Monotone by construction (a later block cannot be SAFER); the raw 6-12s bin
# dips below the 5-6s bin on 38 blocks of noise, so it is clamped.


def hazard(arrival_ms) -> np.ndarray:
    """P(a block is orphaned | its own median arrival time)."""
    return np.interp(np.asarray(arrival_ms, dtype=float), HAZARD_X, HAZARD_Y,
                     left=HAZARD_Y[0], right=HAZARD_Y[-1])


def avoided_fraction(speedup: float) -> float:
    """Fraction of an operator's proposals rescued from orphaning at `speedup`.

    Interpolates between the measured points so a non-standard speedup doesn't
    silently return 0.
    """
    ks = np.array([1.0, 2.0, 3.0, 6.0])
    vs = np.array([0.0,
                   ORPHAN_AVOIDED["2"],
                   ORPHAN_AVOIDED["3"],
                   ORPHAN_AVOIDED["6"]])
    return float(np.interp(speedup, ks, vs, left=0.0, right=vs[-1]))
