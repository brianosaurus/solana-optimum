"""
Real mainnet fleet sizes for the seven operators Optimum publicly named.

These are the operators from the (abandoned) mump2p DiD — see FINDINGS.md. They
remain the right units to price the counterfactual for, because they are exactly
the accounts Optimum is selling to.

Sources and caveats, because these numbers are messier than they look
--------------------------------------------------------------------
Lido counts are HARD on-chain data, read from NodeOperatorsRegistry
(0x55032650b14df07b85bF18A3a3eC8E0Af2e028d5) at block 25,519,863.

Everything else is an estimate of the operator's TOTAL mainnet book (Lido plus
institutional plus retail), reconstructed from archived Rated Network operator
pages. Three traps worth knowing:

  * KILN has ZERO Lido validators. It exited all 10,579 after a Sept-2025
    security incident and never re-entered the curated set. Kiln then reissued
    all keys on new deposit addresses, so deposit-address-based attribution
    (Rated, Dune) LOST it — Rated shows its validator count collapsing from
    ~55,679 to ~5,522. That collapse is an ATTRIBUTION ARTIFACT, not attrition;
    Kiln's real book is ~1.5M ETH. Any figure downstream of Rated badly
    understates Kiln.

  * LUGANODES is not a Lido *curated* operator, though it is in Simple DVT and
    Lido V3 stVaults. Saying "not a Lido operator" would be wrong.

  * Lido's allocator EQUALISES deposits: every active curated operator sits at
    7,574-7,575 validators. So Lido-side counts carry almost no information about
    an operator's real size. All the variance lives in their non-Lido books.

Post-Pectra, validator COUNT and ETH STAKED are no longer interchangeable
(average effective balance is ~46.2 ETH, not 32), so we carry both where known.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operator:
    name: str
    validators: int  # total mainnet fleet (best public estimate)
    lido_validators: int  # of which, Lido curated (hard on-chain)
    confidence: str  # our confidence in `validators`


# Ordered by fleet size. `validators` is the number that drives the economics.
OPERATORS = [
    Operator("Kiln", 47_500, 0, "medium — Rated lost it; reconstructed"),
    Operator("P2P.org", 31_862, 7_575, "high"),
    Operator("Everstake", 29_300, 7_575, "high"),
    Operator("Ebunker", 11_537, 7_574, "high"),
    Operator("InfStones", 11_500, 7_575, "medium"),
    Operator("Blockdaemon", 10_788, 7_575, "high"),
    Operator("Luganodes", 5_250, 0, "very high — SDVT/stVaults only"),
]

# What the seven together represent — the "15% of Ethereum's stake" that Optimum's
# launch PR cited. Note the PR described the PARTNERS' stake share, NOT stake
# running mump2p (which is zero on mainnet).
COMBINED_VALIDATORS = sum(o.validators for o in OPERATORS)
