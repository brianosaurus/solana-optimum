"""
Pricing latency: from milliseconds to ETH.

An RD coefficient of "-0.24 correct-head-rate" is meaningless to anyone outside
econometrics. This module converts the estimated effects into the only unit that
matters commercially: ETH per validator per year.

The consensus reward structure (Altair onwards)
-----------------------------------------------
Each epoch, an attester can earn three components, weighted out of 64:

    TIMELY_SOURCE   14/64   voted for the right justified checkpoint
    TIMELY_TARGET   26/64   voted for the right epoch boundary block
    TIMELY_HEAD     14/64   voted for the right HEAD block
    ------------------------------------------------------------------
    attestation total  54/64 of the base reward

The remaining 10/64 is the proposer's cut (8/64) and sync committee (2/64).

Latency hits TIMELY_HEAD specifically, and this is the crucial asymmetry:

  * TIMELY_HEAD requires the attester to have SEEN the block by the 4s deadline.
    A late block => wrong head vote => this component is forfeited entirely.
  * TIMELY_SOURCE and TIMELY_TARGET are about checkpoints, not the head block.
    They survive a late block (the attester still votes for the correct target,
    which is an epoch-boundary block from up to 32 slots ago).

So the marginal cost of a late block to an ATTESTER is exactly the head
component: 14/64 of the base reward, i.e. ~25.9% of its attestation income for
that epoch (14/54).

The proposer's exposure is far worse. A block that propagates too slowly can be
reorged out entirely, in which case the proposer loses the whole thing: the
proposer reward AND the execution-layer tips AND the MEV. That is orders of
magnitude larger than one attester's head vote, which is why propagation latency
is a proposer-side problem first and an attester-side problem second.

Caveat, stated plainly
----------------------
`base_reward` depends on total_active_balance, which drifts as the validator set
grows. We take it as an input rather than hardcoding it, and the CLI reports the
value used. Anyone re-running this in six months must update it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Consensus spec constants (Altair+). Not tunables. -----------------------
TIMELY_SOURCE_WEIGHT = 14
TIMELY_TARGET_WEIGHT = 26
TIMELY_HEAD_WEIGHT = 14
PROPOSER_WEIGHT = 8
SYNC_REWARD_WEIGHT = 2
WEIGHT_DENOMINATOR = 64

BASE_REWARD_FACTOR = 64
EFFECTIVE_BALANCE_INCREMENT_GWEI = 1_000_000_000  # 1 ETH, in Gwei
GWEI_PER_ETH = 1_000_000_000

SLOTS_PER_EPOCH = 32
SECONDS_PER_SLOT = 12
EPOCHS_PER_YEAR = (365 * 24 * 3600) // (SLOTS_PER_EPOCH * SECONDS_PER_SLOT)  # 82125


def integer_squareroot(n: int) -> int:
    """The spec's integer sqrt. We mirror it exactly rather than using math.sqrt,
    because the reward formula is defined over integer arithmetic and rounding
    differences compound over 82,125 epochs/year."""
    return math.isqrt(n)


def base_reward_gwei(effective_balance_gwei: int, total_active_balance_gwei: int) -> int:
    """Per-epoch base reward for one validator, per the consensus spec.

        base_reward_per_increment =
            EFFECTIVE_BALANCE_INCREMENT * BASE_REWARD_FACTOR
            // integer_squareroot(total_active_balance)

        base_reward = increments * base_reward_per_increment
    """
    per_increment = (
        EFFECTIVE_BALANCE_INCREMENT_GWEI
        * BASE_REWARD_FACTOR
        // integer_squareroot(total_active_balance_gwei)
    )
    increments = effective_balance_gwei // EFFECTIVE_BALANCE_INCREMENT_GWEI
    return increments * per_increment


@dataclass(frozen=True)
class RevenueModel:
    """Prices the estimated latency effects in ETH.

    total_active_balance_eth
        The staked total across the whole beacon chain. Drives base_reward.
    effective_balance_eth
        Per-validator effective balance. 32 pre-Pectra; post-Pectra (EIP-7251)
        a consolidated validator can hold up to 2048, which scales its rewards
        linearly — so a large operator's exposure scales with this too.
    mean_block_value_eth
        Average total value of a block to its proposer (consensus reward +
        priority fees + MEV). Used to price a REORG, where the proposer loses
        everything. This is an empirical input, not a spec constant.
    """

    # Post-Pectra (EIP-7251) mainnet reality, as of 2026-07.
    #
    # DO NOT compute validator counts as staked_eth / 32. Consolidation means the
    # network now holds ~40.72M ETH across ~880,550 validators — an AVERAGE
    # effective balance of ~46.2 ETH, not 32. Dividing by 32 overstates the
    # validator count by ~45%, which in turn understates per-validator rewards and
    # overstates how many blocks any given fleet proposes.
    total_active_balance_eth: float = 40_720_000.0
    effective_balance_eth: float = 46.2
    # Measured from Xatu relay bid traces (see src/mev.py), not guessed: the
    # plateau of V(t) is what a reorg actually destroys.
    mean_block_value_eth: float = 0.0513

    @property
    def base_reward_per_epoch_eth(self) -> float:
        br = base_reward_gwei(
            int(self.effective_balance_eth * GWEI_PER_ETH),
            int(self.total_active_balance_eth * GWEI_PER_ETH),
        )
        return br / GWEI_PER_ETH

    @property
    def head_reward_per_epoch_eth(self) -> float:
        """The slice of each epoch's reward that a late block destroys."""
        return self.base_reward_per_epoch_eth * TIMELY_HEAD_WEIGHT / WEIGHT_DENOMINATOR

    @property
    def attestation_reward_per_epoch_eth(self) -> float:
        w = TIMELY_SOURCE_WEIGHT + TIMELY_TARGET_WEIGHT + TIMELY_HEAD_WEIGHT
        return self.base_reward_per_epoch_eth * w / WEIGHT_DENOMINATOR

    def attester_cost_of_head_misses(self, head_miss_rate: float) -> float:
        """ETH/validator/year forfeited by missing the head vote at `head_miss_rate`.

        A validator attests exactly once per epoch, so a head-miss RATE maps
        directly onto a fraction of the year's epochs.
        """
        return head_miss_rate * self.head_reward_per_epoch_eth * EPOCHS_PER_YEAR

    def attester_cost_per_100ms(self, d_head_miss_per_ms: float) -> float:
        """ETH/validator/year per +100ms of block arrival latency.

        `d_head_miss_per_ms` is the marginal effect of one extra millisecond of
        arrival time on the probability of a head miss — i.e. the slope of the
        dose-response curve.
        """
        return self.attester_cost_of_head_misses(d_head_miss_per_ms * 100.0)

    def proposer_cost_of_reorgs(
        self, reorg_rate: float, blocks_proposed_per_year: float
    ) -> float:
        """ETH/year a proposer forfeits to reorged blocks.

        A reorged block pays its proposer nothing at all — consensus reward,
        tips, and MEV all evaporate. This is the expensive tail of latency.
        """
        return reorg_rate * blocks_proposed_per_year * self.mean_block_value_eth

    def expected_proposals_per_year(self, n_validators: int) -> float:
        """How many blocks a validator (or an operator's fleet) proposes a year.

        Proposer selection is proportional to stake, so a validator's share of
        the ~2.63M slots/year equals its share of the active set.
        """
        active_validators = self.total_active_balance_eth / self.effective_balance_eth
        slots_per_year = EPOCHS_PER_YEAR * SLOTS_PER_EPOCH
        return slots_per_year * (n_validators / active_validators)

    # ------------------------------------------------------------------
    # Reporting helpers: the same annual figure, sliced into the periods a
    # commercial reader actually thinks in.
    # ------------------------------------------------------------------
    def periods(self, eth_per_year: float, eth_price_usd: float) -> dict[str, dict]:
        """Break an annual ETH figure into hour / day / week / month / year.

        These are straight-line divisions of an annual RATE, not separate
        estimates — and the distinction matters more than it looks.

        The underlying events are extremely lumpy. A validator proposes a block
        roughly once every ~2 months at typical fleet ratios, and blocks only
        cross the 4s deadline ~1% of the time. So a "per hour" figure is a
        long-run average, NOT a prediction about any given hour: in almost every
        actual hour, the realised gain is exactly zero, and the annual total
        arrives in a handful of discrete events.

        Quoting the hourly number as though it accrues smoothly would be
        misleading. It is here because it was asked for, and because it is the
        honest linear decomposition of the annual rate — not because anything
        actually happens hourly.
        """
        per = {
            "hour": eth_per_year / (365.0 * 24.0),
            "day": eth_per_year / 365.0,
            "week": eth_per_year / 52.0,
            "month": eth_per_year / 12.0,
            "year": eth_per_year,
        }
        return {
            k: {"eth": v, "usd": v * eth_price_usd} for k, v in per.items()
        }
