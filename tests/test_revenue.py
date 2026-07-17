"""
Tests for the revenue model.

These pin the consensus-spec arithmetic. If a future Ethereum fork changes the
reward weights, these tests should fail loudly rather than let the study quietly
report wrong ETH figures.
"""

from __future__ import annotations

import pytest

from src.revenue import (
    EPOCHS_PER_YEAR,
    GWEI_PER_ETH,
    TIMELY_HEAD_WEIGHT,
    WEIGHT_DENOMINATOR,
    RevenueModel,
    base_reward_gwei,
    integer_squareroot,
)


def test_integer_squareroot_matches_spec():
    # The spec's integer_squareroot floors. 15 -> 3, not 4.
    assert integer_squareroot(15) == 3
    assert integer_squareroot(16) == 4
    assert integer_squareroot(0) == 0


def test_base_reward_matches_hand_computation():
    """Recompute the spec formula by hand and check we agree.

    base_reward_per_increment = 1e9 * 64 // isqrt(total_active_balance_gwei)
    base_reward               = increments * base_reward_per_increment
    """
    total_eth = 35_000_000
    total_gwei = total_eth * GWEI_PER_ETH
    eb_gwei = 32 * GWEI_PER_ETH

    expected_per_inc = (1_000_000_000 * 64) // integer_squareroot(total_gwei)
    expected = 32 * expected_per_inc

    assert base_reward_gwei(eb_gwei, total_gwei) == expected


def test_base_reward_falls_as_more_eth_is_staked():
    """Rewards scale as 1/sqrt(total staked) — doubling the set does NOT halve
    the reward, it divides by sqrt(2). Guards against a linear-scaling bug."""
    eb = 32 * GWEI_PER_ETH
    small = base_reward_gwei(eb, 30_000_000 * GWEI_PER_ETH)
    large = base_reward_gwei(eb, 60_000_000 * GWEI_PER_ETH)

    ratio = small / large
    assert ratio == pytest.approx(2**0.5, rel=0.01)


def test_head_component_is_14_over_64_of_base():
    m = RevenueModel(total_active_balance_eth=35_000_000, effective_balance_eth=32)
    assert m.head_reward_per_epoch_eth == pytest.approx(
        m.base_reward_per_epoch_eth * TIMELY_HEAD_WEIGHT / WEIGHT_DENOMINATOR
    )


def test_head_is_about_26pct_of_attestation_income():
    """A late block costs the head vote: 14 of the 54 weights an attester can
    earn, i.e. ~25.9% of attestation income. This ratio is the single number the
    whole revenue translation hangs on."""
    m = RevenueModel()
    share = m.head_reward_per_epoch_eth / m.attestation_reward_per_epoch_eth
    assert share == pytest.approx(14 / 54, rel=1e-6)


def test_annual_cost_scales_linearly_in_miss_rate():
    m = RevenueModel()
    assert m.attester_cost_of_head_misses(0.02) == pytest.approx(
        2 * m.attester_cost_of_head_misses(0.01)
    )


def test_annual_cost_of_total_head_loss_equals_year_of_head_rewards():
    """Sanity anchor: a validator that misses EVERY head vote forfeits exactly
    one year's worth of head rewards."""
    m = RevenueModel()
    assert m.attester_cost_of_head_misses(1.0) == pytest.approx(
        m.head_reward_per_epoch_eth * EPOCHS_PER_YEAR
    )


def test_pectra_consolidated_validator_scales_rewards():
    """Post-EIP-7251 a 2048-ETH validator earns 64x a 32-ETH one, so its
    exposure to latency scales too."""
    small = RevenueModel(effective_balance_eth=32).head_reward_per_epoch_eth
    big = RevenueModel(effective_balance_eth=2048).head_reward_per_epoch_eth
    assert big / small == pytest.approx(64, rel=0.01)


def test_expected_proposals_per_year():
    """An operator holding 1% of the active set should propose ~1% of slots."""
    m = RevenueModel(total_active_balance_eth=32_000_000, effective_balance_eth=32)
    active = 1_000_000  # 32M ETH / 32
    props = m.expected_proposals_per_year(n_validators=10_000)  # 1% of the set
    slots_per_year = EPOCHS_PER_YEAR * 32
    assert props == pytest.approx(slots_per_year * 0.01, rel=1e-6)


def test_reorg_cost_is_full_block_value():
    """A reorged block pays nothing — the proposer loses the entire block value."""
    m = RevenueModel(mean_block_value_eth=0.06)
    cost = m.proposer_cost_of_reorgs(reorg_rate=0.01, blocks_proposed_per_year=1000)
    assert cost == pytest.approx(0.01 * 1000 * 0.06)
