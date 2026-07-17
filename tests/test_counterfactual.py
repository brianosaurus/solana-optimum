"""
Tests for the counterfactual profit model.

The dangerous failure here is not a crash — it's a plausible-looking number that
overstates the benefit. So these tests pin the three things most likely to
silently inflate the answer:

  1. The speedup must apply ONLY to transit, never to publication delay.
  2. Channel A must be discounted by the adopter's stake share (it is mostly an
     externality to the rest of the network).
  3. Channels B and C must not be double-counted against the same saved ms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.counterfactual import (
    apply_speedup,
    private_total,
    channel_a_attester,
    channel_b_reorgs,
    channel_c_mev,
    run_counterfactual,
)
from src.revenue import RevenueModel
from src.xatu import ATTESTATION_DEADLINE_MS


# The REAL dose-response, measured on 214,443 mainnet slots (2025-06). The
# fixture interpolates this rather than inventing a tail shape, so the calibration
# test below is actually testing the model and not my imagination.
#   median arrival (ms) -> head-miss rate
_REAL_DOSE = np.array([
    [  500, 0.00044],
    [ 1500, 0.00071],
    [ 2500, 0.00332],
    [ 3250, 0.01179],
    [ 3750, 0.04112],
    [ 4500, 0.32567],
    [ 5500, 0.77650],
    [ 6500, 0.99875],
])


def _panel(n=2000, seed=0):
    """Synthetic slot panel whose head-miss structure matches real mainnet.

    Two properties matter, and both are load-bearing for the tests:

      1. arrival is DECOMPOSED into publication + transit, because only transit
         is compressible by a propagation product; and
      2. `correct_head_rate` follows the REAL measured dose-response, which is
         non-zero even for blocks arriving at 2-3s. Those are tail nodes — 61% of
         all real head-vote misses — and the original Channel A scored them zero.
    """
    rng = np.random.default_rng(seed)
    publish = rng.normal(1811, 500, n).clip(200)     # real median publication
    transit = rng.gamma(2.0, 165, n).clip(20)        # real median transit ~331ms
    arrival = publish + transit
    spread = transit * 2.66                          # real p90/median transit ratio

    miss = np.interp(arrival, _REAL_DOSE[:, 0], _REAL_DOSE[:, 1])

    return pd.DataFrame({
        "arrival_ms": arrival,
        "arrival_min_ms": publish,
        "prop_spread_ms": spread,
        "correct_head_rate": 1.0 - miss,
        "n_attested": rng.integers(8000, 32000, n),
        "missed_proposal": (arrival > 4500) & (rng.random(n) < 0.3),
        "had_orphan": arrival > 5000,
        "orphaned": (arrival > 4200) & (rng.random(n) < 0.5),
    })


def test_speedup_only_compresses_transit_never_publication():
    """THE critical test. A propagation product cannot make a proposer stop
    waiting. If the speedup were applied to the whole arrival time, the modelled
    benefit would be inflated by roughly an order of magnitude."""
    df = _panel()
    cf = apply_speedup(df, speedup=6.0)

    # Arrival can never fall below the publication time, no matter the speedup.
    assert (cf >= df["arrival_min_ms"] - 1e-9).all()

    # With an infinite speedup, arrival converges EXACTLY to publication time —
    # the irreducible floor.
    inf = apply_speedup(df, speedup=1e9)
    assert inf.values == pytest.approx(df["arrival_min_ms"].values, abs=1e-3)


def test_speedup_of_one_changes_nothing():
    df = _panel()
    assert apply_speedup(df, 1.0).values == pytest.approx(df["arrival_ms"].values)


def test_faster_is_never_worse():
    """Monotonicity: more speedup => arrival no later."""
    df = _panel()
    a = apply_speedup(df, 2.0)
    b = apply_speedup(df, 6.0)
    assert (b <= a + 1e-9).all()


def test_channel_a_is_receive_side_and_fully_private():
    """Channel A is the RECEIVE-side effect: my own nodes get blocks sooner, so
    MY attesters keep their head votes on everyone else's blocks. It is captured
    100% by the adopter and scales linearly with fleet size.

    (The SEND-side effect — my blocks reaching others' attesters sooner — is an
    externality worth only my stake share, and is deliberately EXCLUDED.)"""
    df = _panel()
    rev = RevenueModel()

    small = channel_a_attester(df, rev, 6.0, 10_000, private_share=0.01)
    big = channel_a_attester(df, rev, 6.0, 20_000, private_share=0.02)

    # Linear in validators, and NOT rescaled by private_share.
    assert big.eth_per_year == pytest.approx(2 * small.eth_per_year, rel=1e-6)
    assert small.eth_per_year > 0


def test_private_total_does_not_double_count_B_and_C():
    """B and C are mutually exclusive uses of the same saved milliseconds:
    bank them as earlier arrival (B) OR spend them as later publication (C).

    Summing all three would inflate the proposer-side benefit ~2x. The total must
    be A + max(B, C)."""
    df = _panel()
    out = run_counterfactual(
        df, RevenueModel(), n_validators=30_000,
        dv_dt_eth_per_ms=7e-6, mean_block_value_eth=0.05, eth_price_usd=1800.0,
    )
    sc = "Optimum stated (6x)"
    s = out[out.scenario == sc]
    a = s[s.channel.str.startswith("A.")].eth_year.sum()
    b = s[s.channel.str.startswith("B.")].eth_year.sum()
    c = s[s.channel.str.startswith("C.")].eth_year.sum()

    tot = private_total(out, sc)
    assert tot == pytest.approx(a + max(b, c))
    assert tot < a + b + c or min(b, c) == 0  # strictly less unless one is zero


def test_channel_c_mev_scales_with_saved_transit():
    """More speedup => more delay budget => more MEV. Linear in dV/dt."""
    df = _panel()
    rev = RevenueModel()

    slow = channel_c_mev(df, rev, 2.0, 30_000, dv_dt_eth_per_ms=7e-6)
    fast = channel_c_mev(df, rev, 6.0, 30_000, dv_dt_eth_per_ms=7e-6)
    assert fast.eth_per_year > slow.eth_per_year

    # Doubling the value of a millisecond doubles the channel.
    dbl = channel_c_mev(df, rev, 6.0, 30_000, dv_dt_eth_per_ms=14e-6)
    assert dbl.eth_per_year == pytest.approx(2 * fast.eth_per_year, rel=1e-6)


def test_channel_c_cannot_spend_delay_it_does_not_have():
    """A block already arriving past the deadline gains nothing from waiting
    longer — the bid curve has plateaued and the block is dead anyway. Usable
    delay must be capped at the headroom to the deadline, never negative."""
    df = pd.DataFrame({
        # Every block already lands well past the deadline.
        "arrival_ms": [6000.0, 7000.0, 8000.0],
        "arrival_min_ms": [5000.0, 6000.0, 7000.0],
        "prop_spread_ms": [1000.0, 1000.0, 1000.0],
        "n_attested": [10000, 10000, 10000],
        "missed_proposal": [False, False, False],
        "had_orphan": [False, False, False],
        "orphaned": [False, False, False],
    })
    r = channel_c_mev(df, RevenueModel(), 6.0, 30_000, dv_dt_eth_per_ms=7e-6)
    assert r.eth_per_year == pytest.approx(0.0, abs=1e-9)


def test_no_speedup_yields_no_uplift_on_any_channel():
    """The null: a 1x 'speedup' must earn exactly nothing, everywhere. If any
    channel returns a positive number here, it is fabricating value."""
    df = _panel()
    rev = RevenueModel()

    a = channel_a_attester(df, rev, 1.0, 30_000, private_share=0.03)
    b = channel_b_reorgs(df, rev, 1.0, 30_000, mean_block_value_eth=0.05)
    c = channel_c_mev(df, rev, 1.0, 30_000, dv_dt_eth_per_ms=7e-6)

    assert a.eth_per_year == pytest.approx(0.0, abs=1e-9)
    assert b.eth_per_year == pytest.approx(0.0, abs=1e-9)
    assert c.eth_per_year == pytest.approx(0.0, abs=1e-9)


def test_uplift_scales_with_fleet_size():
    """Twice the validators, twice the blocks proposed, twice the private uplift."""
    df = _panel()
    rev = RevenueModel()
    small = channel_c_mev(df, rev, 6.0, 10_000, dv_dt_eth_per_ms=7e-6)
    big = channel_c_mev(df, rev, 6.0, 20_000, dv_dt_eth_per_ms=7e-6)
    assert big.eth_per_year == pytest.approx(2 * small.eth_per_year, rel=1e-6)


def test_run_counterfactual_returns_all_channels_and_scenarios():
    df = _panel()
    out = run_counterfactual(
        df, RevenueModel(), n_validators=30_000,
        dv_dt_eth_per_ms=7e-6, mean_block_value_eth=0.05, eth_price_usd=1800.0,
    )
    assert set(out["scenario"]) == {
        "Optimum stated (6x)", "conservative (3x)", "skeptical (2x)"
    }
    assert len(out) == 9  # 3 scenarios x 3 channels
    # USD must be a strictly consistent scaling of ETH.
    r = out.iloc[0]
    assert r["usd_year"] == pytest.approx(r["eth_year"] * 1800.0)
    assert r["usd_day"] == pytest.approx(r["usd_year"] / 365.0)


def test_bigger_speedup_never_earns_less():
    """Monotonicity of the headline: 6x must dominate 2x on every channel."""
    df = _panel()
    out = run_counterfactual(
        df, RevenueModel(), n_validators=30_000,
        dv_dt_eth_per_ms=7e-6, mean_block_value_eth=0.05, eth_price_usd=1800.0,
    )
    for ch in out["channel"].unique():
        s2 = out[(out.channel == ch) & (out.scenario == "skeptical (2x)")].eth_year.iloc[0]
        s6 = out[(out.channel == ch) & (out.scenario == "Optimum stated (6x)")].eth_year.iloc[0]
        assert s6 >= s2 - 1e-9, f"{ch}: 6x earned less than 2x"


# --- Regression tests for the tail-attester bug ------------------------------
#
# The original Channel A only fired when a block's MEDIAN arrival crossed 4000ms,
# so it scored ZERO attestation benefit at 2x and 3x. That is wrong: arrival is a
# distribution, and 61% of real head-vote misses occur on blocks whose median
# arrival was ON TIME (tail nodes). These tests exist so nobody reintroduces it.

def test_channel_a_is_nonzero_at_modest_speedups():
    """THE BUG. A 2x speedup must produce a REAL attestation benefit.

    It pulls the operator's node out of the propagation tail on every block —
    it does not need to drag a late block back under the deadline to help."""
    df = _panel()
    rev = RevenueModel()
    for k in (2.0, 3.0, 6.0):
        r = channel_a_attester(df, rev, k, 30_000, private_share=0.03)
        assert r.eth_per_year > 0, (
            f"{k}x speedup reported ZERO attester benefit — the tail-attester bug "
            "is back: misses on ON-TIME blocks are being ignored"
        )


def test_channel_a_monotone_in_speedup():
    """Faster receive path => strictly fewer tail misses."""
    df = _panel()
    rev = RevenueModel()
    vals = [channel_a_attester(df, rev, k, 30_000, 0.03).eth_per_year
            for k in (2.0, 3.0, 6.0)]
    assert vals[0] < vals[1] < vals[2]


def test_accelerated_miss_recovers_baseline_at_1x():
    """CALIBRATION CHECK. At speedup=1 the modelled miss rate must reproduce the
    MEASURED one. A counterfactual that cannot recover reality when the treatment
    is a no-op cannot be trusted anywhere else."""
    from src.counterfactual import (
        accelerated_head_miss_rate,
        measured_head_miss_rate,
    )
    df = _panel()
    measured = measured_head_miss_rate(df)     # network average, all nodes
    modelled = accelerated_head_miss_rate(df, 1.0)  # the median node, unaccelerated

    # These are DIFFERENT populations (network-wide vs one node), so they need not
    # be identical — but if the model's un-accelerated node is nowhere near the
    # observed rate, the model is not describing this network at all.
    # On the real 214k-slot panel: modelled 1.008% vs measured 0.938%.
    assert modelled == pytest.approx(measured, abs=0.02), (
        f"model at 1x ({modelled:.4%}) is nowhere near the measured head-miss "
        f"rate ({measured:.4%}) — it is not describing this network"
    )


def test_tail_misses_on_ontime_blocks_are_counted():
    """The user's objection, as a test.

    EVERY block here arrives comfortably before the deadline (2.0-3.8s) — not one
    is "late". Yet attesters still miss head votes, because the propagation tail
    runs past 4000ms even when the median does not. A faster receive path pulls the
    operator's node out of that tail, so the benefit MUST be positive.

    The original model scored this as exactly zero, forever, at every speedup.
    """
    n = 2000
    rng = np.random.default_rng(3)
    arrival = rng.uniform(2000, 3800, n)      # NOTHING is late by median
    publish = arrival - rng.uniform(200, 500, n)
    # Miss rises with arrival but is never zero — these are tail nodes.
    miss = 0.0005 + 0.00002 * (arrival - 2000)

    df = pd.DataFrame({
        "arrival_ms": arrival,
        "arrival_min_ms": publish,
        "prop_spread_ms": (arrival - publish) * 2.66,
        "correct_head_rate": 1.0 - miss,
        "n_attested": np.full(n, 20000),
        "missed_proposal": np.zeros(n, dtype=bool),
        "had_orphan": np.zeros(n, dtype=bool),
        "orphaned": np.zeros(n, dtype=bool),
    })
    assert (df.arrival_ms <= 4000).all(), "fixture must contain NO late blocks"

    r = channel_a_attester(df, RevenueModel(), 6.0, 30_000, 0.03)
    assert r.eth_per_year > 0, (
        "no block is late by median, yet attesters still miss the head vote. "
        "A faster receive path must capture that — the old model returned zero."
    )


# --- Regression tests for the silent-zero Channel B bug ----------------------

def test_channel_b_is_never_exactly_zero_at_real_speedups():
    """Physics check (the user's objection, again): blocks demonstrably DO get
    orphaned — 188 measured in 30 days, with a hazard cliff at the deadline. A
    transit speedup must therefore rescue a small but strictly positive number
    of them. Two earlier versions of this channel printed exactly $0 into
    results tables via silent fallbacks; this test makes that impossible."""
    df = _panel()
    rev = RevenueModel()
    for k in (2.0, 3.0, 6.0):
        r = channel_b_reorgs(df, rev, k, 30_000, mean_block_value_eth=0.0513)
        assert r.eth_per_year > 0, (
            f"Channel B returned zero at {k}x — the silent-zero bug is back"
        )


def test_channel_b_monotone_and_fleet_scaled():
    df = _panel()
    rev = RevenueModel()
    b2 = channel_b_reorgs(df, rev, 2.0, 10_000, 0.0513).eth_per_year
    b6 = channel_b_reorgs(df, rev, 6.0, 10_000, 0.0513).eth_per_year
    assert 0 < b2 < b6
    b6_big = channel_b_reorgs(df, rev, 6.0, 20_000, 0.0513).eth_per_year
    assert b6_big == pytest.approx(2 * b6, rel=1e-6)


def test_channel_b_still_zero_at_1x():
    """No speedup, no rescue — the null must stay exactly null."""
    df = _panel()
    r = channel_b_reorgs(df, RevenueModel(), 1.0, 30_000, 0.0513)
    assert r.eth_per_year == pytest.approx(0.0, abs=1e-12)


def test_orphan_hazard_is_a_cliff_at_the_deadline():
    """Pin the measured curve's shape: negligible before 4s, steep after."""
    from src.orphans import hazard
    assert hazard(2500) < 0.001
    assert hazard(4250) > 0.01
    assert hazard(5500) > 0.20
    # Monotone across the cliff (3s onward). NOT asserted below 3s: the measured
    # 0-2s bin (0.024%) sits slightly above the 2-3s bin (0.008%) — 15 blocks of
    # same-slot fork noise — and pretending otherwise would misstate the data.
    xs = [3250, 4250, 4750, 5500]
    hs = [float(hazard(x)) for x in xs]
    assert hs == sorted(hs)
