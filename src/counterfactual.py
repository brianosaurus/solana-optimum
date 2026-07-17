"""
Counterfactual: what does adopting a propagation accelerator actually earn you?

This module answers the commercial question — "if we deploy Optimum/mump2p and
get their stated improvement, what is the profit uplift?" — through three
SEPARATE channels, because they have completely different sizes and completely
different credibility.

    Channel A — ATTESTER head votes
        Faster transit => fewer of MY blocks land past the 4s deadline => the
        (randomly assigned) attesters of my slots can actually see my block.
        BUT: an operator's attesters are voting on OTHER people's blocks 99.99%
        of the time. See the "who captures this" warning below. This channel is
        largely a PUBLIC GOOD, not a private return.

    Channel B — PROPOSER reorgs
        A block that propagates too slowly gets orphaned: the proposer loses the
        consensus reward AND the priority fees AND the MEV. Everything.

    Channel C — MEV timing games  ** the big one **
        This is the real mechanism, and it is NOT "faster is better" — it is
        "faster buys you DELAY BUDGET".

        Block value V(t) rises steeply across the slot (measured: 0.027 ETH at
        t=0 -> 0.051 ETH by 3.5s). Proposers already delay publication to harvest
        this. What stops them delaying further is reorg risk: publish too late,
        the block doesn't propagate before the attestation deadline, and it dies.

        If a faster transport saves you Δ ms of transit, you can publish Δ ms
        LATER and still arrive at the same time — identical reorg risk, strictly
        better bid. The gain is dV/dt × Δ per block proposed.

        Optimum does not sell throughput. It sells delay budget.

The decomposition that makes this honest
----------------------------------------
    arrival_ms = t_publish + t_transit

A propagation product compresses ONLY t_transit. Most of arrival_ms is the
proposer *deliberately waiting* (Channel C's cause), which no networking upgrade
removes. Applying a "6x faster" claim to the whole of arrival_ms would overstate
the benefit by roughly an order of magnitude. We therefore apply the speedup only
to `prop_spread_ms` (the measured p90-minus-min sentry spread ~ transit time).

WHO ACTUALLY CAPTURES CHANNEL A — read this before quoting it
--------------------------------------------------------------
Channel A is the effect on the attesters *of the slots you propose*. But those
attesters are a random RANDAO committee drawn from the WHOLE network — they are
overwhelmingly other people's validators. So when you speed up your own block
propagation, you mostly improve OTHER operators' attestation rewards.

This is a positive externality, not a private return. An operator with 1% of
stake captures ~1% of the Channel A benefit its own blocks create. Reporting
Channel A as if the adopting operator banks all of it would be flatly wrong, and
`private_share` below exists to prevent exactly that mistake.

Channels B and C, by contrast, accrue ENTIRELY to the proposer. They are the real
commercial case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import orphans
from src.revenue import EPOCHS_PER_YEAR, SLOTS_PER_EPOCH, RevenueModel
from src.xatu import ATTESTATION_DEADLINE_MS


@dataclass(frozen=True)
class Scenario:
    """A claimed propagation improvement.

    speedup
        Factor by which transit time is divided. Optimum's public claim is "6x"
        (mump2p ~150ms vs a ~1s gossipsub baseline). NOTE that number comes from
        the HOODI TESTNET compared against an ethPandaOps gossipsub baseline on a
        DIFFERENT network — it is not a mainnet measurement, and we treat it as a
        marketing claim to be stress-tested, not a fact. Hence the conservative
        scenarios below.
    """

    name: str
    speedup: float


# Optimum's stated claim, plus deliberately conservative haircuts. We report all
# of them side by side rather than quoting the vendor number alone.
SCENARIOS = [
    Scenario("Optimum stated (6x)", 6.0),
    Scenario("conservative (3x)", 3.0),
    Scenario("skeptical (2x)", 2.0),
]


@dataclass
class ChannelResult:
    channel: str
    eth_per_year: float
    note: str


def measured_head_miss_rate(df: pd.DataFrame) -> float:
    """The head-miss rate actually OBSERVED, weighted by exposed attesters.

    No modelling: this is 1 - correct_head_rate straight from the chain. It is the
    baseline an average operator lives with today.
    """
    d = df.dropna(subset=["correct_head_rate", "n_attested"])
    if d.empty:
        return 0.0
    return float(
        np.average(1.0 - d["correct_head_rate"], weights=d["n_attested"])
    )


def _irreducible_floor(df: pd.DataFrame) -> float:
    """Head misses that survive even when the block is seen almost instantly.

    Reorged heads, client quirks, validators voting on a block that later loses
    the fork choice. No transport can remove these, so no speedup may claim them.
    Estimated from slots whose block was visible in under a second.
    """
    d = df.dropna(subset=["correct_head_rate", "n_attested", "arrival_ms"])
    early = d[d["arrival_ms"] < 1000]
    if early.empty:
        return 0.0
    return float(
        np.average(1.0 - early["correct_head_rate"], weights=early["n_attested"])
    )


def _dose_response_fn(df: pd.DataFrame, bins: int = 40):
    """Empirical f(median arrival) -> head-miss rate, fitted from the panel.

    THIS FUNCTION IS THE TAIL. That is the entire insight.

    f(m) is the fraction of a slot's attesters who missed the head vote, given
    that the block reached the MEDIAN node at m milliseconds. Measured on real
    data it is non-zero long before the deadline:

        f(1500ms) = 0.07%     f(2500ms) = 0.33%
        f(3250ms) = 1.18%     f(3750ms) = 4.11%

    Those misses on early blocks are nodes out in the propagation tail. f already
    encodes them — we do not have to model the spread ourselves, because the chain
    measured it for us.
    """
    d = df.dropna(subset=["arrival_ms", "correct_head_rate", "n_attested"])
    if d.empty:
        return lambda x: np.zeros_like(np.asarray(x, dtype=float))

    edges = np.linspace(0, 8000, bins + 1)
    idx = np.clip(np.digitize(d["arrival_ms"], edges) - 1, 0, bins - 1)
    miss = (1.0 - d["correct_head_rate"]).to_numpy()
    w = d["n_attested"].to_numpy()

    xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if w[m].sum() > 0:
            xs.append((edges[b] + edges[b + 1]) / 2)
            ys.append(float(np.average(miss[m], weights=w[m])))

    xs, ys = np.array(xs), np.array(ys)
    # Enforce monotonicity: a later block cannot make FEWER attesters miss. Thin
    # bins in the tail are noisy and would otherwise let the curve wobble
    # downward, which would let acceleration invent value out of noise.
    ys = np.maximum.accumulate(ys)

    return lambda x: np.interp(np.asarray(x, dtype=float), xs, ys,
                               left=ys[0], right=ys[-1])


def accelerated_head_miss_rate(df: pd.DataFrame, speedup: float) -> float:
    """Head-miss rate for a node whose TRANSIT is `speedup` times faster.

    THE MISTAKE THIS REPLACES — and it is instructive.
    ---------------------------------------------------
    The first version asked: "how many blocks does the speedup pull back under
    the 4s deadline?" That treats arrival as a single number per slot, so a block
    whose MEDIAN arrival is 2,500ms scores zero attestation loss.

    But arrival is a DISTRIBUTION across nodes, not a point. On a block that
    reaches the median node at 2,500ms, nodes out in the propagation tail still
    see it after 4,000ms — and still miss the head vote. Measured on 214k slots:

        head misses on LATE blocks (median > 4s)  : 0.368%  (39% of all misses)
        head misses on ON-TIME blocks (tail nodes): 0.570%  (61% of all misses)

    SIXTY-ONE PERCENT of head-vote misses happen on blocks that arrived on time.
    The old model scored every one of them as zero, and therefore reported ZERO
    attestation benefit at 2x and 3x — which is plainly wrong, because pulling
    your node out of the propagation tail helps on EVERY block, not just the late
    ones.

    (It also happened to produce a roughly-right total at 6x, by over-counting
    late blocks — it charged 100% of their attesters as missing when only ~33%
    actually do — and under-counting the tail. Two errors that nearly cancelled.
    Right answer, wrong reason: the most dangerous kind.)

    THE MODEL
    ---------
    The miss rate is TAIL MASS, and tail mass shrinks CONTINUOUSLY as transit
    compresses. It is not a step function on the median node's arrival — a second
    version of this code made that mistake too, and reported an identical miss
    rate at 1x, 2x, 3x and 6x, because no slot's MEDIAN ever crossed the deadline.

    The empirical dose-response f() already IS the tail: f(m) is the share of
    attesters who missed, given the block reached the median node at m. So we do
    not model the spread at all — we move each slot ALONG the measured curve:

        my accelerated median arrival = publication + (transit / speedup)
        my miss rate                  = f(that)

    At speedup=1 this returns f(observed arrival) = the MEASURED head-miss rate,
    exactly. The counterfactual recovers reality when the treatment is a no-op,
    by construction rather than by luck.

    (Conservative by design: compressing transit shrinks the SPREAD faster than it
    moves the median, so f(compressed median) slightly overstates the residual
    miss. We under-claim rather than over-claim.)
    """
    d = df.dropna(subset=["arrival_ms", "arrival_min_ms", "n_attested"])
    if d.empty:
        return 0.0

    f = _dose_response_fn(df)
    transit = (d["arrival_ms"] - d["arrival_min_ms"]).clip(lower=0)
    my_arrival = d["arrival_min_ms"] + transit / speedup

    return float(np.average(f(my_arrival), weights=d["n_attested"]))


def apply_speedup(df: pd.DataFrame, speedup: float) -> pd.Series:
    """Counterfactual arrival time under a transit speedup.

        new_arrival = publication + transit / speedup
                    = arrival_min_ms + prop_spread_ms / speedup

    Note we hold PUBLICATION TIME FIXED here. That is the conservative choice: it
    measures the pure "my block arrives sooner" effect (Channels A and B) without
    yet letting the proposer spend the saving on extra delay. Channel C then
    prices spending it.

    A proposer cannot do both — the saved milliseconds are either banked as
    lower reorg risk (A+B) or spent on more MEV (C). We report them separately
    and DO NOT add A+B to C, because that would double-count the same Δ.

    CONSISTENCY BUG THIS FIXES: an earlier version used `prop_spread_ms`
    (p90 minus min) as the transit here. But `arrival_ms` is the MEDIAN node's
    arrival, and p90-min is the spread out to the ninetieth percentile — a much
    larger number (880ms vs 331ms). So `apply_speedup(df, 1.0)` did NOT return
    `arrival_ms`: a 1x "speedup" silently moved every block's arrival by hundreds
    of milliseconds. Any counterfactual built on it was measuring partly its own
    inconsistency.

    The transit of the MEDIAN node is, by definition, (median arrival - first
    sighting). Use that, and a 1x speedup is exactly a no-op.

    (Channel C deliberately still uses `prop_spread_ms`, because a PROPOSER needs
    its block to reach the bulk of the network, not just the median node — so the
    wider p90 spread is the right constraint there. Different question, different
    quantity.)
    """
    transit = (df["arrival_ms"] - df["arrival_min_ms"]).clip(lower=0)
    return df["arrival_min_ms"] + transit / speedup


def channel_a_attester(
    df: pd.DataFrame,
    rev: RevenueModel,
    speedup: float,
    n_validators: int,
    private_share: float,  # retained for the externality note; see below
) -> ChannelResult:
    """RECEIVE-SIDE: my nodes get blocks sooner, so MY attesters keep their head vote.

    Getting the direction of this channel right is the difference between a
    number that is real and one that is a fantasy.

    THE WRONG MODEL (what I first wrote): "my blocks propagate faster, so the
    attesters of the slots I propose keep their head votes." That benefit is real
    but it is NOT MINE — the attesters of my slots are a random RANDAO committee
    drawn from the whole network, i.e. overwhelmingly other operators' validators.
    An operator with 3% of stake would capture ~3% of it. It is a positive
    externality, and counting it as private revenue would overstate this channel
    by ~30x for a large operator.

    THE RIGHT MODEL: mump2p is an acceleration layer on MY OWN nodes, so it
    delivers blocks to ME faster. That means:
      * it applies to EVERY slot I attest to (once per validator per epoch),
        not just the ~0.003% of slots I propose;
      * the beneficiary is MY validator, so I capture 100% of it;
      * the blocks being accelerated are OTHER PEOPLE's blocks — I benefit from
        their slowness being fixed on my end.

    So the counterfactual is: instead of seeing a block at the network-median
    arrival time, my accelerated node sees it at (publication + transit/speedup).
    Head-miss probability falls accordingly, across all my attestation duties.
    """
    # BASELINE and COUNTERFACTUAL must come from the SAME estimator, or the delta
    # is meaningless.
    #
    # A tempting mistake (which this code made): use the MEASURED head-miss rate
    # as the baseline and a MODELLED rate as the counterfactual. Those are not
    # comparable. The measured rate is the network AVERAGE across all nodes, and
    # is dominated by the slow tail. The modelled rate is one specific node's.
    # Subtracting one from the other mixes populations and silently invents (or
    # destroys) value.
    #
    # So both sides run through accelerated_head_miss_rate(): speedup=1 is the
    # operator's node today, speedup=k is the same node accelerated. The delta is
    # then internally consistent — it is one node, before and after.
    #
    # measured_head_miss_rate() is retained as an external SANITY CHECK, reported
    # by run_study.py: on the real 214k-slot panel the model's 1x (1.008%) lands
    # within 0.07pp of the measured 0.938%, which is what earns the model any
    # credibility at all.
    base_rate = accelerated_head_miss_rate(df, 1.0)
    new_rate = accelerated_head_miss_rate(df, speedup)

    delta = max(0.0, base_rate - new_rate)

    # Every validator attests once per epoch, so a head-miss RATE maps directly
    # onto the fraction of the year's epochs in which the head reward is lost.
    # This is captured entirely by the adopting operator.
    eth = rev.attester_cost_of_head_misses(delta) * n_validators

    return ChannelResult(
        channel="A. attester head votes (receive-side)",
        eth_per_year=eth,
        note=(
            f"my nodes' head-miss {base_rate:.4%} -> {new_rate:.4%} "
            f"across all {n_validators:,} validators, every epoch. "
            f"(A separate SEND-side benefit — my blocks reaching others' attesters "
            f"sooner — is an externality; I'd capture only ~{private_share:.2%} of it, "
            f"so it is EXCLUDED here.)"
        ),
    )


def channel_b_reorgs(
    df: pd.DataFrame,
    rev: RevenueModel,
    speedup: float,
    n_validators: int,
    mean_block_value_eth: float,
) -> ChannelResult:
    """Fewer of my blocks get orphaned. This accrues 100% to the proposer.

    We estimate P(orphan | arrival) empirically, then re-evaluate it under the
    counterfactual arrival distribution.
    """
    # THE ZERO THAT WASN'T. Two earlier versions of this channel returned exactly
    # $0 and printed it into results tables:
    #
    #   v1 keyed on `missed_proposal`, whose rows have no arrival time (no block,
    #      no arrival) and so were silently dropped by dropna.
    #   v2 keyed on an `orphaned` panel column that (a) did not exist in the
    #      cached panel, tripping a fallback that returned 0.0 without the report
    #      ever surfacing the note, and (b) could not have worked anyway — the
    #      panel joins arrival on the CANONICAL block, so a truly orphaned
    #      proposal has arrival_ms NULL. The orphan's OWN arrival, the thing the
    #      hazard depends on, was never in the panel at all.
    #
    # Physically, B cannot be zero: blocks demonstrably DO lose the fork choice
    # (188 measured in 30 days), and the hazard is a cliff at the attestation
    # deadline — 1.6% at 4.0-4.5s, 9.9% at 4.5-5s, 28.5% at 5-6s.
    #
    # So Channel B is now priced from that direct measurement (src/orphans.py,
    # built from SEEN blocks — every block sentries observed, canonical or not,
    # with its own arrival). The pre-integrated `avoided_fraction(speedup)` is the
    # share of proposals a transit speedup rescues, model-vs-model.
    #
    # It is genuinely small, and the measurement says why: orphaned blocks were
    # PUBLISHED at median 4,290ms (survivors: 1,831ms) with only modestly worse
    # transit (482 vs 322ms). Orphans are timing-game losers, not transit
    # victims — so a transport can only save the sliver whose lateness was the
    # network's fault. Small is the true answer; zero was a bug.
    avoided = orphans.avoided_fraction(speedup)
    blocks = rev.expected_proposals_per_year(n_validators)

    return ChannelResult(
        channel="B. proposer reorgs avoided",
        eth_per_year=avoided * blocks * mean_block_value_eth,
        note=(
            f"{avoided:.4%} of {blocks:,.0f} proposals/yr rescued from orphaning "
            f"(measured hazard: 188 orphans / 214,712 seen blocks; orphans are "
            f"mostly late-PUBLISHED, so transit can save only a sliver)"
        ),
    )


def channel_c_mev(
    df: pd.DataFrame,
    rev: RevenueModel,
    speedup: float,
    n_validators: int,
    dv_dt_eth_per_ms: float,
) -> ChannelResult:
    """THE BIG ONE: saved transit is spent as extra publication delay.

    Δ = transit_saved = prop_spread * (1 - 1/speedup)

    Publishing Δ ms later at the SAME arrival time means identical reorg risk but
    a strictly better bid: gain = dV/dt × Δ per proposed block.

    This is bounded by the fact that V(t) plateaus after ~3.5s (builders stop
    bidding on blocks that can't beat the deadline), so we cap the usable delay
    at the point where the block would cross the deadline.
    """
    d = df.dropna(subset=["prop_spread_ms", "arrival_ms"]).copy()
    if d.empty:
        return ChannelResult("C. MEV from delay budget", 0.0, "no data")

    saved = (d["prop_spread_ms"].clip(lower=0) * (1 - 1 / speedup))

    # You cannot spend delay you don't have: a block already arriving past the
    # deadline gains nothing from waiting longer, and no proposer can push
    # arrival beyond the deadline on purpose. Cap the usable delay at the
    # headroom to the deadline.
    headroom = (ATTESTATION_DEADLINE_MS - d["arrival_ms"]).clip(lower=0)
    usable = np.minimum(saved, headroom)

    mean_usable_ms = float(usable.mean())
    blocks = rev.expected_proposals_per_year(n_validators)

    return ChannelResult(
        channel="C. MEV from extra delay budget",
        eth_per_year=mean_usable_ms * dv_dt_eth_per_ms * blocks,
        note=(
            f"transit saved {saved.mean():.0f}ms, usable {mean_usable_ms:.0f}ms "
            f"@ {dv_dt_eth_per_ms:.2e} ETH/ms over {blocks:,.0f} blocks/yr"
        ),
    )


def run_counterfactual(
    df: pd.DataFrame,
    rev: RevenueModel,
    n_validators: int,
    dv_dt_eth_per_ms: float,
    mean_block_value_eth: float,
    eth_price_usd: float,
) -> pd.DataFrame:
    """All three channels x all scenarios, in ETH and USD per day/week/month/year."""
    active = rev.total_active_balance_eth / rev.effective_balance_eth
    private_share = n_validators / active

    rows = []
    for sc in SCENARIOS:
        chans = [
            channel_a_attester(df, rev, sc.speedup, n_validators, private_share),
            channel_b_reorgs(df, rev, sc.speedup, n_validators, mean_block_value_eth),
            channel_c_mev(df, rev, sc.speedup, n_validators, dv_dt_eth_per_ms),
        ]
        for c in chans:
            p = rev.periods(c.eth_per_year, eth_price_usd)
            rows.append({
                "scenario": sc.name,
                "channel": c.channel,
                "eth_year": c.eth_per_year,
                "usd_day": p["day"]["usd"],
                "usd_week": p["week"]["usd"],
                "usd_month": p["month"]["usd"],
                "usd_year": p["year"]["usd"],
                "note": c.note,
            })

    return pd.DataFrame(rows)


PERIODS = ["hour", "day", "week", "month", "year"]


def per_validator_table(
    df: pd.DataFrame,
    rev: RevenueModel,
    scenario_name: str,
    dv_dt_eth_per_ms: float,
    mean_block_value_eth: float,
    eth_price_usd: float,
) -> pd.DataFrame:
    """Uplift for ONE staked validator, by channel and period.

    This is the natural unit: every channel is LINEAR in fleet size, so a
    per-validator figure multiplied by any fleet gives that fleet's uplift. It is
    also the only figure that is comparable across operators of different sizes.

    Two things to hold in mind when reading it:

    1. POST-PECTRA, "a validator" is not 32 ETH. Consolidation (EIP-7251) means
       the average effective balance is now ~46.2 ETH. Rewards scale linearly in
       effective balance, so a 2048-ETH consolidated validator earns 64x a
       32-ETH one. We therefore also report per-ETH-staked, which is invariant to
       how an operator has chosen to consolidate.

    2. THE HOURLY FIGURE IS AN ACCOUNTING ARTIFACT. One validator proposes a
       block roughly once every two months. In virtually every actual hour, the
       realised gain is exactly zero; the annual total arrives in a handful of
       discrete events. The hourly number is the annual rate / 8760, not a
       prediction about any given hour.
    """
    sc = next(s for s in SCENARIOS if s.name == scenario_name)
    active = rev.total_active_balance_eth / rev.effective_balance_eth

    chans = [
        channel_a_attester(df, rev, sc.speedup, 1, 1 / active),
        channel_b_reorgs(df, rev, sc.speedup, 1, mean_block_value_eth),
        channel_c_mev(df, rev, sc.speedup, 1, dv_dt_eth_per_ms),
    ]
    a, b, c = (x.eth_per_year for x in chans)

    rows = []
    for ch in chans + [None]:
        if ch is None:
            name, eth_yr = "TOTAL = A + max(B,C)", a + max(b, c)
        else:
            name, eth_yr = ch.channel, ch.eth_per_year
        p = rev.periods(eth_yr, eth_price_usd)
        rows.append({
            "channel": name,
            "eth_year": eth_yr,
            # Per ETH staked — invariant to consolidation choices.
            "eth_year_per_eth_staked": eth_yr / rev.effective_balance_eth,
            **{f"usd_{k}": p[k]["usd"] for k in PERIODS},
        })
    return pd.DataFrame(rows)


def profit_grid(
    df: pd.DataFrame,
    rev: RevenueModel,
    fleets: list[int],
    scenario_name: str,
    dv_dt_eth_per_ms: float,
    mean_block_value_eth: float,
    eth_price_usd: float,
) -> pd.DataFrame:
    """USD profit uplift by CHANNEL x FLEET SIZE x PERIOD, for one scenario.

    Includes a TOTAL row per fleet using `private_total` (A + max(B,C)) — never
    the raw sum, which would double-count the proposer's saved milliseconds.

    A warning about the small fleets in this table: at 1-100 validators the
    numbers are not merely small, they are a long-run average over events that
    almost never happen. A single validator proposes a block roughly once every
    two months; the probability that any given one of its blocks is both late AND
    rescued by the speedup is tiny. The annual figure is real; the hourly figure
    is an accounting artifact of dividing it by 8,760.
    """
    sc = next(s for s in SCENARIOS if s.name == scenario_name)
    active = rev.total_active_balance_eth / rev.effective_balance_eth

    rows = []
    for n in fleets:
        share = n / active
        chans = [
            channel_a_attester(df, rev, sc.speedup, n, share),
            channel_b_reorgs(df, rev, sc.speedup, n, mean_block_value_eth),
            channel_c_mev(df, rev, sc.speedup, n, dv_dt_eth_per_ms),
        ]
        for c in chans:
            p = rev.periods(c.eth_per_year, eth_price_usd)
            rows.append({
                "fleet": n, "channel": c.channel, "eth_year": c.eth_per_year,
                **{f"usd_{k}": p[k]["usd"] for k in PERIODS},
            })

        # TOTAL = A + max(B, C). See private_total().
        a, b, cc = (x.eth_per_year for x in chans)
        tot = a + max(b, cc)
        p = rev.periods(tot, eth_price_usd)
        rows.append({
            "fleet": n, "channel": "TOTAL = A + max(B,C)", "eth_year": tot,
            **{f"usd_{k}": p[k]["usd"] for k in PERIODS},
        })

    return pd.DataFrame(rows)


def private_total(cf: pd.DataFrame, scenario: str) -> float:
    """The defensible headline: ETH/yr the ADOPTING operator actually banks.

        total = A + max(B, C)

    NOT A + B + C. Channels B and C are MUTUALLY EXCLUSIVE uses of the same saved
    milliseconds:

      * Channel B banks the saving as EARLIER ARRIVAL  -> lower reorg risk.
      * Channel C spends the saving as LATER PUBLICATION -> more MEV, at
        unchanged arrival time and therefore unchanged reorg risk.

    A proposer does one or the other with any given millisecond. Summing them
    would double-count Δ and inflate the proposer-side benefit roughly 2x. A
    rational proposer takes whichever is worth more — and given that block value
    nearly doubles across the slot while orphan rates are already <1%, that is
    almost always C.

    Channel A is additive: it is receive-side, concerns the operator's ATTESTERS
    rather than its proposals, and spends none of the proposer's delay budget.
    """
    s = cf[cf.scenario == scenario]
    a = s.loc[s.channel.str.startswith("A."), "eth_year"].sum()
    b = s.loc[s.channel.str.startswith("B."), "eth_year"].sum()
    c = s.loc[s.channel.str.startswith("C."), "eth_year"].sum()
    return float(a + max(b, c))
