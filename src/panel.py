"""
Panel construction: one row per canonical slot.

The unit of analysis is the SLOT, and the causal question is:

    When the block for slot s arrives late, what happens to the validators who
    were assigned to attest to slot s?

Why the slot is the right unit — the identification argument
------------------------------------------------------------
Attesters for slot s are drawn by RANDAO into a committee. That assignment is
random with respect to everything about the attester: an operator cannot choose
to be assigned to slots whose blocks arrive early. So from the attester's point
of view, *exposure to a late block is as-good-as-random*.

That is the natural experiment. We are not comparing good operators to bad ones
(which would be hopelessly confounded); we are comparing the SAME validators
across slots that happened, by lottery, to have early vs late blocks.

The outcomes
------------
correct_head
    Did the attester vote for slot s's block as head? An attester votes at the
    4s deadline for whatever head it can see. If block s hasn't arrived, the
    attester votes for slot s-1's block instead — a "wrong head vote" — and
    forfeits the `timely_head` component of its reward (14/64 of the attestation
    reward). This is the sharpest, most mechanical consequence of latency.

inclusion_distance
    block_slot - slot: how many slots elapsed before the attestation was
    included on chain.

missed_attestation
    Assigned by the committee but never included in any block within the
    lookahead window.

missed_proposal
    Had a proposer duty but produced no canonical block. A proposer whose own
    block propagates too slowly gets reorged out — the most expensive failure
    mode there is, costing the entire block reward plus MEV.

A correctness trap: the day boundary
------------------------------------
An attestation for slot s may be included up to SLOTS_PER_EPOCH (32) slots
later. If we read only day D's attestation file, attestations belonging to
slots late on day D get silently truncated — and truncation looks exactly like
"the validator missed its attestation". That would manufacture a spurious spike
in missed attestations at every midnight, and (worse) it would correlate with
nothing, biasing our estimates toward zero.

So we always read day D *and* day D+1's attestation files, and keep only
attestations whose ATTESTED slot falls in day D. `lookahead=True` below.
"""

from __future__ import annotations

import datetime as dt

import duckdb

from src.xatu import (
    ATTESTATION_DEADLINE_MS,
    SLOTS_PER_EPOCH,
    T_CANON_BLOCK,
    T_COMMITTEE,
    T_ELAB_ATT,
    T_EVENTS_BLOCK,
    T_PROPOSER_DUTY,
    XatuPaths,
)


def _rel(paths: XatuPaths, table: str, date: dt.date, local_dir=None) -> str:
    """A DuckDB-readable source for one table-day.

    If a local copy has been downloaded, read that; otherwise read the URL.
    The fat tables (attestations, committees) MUST be local — DuckDB's httpfs
    corrupts large ZSTD column chunks over HTTP range requests.
    """
    if local_dir is not None:
        p = local_dir / f"{table}_{date.isoformat()}.parquet"
        if p.exists():
            return f"read_parquet('{p}')"
    return f"read_parquet('{paths.day(table, date)}')"


def build_slot_panel(
    con: duckdb.DuckDBPyConnection,
    paths: XatuPaths,
    date: dt.date,
    min_sentries: int,
    local_dir=None,
) -> None:
    """Materialise `slot_panel` for one day into the DuckDB connection.

    Produces one row per slot that had a proposer duty — including slots where
    the proposer failed to produce a block, which is precisely the
    `missed_proposal` outcome.
    """
    nxt = date + dt.timedelta(days=1)

    canon = _rel(paths, T_CANON_BLOCK, date, local_dir)
    events = _rel(paths, T_EVENTS_BLOCK, date, local_dir)
    duty = _rel(paths, T_PROPOSER_DUTY, date, local_dir)
    att_d0 = _rel(paths, T_ELAB_ATT, date, local_dir)
    att_d1 = _rel(paths, T_ELAB_ATT, nxt, local_dir)
    comm = _rel(paths, T_COMMITTEE, date, local_dir)

    con.execute(f"""
    -- The canonical chain for this day: the ground truth of what actually
    -- happened. `root` is what an on-time attester *should* have voted for.
    CREATE OR REPLACE TEMP TABLE canon AS
    SELECT slot,
           epoch,
           lower(hex(block_root)) AS root,
           proposer_index
    FROM {canon};

    -- TREATMENT: network-level block arrival time.
    --
    -- Each Xatu sentry reports when IT first saw the block. Sentries sit in
    -- different regions on different hosts, so any single one is a noisy view
    -- of "when did the network see this block". We take the median across
    -- sentries, and require at least `min_sentries` observers before trusting
    -- the slot at all — one badly-clocked sentry must not drive the estimate.
    --
    -- We join on block ROOT, not just slot: during a reorg two blocks compete
    -- for the same slot, and we want the arrival time of the block that
    -- actually won.
    -- We decompose arrival into its two economically distinct parts, because a
    -- propagation product only attacks ONE of them:
    --
    --     arrival_ms  =  t_publish        +  t_propagate
    --                    (the proposer's      (network transit — the ONLY part
    --                     timing-game delay)   mump2p/RLNC can shrink)
    --
    -- Xatu's sentries sit in different regions, so the SPREAD of their sightings
    -- of the same block identifies transit time:
    --     arrival_min_ms (first sighting)  ~ publication + one hop
    --     arrival_p90_ms (near-full spread) ~ publication + transit
    --     prop_spread_ms = p90 - min        ~ TRANSIT TIME
    --
    -- Treating the whole of arrival_ms as compressible would wildly overstate
    -- any propagation product's benefit — most of arrival_ms is the proposer
    -- deliberately waiting to accrue MEV, which no networking upgrade removes.
    --
    -- Caveat, stated plainly: `min` is the first SENTRY sighting, not the true
    -- publication instant, so it already contains one hop. prop_spread_ms is
    -- therefore a LOWER bound on true transit, which makes the counterfactual
    -- gains we compute from it conservative.
    CREATE OR REPLACE TEMP TABLE arrival AS
    SELECT e.slot,
           median(e.propagation_slot_start_diff)::DOUBLE            AS arrival_ms,
           min(e.propagation_slot_start_diff)::DOUBLE               AS arrival_min_ms,
           quantile_cont(e.propagation_slot_start_diff, 0.9)::DOUBLE AS arrival_p90_ms,
           quantile_cont(e.propagation_slot_start_diff, 0.9)::DOUBLE
             - min(e.propagation_slot_start_diff)::DOUBLE           AS prop_spread_ms,
           count(DISTINCT e.meta_client_name)                       AS n_sentries
    FROM {events} e
    JOIN canon c
      ON c.slot = e.slot
     AND c.root = lower(hex(e.block))
    -- DATA QUALITY GATE. `propagation_slot_start_diff` has a monstrous right
    -- tail: some sightings land tens of SECONDS after slot start. Those are not
    -- propagation — they are a sentry that was syncing, restarting, or
    -- clock-skewed, reporting a block it caught up on later. Left in, they
    -- dragged MEAN transit to 44,000ms against a MEDIAN of 986ms, silently
    -- corrupting every counterfactual figure downstream. A block cannot
    -- meaningfully propagate for longer than its own slot.
    WHERE e.propagation_slot_start_diff BETWEEN 0 AND 12000
    GROUP BY e.slot
    HAVING count(DISTINCT e.meta_client_name) >= {min_sentries};

    -- ORPHANED BLOCKS. A block the sentries SAW but which never made the
    -- canonical chain was reorged out — the proposer built it, published it,
    -- and got nothing: no consensus reward, no priority fees, no MEV. This is
    -- the most expensive consequence of slow propagation, and it is invisible in
    -- `canonical_beacon_block` by construction (that table only holds winners).
    --
    -- We recover it by anti-joining sightings against the canonical chain.
    CREATE OR REPLACE TEMP TABLE orphaned AS
    SELECT e.slot,
           count(DISTINCT lower(hex(e.block)))                    AS n_blocks_seen,
           median(e.propagation_slot_start_diff)::DOUBLE          AS orphan_arrival_ms
    FROM {events} e
    LEFT JOIN canon c
      ON c.slot = e.slot AND c.root = lower(hex(e.block))
    WHERE c.slot IS NULL          -- seen, but never canonical => orphaned
    GROUP BY e.slot;

    -- OUTCOMES (attestation side).
    --
    -- Read day D and day D+1, keep only attestations whose ATTESTED slot is in
    -- day D. Without the D+1 lookahead, attestations for late-in-day slots are
    -- truncated and masquerade as missed attestations. See module docstring.
    CREATE OR REPLACE TEMP TABLE att AS
    SELECT slot, block_slot, committee_index, beacon_block_root, validators
    FROM (
        SELECT slot, block_slot, committee_index, beacon_block_root, validators
        FROM {att_d0}
        UNION ALL
        SELECT slot, block_slot, committee_index, beacon_block_root, validators
        FROM {att_d1}
    )
    WHERE slot IN (SELECT slot FROM canon)
      -- An attestation cannot be included before the slot it attests to, and
      -- cannot be included more than one epoch later. Anything outside that is
      -- corrupt and would poison the inclusion-distance mean.
      AND block_slot >= slot
      AND block_slot - slot <= {SLOTS_PER_EPOCH};

    -- DE-DUPLICATION. A validator attests to slot s exactly ONCE, but that one
    -- attestation can be INCLUDED in several blocks: aggregates can overlap
    -- within a block, and competing forks each carry their own copy. Summing
    -- `len(validators)` over raw rows therefore DOUBLE-COUNTS attesters — badly
    -- enough that it yielded attester counts above the committee size, i.e. a
    -- NEGATIVE missed-attestation rate. That bug is why `validate_panel()` in
    -- run_study.py now refuses to report a rate outside [0,1].
    --
    -- The naive fix (UNNEST to one row per (slot, validator)) explodes a single
    -- day into ~200M rows and runs the host out of disk. We don't need it.
    --
    -- The key fact: a validator signs ONE attestation with ONE beacon_block_root.
    -- Inclusion can be duplicated; the VOTE cannot. So the set of validators who
    -- voted root R at slot s is exactly the distinct union of the `validators`
    -- arrays over rows with that root — computable with list ops, per slot, with
    -- no explosion. `flatten(list(...))` gathers the arrays for a slot;
    -- `list_distinct` collapses the duplicate inclusions.
    -- De-duplicate at the (slot, committee, voted root) grain — NOT the slot
    -- grain. This matters enormously for memory: flattening every attester in a
    -- slot into one list builds a ~30k-element list per slot and OOM-killed the
    -- host. A committee is bounded (~500 validators), so grouping one level
    -- finer keeps every intermediate list small.
    --
    -- Summing these group counts up to the slot is EXACT, because the groups
    -- partition the attesters:
    --   * committees within a slot are disjoint by construction, and
    --   * a validator signs ONE attestation with ONE root, so it appears under
    --     exactly one (committee, root) group.
    -- Only the *inclusion* of that attestation is duplicated (across forks and
    -- across blocks), and list_distinct collapses precisely that.
    CREATE OR REPLACE TEMP TABLE att_grp AS
    SELECT slot,
           committee_index,
           lower(hex(beacon_block_root))                   AS voted_root,
           len(list_distinct(flatten(list(validators))))   AS n_v,
           -- Earliest inclusion is the one that defines the true distance;
           -- later re-inclusions in other blocks are duplicates.
           min(block_slot - slot)                          AS min_dist
    FROM att
    GROUP BY slot, committee_index, beacon_block_root;

    CREATE OR REPLACE TEMP TABLE att_agg AS
    SELECT g.slot,
           sum(g.n_v)                                              AS n_attested,
           sum(g.n_v) FILTER (WHERE g.voted_root = c.root)         AS n_correct_head,
           sum(g.n_v * g.min_dist)::DOUBLE
               / nullif(sum(g.n_v), 0)                             AS mean_inclusion_distance
    FROM att_grp g
    JOIN canon c ON c.slot = g.slot
    GROUP BY g.slot;

    -- DENOMINATOR: who was actually *assigned* to attest at this slot.
    -- Attestations that never landed are invisible in att_agg; only the
    -- committee roster reveals them.
    --
    -- Same two-level trick: distinct within a committee (Xatu can repeat a row
    -- if two sentries both reported it), then sum across the disjoint committees.
    CREATE OR REPLACE TEMP TABLE comm_agg AS
    SELECT slot, sum(n_v) AS committee_size
    FROM (
        SELECT slot,
               committee_index,
               len(list_distinct(flatten(list(validators)))) AS n_v
        FROM {comm}
        GROUP BY slot, committee_index
    )
    GROUP BY slot;

    -- CONTROLS / INSTRUMENT CANDIDATES.
    -- Blob count and transaction count mechanically slow propagation (more
    -- bytes to push through gossip) while being driven by rollup/user demand
    -- rather than by anything about the randomly-assigned attesters. That makes
    -- them candidate instruments for arrival_ms. See src/estimators/iv.py.
    CREATE OR REPLACE TEMP TABLE payload AS
    SELECT slot,
           coalesce(execution_payload_blob_gas_used, 0) / 131072 AS blob_count,
           coalesce(execution_payload_transactions_count, 0)     AS tx_count,
           coalesce(execution_payload_transactions_total_bytes, 0) AS payload_bytes
    FROM {canon};

    -- FINAL PANEL. Left-join from proposer_duty so that slots with NO canonical
    -- block survive as missed proposals rather than vanishing.
    CREATE OR REPLACE TEMP TABLE slot_panel AS
    SELECT
        d.slot,
        d.epoch,
        d.proposer_validator_index                       AS proposer_index,
        (c.slot IS NULL)                                 AS missed_proposal,
        ar.arrival_ms,
        -- The decomposition: what the proposer chose vs what the network cost.
        ar.arrival_min_ms,                               -- ~ publication time
        ar.arrival_p90_ms,                               -- ~ fully propagated
        ar.prop_spread_ms,                               -- ~ TRANSIT (compressible)
        ar.n_sentries,
        (ar.arrival_ms > {ATTESTATION_DEADLINE_MS})      AS late_block,
        ar.arrival_ms - {ATTESTATION_DEADLINE_MS}        AS ms_past_deadline,
        -- Orphaned: a block was seen at this slot but never made the chain.
        coalesce(o.n_blocks_seen, 0)                     AS n_orphaned_blocks,
        (o.slot IS NOT NULL)                             AS had_orphan,
        -- Channel B keys on this: a block built and broadcast, then beaten in
        -- the fork choice. NOT the same as missed_proposal (never built).
        (o.slot IS NOT NULL)                             AS orphaned,
        p.blob_count,
        p.tx_count,
        p.payload_bytes,
        cm.committee_size,
        aa.n_attested,
        aa.n_correct_head,
        aa.mean_inclusion_distance,
        -- Rates. nullif guards the (rare) slot with an empty committee.
        aa.n_correct_head::DOUBLE / nullif(aa.n_attested, 0)          AS correct_head_rate,
        1.0 - (aa.n_attested::DOUBLE / nullif(cm.committee_size, 0))  AS missed_attestation_rate
    FROM {duty} d
    LEFT JOIN canon    c  ON c.slot  = d.slot
    LEFT JOIN arrival  ar ON ar.slot = d.slot
    LEFT JOIN att_agg  aa ON aa.slot = d.slot
    LEFT JOIN comm_agg cm ON cm.slot = d.slot
    LEFT JOIN payload  p  ON p.slot  = d.slot
    LEFT JOIN orphaned o  ON o.slot  = d.slot;
    """)
