"""
Xatu data access layer.

Xatu is ethPandaOps' public Ethereum beacon-chain dataset, published as plain
parquet over HTTPS with no authentication:

    https://data.ethpandaops.io/xatu/{network}/databases/default/{table}/{Y}/{M}/{D}.parquet

We read it *in place* with DuckDB's httpfs extension. DuckDB issues HTTP range
requests and prunes columns, so a query that touches 3 of a table's 40 columns
only downloads those 3 column chunks. That is the entire reason this study needs
no infrastructure: there is no ETL, no warehouse, no ingestion. We point SQL at
a URL.

A caveat that shapes the whole pipeline, learned by measurement rather than
assumption: `canonical_beacon_elaborated_attestation` is ~800MB/day, and almost
all of that mass sits in the `validators uint32[]` array column. Any query that
touches `validators` pays for it. So we aggregate *inside* the scan and cache the
small result — we never materialise raw attestation rows locally.

Tables this study uses
----------------------
beacon_api_eth_v1_events_block
    One row per (sentry node, block) sighting. `propagation_slot_start_diff` is
    the milliseconds between the slot's start and when THAT sentry first saw the
    block. Median across sentries is our network-level arrival time — the
    treatment variable.

canonical_beacon_block
    The canonical chain. Gives us, per slot, the true block root and proposer.
    A slot absent here but present in proposer_duty is a *missed proposal*.

canonical_beacon_elaborated_attestation
    Attestations with the aggregation bits already expanded into a `validators`
    array of validator indices. `slot` is the slot being attested to;
    `block_slot` is the slot of the block the attestation landed in, so
    inclusion_distance = block_slot - slot. `beacon_block_root` is the head the
    attester voted for — comparing it to the canonical root of `slot` tells us
    whether the attester saw the block in time.

canonical_beacon_committee
    The attestation duty roster: which validators were *assigned* to attest at
    each slot. This is the denominator. Without it you can only see attestations
    that landed, and cannot distinguish "validator was offline" from "validator
    attested but was never included".

canonical_beacon_proposer_duty
    Who was assigned to propose each slot, by both index and pubkey. Doubles as
    a free pubkey<->index bridge for operator labelling.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterator

import duckdb

from config import Config

# Xatu table names, centralised so a rename upstream is a one-line fix here.
T_EVENTS_BLOCK = "beacon_api_eth_v1_events_block"
T_CANON_BLOCK = "canonical_beacon_block"
T_ELAB_ATT = "canonical_beacon_elaborated_attestation"
T_COMMITTEE = "canonical_beacon_committee"
T_PROPOSER_DUTY = "canonical_beacon_proposer_duty"

# Beacon-chain constants (mainnet, post-Bellatrix). These are consensus-spec
# values, not tunables — changing them changes what the numbers *mean*.
SECONDS_PER_SLOT = 12
SLOTS_PER_EPOCH = 32

# The attestation deadline. Validators are specced to attest at 1/3 into the
# slot (4s), voting for whatever head they can see at that instant. A block that
# arrives after this cannot be voted for by an honest, on-time attester.
#
# This threshold is the identifying discontinuity of the whole study: it is a
# sharp, spec-mandated, exogenous cutoff in a continuous running variable
# (block arrival time). Blocks landing at 3.9s vs 4.1s are alike in every way
# except that one is votable and the other is not.
ATTESTATION_DEADLINE_MS = (SECONDS_PER_SLOT * 1000) // 3  # 4000ms


@dataclass(frozen=True)
class XatuPaths:
    """Builds Xatu parquet URLs. Frozen because a study run must be reproducible."""

    base_url: str

    def day(self, table: str, date: dt.date) -> str:
        """URL for one table-day.

        Note Xatu does NOT zero-pad month/day: it is `/2025/6/1.parquet`, not
        `/2025/06/01.parquet`. Zero-padding silently 404s.
        """
        return f"{self.base_url}/{table}/{date.year}/{date.month}/{date.day}.parquet"

    def days(self, table: str, start: dt.date, end: dt.date) -> list[str]:
        """URLs for an inclusive date range."""
        return [self.day(table, d) for d in daterange(start, end)]


def daterange(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    """Inclusive date iterator."""
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


# Genesis per network. Mainnet is the well-known constant; hoodi was DERIVED
# from Xatu data itself (min(slot_start_date_time) - min(slot)*12) rather than
# trusted from memory — do the same if adding a network. Sepolia likewise.
NETWORK_GENESIS = {
    "mainnet": 1_606_824_023,  # 2020-12-01 12:00:23 UTC
    "hoodi": 1_742_213_400,    # 2025-03-17, derived from canonical_beacon_block
    "sepolia": 1_655_733_600,
}


def slot_to_datetime(slot: int, genesis_unix: int = NETWORK_GENESIS["mainnet"]) -> dt.datetime:
    """Beacon slot -> UTC wall clock. Pass the right network's genesis —
    a hoodi slot number run through mainnet genesis is off by ~4.4 years."""
    return dt.datetime.fromtimestamp(
        genesis_unix + slot * SECONDS_PER_SLOT, tz=dt.timezone.utc
    )


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with httpfs loaded and resource limits applied.

    We deliberately do NOT enable the HTTP metadata cache. It caused
    `TProtocolException: Invalid data` on large remote parquet files during
    development — DuckDB reused stale footer metadata across range requests.
    Correctness beats the small speedup.
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute(f"SET memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute(f"SET threads={cfg.duckdb_threads}")

    # Deduplicating attestations means UNNESTing the `validators` arrays, which
    # explodes one day into ~200M (slot, validator) rows — far more than fits in
    # RAM on a 15GB box. Give DuckDB somewhere to spill so it degrades to disk
    # instead of dying with an out-of-memory error.
    tmp = cfg.cache_dir / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con
