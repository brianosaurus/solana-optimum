"""
Solana data access layer — the analog of `src/xatu.py`.

Where the Ethereum study points DuckDB at free ethPandaOps parquet, Solana has no
equivalent public dataset of per-node block sighting times (see SOLANA_PORT.md
§3). This module therefore does two things:

  1. Wraps the JSON-RPC endpoints that DO give us a complete OUTCOME panel and a
     PROXY treatment for free (vote latency, skip rate, rewards, leader schedule).
  2. Centralises the Solana consensus constants — the Timely Vote Credit ramp in
     particular, which replaces Ethereum's 4-second attestation deadline as the
     study's identifying feature (a KINK, not a cliff → RKD, not RDD).

It deliberately holds no keypair and signs nothing: like the Ethereum repo, this
is read-only analysis of public data. Any RPC provider works (public mainnet-beta,
Helius, Triton, QuickNode); a paid endpoint is strongly advised because walking a
whole epoch of blocks hammers rate limits.

CAVEAT that shapes the whole port: vote latency conflates transit ("the block was
slow to reach me") with publication ("my vote was slow to land"). It is a good
running variable for the credit-revenue RKD, but it is NOT clean transit. A clean
transit counterfactual needs a self-run shred-arrival fleet (SOLANA_PORT.md §3
Path B). This module is the Path A (free, on-chain) foundation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

# --- Consensus constants (agave/mainnet-beta). Verify against current agave -----
# --- source before publishing; these are spec values, not tunables. ------------

# Nominal slot time. Real slot time drifts (skips, congestion); for wall-clock
# use getBlockTime rather than assuming 400ms.
MS_PER_SLOT_NOMINAL = 400
SLOTS_PER_EPOCH = 432_000
# ~78.8M slots/year at the nominal rate; real number is lower due to skips.
SLOTS_PER_YEAR_NOMINAL = (365 * 24 * 3600 * 1000) // MS_PER_SLOT_NOMINAL
NUM_CONSECUTIVE_LEADER_SLOTS = 4

# TIMELY VOTE CREDITS (SIMD-0033) — the identifying feature of the Solana port.
#
# A vote for slot v that lands in a block at slot L has latency = L - v. The
# validator earns:
#
#     credit(latency) = max(CREDITS_FLOOR, MAX_CREDITS - max(0, latency - GRACE))
#
# So latency <= 2 earns the full 16; each extra slot costs one credit down to a
# floor of 1 reached at latency >= 17. Inflation rewards are distributed in
# proportion to credits earned, so this schedule is LITERALLY the price of
# latency in SOL — and its KINK at latency = GRACE_SLOTS is where the Regression
# Kink Design identifies off (the heir to rdd.py's 4000ms cutoff).
VOTE_CREDITS_MAXIMUM = 16
VOTE_CREDITS_GRACE_SLOTS = 2
VOTE_CREDITS_FLOOR = 1

# The on-chain vote program. Vote transactions carry the (voted slots, hash) we
# parse latency from.
VOTE_PROGRAM_ID = "Vote111111111111111111111111111111111111111"


def vote_credit(latency_slots: int) -> int:
    """Credits earned by a vote that landed `latency_slots` after the voted slot.

    This is the Solana analog of Ethereum's binary timely-head reward, except it
    is a graded ramp rather than an all-or-nothing cliff — which is exactly why
    the Solana identification is an RKD (kink in a continuous reward) rather than
    a sharp RD (jump at a deadline).
    """
    if latency_slots < 0:
        raise ValueError(f"negative vote latency {latency_slots}")
    penalty = max(0, latency_slots - VOTE_CREDITS_GRACE_SLOTS)
    return max(VOTE_CREDITS_FLOOR, VOTE_CREDITS_MAXIMUM - penalty)


def slot_to_epoch(slot: int) -> int:
    """Epoch containing a slot. (Mainnet-beta ran a shorter warmup epoch schedule
    at genesis; for any modern slot the fixed 432k length holds.)"""
    return slot // SLOTS_PER_EPOCH


def epoch_slot_range(epoch: int) -> tuple[int, int]:
    """Inclusive [first, last] slot of an epoch under the steady-state schedule."""
    first = epoch * SLOTS_PER_EPOCH
    return first, first + SLOTS_PER_EPOCH - 1


@dataclass(frozen=True)
class SolanaRPC:
    """Minimal JSON-RPC client. Frozen so a run cannot mutate its own endpoint.

    Only the read-only methods the panel needs are wrapped; everything goes
    through `call()`, so adding one is a two-line method.
    """

    url: str
    timeout_s: float = 30.0
    max_retries: int = 5
    # Politeness / rate-limit backoff between retries, seconds.
    backoff_s: float = 0.5

    def call(self, method: str, params: list[Any] | None = None) -> Any:
        """One JSON-RPC call with retry on transient failure.

        Note: NOT using Date/random for jitter — a fixed backoff keeps runs
        reproducible, matching the Ethereum repo's determinism stance.
        """
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                body = r.json()
                if "error" in body:
                    # 429 / node-behind are retryable; a bad request is not.
                    code = body["error"].get("code")
                    if code in (-32005, 429, -32004) and attempt < self.max_retries - 1:
                        time.sleep(self.backoff_s * (attempt + 1))
                        continue
                    raise RuntimeError(f"RPC {method} error: {body['error']}")
                return body["result"]
            except (requests.RequestException, ValueError) as e:  # noqa: PERF203
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_s * (attempt + 1))
                    continue
        raise RuntimeError(f"RPC {method} failed after {self.max_retries} tries: {last_err}")

    # --- Outcome-panel sources (Path A: free, on-chain) -----------------------

    def get_epoch_info(self) -> dict:
        """Current epoch, slot index, absolute slot. Start here to pick a range."""
        return self.call("getEpochInfo")

    def get_leader_schedule(self, slot: int | None = None) -> dict[str, list[int]]:
        """Identity pubkey -> list of leader slot INDICES (relative to epoch start)
        for the epoch containing `slot`. The proposer duty roster — deterministic
        and known an epoch ahead, unlike Ethereum's RANDAO."""
        params: list[Any] = [slot] if slot is not None else []
        return self.call("getLeaderSchedule", params + ([{"identity": None}] if False else []))

    def get_block_production(
        self, first_slot: int | None = None, last_slot: int | None = None
    ) -> dict:
        """Per-identity (leaderSlots, blocksProduced) over a slot range.

        skip_rate = 1 - blocksProduced / leaderSlots. This is the Solana
        `missed_proposal` outcome, and unlike Ethereum it comes pre-aggregated —
        no need to anti-join a canonical table.
        """
        cfg: dict[str, Any] = {}
        if first_slot is not None:
            rng: dict[str, Any] = {"firstSlot": first_slot}
            if last_slot is not None:
                rng["lastSlot"] = last_slot
            cfg["range"] = rng
        return self.call("getBlockProduction", [cfg] if cfg else [])

    def get_blocks(self, start_slot: int, end_slot: int) -> list[int]:
        """Rooted slots in [start, end]. Gaps vs the leader schedule = skips."""
        return self.call("getBlocks", [start_slot, end_slot])

    def get_block(self, slot: int, *, full: bool = True) -> dict | None:
        """One block. `full` keeps transactions so vote latency can be parsed;
        pass full=False for a cheap header+rewards read.

        Returns None for a skipped slot (RPC raises "slot skipped"); the caller
        treats None as the missed-proposal signal.
        """
        cfg = {
            "encoding": "json",
            "maxSupportedTransactionVersion": 0,
            "transactionDetails": "full" if full else "none",
            "rewards": True,
        }
        try:
            return self.call("getBlock", [slot, cfg])
        except RuntimeError as e:
            if "skipped" in str(e).lower() or "not available" in str(e).lower():
                return None
            raise

    def get_inflation_reward(self, addresses: list[str], epoch: int) -> list[dict | None]:
        """Per-validator inflation reward for one epoch. The SOL that vote
        credits translate into — the revenue side of the RKD."""
        return self.call("getInflationReward", [addresses, {"epoch": epoch}])

    def get_vote_accounts(self) -> dict:
        """Current + delinquent vote accounts, each with `epochCredits` and stake.
        The credit denominator and the stake weights for revenue."""
        return self.call("getVoteAccounts")


@dataclass(frozen=True)
class VoteObservation:
    """One (validator, voted_slot) vote landing — the row the RKD consumes.

    latency = landed_slot - voted_slot is the running variable; `credit` is the
    outcome (SOL, up to the inflation conversion). Parsed from the vote-program
    instructions inside a block's transactions.
    """

    vote_pubkey: str
    voted_slot: int
    landed_slot: int

    @property
    def latency(self) -> int:
        return self.landed_slot - self.voted_slot

    @property
    def credit(self) -> int:
        return vote_credit(self.latency)


def parse_vote_latencies(block: dict, landed_slot: int) -> Iterator[VoteObservation]:
    """Extract vote latencies from one full block.

    A vote transaction's vote-program instruction (Vote / VoteStateUpdate /
    TowerSync / their compact forms) lists the slots being voted on. The block it
    lands in is `landed_slot`. For each voted slot we emit latency = landed - voted.

    This is the Solana heir to Xatu's `canonical_beacon_elaborated_attestation`:
    the fat table of the study. One epoch is ~432k blocks, so at scale you index
    this with Helius/Flipside/Dune rather than walking getBlock yourself — exactly
    the "aggregate inside the scan, cache the small result" discipline xatu.py
    describes.

    NOTE: this parses the human-readable `parsed` JSON shape when the RPC returns
    it (jsonParsed encoding); with plain `json` encoding the vote instruction data
    is base58 and must be decoded against the vote-program layout. Kept as the
    integration point rather than a finished decoder because the exact instruction
    variant in use (TowerSync post-SIMD-0326) should be pinned at implementation
    time against the agave version you target.
    """
    txs = (block or {}).get("transactions", [])
    for tx in txs:
        msg = tx.get("transaction", {}).get("message", {})
        for ix in msg.get("instructions", []):
            parsed = ix.get("parsed")
            if not isinstance(parsed, dict) or parsed.get("type", "").lower().find("vote") < 0:
                continue
            info = parsed.get("info", {})
            vote_pubkey = info.get("voteAccount") or info.get("voteAccount1") or ""
            # Different vote-instruction variants expose the slots differently;
            # cover the common shapes.
            slots = []
            if "vote" in info and isinstance(info["vote"], dict):
                slots = info["vote"].get("slots", [])
            elif "voteStateUpdate" in info:
                lockouts = info["voteStateUpdate"].get("lockouts", [])
                slots = [lo.get("slot") for lo in lockouts if lo.get("slot") is not None]
            elif "hash" in info and "slots" in info:
                slots = info["slots"]
            for v in slots:
                if v is not None and landed_slot >= v:
                    yield VoteObservation(vote_pubkey, int(v), int(landed_slot))
