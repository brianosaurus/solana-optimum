#!/usr/bin/env python3
"""
Find validator operator groups by size, so the study can be run per-group.

    SOLANA_RPC_URL=... python find_groups.py

Prints a ranked table of operators (by validator count and total stake) and
writes data/identity_to_group.json for joining to the slot panel's `leader`.
"""

from __future__ import annotations

import json
import os

from src.sol_groups import (
    attach_names,
    attach_withdrawers,
    build_groups,
    get_validators,
    identity_to_group,
)


def main() -> None:
    rpc = os.getenv("SOLANA_RPC_URL", "")
    dd = os.getenv("SENTRY_DATA_DIR", "data")
    top = int(os.getenv("TOP", "40"))
    if not rpc:
        raise SystemExit("set SOLANA_RPC_URL")

    SLOTS_PER_EPOCH = 432_000
    print("fetching vote accounts ...", flush=True)
    vals = get_validators(rpc)
    print("fetching withdrawers (operator key) ...", flush=True)
    attach_withdrawers(rpc, vals)
    print("fetching validator names ...", flush=True)
    attach_names(rpc, vals)

    # Rank on real presence: drop unstaked/testnet/decommissioned validators.
    vals = [v for v in vals if v.stake_sol > 0]
    total_stake = sum(v.stake_sol for v in vals)
    groups = build_groups(vals)
    multi = [g for g in groups if len(g.identities) > 1]

    print(f"\n  {len(vals)} staked validators, {total_stake:,.0f} SOL")
    print(f"=== {len(groups)} operators ({len(multi)} run >1 validator) ===")
    print(f"{'rank':>4} {'#val':>5} {'stake_SOL':>14} {'%net':>6} {'lead/epoch':>10}  operator")
    cum = 0.0
    for i, g in enumerate(groups[:top], 1):
        pct = 100 * g.stake_sol / total_stake
        cum += pct
        lead = g.stake_sol / total_stake * SLOTS_PER_EPOCH  # expected leader slots/epoch
        name = g.name or f"(unnamed {g.key[:8]})"
        print(f"{i:>4} {len(g.identities):>5} {g.stake_sol:>14,.0f} {pct:>5.2f}% "
              f"{lead:>10,.0f}  {name}")
    print(f"\ntop {top} operators = {cum:.1f}% of stake. "
          f"'lead/epoch' = expected leader slots per ~2.4-day epoch — your per-group "
          f"sample size for Channel C / skips.")

    m = identity_to_group(groups)
    out = os.path.join(dd, "identity_to_group.json")
    with open(out, "w") as f:
        json.dump(m, f)
    print(f"\nwrote {out} ({len(m)} identities) — join to panel.leader for per-group analysis")


if __name__ == "__main__":
    main()
