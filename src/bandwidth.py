"""
Channel D: bandwidth savings from less redundant data transmission.

The mechanism
-------------
Gossipsub floods. Every node in a topic mesh receives the same block and blob
sidecars from several peers — the duplication factor is roughly the mesh degree
(D=8) attenuated by IDONTWANT/IHAVE gating. RLNC-coded pubsub (mump2p's design)
sends coded chunks instead of full copies: a node needs only ~1 "block's worth"
of innovative chunks plus a small overhead, then stops. The redundant copies —
which are most of gossip traffic — go away.

This channel is DIFFERENT IN KIND from A, B and C:

  * It is a COST saving (opex), not revenue. It does not touch consensus rewards.
  * It scales per NODE, not per validator. An operator with 47,000 validators on
    150 beacon nodes saves 150 nodes' worth, not 47,000 validators' worth.
  * It is ADDITIVE to A + max(B,C) — it spends no delay budget, so it does not
    participate in the B/C exclusivity.
  * Its dollar value depends almost entirely on WHERE you host. Cloud egress
    (~$0.09/GB on AWS) makes it real money; a Hetzner box with 20TB included
    makes it nearly free. We therefore price both and refuse to pick one.

Measured inputs (Xatu, mainnet 2025-06-15, 7,136 blocks)
--------------------------------------------------------
    block on the wire (snappy)  : 42.2 KB average
    blobs per block             : 3.81 average, ~131 KB each on the wire
    => gossip payload per slot  : ~542 KB before duplication

Parameters we CANNOT measure and therefore expose as assumptions
----------------------------------------------------------------
    DUP_GOSSIPSUB : how many copies of each message a node actually moves.
                    Literature on gossipsub duplication for large messages
                    reports ~3-6x with modern IDONTWANT; we default to 4 and
                    show sensitivity.
    DUP_RLNC      : residual overhead of the coded scheme (coding headers,
                    slightly-more-than-1 chunks needed). Default 1.2.
    Neither is a vendor number: Optimum's own bandwidth claims are testnet
    marketing, so we parameterise instead of quoting them.

One more honesty note: blob count is about to change under PeerDAS (Fusaka),
which re-architects blob distribution entirely. This channel's blob share should
be re-measured after that fork ships.
"""

from __future__ import annotations

from dataclasses import dataclass

# Measured payload per slot, on the wire.
BLOCK_WIRE_KB = 42.2
BLOBS_PER_BLOCK = 3.81
BLOB_WIRE_KB = 131.0  # 128KB blob + sidecar/framing overhead
PAYLOAD_PER_SLOT_KB = BLOCK_WIRE_KB + BLOBS_PER_BLOCK * BLOB_WIRE_KB  # ~541.3

SLOTS_PER_YEAR = 365 * 24 * 3600 // 12  # 2,628,000

# Assumption knobs — see module docstring. These are NOT measurements.
DUP_GOSSIPSUB = 4.0
DUP_RLNC = 1.2

# Egress pricing scenarios, $/GB. Clouds bill egress only (ingress is free), so
# we price the egress half of gossip traffic: in a symmetric mesh a node
# forwards roughly what it receives.
PRICE_CLOUD = 0.09      # AWS inter-region/internet egress
PRICE_MIDCLOUD = 0.05   # discounted / committed-use cloud
PRICE_BAREMETAL = 0.001  # Hetzner-class: ~€1/TB beyond generous included quota


@dataclass(frozen=True)
class BandwidthSaving:
    """Per-NODE annual saving."""

    gb_moved_gossipsub: float
    gb_moved_rlnc: float
    gb_saved: float
    usd_cloud: float
    usd_midcloud: float
    usd_baremetal: float


def per_node_saving(
    dup_gossipsub: float = DUP_GOSSIPSUB,
    dup_rlnc: float = DUP_RLNC,
) -> BandwidthSaving:
    """Annual egress saved by one beacon node switching gossip -> RLNC transport.

    egress/yr = slots/yr x payload x duplication_factor
    (the egress half of mesh traffic; ingress is free on every pricing model
    that matters).
    """
    payload_gb_yr = SLOTS_PER_YEAR * PAYLOAD_PER_SLOT_KB / 1024 / 1024

    gossip = payload_gb_yr * dup_gossipsub
    rlnc = payload_gb_yr * dup_rlnc
    saved = max(0.0, gossip - rlnc)

    return BandwidthSaving(
        gb_moved_gossipsub=gossip,
        gb_moved_rlnc=rlnc,
        gb_saved=saved,
        usd_cloud=saved * PRICE_CLOUD,
        usd_midcloud=saved * PRICE_MIDCLOUD,
        usd_baremetal=saved * PRICE_BAREMETAL,
    )


def nodes_estimate(validators: int, validators_per_node: int = 500) -> int:
    """Illustrative beacon-node count for a fleet.

    Operators do not publish node counts. Institutional setups run hundreds to
    thousands of keys per node, pulled the other way by redundancy, geo-spread
    and DVT. 500 keys/node is a stated, adjustable middle — not a fact. Floor of
    8 because nobody runs a serious fleet on fewer.
    """
    return max(8, round(validators / validators_per_node))
