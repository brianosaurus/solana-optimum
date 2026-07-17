"""Channel D (bandwidth savings) — pinning the arithmetic and the honesty rules."""
import pytest

from src.bandwidth import (
    DUP_GOSSIPSUB,
    DUP_RLNC,
    PAYLOAD_PER_SLOT_KB,
    SLOTS_PER_YEAR,
    nodes_estimate,
    per_node_saving,
)


def test_payload_matches_measured_inputs():
    # 42.2 block + 3.81 x 131 blobs = 541.3 KB/slot. If someone edits one input
    # they must consciously update the expectation.
    assert PAYLOAD_PER_SLOT_KB == pytest.approx(541.3, abs=0.5)


def test_saving_is_dup_minus_residual():
    s = per_node_saving()
    payload_gb = SLOTS_PER_YEAR * PAYLOAD_PER_SLOT_KB / 1024 / 1024
    assert s.gb_moved_gossipsub == pytest.approx(payload_gb * DUP_GOSSIPSUB)
    assert s.gb_saved == pytest.approx(payload_gb * (DUP_GOSSIPSUB - DUP_RLNC))


def test_no_negative_saving_if_rlnc_worse():
    # If the coded scheme somehow moved MORE data, the saving clamps to zero
    # rather than going negative and being "banked" as a cost elsewhere.
    s = per_node_saving(dup_gossipsub=1.0, dup_rlnc=2.0)
    assert s.gb_saved == 0.0
    assert s.usd_cloud == 0.0


def test_hosting_dominates_the_dollar_value():
    # The whole point of pricing three scenarios: on bare metal this channel is
    # nearly worthless, on cloud egress it is real money. Both must be visible.
    s = per_node_saving()
    assert s.usd_cloud > 50 * s.usd_baremetal


def test_nodes_estimate_floors_and_scales():
    assert nodes_estimate(100) == 8            # floor
    assert nodes_estimate(50_000) == 100       # 500 keys/node
    assert nodes_estimate(50_000, 250) == 200  # knob is honoured
