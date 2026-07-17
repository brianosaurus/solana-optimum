"""
Tests for the Xatu access layer.

Mostly guarding against silent-404 traps: Xatu's URL scheme has a
non-obvious quirk (unpadded month/day) that produces a 404 rather than an error
if you get it wrong, and a 404 mid-study looks like "no data that day".
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.xatu import (
    ATTESTATION_DEADLINE_MS,
    SECONDS_PER_SLOT,
    SLOTS_PER_EPOCH,
    XatuPaths,
    daterange,
    slot_to_datetime,
)

BASE = "https://data.ethpandaops.io/xatu/mainnet/databases/default"


def test_url_does_not_zero_pad_month_and_day():
    """THE trap. Xatu serves /2025/6/1.parquet — /2025/06/01.parquet is a 404."""
    p = XatuPaths(BASE)
    url = p.day("canonical_beacon_block", dt.date(2025, 6, 1))
    assert url == f"{BASE}/canonical_beacon_block/2025/6/1.parquet"
    assert "/06/" not in url
    assert "/01." not in url


def test_url_double_digit_dates_are_untouched():
    p = XatuPaths(BASE)
    url = p.day("canonical_beacon_block", dt.date(2025, 12, 25))
    assert url.endswith("/2025/12/25.parquet")


def test_daterange_is_inclusive_of_both_ends():
    days = list(daterange(dt.date(2025, 6, 1), dt.date(2025, 6, 3)))
    assert days == [dt.date(2025, 6, 1), dt.date(2025, 6, 2), dt.date(2025, 6, 3)]


def test_daterange_single_day():
    assert list(daterange(dt.date(2025, 6, 1), dt.date(2025, 6, 1))) == [
        dt.date(2025, 6, 1)
    ]


def test_daterange_rejects_reversed_range():
    with pytest.raises(ValueError):
        list(daterange(dt.date(2025, 6, 3), dt.date(2025, 6, 1)))


def test_days_spans_month_boundary():
    p = XatuPaths(BASE)
    urls = p.days("canonical_beacon_block", dt.date(2025, 6, 29), dt.date(2025, 7, 2))
    assert len(urls) == 4
    assert urls[0].endswith("/2025/6/29.parquet")
    assert urls[-1].endswith("/2025/7/2.parquet")


def test_attestation_deadline_is_one_third_of_slot():
    """4000ms. This constant IS the study's identifying threshold — if it were
    wrong, the RD would be centred on the wrong cutoff and estimate nothing."""
    assert ATTESTATION_DEADLINE_MS == 4000
    assert ATTESTATION_DEADLINE_MS == (SECONDS_PER_SLOT * 1000) // 3


def test_slot_to_datetime_at_genesis():
    # Mainnet beacon genesis: 2020-12-01 12:00:23 UTC.
    assert slot_to_datetime(0) == dt.datetime(
        2020, 12, 1, 12, 0, 23, tzinfo=dt.timezone.utc
    )


def test_slot_to_datetime_advances_12s_per_slot():
    a = slot_to_datetime(1_000_000)
    b = slot_to_datetime(1_000_001)
    assert (b - a).total_seconds() == SECONDS_PER_SLOT


def test_epoch_geometry():
    assert SLOTS_PER_EPOCH == 32
    assert SECONDS_PER_SLOT * SLOTS_PER_EPOCH == 384  # 6.4 min per epoch
