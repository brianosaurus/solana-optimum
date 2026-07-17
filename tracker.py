#!/usr/bin/env python3
"""
Realtime Ethereum mainnet slot tracker.

Polls a public beacon node's head, times each block's arrival relative to its
slot start, prices the counterfactual uplift for a set of operators, and writes
it to SQLite for the watcher dashboard to read.

    python tracker.py                 # run forever
    python tracker.py --once          # single poll (for testing)

WHAT THIS IS NOT
----------------
It is not a record of money anyone is making. mump2p is not deployed on Ethereum
mainnet — see FINDINGS.md. Every figure here is a MODELLED COUNTERFACTUAL: what
an operator WOULD gain if a propagation accelerator delivering the stated speedup
were running, AND (for the dominant MEV channel) if they re-tuned their timing
games to spend the saved milliseconds. The dashboard labels it as such, and so
does this file, because a number that climbs on a public page is very easily
mistaken for revenue.

Design
------
We poll `/eth/v1/beacon/headers/head` rapidly. When the head advances to a new
slot, the moment we first see it is our arrival timestamp. `arrival_ms` is then
(first_seen - slot_start).

The honest caveats (also in src/live.py):
  * One vantage point, so arrival is noisier and later than Xatu's
    many-sentry median.
  * Poll interval bounds our resolution. At 250ms we cannot resolve better
    than that, which is coarse next to a 4000ms deadline but adequate for
    classifying late vs on-time.
  * Transit time is not observable from one vantage (it needs the SPREAD across
    sentries), so it comes from the calibrated batch study.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import load_config
from src.live import Calibration, SlotGain, price_slot, slot_start_unix
from src.operators import COMBINED_VALIDATORS, OPERATORS
from src.revenue import RevenueModel

BEACON_URL = "https://ethereum-beacon-api.publicnode.com"
HEAD_ENDPOINT = "/eth/v1/beacon/headers/head"

POLL_INTERVAL_S = 0.25  # bounds our arrival-time resolution
HTTP_TIMEOUT_S = 4.0

# The speedups we track concurrently, so the dashboard can show the vendor claim
# next to sober haircuts rather than only the flattering number.
SPEEDUPS = [6.0, 3.0, 2.0]

# Transitions to discard at startup before we trust an arrival time. See the
# warm-up block in main() for the two artifacts this defends against.
WARMUP_SLOTS = 2

DDL = """
CREATE TABLE IF NOT EXISTS slots (
    slot          INTEGER PRIMARY KEY,
    seen_unix     REAL NOT NULL,
    arrival_ms    REAL NOT NULL,
    late          INTEGER NOT NULL,
    proposer_index INTEGER
);
CREATE TABLE IF NOT EXISTS gains (
    slot        INTEGER NOT NULL,
    operator    TEXT NOT NULL,
    validators  INTEGER NOT NULL,
    speedup     REAL NOT NULL,
    eth_a       REAL NOT NULL,
    eth_b       REAL NOT NULL,
    eth_c       REAL NOT NULL,
    eth_total   REAL NOT NULL,
    PRIMARY KEY (slot, operator, speedup)
);
CREATE INDEX IF NOT EXISTS idx_gains_op ON gains(operator, speedup);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


# --- Clock ------------------------------------------------------------------
#
# We CANNOT trust the local system clock. The deployment host was found running
# ~94 seconds behind real time with NTP disabled entirely
# ("System clock synchronized: no"). Since arrival_ms is
# (now - slot_start), a 94s error makes every arrival time nonsense — in
# testing it produced arrivals of -91,000 ms, i.e. blocks appearing to arrive a
# minute and a half BEFORE their own slot began.
#
# We deliberately do NOT fix this by stepping the system clock. That host also
# runs live trading bots, and yanking the wall clock by 94 seconds underneath a
# process that is signing time-sensitive transactions is an excellent way to
# break something expensive.
#
# Instead we query NTP ourselves over UDP, compute the OFFSET, and correct our
# own timestamps in-process. Nothing outside this program is affected.

NTP_SERVER = "pool.ntp.org"
NTP_PORT = 123
NTP_UNIX_EPOCH_DELTA = 2_208_988_800  # NTP epoch (1900) -> Unix epoch (1970)
NTP_REFRESH_S = 900.0  # re-measure the offset every 15 min to catch drift


def ntp_offset(server: str = NTP_SERVER, timeout: float = 5.0) -> float | None:
    """Seconds to ADD to the local clock to get true time. None if unreachable.

    Plain SNTP: a 48-byte packet, mode 3 (client), version 3. We use the
    round-trip midpoint as the local reference, which is the standard estimator
    and is accurate to well under the ~250ms we actually need.
    """
    import socket
    import struct

    pkt = b"\x1b" + 47 * b"\0"  # LI=0, VN=3, Mode=3
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(timeout)
        t0 = time.time()
        s.sendto(pkt, (server, NTP_PORT))
        data, _ = s.recvfrom(48)
        t3 = time.time()
    except (OSError, socket.timeout):
        return None
    finally:
        s.close()

    if len(data) < 48:
        return None

    # Transmit timestamp sits at byte 40 (seconds) / 44 (fraction).
    secs, frac = struct.unpack("!II", data[40:48])
    server_time = (secs - NTP_UNIX_EPOCH_DELTA) + frac / 2**32

    # Midpoint of our send/receive brackets the server's reading.
    local_mid = (t0 + t3) / 2.0
    return server_time - local_mid


class Clock:
    """True time, independent of a possibly-broken system clock."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.synced = False
        self._last_sync = 0.0
        self.resync()

    def resync(self) -> None:
        off = ntp_offset()
        if off is None:
            # Keep the last known offset rather than silently reverting to a
            # clock we know is wrong.
            print("ntp: unreachable, keeping previous offset "
                  f"{self.offset:+.3f}s", file=sys.stderr)
            return
        self.offset = off
        self.synced = True
        self._last_sync = time.time()
        print(f"ntp: offset {off:+.3f}s "
              f"({'local clock is fine' if abs(off) < 1 else 'LOCAL CLOCK IS WRONG'})",
              file=sys.stderr)

    def now(self) -> float:
        if time.time() - self._last_sync > NTP_REFRESH_S:
            self.resync()
        return time.time() + self.offset


def db_connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=10)
    # WAL so the FastAPI reader never blocks the writer (and vice versa).
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(DDL)
    return con


def fetch_head() -> tuple[int, int] | None:
    """Return (slot, proposer_index) of the current head, or None on failure.

    A transient failure must never kill the tracker — public endpoints rate-limit
    and blip. We simply miss that poll and try again.
    """
    try:
        req = urllib.request.Request(
            BEACON_URL + HEAD_ENDPOINT,
            headers={"Accept": "application/json", "User-Agent": "optimum-tracker"},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
            d = json.loads(r.read())
        msg = d["data"]["header"]["message"]
        return int(msg["slot"]), int(msg["proposer_index"])
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError, OSError):
        return None


def record_slot(
    con: sqlite3.Connection,
    slot: int,
    seen_unix: float,
    proposer_index: int,
    rev: RevenueModel,
    cal: Calibration,
) -> None:
    """Price a newly-observed slot for every operator x speedup and persist it."""
    arrival_ms = (seen_unix - slot_start_unix(slot)) * 1000.0

    # Guard: a block "arriving" before its slot began, or more than a slot late,
    # means our clock or the endpoint is lying. Drop it rather than let a
    # nonsense arrival poison the accruals.
    if arrival_ms < 0 or arrival_ms > 12_000:
        return

    late = arrival_ms > 4000
    con.execute(
        "INSERT OR IGNORE INTO slots(slot, seen_unix, arrival_ms, late, proposer_index)"
        " VALUES (?,?,?,?,?)",
        (slot, seen_unix, arrival_ms, int(late), proposer_index),
    )

    rows = []
    targets = [(o.name, o.validators) for o in OPERATORS]
    targets.append(("ALL SEVEN", COMBINED_VALIDATORS))
    # A single validator, so the dashboard can show the per-validator unit too.
    targets.append(("per validator", 1))

    for name, n in targets:
        for sp in SPEEDUPS:
            g: SlotGain = price_slot(slot, arrival_ms, sp, n, rev, cal)
            rows.append((slot, name, n, sp, g.eth_a, g.eth_b, g.eth_c, g.eth_total))

    con.executemany(
        "INSERT OR IGNORE INTO gains"
        "(slot, operator, validators, speedup, eth_a, eth_b, eth_c, eth_total)"
        " VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="single poll then exit")
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config()
    db_path = args.db or (cfg.cache_dir / "optimum_live.db")
    con = db_connect(db_path)

    rev = RevenueModel()
    cal = Calibration.load(cfg.cache_dir / "calibration.json")

    # True time, independent of the host clock (which on frankfurt was found
    # ~94s behind with NTP disabled). Every arrival time below uses clock.now(),
    # never time.time().
    clock = Clock()
    if not clock.synced:
        print("REFUSING TO START: no NTP reference. Arrival times would be "
              "meaningless without a trustworthy clock.", file=sys.stderr)
        return 1
    if abs(clock.offset) > 1.0:
        print(f"WARNING: host clock is {clock.offset:+.1f}s off true time. "
              "Correcting in-process; the system clock is NOT being touched "
              "(other services on this host depend on it).", file=sys.stderr)

    con.execute(
        "INSERT OR REPLACE INTO meta(k,v) VALUES ('eth_price_usd', ?)",
        (str(cfg.eth_price_usd),),
    )
    con.commit()

    print(f"tracker -> {db_path}", file=sys.stderr)
    print(f"  beacon      : {BEACON_URL}", file=sys.stderr)
    print(f"  speedups    : {SPEEDUPS}", file=sys.stderr)
    print(f"  transit(cal): {cal.mean_transit_ms:.0f} ms", file=sys.stderr)
    print(f"  dV/dt (cal) : {cal.dv_dt_eth_per_ms:.3e} ETH/ms", file=sys.stderr)

    last_slot = None
    warmed = 0
    consecutive_failures = 0

    while True:
        head = fetch_head()
        now = clock.now()   # NTP-corrected; never the raw system clock

        if head is None:
            consecutive_failures += 1
            # Back off politely rather than hammering a struggling endpoint, but
            # never give up — the tracker is meant to run for months.
            time.sleep(min(30.0, POLL_INTERVAL_S * (2 ** min(consecutive_failures, 7))))
            if args.once:
                print("head fetch failed", file=sys.stderr)
                return 1
            continue

        consecutive_failures = 0
        slot, proposer = head

        # WARM-UP. A slot is only timeable if we observed the TRANSITION into it.
        # Two distinct startup artifacts bite here, and both manufacture fake
        # "LATE" blocks that would fire Channel A for free:
        #
        #  1. The slot in progress when we start. We have no idea when it first
        #     became visible — only when we happened to begin polling. (Measured:
        #     8,086ms for a perfectly ordinary block.)
        #  2. The FIRST transition we see. The public endpoint's head can itself
        #     lag, so our first observed transition may fire long after the block
        #     really appeared. (Measured: 9,712ms, against a ~2,700ms norm.)
        #
        # So we discard the first WARMUP_SLOTS transitions and only start
        # accruing once our polling is genuinely locked on to the head.
        if warmed < WARMUP_SLOTS:
            if last_slot is None or slot > last_slot:
                warmed += 1
                print(f"warm-up {warmed}/{WARMUP_SLOTS}: skipping slot {slot} "
                      f"(arrival not yet trustworthy)", file=sys.stderr)
                last_slot = slot
            if args.once:
                return 0
            time.sleep(POLL_INTERVAL_S)
            continue

        if slot > last_slot:
            record_slot(con, slot, now, proposer, rev, cal)
            arrival = (now - slot_start_unix(slot)) * 1000.0
            flag = "LATE" if arrival > 4000 else ""
            print(f"slot {slot}  arrival {arrival:7.0f} ms  {flag}", file=sys.stderr)
            last_slot = slot

        if args.once:
            return 0

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
