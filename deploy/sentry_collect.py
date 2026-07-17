#!/usr/bin/env python3
"""
Solana propagation sentry — the Xatu-for-Solana capture layer (SOLANA_PORT.md §3 Path B).

Subscribes to a Yellowstone Geyser gRPC endpoint's SLOT-STATUS stream and logs,
per slot, the server-side `created_at` timestamp of every status transition:

    SLOT_FIRST_SHRED_RECEIVED  <- first Turbine shred of the slot reached the node
    SLOT_COMPLETED             <- all shreds received (block fully downloaded)
    SLOT_PROCESSED             <- bank replayed
    SLOT_CONFIRMED / FINALIZED <- consensus milestones
    SLOT_DEAD                  <- fork abandoned (dead slot)

Why this is the right primitive
-------------------------------
The two economically load-bearing quantities fall straight out of it:

  * PROPAGATION DURATION per slot = t(COMPLETED) - t(FIRST_SHRED_RECEIVED).
    Both timestamps are the SAME node's `created_at` clock, so delivery latency
    from HelloMoon's node to this sentry CANCELS in the difference — the interval
    is clean even though the vantage point is HelloMoon's, not ours.
  * SKIPS / FORKS from the `parent` field: parent < slot-1 means slots were
    skipped; SLOT_DEAD marks an abandoned fork. This is the Solana `missed_proposal`
    / orphan signal, live.

We log RAW timestamps only and derive everything downstream — the collector stays
dumb and hard to kill, which is what a data-collection process on a remote box
must be. Propagation timing cannot be backfilled: every minute not running is lost.

Caveat, stated plainly: this is ONE vantage point (HelloMoon's fleet). Absolute
"ms into slot" carries HelloMoon's clock; the COMPLETED-minus-FIRST_SHRED interval
does not. A transit-SPREAD estimate needs a second INDEPENDENT endpoint — run a
second instance against SHRED_ENDPOINT (elite-cache) or a non-HelloMoon provider.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import grpc

# The generated Yellowstone stubs live with the sibling bots on this box; reuse
# them (and their solana_storage_pb2 dependency) rather than re-vendoring.
STUBS_DIR = os.getenv("GEYSER_STUBS_DIR", "/home/ubuntu/cointegration_scanner")
sys.path.insert(0, STUBS_DIR)
import geyser_pb2 as pb  # noqa: E402
import geyser_pb2_grpc as pb_grpc  # noqa: E402

CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]

# Human-readable status names, resolved from the enum so a proto bump can't
# silently desync the labels.
STATUS_NAME = {v: k for k, v in pb.SlotStatus.items()}


def _load_env(path: Path) -> None:
    """Tiny .env reader — no dependency on python-dotenv on the remote box."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        # strip trailing inline comments on unquoted values
        if " #" in v:
            v = v.split(" #", 1)[0].strip()
        os.environ.setdefault(k, v)


def _created_ms(update) -> float | None:
    """Server-side event timestamp (ms) from SubscribeUpdate.created_at."""
    if not update.HasField("created_at"):
        return None
    ts = update.created_at
    return ts.seconds * 1000.0 + ts.nanos / 1e6


def _open_channel(endpoint: str, insecure: bool):
    if insecure:
        return grpc.aio.insecure_channel(endpoint, options=CHANNEL_OPTIONS)
    return grpc.aio.secure_channel(
        endpoint, grpc.ssl_channel_credentials(), options=CHANNEL_OPTIONS
    )


def _build_request() -> pb.SubscribeRequest:
    req = pb.SubscribeRequest()
    # interslot_updates=True is what surfaces FIRST_SHRED_RECEIVED / COMPLETED /
    # DEAD, not just the committed milestones. filter_by_commitment=False so we
    # receive ALL status transitions regardless of the request commitment.
    req.slots["sentry"].interslot_updates = True
    req.slots["sentry"].filter_by_commitment = False
    req.commitment = pb.CommitmentLevel.PROCESSED
    return req


async def _request_iter(req: pb.SubscribeRequest):
    """Yield the subscribe request, then keep the stream warm with periodic pings."""
    yield req
    ping = pb.SubscribeRequest()
    ping.ping.id = 1
    while True:
        await asyncio.sleep(30)
        yield ping


def _writer(data_dir: Path, name: str):
    """Daily-rotated JSONL writer. Returns a `write(dict)` closure.

    `name` labels the file so multiple sentries (distinct vantage points) write
    to distinct files: slots_<name>_<day>.jsonl.
    """
    state = {"day": None, "fh": None, "n": 0}

    def _rotate(now_utc: datetime):
        day = now_utc.strftime("%Y%m%d")
        if state["day"] != day:
            if state["fh"]:
                state["fh"].flush()
                state["fh"].close()
            path = data_dir / f"slots_{name}_{day}.jsonl"
            state["fh"] = open(path, "a", buffering=1)  # line-buffered
            state["day"] = day
            print(f"[sentry] writing {path}", flush=True)

    def write(row: dict):
        _rotate(datetime.now(timezone.utc))
        state["fh"].write(json.dumps(row, separators=(",", ":")) + "\n")
        state["n"] += 1

    write.state = state  # type: ignore[attr-defined]
    return write


async def run(endpoint: str, token: str, insecure: bool, data_dir: Path, name: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    write = _writer(data_dir, name)
    metadata = [("x-token", token)] if token else []
    backoff = 1.0
    last_report = time.time()
    seen = 0

    while True:
        try:
            print(f"[sentry] connecting {endpoint} (insecure={insecure})", flush=True)
            async with _open_channel(endpoint, insecure) as channel:
                stub = pb_grpc.GeyserStub(channel)
                stream = stub.Subscribe(_request_iter(_build_request()), metadata=metadata)
                backoff = 1.0  # reset on a good connection
                async for update in stream:
                    recv_ms = time.time() * 1000.0
                    if not update.HasField("slot"):
                        continue  # ignore pong/account/etc.
                    s = update.slot
                    write({
                        "src": name,
                        "slot": s.slot,
                        "parent": s.parent,
                        "status": STATUS_NAME.get(s.status, s.status),
                        "srv_ms": _created_ms(update),   # HelloMoon node clock
                        "recv_ms": recv_ms,               # this sentry's clock
                    })
                    seen += 1
                    now = time.time()
                    if now - last_report >= 60:
                        print(f"[sentry] {seen} slot-events, "
                              f"{write.state['n']} rows written, latest slot {s.slot}",
                              flush=True)
                        last_report = now
        except asyncio.CancelledError:
            print("[sentry] cancelled, flushing", flush=True)
            if write.state["fh"]:
                write.state["fh"].flush()
            raise
        except Exception as e:  # noqa: BLE001 — a sentry must survive everything
            print(f"[sentry] stream error: {e!r}; reconnecting in {backoff:.0f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main() -> None:
    here = Path(__file__).resolve().parent
    _load_env(here.parent / ".env")           # ~/solana-optimum/.env
    _load_env(Path.home() / "solana-optimum" / ".env")

    endpoint = os.getenv("SENTRY_ENDPOINT") or os.getenv("GRPC_ENDPOINT", "")
    token = os.getenv("GRPC_TOKEN", "")
    # :889 parallel-titan is TLS; :2096 elite-cache shred feed is plaintext.
    insecure_env = os.getenv("SENTRY_INSECURE")
    insecure = (insecure_env == "1") if insecure_env is not None else endpoint.endswith(":2096")
    data_dir = Path(os.getenv("SENTRY_DATA_DIR", str(here.parent / "data")))
    name = os.getenv("SENTRY_NAME", "titan")

    if not endpoint:
        sys.exit("no GRPC_ENDPOINT / SENTRY_ENDPOINT configured")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(run(endpoint, token, insecure, data_dir, name))
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
