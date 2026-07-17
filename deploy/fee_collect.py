#!/usr/bin/env python3
"""
Leader-fee collector — Channel C' (priority + base fees), the other delay-budget
revenue source (SOLANA_PORT.md §8; the priority-fee channel).

Priority fees are probably the LARGER performance-sensitive pool than Jito tips,
and they respond to the same mechanism (seal later → include more late high-fee
txs) plus a second lever (better connectivity → receive more orderflow to pack).

Leanest capture: the block-meta stream carries the leader's `Fee` reward per
block — one message per block, not per transaction. That reward's lamports are
the validator's total fee income for the block (post-SIMD-0096: 100% of priority
fees + 50% of base fees), and its `pubkey` is the leader identity (a free
cross-check on the leader schedule). Priority fees dominate this total in
congestion, so we log the total and label it base+priority.

Join to the sentry (sealing lateness) and jito (tips) panels on SLOT.
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

STUBS_DIR = os.getenv("GEYSER_STUBS_DIR", "/home/ubuntu/cointegration_scanner")
sys.path.insert(0, STUBS_DIR)
import geyser_pb2 as pb  # noqa: E402
import geyser_pb2_grpc as pb_grpc  # noqa: E402
import solana_storage_pb2 as sto  # noqa: E402

FEE_REWARD = sto.RewardType.Value("Fee")

CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if " #" in v:
            v = v.split(" #", 1)[0].strip()
        os.environ.setdefault(k, v)


def _created_ms(update) -> float | None:
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
    req.blocks_meta["fees"].SetInParent()
    req.commitment = pb.CommitmentLevel.CONFIRMED
    return req


async def _request_iter(req: pb.SubscribeRequest):
    yield req
    ping = pb.SubscribeRequest()
    ping.ping.id = 1
    while True:
        await asyncio.sleep(30)
        yield ping


def _writer(data_dir: Path, name: str):
    state = {"day": None, "fh": None, "n": 0}

    def _rotate(now_utc: datetime):
        day = now_utc.strftime("%Y%m%d")
        if state["day"] != day:
            if state["fh"]:
                state["fh"].flush()
                state["fh"].close()
            path = data_dir / f"fees_{name}_{day}.jsonl"
            state["fh"] = open(path, "a", buffering=1)
            state["day"] = day
            print(f"[fees] writing {path}", flush=True)

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
    n = 0
    lamports = 0

    while True:
        try:
            print(f"[fees] connecting {endpoint} (insecure={insecure})", flush=True)
            async with _open_channel(endpoint, insecure) as channel:
                stub = pb_grpc.GeyserStub(channel)
                stream = stub.Subscribe(_request_iter(_build_request()), metadata=metadata)
                backoff = 1.0
                async for update in stream:
                    recv_ms = time.time() * 1000.0
                    if not update.HasField("block_meta"):
                        continue
                    bm = update.block_meta
                    fee = 0
                    leader = ""
                    for r in bm.rewards.rewards:
                        if r.reward_type == FEE_REWARD:
                            fee += int(r.lamports)
                            leader = r.pubkey
                    write({
                        "src": name,
                        "slot": bm.slot,
                        "fee_lamports": fee,
                        "leader": leader,
                        "tx_count": int(bm.executed_transaction_count),
                        "srv_ms": _created_ms(update),
                        "recv_ms": recv_ms,
                    })
                    n += 1
                    lamports += fee
                    now = time.time()
                    if now - last_report >= 60:
                        print(f"[fees] {n} blocks, {lamports/1e9:.2f} SOL fees, "
                              f"latest slot {bm.slot}", flush=True)
                        last_report = now
        except asyncio.CancelledError:
            print("[fees] cancelled, flushing", flush=True)
            if write.state["fh"]:
                write.state["fh"].flush()
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[fees] stream error: {e!r}; reconnecting in {backoff:.0f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main() -> None:
    here = Path(__file__).resolve().parent
    _load_env(here.parent / ".env")
    _load_env(Path.home() / "solana-optimum" / ".env")

    endpoint = os.getenv("FEE_ENDPOINT") or os.getenv("GRPC_ENDPOINT", "")
    token = os.getenv("GRPC_TOKEN", "")
    insecure_env = os.getenv("FEE_INSECURE")
    insecure = (insecure_env == "1") if insecure_env is not None else endpoint.endswith(":2096")
    data_dir = Path(os.getenv("SENTRY_DATA_DIR", str(here.parent / "data")))
    name = os.getenv("FEE_NAME", "titan")

    if not endpoint:
        sys.exit("no GRPC_ENDPOINT / FEE_ENDPOINT configured")

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
