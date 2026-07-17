#!/usr/bin/env python3
"""
Jito-tip collector — the Channel C outcome variable (SOLANA_PORT.md §8).

Subscribes to a Yellowstone Geyser TRANSACTIONS stream filtered to the 8 Jito tip
payment accounts, and logs, per tip-bearing transaction, the lamports that landed
on a tip account. Summed per slot this is the MEV tip the leader captured — the
on-chain value curve that the "delay accrues more tips" hypothesis is about.

Why balance deltas, not instruction parsing
--------------------------------------------
A tip can arrive as a plain SystemProgram transfer OR embedded inside a bundle's
instructions. The robust, parser-free measurement is the tip account's balance
change: post_balance - pre_balance over the transaction. Balances are indexed by
the FULL ordered account list — static message.account_keys first, then meta's
loaded_writable_addresses, then loaded_readonly_addresses (the order the runtime
uses) — so we reconstruct that list to map a tip account to its balance slot. A
positive delta on a tip account is the tip.

Commitment = CONFIRMED: we count tips in blocks that actually stuck, so a tip that
landed in a slot later abandoned as a dead fork is not counted as captured. Join
to the sentry panel on SLOT (not timestamp) to get (tips, sealing-lateness).

Caveat: the 8 tip accounts are the well-known Jito mainnet set (stable, but verify
against Jito's getTipAccounts if a discrepancy appears). This measures Jito-routed
MEV specifically — the dominant but not the only MEV channel on Solana.
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

# Canonical Jito mainnet tip payment accounts.
TIP_ACCOUNTS_B58 = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]

CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", 128 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 20_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58decode(s: str) -> bytes:
    """Minimal base58 decode (no dependency) — used once, for the tip accounts."""
    num = 0
    for ch in s:
        num = num * 58 + _B58.index(ch)
    body = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + body


TIP_ACCOUNT_BYTES = {b58decode(a) for a in TIP_ACCOUNTS_B58}


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
    f = req.transactions["jito_tips"]
    f.vote = False
    f.failed = False
    for a in TIP_ACCOUNTS_B58:
        f.account_include.append(a)
    req.commitment = pb.CommitmentLevel.CONFIRMED
    return req


async def _request_iter(req: pb.SubscribeRequest):
    yield req
    ping = pb.SubscribeRequest()
    ping.ping.id = 1
    while True:
        await asyncio.sleep(30)
        yield ping


def _tip_lamports(info) -> int:
    """Lamports that landed on tip accounts in this transaction.

    Full account order = static account_keys ++ loaded_writable ++ loaded_readonly,
    which is how pre_balances/post_balances are indexed.
    """
    meta = info.meta
    keys = list(info.transaction.message.account_keys)
    keys += list(meta.loaded_writable_addresses)
    keys += list(meta.loaded_readonly_addresses)
    pre, post = meta.pre_balances, meta.post_balances
    total = 0
    for i, k in enumerate(keys):
        if bytes(k) in TIP_ACCOUNT_BYTES and i < len(pre) and i < len(post):
            delta = int(post[i]) - int(pre[i])
            if delta > 0:
                total += delta
    return total


def _writer(data_dir: Path, name: str):
    state = {"day": None, "fh": None, "n": 0}

    def _rotate(now_utc: datetime):
        day = now_utc.strftime("%Y%m%d")
        if state["day"] != day:
            if state["fh"]:
                state["fh"].flush()
                state["fh"].close()
            path = data_dir / f"jitotips_{name}_{day}.jsonl"
            state["fh"] = open(path, "a", buffering=1)
            state["day"] = day
            print(f"[jito] writing {path}", flush=True)

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
    n_tx = 0
    lamports = 0

    while True:
        try:
            print(f"[jito] connecting {endpoint} (insecure={insecure})", flush=True)
            async with _open_channel(endpoint, insecure) as channel:
                stub = pb_grpc.GeyserStub(channel)
                stream = stub.Subscribe(_request_iter(_build_request()), metadata=metadata)
                backoff = 1.0
                async for update in stream:
                    recv_ms = time.time() * 1000.0
                    if not update.HasField("transaction"):
                        continue
                    info = update.transaction.transaction
                    if info.is_vote:
                        continue
                    tip = _tip_lamports(info)
                    if tip <= 0:
                        continue
                    n_tx += 1
                    lamports += tip
                    write({
                        "src": name,
                        "slot": update.transaction.slot,
                        "sig": info.signature.hex()[:32],
                        "tip_lamports": tip,
                        "cu": int(info.meta.compute_units_consumed),
                        "srv_ms": _created_ms(update),
                        "recv_ms": recv_ms,
                    })
                    now = time.time()
                    if now - last_report >= 60:
                        print(f"[jito] {n_tx} tip-txs, {lamports/1e9:.3f} SOL total, "
                              f"latest slot {update.transaction.slot}", flush=True)
                        last_report = now
        except asyncio.CancelledError:
            print("[jito] cancelled, flushing", flush=True)
            if write.state["fh"]:
                write.state["fh"].flush()
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[jito] stream error: {e!r}; reconnecting in {backoff:.0f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main() -> None:
    here = Path(__file__).resolve().parent
    _load_env(here.parent / ".env")
    _load_env(Path.home() / "solana-optimum" / ".env")

    endpoint = os.getenv("JITO_ENDPOINT") or os.getenv("GRPC_ENDPOINT", "")
    token = os.getenv("GRPC_TOKEN", "")
    insecure_env = os.getenv("JITO_INSECURE")
    insecure = (insecure_env == "1") if insecure_env is not None else endpoint.endswith(":2096")
    data_dir = Path(os.getenv("SENTRY_DATA_DIR", str(here.parent / "data")))
    name = os.getenv("JITO_NAME", "titan")

    if not endpoint:
        sys.exit("no GRPC_ENDPOINT / JITO_ENDPOINT configured")

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
