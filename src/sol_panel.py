"""
Solana per-leader-slot panel: joins the two frankfurt collectors to the leader
schedule so the Channel-C and Channel-B designs can be run.

Inputs (produced live on frankfurt):
  data/slots_titan_*.jsonl[.gz]     -- slot-status sentry (propagation timing, skips)
  data/jitotips_titan_*.jsonl[.gz]  -- per-tx Jito tips (Channel-C outcome)

Join key: SLOT NUMBER (never timestamp — the HelloMoon `created_at` clock is
offset from wall clock; see SOLANA_PORT.md §7). Leader identity + window position
come from getLeaderSchedule, which is deterministic and joinable after the fact.

The unit is the slot. Per slot we assemble:
  * propagation      : shred_ms = t(COMPLETED) - t(FIRST_SHRED)   [same-clock, clean]
  * outcome (C)      : tips_sol, n_tip_tx, cu  (summed from the tip stream)
  * leader structure : leader identity, window_id, pos_in_window (0..3)
  * outcome (B)      : dead (SLOT_DEAD seen), produced (a block completed)

Leader windows: Solana assigns NUM_CONSECUTIVE_LEADER_SLOTS=4 consecutive slots
per leader, aligned to 4-slot boundaries from the epoch start (epoch length 432000
is divisible by 4). So within a window the leader is fixed and network load is
near-fixed — that is the variation the within-window Channel-C design exploits.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

SLOTS_PER_EPOCH = 432_000
NUM_CONSECUTIVE_LEADER_SLOTS = 4


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _rpc(url: str, method: str, params: list) -> dict | None:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("result")


def load_sentry(data_dir: str, name: str = "titan") -> pd.DataFrame:
    """One row per slot from the slot-status stream."""
    rows: dict[int, dict] = {}
    for f in sorted(glob.glob(f"{data_dir}/slots_{name}_*.jsonl*")):
        with _open(f) as fh:
            for line in fh:
                r = json.loads(line)
                s, st, srv = r["slot"], r["status"], r.get("srv_ms")
                d = rows.setdefault(s, {"slot": s, "parent": None, "dead": False})
                if r.get("parent"):
                    d["parent"] = r["parent"]
                if st == "SLOT_FIRST_SHRED_RECEIVED" and srv:
                    d["t_first_shred"] = srv
                elif st == "SLOT_COMPLETED" and srv:
                    d["t_completed"] = srv
                elif st == "SLOT_PROCESSED" and srv:
                    d["t_processed"] = srv
                elif st == "SLOT_DEAD":
                    d["dead"] = True
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df["shred_ms"] = df.get("t_completed") - df.get("t_first_shred")
    df["produced"] = df.get("t_completed").notna()
    return df


def load_jito(data_dir: str, name: str = "titan") -> pd.DataFrame:
    """Per-slot Jito tip totals from the per-tx stream."""
    agg: dict[int, dict] = {}
    for f in sorted(glob.glob(f"{data_dir}/jitotips_{name}_*.jsonl*")):
        with _open(f) as fh:
            for line in fh:
                r = json.loads(line)
                s = r["slot"]
                a = agg.setdefault(s, {"slot": s, "tip_lamports": 0, "n_tip_tx": 0, "cu": 0})
                a["tip_lamports"] += r["tip_lamports"]
                a["n_tip_tx"] += 1
                a["cu"] += r.get("cu", 0)
    df = pd.DataFrame(agg.values())
    if not df.empty:
        df["tips_sol"] = df["tip_lamports"] / 1e9
    return df


def load_fees(data_dir: str, name: str = "titan") -> pd.DataFrame:
    """Per-slot leader fee (base+priority) from the block-meta stream."""
    rows = []
    for f in sorted(glob.glob(f"{data_dir}/fees_{name}_*.jsonl*")):
        with _open(f) as fh:
            for line in fh:
                r = json.loads(line)
                rows.append({"slot": r["slot"],
                             "fee_sol": r["fee_lamports"] / 1e9,
                             "tx_count": r.get("tx_count", 0)})
    df = pd.DataFrame(rows)
    return df.drop_duplicates("slot") if not df.empty else df


def leader_map(rpc_url: str, slots) -> dict[int, str]:
    """Absolute slot -> leader identity, over every epoch the data spans."""
    epochs = sorted({int(s) // SLOTS_PER_EPOCH for s in slots})
    m: dict[int, str] = {}
    for ep in epochs:
        first = ep * SLOTS_PER_EPOCH
        sched = _rpc(rpc_url, "getLeaderSchedule", [first])
        if not sched:
            continue
        for identity, idxs in sched.items():
            for i in idxs:
                m[first + i] = identity
    return m


def build_panel(data_dir: str, rpc_url: str, name: str = "titan",
                window_hours: float | None = None) -> pd.DataFrame:
    """The joined per-slot panel, restricted to a trailing `window_hours` window.

    The rolling window keeps the study to a fixed, recent horizon (default 48h =
    two days — enough to average two full day/night cycles and stabilise the
    slopes) and bounds the panel size, so the per-run bootstrap cost stays
    constant instead of growing with every day of collection.
    """
    if window_hours is None:
        window_hours = float(os.getenv("WINDOW_HOURS", "48"))
    sentry = load_sentry(data_dir, name)
    jito = load_jito(data_dir, name)
    fees = load_fees(data_dir, name)
    if sentry.empty:
        raise SystemExit("no sentry data yet")

    # SKIP ACCOUNTING needs the FULL contiguous slot range, not just the slots
    # the sentry emitted a status for: a slot skipped outright produces no shreds
    # and therefore no status, so it is ABSENT from the sentry stream. Reindex to
    # [lo, hi] and treat absent slots as not-produced — that is the skip.
    hi = int(sentry["slot"].max())
    # trailing window: 1 slot ~ 0.4 s, so window_hours * 9000 slots.
    window_slots = int(window_hours * 3600 / 0.4)
    lo = max(int(sentry["slot"].min()), hi - window_slots + 1)
    sentry = sentry[sentry["slot"] >= lo]
    full = pd.DataFrame({"slot": range(lo, hi + 1)})
    df = (full.merge(sentry, on="slot", how="left")
          .merge(jito, on="slot", how="left"))
    if not fees.empty:
        df = df.merge(fees, on="slot", how="left")
    df["produced"] = df["produced"].fillna(False).astype(bool)
    df["dead"] = df["dead"].fillna(False).astype(bool)
    for c in ("tip_lamports", "n_tip_tx", "cu", "tips_sol", "fee_sol"):
        if c in df:
            df[c] = df[c].fillna(0)

    lm = leader_map(rpc_url, df["slot"])
    df["leader"] = df["slot"].map(lm)
    df["epoch_first"] = (df["slot"] // SLOTS_PER_EPOCH) * SLOTS_PER_EPOCH
    within = df["slot"] - df["epoch_first"]
    df["window_id"] = df["slot"] // NUM_CONSECUTIVE_LEADER_SLOTS
    df["pos_in_window"] = within % NUM_CONSECUTIVE_LEADER_SLOTS

    # SEALING LATENESS: how much later a block COMPLETED than its on-schedule
    # position predicts, measured WITHIN its window so the HelloMoon clock offset
    # cancels. anchor = pos-0's first-shred; expected completion of pos p is
    # anchor + p*400ms; the residual is the leader's discretionary sealing delay —
    # the sharper Channel-C running variable than coarse position.
    anchor = (df[df["pos_in_window"] == 0][["window_id", "t_first_shred"]]
              .rename(columns={"t_first_shred": "window_start"}))
    df = df.merge(anchor, on="window_id", how="left")
    df["seal_lateness_ms"] = (
        df["t_completed"] - df["window_start"] - df["pos_in_window"] * 400.0
    )
    # guard against clock-jitter outliers driving the estimate
    df.loc[df["seal_lateness_ms"].abs() > 2000, "seal_lateness_ms"] = pd.NA
    return df.sort_values("slot").reset_index(drop=True)


if __name__ == "__main__":  # quick manual build/inspect
    import os
    dd = os.getenv("SENTRY_DATA_DIR", "data")
    rpc = os.getenv("SOLANA_RPC_URL", "")
    p = build_panel(dd, rpc)
    out = Path(dd) / "panel.parquet"
    p.to_parquet(out)
    print(f"panel: {len(p)} slots -> {out}")
    print(p[["slot", "leader", "pos_in_window", "shred_ms", "tips_sol", "dead", "produced"]].head())
