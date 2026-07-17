"""
Configuration, loaded from `.env`.

Deliberately minimal and deliberately secret-free. This project is a read-only
analysis of public data: it has no wallet, no RPC credentials, no API keys, and
submits no transactions. If you ever find yourself adding a private key here,
you are in the wrong repository.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Immutable run configuration. Frozen so a run can't mutate its own params."""

    xatu_base_url: str
    network: str
    start_date: dt.date
    end_date: dt.date
    cache_dir: Path
    raw_dir: Path
    duckdb_memory_limit: str
    duckdb_threads: int
    min_sentries: int
    bootstrap_reps: int
    seed: int
    eth_price_usd: float

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError(
                f"END_DATE ({self.end_date}) precedes START_DATE ({self.start_date})"
            )
        if self.min_sentries < 1:
            raise ValueError("MIN_SENTRIES must be >= 1")
        if self.bootstrap_reps < 99:
            # Below ~99 the bootstrap p-value grid is too coarse to be meaningful.
            raise ValueError("BOOTSTRAP_REPS must be >= 99")

    @property
    def n_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


def _date(name: str, default: str) -> dt.date:
    return dt.date.fromisoformat(os.getenv(name, default))


def load_config(env_file: str | os.PathLike[str] | None = ".env") -> Config:
    """Read `.env` into a Config.

    `.env` is optional — every setting has a working default, so a fresh clone
    runs with zero setup.
    """
    if env_file is not None and Path(env_file).exists():
        load_dotenv(env_file)

    cache_dir = Path(os.getenv("CACHE_DIR", "./data"))
    cfg = Config(
        xatu_base_url=os.getenv(
            "XATU_BASE_URL",
            "https://data.ethpandaops.io/xatu/mainnet/databases/default",
        ),
        network=os.getenv("NETWORK", "mainnet"),
        start_date=_date("START_DATE", "2025-06-01"),
        end_date=_date("END_DATE", "2025-06-30"),
        cache_dir=cache_dir,
        # Raw Xatu parquet is downloaded here before querying. We do NOT query
        # the fat tables over HTTP: DuckDB's httpfs range requests corrupt large
        # ZSTD column chunks ("ZSTD Decompression failure") on the ~800MB/day
        # attestation table. Download-then-query is both correct and faster.
        raw_dir=cache_dir / "raw",
        duckdb_memory_limit=os.getenv("DUCKDB_MEMORY_LIMIT", "8GB"),
        duckdb_threads=int(os.getenv("DUCKDB_THREADS", "4")),
        min_sentries=int(os.getenv("MIN_SENTRIES", "5")),
        bootstrap_reps=int(os.getenv("BOOTSTRAP_REPS", "9999")),
        seed=int(os.getenv("SEED", "20250624")),
        # Used only to translate ETH figures into USD for reporting.
        # It does not enter any estimate. Set it to today's price.
        eth_price_usd=float(os.getenv("ETH_PRICE_USD", "3000")),
    )
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    return cfg
