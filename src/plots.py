"""
The charts.

The headline figure is the RD plot: binned means of the outcome against block
arrival time, with the 4-second attestation deadline marked. It is the visual
form of the entire argument — flat, flat, flat, cliff.

Design notes (deliberate, not incidental):
  * BINNED MEANS, not a scatter of 650k slots. A scatter of that many points is
    an ink blob that hides the structure. Binned means with the bin's sample
    size encoded in the marker area shows both the shape AND where the data
    actually is — which matters enormously here, because the interesting region
    (past the deadline) is thinly populated.
  * SEPARATE local-linear fits either side of the cutoff. Fitting one smooth
    curve through the discontinuity would literally draw away the finding.
  * The cutoff line is annotated with the estimated jump, so the figure is
    readable without the caption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.xatu import ATTESTATION_DEADLINE_MS


def rd_plot(
    df: pd.DataFrame,
    outcome: str,
    tau: float | None,
    out_path: Path,
    window: float = 2500.0,
    bin_width: float = 100.0,
    ylabel: str | None = None,
) -> Path | None:
    """Binned-means RD figure around the attestation deadline.

    Returns the written path, or None if matplotlib isn't installed (the study
    must still run headless on a box with no plotting stack).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: frankfurt has no display
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    d = df.dropna(subset=["arrival_ms", outcome]).copy()
    lo = ATTESTATION_DEADLINE_MS - window
    hi = ATTESTATION_DEADLINE_MS + window
    d = d[(d["arrival_ms"] >= lo) & (d["arrival_ms"] <= hi)]
    if d.empty:
        return None

    # Bin the running variable and take the mean outcome per bin.
    edges = np.arange(lo, hi + bin_width, bin_width)
    d["bin"] = pd.cut(d["arrival_ms"], bins=edges, right=False)
    g = (
        d.groupby("bin", observed=True)
        .agg(
            x=("arrival_ms", "mean"),
            y=(outcome, "mean"),
            n=(outcome, "size"),
        )
        .dropna()
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    left = g[g["x"] < ATTESTATION_DEADLINE_MS]
    right = g[g["x"] >= ATTESTATION_DEADLINE_MS]

    # Marker area encodes bin sample size: the eye should not give a 12-slot bin
    # the same weight as a 40,000-slot bin.
    for part, colour, label in (
        (left, "#2c7fb8", "block beats the deadline"),
        (right, "#d95f0e", "block misses the deadline"),
    ):
        if part.empty:
            continue
        sizes = 12 + 180 * (part["n"] / g["n"].max())
        ax.scatter(part["x"], part["y"], s=sizes, alpha=0.75, color=colour,
                   label=label, zorder=3, edgecolors="white", linewidths=0.5)

        # Local linear fit, weighted by bin size — fitted SEPARATELY either side.
        if len(part) >= 2:
            coef = np.polyfit(part["x"], part["y"], 1, w=part["n"])
            xs = np.linspace(part["x"].min(), part["x"].max(), 100)
            ax.plot(xs, np.polyval(coef, xs), color=colour, lw=2, zorder=2)

    ax.axvline(
        ATTESTATION_DEADLINE_MS, color="#444", ls="--", lw=1.5, zorder=1,
        label=f"{ATTESTATION_DEADLINE_MS}ms attestation deadline (spec)",
    )

    if tau is not None:
        ax.annotate(
            f"RD τ = {tau:+.3f}",
            xy=(ATTESTATION_DEADLINE_MS, ax.get_ylim()[0]),
            xytext=(ATTESTATION_DEADLINE_MS + 150, ax.get_ylim()[0] + 0.02),
            fontsize=12, fontweight="bold", color="#d95f0e",
        )

    ax.set_xlabel("block arrival time (ms into slot)")
    ax.set_ylabel(ylabel or outcome.replace("_", " "))
    ax.set_title(
        "Block propagation latency and attester head votes — Ethereum mainnet\n"
        "marker area ∝ slots in bin",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def dose_response_plot(df: pd.DataFrame, out_path: Path) -> Path | None:
    """The full dose-response across the whole arrival range, not just near the
    cutoff. This is the chart that shows latency is nearly FREE until the
    deadline and then catastrophic — the non-linearity is the commercial point,
    because it means 'average propagation latency' is the wrong KPI. What matters
    is the share of blocks in the tail."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    d = df.dropna(subset=["arrival_ms", "correct_head_rate"]).copy()
    d = d[d["arrival_ms"] <= 8000]
    if d.empty:
        return None

    edges = np.arange(0, 8200, 200)
    d["bin"] = pd.cut(d["arrival_ms"], bins=edges, right=False)
    g = (
        d.groupby("bin", observed=True)
        .agg(x=("arrival_ms", "mean"),
             y=("correct_head_rate", "mean"),
             n=("correct_head_rate", "size"))
        .dropna()
        .reset_index(drop=True)
    )

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.plot(g["x"], 100 * g["y"], "o-", color="#2c7fb8", lw=2, ms=5)
    ax.axvline(ATTESTATION_DEADLINE_MS, color="#d95f0e", ls="--", lw=2,
               label=f"{ATTESTATION_DEADLINE_MS}ms deadline")
    ax.set_ylabel("correct head vote (%)")
    ax.set_title("Latency is almost free — until it isn't", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)

    # The density panel matters: it shows the effect lives in a thin tail. A
    # reader who sees only the top panel will overestimate how much of the
    # network is exposed.
    ax2.bar(g["x"], g["n"], width=180, color="#bbb")
    ax2.axvline(ATTESTATION_DEADLINE_MS, color="#d95f0e", ls="--", lw=2)
    ax2.set_yscale("log")
    ax2.set_ylabel("slots (log)")
    ax2.set_xlabel("block arrival time (ms into slot)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
