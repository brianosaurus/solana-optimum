#!/usr/bin/env python3
"""
Adoption sweep: per-adopter and total profit as Optimum adoption goes 0% -> 100%.

Why profit is NOT flat in adoption
----------------------------------
Channel C (MEV delay budget, ~90% of the value) is shaped by two opposing forces:

RAMP-UP — you need receivers. A proposer can only SPEND saved transit as extra
publication delay if enough of the slot's attesters actually receive the block
in time. Fork choice with proposer boost (40% of committee weight) means a block
survives a next-slot reorg attempt only if roughly THETA = 40% of attesters saw
it by the deadline. The committee is a uniform random draw, so the fraction of
fast receivers equals the adoption fraction alpha. Below theta, the adopter must
still wait for gossipsub to reach non-adopters:

    usable_delay(alpha) = full_delay x min(1, alpha / THETA)

ARMS-RACE DECAY — the captured MEV is flow stolen from the NEXT slot's proposer.
With probability alpha your PREDECESSOR is also an adopter and steals the same
way from you:

    net_gain(alpha) = gross_gain x (1 - alpha)

So Channel C per adopter is C_max x min(1, alpha/THETA) x (1 - alpha):
zero at both ends, peaked in the middle. That peak is the sweet spot.

Channels A (receive-side attestation) and D (bandwidth) are alpha-independent to
first order — the overlay ingests from gossipsub, so a lone adopter's RECEIVE
path is already fast. Channel B (orphan rescue) needs fast receivers like C.

All dollar constants are the measured 6x calibration (30-day panel, 214k slots;
relay bid traces; ETH @ $1,805.50).
"""

from __future__ import annotations

import numpy as np

# Measured per-validator/yr at 6x, "mature network" values (the alpha-shaping
# is applied to these).
A_USD = 2.48        # attester tail rescue (receive-side; alpha-independent)
B_USD = 0.135       # orphan rescue (needs fast receivers -> ramp only)
C_USD = 29.43       # MEV delay budget (ramp x arms-race)
D_USD = 342.0 / 500 # bandwidth saving per validator (500 keys/node)

N_VALIDATORS = 880_550
SEVEN_PARTNERS = 147_737  # the publicly named partner set

THETA = 0.40  # attester share needed for reorg safety (proposer boost = 40%)


def per_adopter_usd(alpha: float, theta: float = THETA) -> tuple[float, float, float, float]:
    """(A, B, C, total) $/validator/yr for an adopter at adoption `alpha`."""
    ramp = min(1.0, alpha / theta) if theta > 0 else 1.0
    a = A_USD
    b = B_USD * ramp
    c = C_USD * ramp * (1.0 - alpha)
    return a, b, c, a + max(b, c) + D_USD


def non_adopter_usd(alpha: float, theta: float = THETA) -> float:
    """A NON-adopter's per-validator/yr change vs the pre-Optimum baseline.

    They lose the flow adopting predecessors steal: -C_max x ramp x alpha.
    """
    ramp = min(1.0, alpha / theta) if theta > 0 else 1.0
    return -C_USD * ramp * alpha


def main() -> None:
    print(f"theta = {THETA:.0%} (attester share needed for safe delay; proposer boost)")
    print(f"{'adopt%':>7} {'validators':>11} | {'A':>6} {'B':>6} {'C':>7} {'TOTAL/val':>10} | "
          f"{'non-adopter':>11} {'advantage':>10} | {'ALL adopters $M/yr':>18}")
    best_per, best_tot = (0, 0.0), (0, 0.0)
    for pct in range(0, 101, 5):
        alpha = pct / 100
        a, b, c, tot = per_adopter_usd(alpha)
        non = non_adopter_usd(alpha)
        adv = tot - D_USD - a - max(0, non) - (non if non < 0 else 0) - 0  # see note below
        advantage = (a + max(b, c)) - non  # reward-side advantage vs non-adopter
        total_m = tot * alpha * N_VALIDATORS / 1e6
        if tot > best_per[1]:
            best_per = (pct, tot)
        if total_m > best_tot[1]:
            best_tot = (pct, total_m)
        mark = ""
        if abs(alpha - SEVEN_PARTNERS / N_VALIDATORS) < 0.025:
            mark = "  <- the 7 named partners (~16.8%)"
        print(f"{pct:>6}% {int(alpha * N_VALIDATORS):>11,} | {a:>6.2f} {b:>6.2f} {c:>7.2f} "
              f"{tot:>10.2f} | {non:>11.2f} {advantage:>10.2f} | {total_m:>18.2f}{mark}")

    print(f"\nPER-ADOPTER sweet spot : {best_per[0]}% adoption -> ${best_per[1]:.2f}/validator/yr")
    print(f"TOTAL-VALUE sweet spot : {best_tot[0]}% adoption -> ${best_tot[1]:.1f}M/yr across adopters")

    print("\ntheta sensitivity (per-adopter peak):")
    for th in (0.2, 0.4, 0.6):
        grid = [(al, per_adopter_usd(al, th)[3]) for al in np.arange(0.01, 1.0, 0.01)]
        al, v = max(grid, key=lambda t: t[1])
        print(f"  theta={th:.0%}: peak at {al:.0%} adoption, ${v:.2f}/validator/yr")


if __name__ == "__main__":
    main()
