"""
Regression discontinuity at the 4-second attestation deadline.

This is the identification core of the study.

The design
----------
The Ethereum consensus spec tells an attester to vote at 1/3 of the way into its
slot — 4000ms — for whatever head it can see AT THAT INSTANT. This creates a
sharp, mechanical, exogenous threshold in a continuous running variable:

    running variable : arrival_ms   (when the block reached the network)
    cutoff           : 4000ms       (spec-mandated, not chosen by us or anyone)
    treatment        : block is visible to the attester in time
    outcome          : correct_head_rate, inclusion distance, ...

Blocks landing at 3,950ms and 4,050ms are alike in every respect that could
plausibly matter — same proposers, same builders, same network conditions, same
block sizes. The only difference is that one is votable and the other is not.
Comparing outcomes just below vs just above 4000ms therefore identifies the
causal effect of the block being late, free of the selection problems that
plague any comparison of "fast operators" to "slow operators".

Why this beats the naive regression
-----------------------------------
A naive OLS of correct_head_rate on arrival_ms is confounded: slow blocks are
also big blocks, blob-heavy blocks, blocks from overloaded builders. Those
things could independently affect attesters. The RD does not care — it only uses
variation in a tiny neighbourhood of the cutoff, where those confounders are
continuous and therefore differenced away.

What could break it (and how we check)
--------------------------------------
The RD assumption is that nothing ELSE jumps at exactly 4000ms.

1. MANIPULATION of the running variable. If proposers could precisely target
   arrival just under 4000ms, the density of arrival_ms would jump at the
   cutoff and units either side would no longer be comparable. We run a
   McCrary-style density test (`density_test`). Note the strong prior here:
   proposers WANT to be early and cannot control network propagation to the
   millisecond, so manipulation at exactly 4000ms is implausible — but we test
   it rather than assert it.

2. COVARIATES jumping at the cutoff. If blob count or block size discontinuously
   jumped at 4000ms, the "effect" could be theirs. We run the same RD on each
   covariate as a placebo (`covariate_balance`); those effects should be zero.

3. The wrong functional form masquerading as a jump. We use LOCAL LINEAR
   regression with a triangular kernel (Hahn-Todd-Van der Klaauw; Imbens-Lemieux),
   which is the standard, and we vary the bandwidth to show the estimate is not
   an artifact of one choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.estimators.inference import cluster_robust_fit, wild_cluster_bootstrap
from src.xatu import ATTESTATION_DEADLINE_MS


@dataclass(frozen=True)
class RDResult:
    """Estimated jump at the cutoff."""

    tau: float  # the discontinuity: effect of crossing the deadline
    se: float
    t_stat: float
    p_boot: float | None
    ci_boot: tuple[float, float] | None
    bandwidth: float
    n_left: int
    n_right: int
    n_clusters: int

    def __str__(self) -> str:
        p = f"{self.p_boot:.4f}" if self.p_boot is not None else "n/a"
        return (
            f"RD tau = {self.tau:+.4f} (se {self.se:.4f}, t {self.t_stat:+.2f}, "
            f"wild-boot p {p})  h={self.bandwidth:.0f}ms  "
            f"n=[{self.n_left} left | {self.n_right} right]  G={self.n_clusters}"
        )


def triangular_kernel(x: np.ndarray, h: float) -> np.ndarray:
    """Triangular (edge) kernel weights: full weight at the cutoff, zero at ±h.

    Triangular is the boundary-optimal kernel for local linear RD (Cheng, Fan &
    Marron 1997) — it minimises MSE at an evaluation point on the boundary of
    the support, which is exactly where an RD estimates.
    """
    return np.maximum(0.0, 1.0 - np.abs(x) / h)


def rd_estimate(
    running: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    cutoff: float = ATTESTATION_DEADLINE_MS,
    bandwidth: float = 1500.0,
    weights: np.ndarray | None = None,
    bootstrap_reps: int = 0,
    seed: int = 0,
) -> RDResult:
    """Sharp RD via local linear regression with a triangular kernel.

    We fit, on observations within `bandwidth` of the cutoff:

        y = a + tau*D + b1*(x - c) + b2*(x - c)*D + e

    where x is the running variable, c the cutoff, and D = 1{x >= c}. Allowing
    separate slopes either side (the interaction term) is what makes this "local
    linear" rather than a kinked single line — without it, a difference in slopes
    would leak into tau and manufacture a fake jump.

    `tau` is the estimand: the discontinuous jump in the outcome at the cutoff.

    `weights` optionally weights each slot by its number of attesters, so a slot
    with a 30k-validator committee counts more than a thin one. These multiply
    the kernel weights.
    """
    running = np.asarray(running, dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    # Restrict to the bandwidth window. Everything outside gets zero kernel
    # weight anyway; dropping it keeps the design matrix well-conditioned.
    x = running - cutoff
    inwin = np.abs(x) <= bandwidth
    # Drop rows with a missing outcome (e.g. a slot whose committee is unknown).
    inwin &= ~np.isnan(outcome)

    x = x[inwin]
    y = outcome[inwin]
    cl = np.asarray(clusters)[inwin]
    D = (x >= 0).astype(float)

    kw = triangular_kernel(x, bandwidth)
    if weights is not None:
        kw = kw * np.asarray(weights, dtype=float)[inwin]

    # Design: [1, D, x, x*D]. tau is the coefficient on D -> index 1.
    X = np.column_stack([np.ones_like(x), D, x, x * D])
    names = ["intercept", "tau (past deadline)", "slope", "slope x D"]

    # Weighted least squares == OLS on sqrt(w)-scaled data. We scale both X and
    # y so the cluster-robust sandwich below is computed on the correct
    # (transformed) residuals.
    sw = np.sqrt(kw)
    Xw = X * sw[:, None]
    yw = y * sw

    fit = cluster_robust_fit(Xw, yw, cl, names=names)
    TAU = 1

    p_boot: float | None = None
    ci_boot: tuple[float, float] | None = None
    if bootstrap_reps > 0:
        _, p_boot, ci_boot = wild_cluster_bootstrap(
            Xw, yw, cl, test_idx=TAU, reps=bootstrap_reps, seed=seed
        )

    return RDResult(
        tau=float(fit.coef[TAU]),
        se=float(fit.se[TAU]),
        t_stat=fit.t_stat(TAU),
        p_boot=p_boot,
        ci_boot=ci_boot,
        bandwidth=bandwidth,
        n_left=int((D == 0).sum()),
        n_right=int((D == 1).sum()),
        n_clusters=fit.n_clusters,
    )


def density_test(
    running: np.ndarray,
    cutoff: float = ATTESTATION_DEADLINE_MS,
    bin_width: float = 100.0,
    window: float = 1000.0,
) -> dict[str, float]:
    """McCrary-style manipulation test: does the DENSITY of the running variable
    jump at the cutoff?

    If proposers could steer their block's arrival time to land just under the
    deadline, we would see a pile-up just left of 4000ms and a hole just right.
    That would invalidate the RD, because units either side would be selected
    rather than comparable.

    We compare the count in the last bin before the cutoff to the first bin
    after, and report a simple two-proportion z-statistic. A |z| above ~2 is a
    red flag worth investigating.

    (This is the lightweight version. The formal Cattaneo-Jansson-Ma test is
    better but needs a local-polynomial density estimator; the crude version is
    sufficient given the strong prior that millisecond-precision manipulation of
    gossip propagation is not physically available to a proposer.)
    """
    x = np.asarray(running, dtype=float) - cutoff
    x = x[(np.abs(x) <= window) & ~np.isnan(x)]

    left = float(((x >= -bin_width) & (x < 0)).sum())
    right = float(((x >= 0) & (x < bin_width)).sum())
    total = left + right

    if total == 0:
        return {"left": 0.0, "right": 0.0, "z": float("nan")}

    # Under "no manipulation", an observation near the cutoff is equally likely
    # to fall either side: p = 0.5. Binomial z on that null.
    z = (left - total / 2) / np.sqrt(total * 0.25)
    return {"left": left, "right": right, "z": float(z)}
