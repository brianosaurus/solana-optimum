"""
Tests for the RD estimator.

The important tests here are the two-sided ones: an estimator that finds effects
is worthless unless it also *fails* to find them when they are absent. So we
plant a known jump and check we recover it, and plant no jump and check we
report none.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.estimators.rdd import density_test, rd_estimate, triangular_kernel
from src.xatu import ATTESTATION_DEADLINE_MS


def _synthetic(
    n: int,
    tau: float,
    slope_left: float = -0.0001,
    slope_right: float = -0.0002,
    noise: float = 0.02,
    seed: int = 7,
    n_clusters: int = 40,
):
    """Synthetic RD data with a planted discontinuity of size `tau` at 4000ms.

    Deliberately includes DIFFERENT slopes either side of the cutoff. A naive
    estimator that forces a common slope would mistake the slope change for a
    jump; the local-linear design with an interaction term should not.
    """
    rng = np.random.default_rng(seed)
    running = rng.uniform(
        ATTESTATION_DEADLINE_MS - 1500, ATTESTATION_DEADLINE_MS + 1500, n
    )
    x = running - ATTESTATION_DEADLINE_MS
    past = (x >= 0).astype(float)

    clusters = rng.integers(0, n_clusters, n)
    # Cluster-level shocks: this is what makes naive (non-clustered) SEs wrong.
    cluster_effect = rng.normal(0, 0.01, n_clusters)[clusters]

    y = (
        0.99
        + tau * past
        + slope_left * x * (1 - past)
        + slope_right * x * past
        + cluster_effect
        + rng.normal(0, noise, n)
    )
    return running, y, clusters


def test_recovers_planted_discontinuity():
    """Plant tau = -0.25 (a 25pp collapse in correct-head rate). Recover it."""
    running, y, clusters = _synthetic(n=8000, tau=-0.25)
    res = rd_estimate(running, y, clusters, bandwidth=1500.0)

    assert res.tau == pytest.approx(-0.25, abs=0.02), (
        f"failed to recover planted jump: got {res.tau}"
    )
    # And it should be unambiguously significant.
    assert abs(res.t_stat) > 5


def test_no_false_positive_when_no_jump():
    """The critical negative control: tau = 0 must NOT produce a 'finding'.

    We keep the differing slopes either side, so this also verifies that a kink
    is not misread as a jump.
    """
    running, y, clusters = _synthetic(n=8000, tau=0.0)
    res = rd_estimate(running, y, clusters, bandwidth=1500.0)

    assert abs(res.tau) < 0.02, f"fabricated a discontinuity: {res.tau}"
    assert abs(res.t_stat) < 3


def test_bandwidth_robustness():
    """The estimate must not be an artifact of one bandwidth choice."""
    running, y, clusters = _synthetic(n=20000, tau=-0.20)
    taus = [
        rd_estimate(running, y, clusters, bandwidth=h).tau for h in (500, 1000, 1500)
    ]
    for t in taus:
        assert t == pytest.approx(-0.20, abs=0.03)


def test_wild_bootstrap_runs_and_agrees():
    """Bootstrap p-value should be tiny for a large, real effect."""
    running, y, clusters = _synthetic(n=3000, tau=-0.25)
    res = rd_estimate(
        running, y, clusters, bandwidth=1500.0, bootstrap_reps=299, seed=1
    )
    assert res.p_boot is not None
    assert res.p_boot < 0.05
    assert res.ci_boot is not None
    lo, hi = res.ci_boot
    assert lo < res.tau < hi


def test_clustered_se_exceeds_naive_se():
    """Cluster-correlated errors must inflate the SE.

    If our clustering were a no-op, this would fail — and every p-value in the
    study would be too small.
    """
    running, y, clusters = _synthetic(n=5000, tau=-0.2, n_clusters=15)
    clustered = rd_estimate(running, y, clusters, bandwidth=1500.0)
    # Pretend every observation is its own cluster == the iid/naive case.
    naive = rd_estimate(running, y, np.arange(len(y)), bandwidth=1500.0)
    assert clustered.se > naive.se


def test_triangular_kernel_shape():
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0, 150.0])
    w = triangular_kernel(x, h=100.0)
    assert w[2] == pytest.approx(1.0)  # full weight at the cutoff
    assert w[0] == pytest.approx(0.0)  # zero at the edge
    assert w[5] == pytest.approx(0.0)  # never negative beyond the edge
    assert np.all(w >= 0)


def test_density_test_flags_manipulation():
    """A pile-up just below the cutoff must be detected."""
    rng = np.random.default_rng(0)
    clean = rng.uniform(3000, 5000, 5000)
    assert abs(density_test(clean)["z"]) < 3  # no manipulation -> small z

    # Now bunch 800 extra observations just under the deadline.
    bunched = np.concatenate([clean, rng.uniform(3900, 4000, 800)])
    assert density_test(bunched)["z"] > 3  # detected


def test_nan_outcomes_are_dropped():
    """Slots with an unknown committee (NaN outcome) must not poison the fit."""
    running, y, clusters = _synthetic(n=3000, tau=-0.25)
    y[::10] = np.nan
    res = rd_estimate(running, y, clusters, bandwidth=1500.0)
    assert np.isfinite(res.tau)
    assert res.tau == pytest.approx(-0.25, abs=0.03)
