"""
Tests for cluster-robust inference and the wild cluster bootstrap.

The headline test is `test_wild_bootstrap_size`: under the null, p-values must be
approximately uniform. An inference procedure that over-rejects under the null is
worse than useless — it manufactures findings. This is the test that would catch
that.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.estimators.inference import (
    cluster_robust_fit,
    ols,
    wild_cluster_bootstrap,
)


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 2000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    true = np.array([1.5, -2.0, 0.75])
    y = X @ true + rng.normal(0, 0.1, n)

    assert ols(X, y) == pytest.approx(true, abs=0.02)


def test_cluster_robust_se_grows_with_intracluster_correlation():
    """The whole point of clustering: correlated errors within a cluster mean
    less independent information than the raw n suggests, so SEs must widen."""
    rng = np.random.default_rng(1)
    n, G = 3000, 30
    clusters = rng.integers(0, G, n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])

    # No intra-cluster correlation.
    y_iid = X @ np.array([1.0, 0.5]) + rng.normal(0, 1.0, n)
    se_iid = cluster_robust_fit(X, y_iid, clusters).se[1]

    # Strong cluster-level shocks.
    shock = rng.normal(0, 3.0, G)[clusters]
    y_corr = X @ np.array([1.0, 0.5]) + shock + rng.normal(0, 1.0, n)
    se_corr = cluster_robust_fit(X, y_corr, clusters).se[1]

    assert se_corr > se_iid


def test_cluster_count_reported():
    rng = np.random.default_rng(2)
    n = 500
    clusters = rng.integers(0, 17, n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.normal(size=n)
    fit = cluster_robust_fit(X, y, clusters)
    assert fit.n_clusters == len(np.unique(clusters))
    assert fit.n == n


def test_wild_bootstrap_size():
    """SIZE TEST. Under a true null (beta = 0), bootstrap p-values should be
    roughly uniform, so a nominal 10% test should reject about 10% of the time.

    We use few clusters (12) deliberately — that is exactly the regime where the
    plain cluster-robust t-test over-rejects and the wild bootstrap is supposed
    to rescue us. If this test fails, our p-values are lies.
    """
    rng = np.random.default_rng(3)
    G, per = 12, 40
    n = G * per
    clusters = np.repeat(np.arange(G), per)

    rejections = 0
    trials = 60
    for t in range(trials):
        r = np.random.default_rng(1000 + t)
        x = r.normal(size=n)
        # True beta on x is ZERO. Errors carry a cluster shock.
        shock = r.normal(0, 1.0, G)[clusters]
        y = 0.0 * x + shock + r.normal(0, 1.0, n)
        X = np.column_stack([np.ones(n), x])

        _, p, _ = wild_cluster_bootstrap(
            X, y, clusters, test_idx=1, reps=199, seed=t
        )
        if p < 0.10:
            rejections += 1

    rate = rejections / trials
    # Should sit near 0.10. Allow slack for 60 trials of Monte Carlo noise, but
    # catch gross over-rejection (the failure mode that matters).
    assert rate < 0.30, f"wild bootstrap over-rejects under the null: {rate:.2%}"


def test_wild_bootstrap_power():
    """Under a large true effect, the bootstrap must actually reject."""
    rng = np.random.default_rng(4)
    G, per = 25, 40
    n = G * per
    clusters = np.repeat(np.arange(G), per)
    x = rng.normal(size=n)
    shock = rng.normal(0, 0.3, G)[clusters]
    y = 2.0 * x + shock + rng.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), x])

    _, p, ci = wild_cluster_bootstrap(X, y, clusters, test_idx=1, reps=299, seed=0)
    assert p < 0.05
    lo, hi = ci
    assert lo < 2.0 < hi  # CI covers the truth


def test_degenerate_se_yields_inf_not_a_crash():
    """With too few clusters the sandwich's meat is near rank-deficient and the
    SE can collapse to zero. That must surface as an obviously-broken inf
    t-statistic, not a silently-swallowed RuntimeWarning that leaves a
    plausible-looking number in a results table."""
    from src.estimators.inference import FitResult

    fit = FitResult(
        coef=np.array([1.0, 2.0]),
        se=np.array([0.5, 0.0]),  # second SE degenerate
        n=10,
        n_clusters=2,
        names=["a", "b"],
    )
    assert fit.t_stat(0) == pytest.approx(2.0)
    assert np.isinf(fit.t_stat(1))


def test_wild_bootstrap_is_deterministic_given_seed():
    """Reproducibility is non-negotiable for a study someone else must re-run."""
    rng = np.random.default_rng(5)
    n = 400
    clusters = rng.integers(0, 10, n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.normal(size=n)

    a = wild_cluster_bootstrap(X, y, clusters, test_idx=1, reps=99, seed=42)
    b = wild_cluster_bootstrap(X, y, clusters, test_idx=1, reps=99, seed=42)
    assert a == b
