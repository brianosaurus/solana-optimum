"""
Cross-validate the hand-rolled estimators against trusted libraries.

The existing `test_inference.py` checks the estimators against *themselves* and
against their statistical properties (size under the null, power, determinism).
This file does the complementary thing: it pins every hand-rolled formula to an
independent, widely-used implementation — `statsmodels` for the regression and
sandwich variances, `scipy` for the distributions — so a subtle degrees-of-
freedom or small-sample-correction bug can't hide.

Scope is the SOLANA study's statistics only: the sandwiches in
`src.estimators.inference` and the within-leader-window designs in
`src.estimators.channel_c` that wrap them. It deliberately imports neither
`rdd` nor `xatu` (the Ethereum-only RD path), so this suite needs no duckdb.

Why exact equality is the right assertion here: these are closed-form estimators,
not Monte Carlo, so a correct implementation must reproduce statsmodels to
floating-point tolerance. Anything looser would let a real bug through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip("statsmodels.api")
from scipy import stats as sps  # noqa: E402

from src.estimators.channel_c import sealing_design, sealing_two_way  # noqa: E402
from src.estimators.inference import (  # noqa: E402
    cluster_robust_fit,
    cluster_robust_fit_2way,
    ols,
    wild_cluster_bootstrap,
)

EXACT = dict(rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _clustered(n=1200, k=2, g1=30, g2=13, seed=0):
    """A design with genuine two-level error dependence (cluster shocks in both
    dimensions), so the robust variances actually differ from OLS."""
    rng = np.random.default_rng(seed)
    c1 = rng.integers(0, g1, n)
    c2 = rng.integers(0, g2, n)
    X = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(k)])
    beta = rng.normal(size=k + 1)
    y = (X @ beta
         + rng.normal(0, 2.0, g1)[c1]      # leader-level shock
         + rng.normal(0, 1.5, g2)[c2]      # time-level shock
         + rng.normal(0, 1.0, n))
    return X, y, c1, c2


def _within_window_panel(seed=0):
    """A synthetic slot panel shaped like the real one, so `sealing_design` and
    `sealing_two_way` run their exact filtering + window-demeaning path.

    One leader owns several consecutive 4-slot windows; slots span enough range
    (>9000) to yield multiple hour-clusters for the two-way test.
    """
    rng = np.random.default_rng(seed)
    n_windows = 600
    rows = []
    for w in range(n_windows):
        leader = f"L{w // 6}"          # each leader owns 6 windows
        base = w * 64                  # spread slots so hour = slot//9000 varies
        lshock = rng.normal(0, 0.004)
        for pos in range(4):
            seal = rng.normal(0, 120)
            cu = rng.uniform(2e6, 4.8e7)
            tips = 1e-5 * seal + 2e-11 * cu + lshock + rng.normal(0, 0.003)
            rows.append(dict(slot=base + pos, window_id=w, pos_in_window=pos,
                             leader=leader, produced=True,
                             seal_lateness_ms=seal, cu=cu, tips_sol=tips,
                             fee_sol=tips * 1.7))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# OLS                                                                          #
# --------------------------------------------------------------------------- #

def test_ols_matches_statsmodels():
    X, y, _, _ = _clustered()
    assert np.allclose(ols(X, y), sm.OLS(y, X).fit().params, **EXACT)


# --------------------------------------------------------------------------- #
# one-way CR1 cluster-robust sandwich                                          #
# --------------------------------------------------------------------------- #

def test_cr1_coefficients_and_se_match_statsmodels():
    X, y, c1, _ = _clustered(k=3, seed=1)
    mine = cluster_robust_fit(X, y, c1)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c1})
    assert np.allclose(mine.coef, ref.params, **EXACT)
    # this is the assertion that catches a wrong dof correction:
    assert np.allclose(mine.se, ref.bse, **EXACT)
    assert mine.n_clusters == len(np.unique(c1))


@pytest.mark.parametrize("g1", [8, 30, 120])
def test_cr1_se_matches_across_cluster_counts(g1):
    """The finite-sample correction is (G/(G-1))·((n-1)/(n-k)); if it were wrong
    the error would scale with G, so sweep it."""
    X, y, c1, _ = _clustered(g1=g1, seed=g1)
    mine = cluster_robust_fit(X, y, c1)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c1})
    assert np.allclose(mine.se, ref.bse, **EXACT)


def test_cr1_t_and_wald_agree_with_statsmodels():
    X, y, c1, _ = _clustered(seed=7)
    mine = cluster_robust_fit(X, y, c1)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c1})
    assert mine.t_stat(1) == pytest.approx(ref.tvalues[1], rel=1e-9)


# --------------------------------------------------------------------------- #
# two-way (Cameron–Gelbach–Miller) sandwich                                    #
# --------------------------------------------------------------------------- #

def test_two_way_matches_statsmodels_native():
    X, y, c1, c2 = _clustered(seed=3)
    mine = cluster_robust_fit_2way(X, y, c1, c2)
    ref = sm.OLS(y, X).fit(cov_type="cluster",
                           cov_kwds={"groups": np.column_stack([c1, c2])})
    assert np.allclose(mine.coef, ref.params, **EXACT)
    assert np.allclose(mine.se, ref.bse, **EXACT)


def test_two_way_reduces_to_one_way_when_clusterings_identical():
    """V = V(c) + V(c) − V(c∩c) = V(c): the two-way SE must equal the one-way SE
    exactly when both dimensions are the same partition."""
    X, y, c1, _ = _clustered(seed=4)
    one = cluster_robust_fit(X, y, c1)
    two = cluster_robust_fit_2way(X, y, c1, c1)
    assert np.allclose(two.se, one.se, **EXACT)


def test_two_way_reports_smaller_cluster_dimension():
    X, y, c1, c2 = _clustered(g1=30, g2=13, seed=5)
    two = cluster_robust_fit_2way(X, y, c1, c2)
    assert two.n_clusters == min(len(np.unique(c1)), len(np.unique(c2)))


# --------------------------------------------------------------------------- #
# wild cluster bootstrap                                                       #
# --------------------------------------------------------------------------- #

def test_wild_bootstrap_observed_t_matches_statsmodels():
    """The bootstrap resamples, so its *distribution* isn't a closed form — but
    the observed t-statistic it studentises against must equal the analytic
    cluster-robust t exactly."""
    X, y, c1, _ = _clustered(seed=8)
    t_obs, _, _ = wild_cluster_bootstrap(X, y, c1, test_idx=1, reps=99, seed=0)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c1})
    assert t_obs == pytest.approx(ref.tvalues[1], rel=1e-9)


def test_wild_bootstrap_pvalue_tracks_asymptotic_with_many_clusters():
    """With many clusters the restricted wild bootstrap and the asymptotic
    normal test should give similar p-values; this catches a bootstrap that is
    silently miscalibrated in the easy regime."""
    X, y, c1, _ = _clustered(g1=200, n=6000, seed=9)
    _, p_boot, _ = wild_cluster_bootstrap(X, y, c1, test_idx=1, reps=999, seed=1)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": c1})
    p_asym = 2 * sps.norm.sf(abs(ref.tvalues[1]))
    assert p_boot == pytest.approx(p_asym, abs=0.05)


def test_wild_bootstrap_ci_brackets_point_estimate():
    X, y, c1, _ = _clustered(seed=10)
    _, _, (lo, hi) = wild_cluster_bootstrap(X, y, c1, test_idx=1, reps=299, seed=2)
    b1 = ols(X, y)[1]
    assert lo < b1 < hi


# --------------------------------------------------------------------------- #
# the Solana within-window design (channel_c) end-to-end                      #
# --------------------------------------------------------------------------- #

def test_sealing_design_matches_statsmodels_on_demeaned_design():
    """`sealing_design` filters to single-leader windows, demeans within window,
    then runs a leader-clustered regression. Reproduce that pipeline by hand and
    confirm statsmodels lands on the same slope and SE (the design reports them
    per +100ms, i.e. ×100)."""
    panel = _within_window_panel(seed=11)
    res = sealing_design(panel, outcome="tips_sol")

    d = panel.copy()
    for c in ("tips_sol", "seal_lateness_ms", "cu"):
        d[c + "_w"] = d[c] - d.groupby("window_id")[c].transform("mean")
    X = np.column_stack([d["seal_lateness_ms_w"], d["cu_w"]])
    y = d["tips_sol_w"].to_numpy()
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["leader"].to_numpy()})

    assert res.slope == pytest.approx(ref.params[0] * 100, rel=1e-9)
    assert res.se == pytest.approx(ref.bse[0] * 100, rel=1e-9)
    assert res.n == len(d)


def test_sealing_two_way_matches_statsmodels_native():
    """The two-way (leader × hour) result the page renders must equal a native
    statsmodels two-way clustering on the same demeaned design."""
    panel = _within_window_panel(seed=12)
    out = sealing_two_way(panel, outcome="fee_sol")
    assert out is not None

    d = panel.copy()
    for c in ("fee_sol", "seal_lateness_ms", "cu"):
        d[c + "_w"] = d[c] - d.groupby("window_id")[c].transform("mean")
    X = np.column_stack([d["seal_lateness_ms_w"], d["cu_w"]])
    y = d["fee_sol_w"].to_numpy()
    hour = (d["slot"] // 9000).astype(str).to_numpy()
    # integer-code the labels: statsmodels' native two-way can't view() a 2-D
    # string array under numpy 2.x. Encoding is a bijection, so the clustering
    # (and thus the SEs) is identical — the estimator under test handles strings.
    lead_c = pd.factorize(d["leader"])[0]
    hour_c = pd.factorize(hour)[0]
    ref = sm.OLS(y, X).fit(cov_type="cluster",
                           cov_kwds={"groups": np.column_stack([lead_c, hour_c])})

    assert out["slope"] == pytest.approx(ref.params[0] * 100, rel=1e-9)
    assert out["t"] == pytest.approx(ref.tvalues[0], rel=1e-9)
    assert out["p"] == pytest.approx(2 * sps.norm.sf(abs(ref.tvalues[0])), rel=1e-9)
    assert out["n_hours"] == len(np.unique(hour))


def test_sealing_two_way_se_inflation_is_ratio_of_the_two_sandwiches():
    """se_inflation is advertised as two-way SE ÷ one-way SE; verify it against
    statsmodels computing each sandwich independently."""
    panel = _within_window_panel(seed=13)
    out = sealing_two_way(panel, outcome="fee_sol")

    d = panel.copy()
    for c in ("fee_sol", "seal_lateness_ms", "cu"):
        d[c + "_w"] = d[c] - d.groupby("window_id")[c].transform("mean")
    X = np.column_stack([d["seal_lateness_ms_w"], d["cu_w"]])
    y = d["fee_sol_w"].to_numpy()
    hour = (d["slot"] // 9000).astype(str).to_numpy()
    lead_c = pd.factorize(d["leader"])[0]
    hour_c = pd.factorize(hour)[0]
    se1 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": lead_c}).bse[0]
    se2 = sm.OLS(y, X).fit(cov_type="cluster",
                           cov_kwds={"groups": np.column_stack([lead_c, hour_c])}).bse[0]
    assert out["se_inflation"] == pytest.approx(se2 / se1, rel=1e-9)
