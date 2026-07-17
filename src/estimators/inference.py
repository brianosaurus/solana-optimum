"""
Inference: cluster-robust standard errors and the wild cluster bootstrap.

Why this module exists
----------------------
Our observations are slots, but the errors are not independent across slots.
A proposer's infrastructure is persistent: if an operator's blocks are slow
today, they are slow tomorrow. Network-wide conditions (a big NFT mint, a spike
in blob demand) shift arrival times for every slot in an hour at once. Treating
~650k slots as 650k independent draws would produce absurdly tight confidence
intervals.

So we cluster. But clustering creates a second problem: cluster-robust variance
estimators are only consistent as the NUMBER OF CLUSTERS grows. With few
clusters, the CRVE is badly biased downward and t-statistics over-reject — you
find significance that isn't there.

The wild cluster bootstrap (Cameron, Gelbach & Miller 2008) fixes this. We
impose the null, resample cluster-level residuals with random sign flips, and
build the reference distribution of the t-statistic empirically rather than
assuming it is Normal.

An honesty note carried over from the original (abandoned) design
-----------------------------------------------------------------
The first version of this study wanted operator-clustered SEs with 7 treated
operators. MacKinnon & Webb (2017) show the wild cluster bootstrap is NOT a
cure-all there: with few TREATED clusters it can under- or over-reject severely
depending on the weight scheme. That design is gone, but the lesson stays:
we report the number of clusters, and we use the RESTRICTED (WCR) bootstrap,
which imposes the null when generating the residuals and is the variant that
behaves best in simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitResult:
    """A fitted linear model with cluster-robust inference."""

    coef: np.ndarray  # (k,) coefficients
    se: np.ndarray  # (k,) cluster-robust standard errors
    n: int  # observations
    n_clusters: int
    names: list[str]

    def t_stat(self, idx: int) -> float:
        """t = coef / se, guarding against a degenerate zero standard error.

        se can collapse to ~0 when there are very few clusters (the sandwich's
        meat is a sum of G outer products, so with G=2 it is nearly rank-
        deficient). Returning inf rather than raising a RuntimeWarning keeps the
        failure visible in the output instead of buried in stderr — an infinite
        t-statistic is a loud signal that the clustering is degenerate.
        """
        if self.se[idx] == 0 or not np.isfinite(self.se[idx]):
            return float("inf") if self.coef[idx] != 0 else float("nan")
        return float(self.coef[idx] / self.se[idx])

    def summary_row(self, idx: int) -> str:
        return (
            f"{self.names[idx]:>28}: {self.coef[idx]:+.6f} "
            f"(se {self.se[idx]:.6f}, t {self.t_stat(idx):+.2f})"
        )


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain OLS via least squares.

    lstsq (not the normal equations) because our design matrices include
    near-collinear polynomial terms in the RD, where X'X is ill-conditioned.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def cluster_robust_fit(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    names: list[str] | None = None,
) -> FitResult:
    """OLS with a cluster-robust (CR1) sandwich variance estimator.

        V = (X'X)^-1  [ sum_g X_g' u_g u_g' X_g ]  (X'X)^-1  * c

    where the sum runs over clusters g and c is the standard small-sample
    correction. The middle term is what changes: instead of assuming errors are
    independent, we let them be arbitrarily correlated *within* a cluster and
    only assume independence *across* clusters.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if names is None:
        names = [f"x{i}" for i in range(k)]

    beta = ols(X, y)
    resid = y - X @ beta

    XtX_inv = np.linalg.pinv(X.T @ X)

    # Meat of the sandwich: sum over clusters of (X_g' u_g)(X_g' u_g)'.
    uniq = np.unique(clusters)
    G = len(uniq)
    meat = np.zeros((k, k))
    for g in uniq:
        m = clusters == g
        Xg_ug = X[m].T @ resid[m]  # (k,)
        meat += np.outer(Xg_ug, Xg_ug)

    # CR1 finite-sample correction. Without it, SEs are too small.
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    V = c * (XtX_inv @ meat @ XtX_inv)

    # Numerical guard: a near-singular design can yield a tiny negative variance.
    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    return FitResult(coef=beta, se=se, n=n, n_clusters=G, names=names)


def _crve_V(X: np.ndarray, resid: np.ndarray, clusters: np.ndarray,
            XtX_inv: np.ndarray) -> tuple[np.ndarray, int]:
    """The CR1 sandwich variance matrix for one clustering dimension."""
    n, k = X.shape
    uniq = np.unique(clusters)
    G = len(uniq)
    meat = np.zeros((k, k))
    for g in uniq:
        m = clusters == g
        Xg_ug = X[m].T @ resid[m]
        meat += np.outer(Xg_ug, Xg_ug)
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    return c * (XtX_inv @ meat @ XtX_inv), G


def cluster_robust_fit_2way(
    X: np.ndarray,
    y: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    names: list[str] | None = None,
) -> FitResult:
    """Two-way cluster-robust variance (Cameron, Gelbach & Miller 2011).

        V = V(c1) + V(c2) - V(c1 x c2)

    Why this matters here: clustering by leader alone assumes independence ACROSS
    leaders. That is false for congestion — a busy minute hits every leader at
    once, so shocks cluster in TIME as well as in identity. One-way leader
    clustering therefore understates the standard errors on anything driven by
    load, which is precisely what the sealing slope is suspected of being.

    The subtraction can leave V non-PSD in finite samples; we apply the standard
    eigenvalue floor (negative eigenvalues truncated to zero) rather than
    reporting a negative variance.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    k = X.shape[1]
    if names is None:
        names = [f"x{i}" for i in range(k)]

    beta = ols(X, y)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)

    V1, G1 = _crve_V(X, resid, np.asarray(c1), XtX_inv)
    V2, G2 = _crve_V(X, resid, np.asarray(c2), XtX_inv)
    inter = np.array([f"{a}|{b}" for a, b in zip(np.asarray(c1), np.asarray(c2))])
    V12, _ = _crve_V(X, resid, inter, XtX_inv)

    V = V1 + V2 - V12
    w, Q = np.linalg.eigh(V)
    V = Q @ np.diag(np.maximum(w, 0.0)) @ Q.T
    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    # the effective cluster count for inference is the SMALLER dimension
    return FitResult(coef=beta, se=se, n=X.shape[0], n_clusters=min(G1, G2), names=names)


def wild_cluster_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    test_idx: int,
    reps: int = 9999,
    seed: int = 0,
    null_value: float = 0.0,
) -> tuple[float, float, tuple[float, float]]:
    """Wild cluster bootstrap-t (WCR: restricted, Rademacher weights).

    Returns (t_observed, p_value, 95% CI by inversion... see note).

    Procedure
    ---------
    1. Fit the RESTRICTED model — the one with the null imposed, i.e. with the
       coefficient of interest fixed at `null_value`. Keep its residuals.
       Imposing the null is what makes this the WCR variant; it is the one that
       performs best in MacKinnon-Webb simulations.
    2. For each replication, draw ONE Rademacher weight per CLUSTER (+1 or -1
       with prob 1/2). Every observation in a cluster gets the same weight, which
       is what preserves the within-cluster correlation structure.
    3. Build a synthetic y* = X @ beta_restricted + w_g * resid_restricted, refit
       the UNRESTRICTED model, and compute the t-statistic against null_value.
    4. The bootstrap p-value is the share of |t*| exceeding the observed |t|.

    The reference distribution is built from the data's own cluster structure,
    so it does not rely on the CRVE being asymptotically Normal — which is
    precisely the assumption that fails when clusters are few.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    # Observed (unrestricted) t-statistic.
    fit = cluster_robust_fit(X, y, clusters)
    t_obs = (fit.coef[test_idx] - null_value) / fit.se[test_idx]

    # --- Step 1: restricted fit, with beta[test_idx] pinned to null_value. ---
    # Drop the tested column, regress the offset-adjusted y on what remains.
    keep = [j for j in range(X.shape[1]) if j != test_idx]
    X_r = X[:, keep]
    y_offset = y - null_value * X[:, test_idx]
    beta_r = ols(X_r, y_offset)
    resid_r = y_offset - X_r @ beta_r

    # Fitted values under the null, on the FULL design.
    fitted_null = X_r @ beta_r + null_value * X[:, test_idx]

    uniq = np.unique(clusters)
    # Map each observation to its cluster's position, so we can broadcast one
    # weight per cluster across all its rows in a single vectorised step.
    cluster_pos = np.searchsorted(uniq, clusters)

    t_star = np.empty(reps)
    for b in range(reps):
        # Step 2: one Rademacher weight per cluster.
        w = rng.choice([-1.0, 1.0], size=len(uniq))
        y_star = fitted_null + resid_r * w[cluster_pos]

        # Step 3: refit unrestricted on the synthetic outcome.
        fit_b = cluster_robust_fit(X, y_star, clusters)
        se_b = fit_b.se[test_idx]
        t_star[b] = (
            (fit_b.coef[test_idx] - null_value) / se_b if se_b > 0 else np.nan
        )

    # Step 4: symmetric bootstrap p-value.
    valid = t_star[~np.isnan(t_star)]
    p = float(np.mean(np.abs(valid) >= abs(t_obs)))

    # A bootstrap-t percentile interval. Note this is the studentised interval,
    # NOT a formal inversion of the bootstrap test (which would require re-running
    # the whole bootstrap on a grid of null values). It is the conventional
    # reporting choice and is what `boottest` calls the "bootstrap-t CI".
    lo, hi = np.percentile(valid, [2.5, 97.5])
    ci = (
        float(fit.coef[test_idx] - hi * fit.se[test_idx]),
        float(fit.coef[test_idx] - lo * fit.se[test_idx]),
    )

    return float(t_obs), p, ci
