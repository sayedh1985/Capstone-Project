# src/surrogate.py
"""Surrogate model module for Bayesian black-box optimisation.

Fits a Gaussian Process per function, optimises an acquisition function,
and proposes the next query point.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic,
)
from scipy.optimize import differential_evolution
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]

FUNCTION_DIMENSIONS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}


def load_all_data(fid):
    """Load baseline + all weekly data for a single function.

    Returns X (n_samples, dim) and y (n_samples,).

    Handles a CSV quirk in the weekly files: some rows have 9 fields
    instead of 10, so the y value ends up in the x8 column instead
    of the y column.  We detect this by checking whether y is NaN and
    falling back to the last non-NaN value in the row.
    """
    dim = FUNCTION_DIMENSIONS[fid]
    x_cols = [f"x{i}" for i in range(1, dim + 1)]

    # --- Baseline (clean CSVs, no quirk) ---
    baseline_path = ROOT / "data" / "baseline" / f"function_{fid}_baseline.csv"
    bl = pd.read_csv(baseline_path)
    X_bl = bl[x_cols].values
    y_bl = bl["y"].values

    # --- Weekly ---
    weekly_dir = ROOT / "data" / "weekly"
    X_parts = [X_bl]
    y_parts = [y_bl]

    for csv_path in sorted(weekly_dir.glob("week_*.csv")):
        wk = pd.read_csv(csv_path)
        row = wk[wk["function_id"] == fid]
        if row.empty:
            continue
        row = row.iloc[0]

        # Extract x values
        x_vals = row[x_cols].values.astype(float)
        if np.any(np.isnan(x_vals)):
            continue

        # Extract y — handle the CSV quirk where y may be NaN
        y_val = row["y"]
        if pd.isna(y_val):
            # y landed in the wrong column; take last non-NaN value
            non_nan = row.dropna()
            y_val = float(non_nan.iloc[-1])

        X_parts.append(x_vals.reshape(1, -1))
        y_parts.append(np.array([y_val]))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def _build_candidate_kernels(dim):
    """Build a list of kernels to try for a given number of dimensions.

    Each kernel gets a constant scaling factor and a noise term on top.
    We try a few different "shapes" — some assume the function is very
    smooth, others allow rougher landscapes. The GP fitting will pick
    the best hyperparameters within each kernel, and then we compare
    across kernels using the log-marginal likelihood score.
    """
    noise = WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    amp = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
    ls_bounds = (1e-2, 1e1)

    kernels = [
        # Matern 5/2 — smooth-ish, our previous default
        amp * Matern(length_scale=np.ones(dim), length_scale_bounds=ls_bounds, nu=2.5) + noise,
        # Matern 3/2 — allows rougher functions
        amp * Matern(length_scale=np.ones(dim), length_scale_bounds=ls_bounds, nu=1.5) + noise,
        # RBF — assumes very smooth functions (infinitely differentiable)
        amp * RBF(length_scale=np.ones(dim), length_scale_bounds=ls_bounds) + noise,
        # RationalQuadratic — kind of a mix of RBFs at different scales
        amp * RationalQuadratic(length_scale=1.0, length_scale_bounds=ls_bounds) + noise,
    ]
    return kernels


def fit_gp(fid):
    """Fit a GP surrogate to all available data for function `fid`.

    Tries several different kernels and picks the one with the best
    log-marginal likelihood — basically letting the data tell us which
    smoothness assumption fits best for this particular function.

    normalize_y=True keeps the GP numerically stable across
    the wide output ranges we see (e.g. F5 spans 0–3300).

    Returns (fitted_gp, X, y).
    """
    X, y = load_all_data(fid)
    dim = X.shape[1]
    candidates = _build_candidate_kernels(dim)

    best_gp = None
    best_score = -np.inf

    for kernel in candidates:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=15,
            random_state=42,
        )
        try:
            gp.fit(X, y)
            score = gp.log_marginal_likelihood_value_
            if score > best_score:
                best_score = score
                best_gp = gp
        except Exception:
            # Some kernels might not converge — just skip them
            continue

    return best_gp, X, y


def _ucb(gp, X_cand, beta=2.0):
    """Upper confidence bound (maximisation)."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    return mu + beta * sigma


def _ei(gp, X_cand, y_best, xi=0.01):
    """Expected improvement (maximisation)."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def optimize_acquisition(fid, gp=None, X=None, y=None, acq="ucb", beta=2.0, xi=0.01):
    """Find the point that maximises the acquisition function.

    Uses differential evolution over [0.001, 0.999]^dim so that
    proposed points stay away from exact boundary artefacts.
    """
    if gp is None:
        gp, X, y = fit_gp(fid)

    dim = FUNCTION_DIMENSIONS[fid]
    bounds = [(0.001, 0.999)] * dim
    y_best = y.max()

    def neg_acq(x):
        x2d = x.reshape(1, -1)
        if acq == "ei":
            return -_ei(gp, x2d, y_best, xi=xi)[0]
        return -_ucb(gp, x2d, beta=beta)[0]

    result = differential_evolution(neg_acq, bounds, seed=42, maxiter=200, tol=1e-8)
    return result.x


def _auto_select_acquisition(fid, gp, X, y):
    """Run both UCB and EI, return whichever proposes a point with higher predicted mean.

    Neither acquisition function is universally better — UCB explores uncertain
    regions while EI focuses on beating the current best. When no explicit strategy
    is specified, we let the model compare both proposals and pick the one the GP
    predicts is more promising.
    """
    x_ucb = optimize_acquisition(fid, gp=gp, X=X, y=y, acq="ucb")
    x_ei  = optimize_acquisition(fid, gp=gp, X=X, y=y, acq="ei")

    mu_ucb = float(gp.predict(x_ucb.reshape(1, -1))[0])
    mu_ei  = float(gp.predict(x_ei.reshape(1,  -1))[0])

    if mu_ucb >= mu_ei:
        return x_ucb, "ucb"
    return x_ei, "ei"


def propose_week_queries(function_ids=range(1, 9), acq_map=None):
    """Generate model-driven proposals for all functions.

    acq_map: optional dict {fid: "ucb" or "ei"}.
      - If provided, uses the specified acquisition function for that function.
      - Any function not in the dict falls back to auto-selection: runs both UCB
        and EI and picks whichever proposes a point with the higher predicted mean.
      - If acq_map is None entirely, auto-selection is used for every function.

    This means the dict is an explicit override, not a blanket default. The model
    will always do something sensible on its own; supply the dict only when you
    want to force a specific strategy based on what you know from recent results.

    Returns a dict {fid: (proposed_x_list, acq_used)}.
    """
    if acq_map is None:
        acq_map = {}
    proposals = {}
    for fid in function_ids:
        gp, X, y = fit_gp(fid)
        if fid in acq_map:
            acq = acq_map[fid]
            x_star = optimize_acquisition(fid, gp=gp, X=X, y=y, acq=acq)
        else:
            x_star, acq = _auto_select_acquisition(fid, gp, X, y)
        proposals[fid] = (x_star.tolist(), acq)
    return proposals


if __name__ == "__main__":
    # Week 7 acquisition strategy:
    # F1, F2, F3 — force UCB: these functions haven't found a good region yet,
    #   auto-selection might prematurely switch to EI on a weak local best.
    # F4, F5, F6, F7, F8 — force EI: known good regions to exploit.
    acq_map = {1: "ucb", 2: "ucb", 3: "ucb", 4: "ei", 5: "ei", 6: "ei", 7: "ei", 8: "ei"}
    proposals = propose_week_queries(acq_map=acq_map)
    for fid, (vec, acq) in proposals.items():
        formatted = ", ".join(f"{v:.6f}" for v in vec)
        print(f"Function {fid} ({acq.upper()}): [{formatted}]")
