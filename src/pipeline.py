# src/pipeline.py
import numpy as np
import pandas as pd
from src.acquisition import choose_acquisition
from src.utils import load_baseline, pick_cluster_center, save_query_text
from src.functions_map import get_dim

def propose_next_query_for_function(fid, method="heuristic", acq="auto"):
    """Propose next query for a function.

    method="heuristic" — Weeks 1-4 cluster-center approach (no model)
    method="surrogate"  — Week 5+ GP-based optimisation

    acq: acquisition function to use when method="surrogate".
      "auto"  — run both UCB and EI, pick whichever proposes the higher predicted mean
      "ucb"   — Upper Confidence Bound (exploration-heavy)
      "ei"    — Expected Improvement (exploitation-heavy)

    Note: Weeks 1-4 used the heuristic method with no acquisition function at all —
    query points were chosen by cluster centres and geometric spread. The acquisition
    function logic was introduced in Week 5 with the surrogate model.
    """
    if method == "surrogate":
        from src.surrogate import fit_gp, optimize_acquisition, _auto_select_acquisition
        gp, X, y = fit_gp(fid)
        if acq == "auto":
            x_star, _ = _auto_select_acquisition(fid, gp, X, y)
        else:
            x_star = optimize_acquisition(fid, gp=gp, X=X, y=y, acq=acq)
        return x_star.tolist()

    # Original heuristic approach
    baseline = load_baseline(fid)
    X = baseline.iloc[:, :-1].values
    dim = get_dim(fid)
    if X.shape[1] != dim:
        # fallback: pad or trim
        X = X[:, :dim]
    acq = choose_acquisition(fid)
    # Simple heuristic: propose cluster center of baseline
    center = pick_cluster_center(X)
    # Clip to [0,1]
    center = np.clip(center, 0.0, 1.0)
    return center.tolist()

def generate_week_queries(week, function_ids=range(1,9), method="heuristic"):
    """Generate query proposals for all functions.

    Use method="surrogate" for Week 5+ GP-based proposals.
    """
    lines = []
    for fid in function_ids:
        vec = propose_next_query_for_function(fid, method=method)
        lines.append(f"Function {fid}: {vec}")
    save_query_text(week, lines)
    return lines

if __name__ == "__main__":
    # Example: generate queries for week 05 using surrogate model
    print(generate_week_queries(5, method="surrogate"))
