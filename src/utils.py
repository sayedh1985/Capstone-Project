# src/utils.py
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_baseline(fid):
    path = ROOT / "data" / "baseline" / f"function_{fid}_baseline.csv"
    return pd.read_csv(path)

def load_week(week):
    path = ROOT / "data" / "weekly" / f"week_{week:02d}.csv"
    return pd.read_csv(path)

def save_query_text(week, lines):
    path = ROOT / "queries" / f"week_{week:02d}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))

def best_so_far(fid, weekly_df):
    """Return best observed y for a function across weekly_df (maximisation)."""
    rows = weekly_df[weekly_df["function_id"] == fid]
    if rows.empty:
        return None
    return rows["y"].max()

def pick_cluster_center(X):
    """Simple cluster center: mean of provided points (rows)."""
    return np.mean(X, axis=0)
