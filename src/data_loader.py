import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_baseline(fid):
    return pd.read_csv(DATA_DIR / "baseline" / f"function_{fid}_baseline.csv")

def load_week(week):
    return pd.read_csv(DATA_DIR / "weekly" / f"week_{week:02d}.csv")
