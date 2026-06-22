# src/evaluate.py
import pandas as pd
from src.data_loader import load_week

def best_per_function(week):
    df = load_week(week)
    bests = df.groupby("function_id")["y"].max().reset_index()
    return bests

if __name__ == "__main__":
    print(best_per_function(4))
