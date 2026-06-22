# src/acquisition.py
import numpy as np
from scipy.stats import norm

def ucb(mean, std, beta=2.0):
    """Upper Confidence Bound acquisition (higher is better)."""
    return mean + beta * std

def ei(mean, best, std, eps=1e-9):
    """Expected Improvement acquisition (higher is better)."""
    std_safe = std + eps
    z = (mean - best) / std_safe
    return (mean - best) * norm.cdf(z) + std_safe * norm.pdf(z)

def choose_acquisition(function_id):
    """Rule-based acquisition choice used in Weeks 1–4."""
    if function_id in [1, 3, 4, 6, 7, 8]:
        return "ucb"
    if function_id in [2, 5]:
        return "ei"
    return "ucb"
