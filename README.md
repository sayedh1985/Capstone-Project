# Black-Box Optimisation (BBO) Capstone Project

*A Bayesian optimisation project for exploring and maximising unknown black-box functions. Built using iterative modelling, uncertainty-aware decision-making and practical ML heuristics.*

---

## Non-Technical Explanation

Imagine you need to find the best settings for a machine but you have no manual and can only test one combination of settings per week. That is this project. Eight hidden functions act as the machines, each taking a set of inputs and returning a score. Over 13 weeks, I submitted one test per function per week and used the results to build a predictive model that learned where the good settings were. The model balanced trying new areas against refining what it had already found to be promising. By the end, 7 out of 8 functions returned scores better than the starting baseline, with some improving by several orders of magnitude.

---

## 1. Project Overview

This repository contains my work for the Black-Box Optimisation (BBO) capstone challenge. The task is to optimise eight unknown functions using only query-response feedback. Each function varies in dimensionality, smoothness and noise, and their internal structure is completely hidden.

The aim is to design a strategy that learns efficiently from limited data, balances exploration and exploitation, and adapts as more information becomes available. This mirrors real-world ML scenarios where we often work with incomplete information, expensive evaluations or systems we cannot directly inspect.

---

## 2. Data

### Sources

- **Baseline data:** provided at the start of the project - between 6 and 30 evaluations per function, covering the input space broadly.
- **Weekly data:** one query per function per week, submitted over 13 weeks. Results returned as a single scalar per query.

All data is stored in this repository under `data/`. The datasets are small (at most 52 rows per function) and are hosted directly.

### Structure

| Folder | Contents |
|--------|----------|
| `data/baseline/` | 8 CSV files, one per function. Columns: x1...xD, y |
| `data/weekly/` | 13 CSV files (week_01 through week_13). All 8 functions per file, shared x1...x8 schema with empty cells for unused dimensions |

No external data sources were used. The functions and their evaluations were provided by the course.

### Limitations

- Tiny data budget: 22-52 total observations per function across all 13 weeks
- No repeated evaluations - cannot measure noise directly
- No access to the underlying function - cannot verify proximity to the true optimum
- See [docs/datasheet.md](docs/datasheet.md) for full data documentation

---

## 3. Model

A separate **Gaussian Process (GP)** surrogate was fitted per function each week using all observations up to that point. The GP predicts outputs across the full input space and quantifies uncertainty at every point. This uncertainty is used directly by the acquisition function to decide where to query next.

**Why a GP surrogate?** With at most 52 data points and no access to gradients or function structure, a GP was the only practical choice. It is data-efficient, provides calibrated uncertainty estimates, and naturally handles the exploration-exploitation trade-off through acquisition functions. Neural networks and tree-based models need far more data; random search ignores information from previous queries entirely.

**Kernel selection:** four candidate kernel shapes are evaluated per function (Matern 5/2, Matern 3/2, RBF, RationalQuadratic). The kernel with the highest log-marginal likelihood is selected automatically each week.

**Acquisition functions:**
- **UCB (Upper Confidence Bound):** explores uncertain regions - used when the best area has not yet been identified
- **EI (Expected Improvement):** exploits known good regions - used once a promising area has been found

The strategy per function switched between UCB and EI based on weekly results. An auto-selection mode (added Week 7) runs both and picks whichever proposes the higher-value point. Manual overrides were applied when the model got stuck proposing near-duplicate queries.

See [docs/model_card.md](docs/model_card.md) for full model documentation.

---

## 4. Hyperparameter Optimisation

The GP surrogate has several settings that affect how it learns and where it queries next.

| Hyperparameter | What it controls | How it's set |
|---|---|---|
| **Kernel shape** | How smooth or rough the model assumes the function is | Auto-selected per function - tries 4 options, picks best by log-marginal likelihood |
| **Length scales** (per dimension) | How far apart inputs need to be before output changes noticeably | Learned from data during model fitting |
| **Noise level** | How much random noise the model expects in outputs | Learned from data during model fitting |
| **Acquisition function** (UCB or EI) | Whether to explore uncertain areas or exploit known good regions | Chosen per function based on weekly results |
| **UCB beta** (2.0) | Weight on uncertainty vs predicted value | Fixed - standard default |
| **EI xi** (0.01) | Minimum improvement threshold | Fixed - standard default |
| **Query bounds** ([0.001, 0.999]) | Keeps queries away from exact domain edges | Fixed |

All internal parameters (length scales, noise, amplitude) are re-learned from scratch each week as new data arrives. The model runs 15 random restarts during fitting to avoid poor local solutions.

### How the approach evolved

| Week | Kernel | Acquisition |
|------|--------|-------------|
| W1-W4 | No model | Manual heuristics |
| W5 | Matern 5/2 for all | UCB for all |
| W6 | Auto-selected per function | UCB or EI per function |
| W7-W13 | Auto-selected per function | Auto-select with per-function overrides |

---

## 5. Results

### Final best values across all 13 weeks

| Function | Best Value | Week | Baseline Best | Beats Baseline? |
|----------|-----------|------|--------------|----------------|
| F1 | 1.6e-15 | W4 | 7.7e-16 | Yes |
| F2 | **0.694** | W10 | 0.611 | Yes |
| F3 | -0.035 | Baseline | -0.035 | No |
| F4 | **0.696** | W5 | -4.026 | Yes |
| F5 | **8585.3** | W6 | 1088.9 | Yes |
| F6 | **-0.328** | W7 | -0.714 | Yes |
| F7 | **1.756** | W12 | 1.365 | Yes |
| F8 | **9.972** | W9 | 9.598 | Yes |

**7 out of 8 functions beat the baseline.** F3 was the only function that could not surpass its starting best, though Week 12 came within 0.003 of matching it.

### Key findings

- **F4:** the surrogate found a region in Week 5 that the baseline had completely missed - output jumped from -4.026 to 0.696 in the model's first week.
- **F5:** systematic corner probing across Weeks 7-10 confirmed all four input dimensions must be near 0.999 simultaneously for the 8585 peak. Pulling any single dimension to 0.001 dropped the output to ~4399.
- **F7:** improved steadily across multiple weeks (1.365 baseline -> 1.536 -> 1.685 -> 1.756) through a combination of UCB exploration and EI exploitation.
- **F1:** effectively unoptimisable - outputs span 80+ orders of magnitude near machine zero with no learnable gradient.

See [docs/strategy_evolution.md](docs/strategy_evolution.md) for the full week-by-week account.

---

## 6. Technical Approach (Weeks 1-13)

### Weeks 1-4: Manual Exploration

Queries chosen by hand using geometric spread, cluster identification and structural patterns in the baseline data. No surrogate model - too little data to fit one reliably.

### Week 5: Surrogate Model Introduced

With 14-44 observations per function, a GP surrogate was fitted for each function. Queries switched from manual heuristics to model-driven proposals. First result: F4 jumped from -4.026 to 0.696 - the model found a completely different region.

### Week 6: Kernel Selection + Per-Function Acquisition

Added automatic kernel selection (4 candidates, best by log-marginal likelihood). Per-function acquisition strategy: functions with promising regions switched to EI, others kept UCB. F5 hit 8585.3 using the full upper-boundary query.

### Week 7: Acquisition Auto-Selection

Added a mode that runs both UCB and EI and picks whichever proposes the better point. Explicit overrides still available per function.

### Weeks 8-10: Systematic Probing

F5 corner dimensions probed one by one to confirm which inputs drive the peak. F4 guided with manual overrides near the known good region. F2 beat its baseline for the first time in Week 10 (0.694 vs 0.611).

### Weeks 11-13: Exploitation and Peak Sharpness

F5 peak gradient measured - approximately 4% output drop per 0.019 shift from the boundary, confirming a gradual rather than cliff-like peak. F7 reached a new best of 1.756 in Week 12. Final week targeted best-known regions for all functions.

---

## 7. Repository Structure

```
bayesian-bbo-capstone/
├── README.md
├── requirements.txt
├── data/
│   ├── baseline/          # Initial sample data (8 CSVs, one per function)
│   └── weekly/            # Weekly query results (week_01 through week_13)
├── src/
│   ├── surrogate.py       # GP surrogate model and acquisition optimisation
│   ├── pipeline.py        # Query generation pipeline
│   ├── acquisition.py     # UCB and EI scoring functions
│   ├── utils.py           # Data loading and helper utilities
│   ├── functions_map.py   # Function dimension mappings
│   ├── data_loader.py     # Simple CSV loaders
│   └── evaluate.py        # Result evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Baseline analysis and weekly progress
│   ├── 02_surrogate_model_demo.ipynb      # Surrogate model fitting and predicted vs actual
│   └── 03_acquisition_and_pipeline.ipynb  # Kernel selection, UCB vs EI, end-to-end pipeline
├── queries/               # Submitted query vectors (week_01 through week_13)
├── results/               # Weekly result summaries (week_01 through week_13)
└── docs/
    ├── datasheet.md           # Data documentation
    ├── model_card.md          # Model documentation
    └── strategy_evolution.md  # Week-by-week strategy and results log
```

---

## 8. Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate surrogate model proposals for all 8 functions:

```python
from src.surrogate import propose_week_queries
print(propose_week_queries())
```

Or run the full pipeline:

```python
from src.pipeline import generate_week_queries
generate_week_queries(5, method='surrogate')
```

The notebooks can be run from the `notebooks/` directory. They expect the project root as the parent folder (standard Jupyter setup).

---

## 9. Key Lessons

- **Data over intuition.** Early manual assumptions about which input dimensions mattered were often wrong. The surrogate model revealed the true structure.
- **Knowing when to model.** Weeks 1-4 used heuristics because the data was too sparse for a reliable surrogate. Week 5 was the right time to switch.
- **Exploration-exploitation discipline.** With one query per function per week, every submission has a cost. Acquisition functions make the trade-off principled rather than guesswork.
- **Knowing when to override the model.** The surrogate occasionally got stuck proposing near-duplicate points. Recognising this and applying manual overrides was essential for F4 and F5.
