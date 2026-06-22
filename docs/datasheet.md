# Datasheet: Bayesian Black-Box Optimisation Capstone

This datasheet documents the optimisation decisions, learning and reasoning across all eight black-box functions in the BBO capstone project.

---

## Function Overview

**1. Which functions does this datasheet describe?**

Functions 1 through 8. Each is a separate unknown black-box function with a different dimensionality, output range and landscape character.

| Function | Dimensions | Output range observed | Character |
|----------|------------|----------------------|-----------|
| F1 | 2D | ~10⁻¹⁵ to 10⁻⁹⁸ | Near-zero everywhere, not learnable |
| F2 | 2D | -0.28 to 0.69 | Smooth, structured, exploitable |
| F3 | 3D | -0.44 to -0.035 | Flat, negative, hard to improve |
| F4 | 4D | -4.03 to 0.70 | Narrow peak, highly sensitive to x3 |
| F5 | 4D | ~1 to 8585 | Sharp corner maximum, all dims critical |
| F6 | 5D | -0.75 to -0.33 | Volatile, many local optima |
| F7 | 6D | 0.12 to 1.76 | Gradual improvement, exploitable |
| F8 | 8D | 9.52 to 9.97 | High-dimensional, narrow good region |

**2. What real-world scenarios do these functions simulate?**

The functions are synthetic but represent the class of problems where a process or system must be tuned through repeated experiments — for example, chemical yield optimisation, drug dosing, engineering parameter tuning, or hyperparameter search in machine learning. In each case, the internal mechanism is hidden and only the output of each trial is observed.

**3. What is the dimensionality of the input?**

Varies by function: F1 and F2 are 2D, F3 is 3D, F4 and F5 are 4D, F6 is 5D, F7 is 6D, and F8 is 8D. All inputs are continuous values in the range [0, 1].

**4. How many initial data points were provided?**

Each function came with a baseline dataset of between 6 and 30 initial evaluations. The exact sizes varied:

| Function | Baseline points | Weekly points added | Total at end |
|----------|----------------|---------------------|-------------|
| F1 | 9 | 13 | 22 |
| F2 | 9 | 13 | 22 |
| F3 | 14 | 13 | 27 |
| F4 | 29 | 13 | 42 |
| F5 | 19 | 13 | 32 |
| F6 | 19 | 13 | 32 |
| F7 | 29 | 13 | 42 |
| F8 | 39 | 13 | 52 |

**5. What does the output represent?**

A single real-valued scalar — the result of evaluating the hidden function at the queried input point. No units are provided. The goal is to maximise this value.

---

## Nature of the Data

**1. Describe the structure of the initial dataset.**

Each baseline CSV contains rows of (x1, x2, ..., xD, y) where D is the function's dimensionality and y is the observed output. All inputs lie in [0, 1]. Weekly results follow the same format and are stored as separate files (week_01.csv through week_13.csv), with all 8 functions in each file using a shared schema of x1 through x8 (unused dimensions are left empty).

**2. How does the dataset evolve as you add new queries weekly?**

One query per function per week was added over 13 weeks. Early weeks used geometric exploration — spreading points to cover the domain — while later weeks targeted specific regions identified as promising. By the final week, each function had shifted from broad coverage to repeated exploitation of its best-known neighbourhood.

**3. Does the function include noise or randomness?**

Most functions appeared deterministic — re-querying at the same point would likely return the same value. However, F6 and F7 showed volatile behaviour where nearby points returned very different outputs, suggesting either a highly irregular landscape or mild evaluation noise. F1 appeared degenerate — outputs spanned 80+ orders of magnitude with no learnable structure.

**4. Do the functions appear unimodal, multimodal, noisy or smooth?**

- F2, F4, F8: Smooth with a single exploitable peak region. GP surrogate predicted well.
- F5: Unimodal with a sharp corner maximum — all four inputs needed to be near 0.999 simultaneously.
- F6, F7: Multimodal or highly irregular. The GP surrogate struggled to predict accurately, and small moves produced large output changes.
- F3: Essentially flat negative surface. Near-impossible to improve on the baseline.
- F1: Degenerate — outputs near machine zero, no gradient signal visible.

---

## Optimisation Strategy

**1. Which optimisation methods were used?**

- Weeks 1–4: Manual heuristics — geometric point spreading, cluster identification, intuition-driven targeting
- Weeks 5–13: Gaussian Process (GP) surrogate models with Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions, optimised via differential evolution

**2. Why this method for these functions?**

With at most 52 data points and no access to gradients, a GP surrogate was the only practical choice. It predicts outputs across the full input space, quantifies uncertainty, and uses that uncertainty to decide where to query next. Random search would waste queries. Grid search is intractable in 6–8 dimensions. Neural networks need far more data than was available.

**3. How was exploration and exploitation balanced?**

Two acquisition functions were used:
- **UCB (Upper Confidence Bound):** adds a multiple of the predicted uncertainty to the predicted mean. Favours uncertain regions — good for functions where the best area hasn't been found yet.
- **EI (Expected Improvement):** scores points by how much better than the current best they are predicted to be. Favours known good regions — good for squeezing out improvements once a promising area is found.

The strategy switched between them per function based on weekly results. An auto-selection mode was also added (W7 onward) that runs both and picks whichever proposes the higher-value point.

**4. Did the strategy change over the weeks?**

Significantly. Early weeks had no model at all — queries were chosen by hand. The surrogate was introduced in W5. W6 added automatic kernel selection. W7 added acquisition auto-selection. From W8 onward, per-function manual overrides were used whenever the model got stuck producing near-duplicate proposals. F5 was systematically probed corner-by-corner across W7–W10 to map which dimensions were critical.

---

## Data Handling and Preprocessing

**1. Were inputs rescaled or normalised?**

No. All inputs were already bounded in [0, 1] by the problem definition, so no additional normalisation was needed. The GP model was trained directly on the raw input vectors.

**2. Were surrogate models trained?**

Yes — a separate Gaussian Process was fitted per function each week using all available data up to that point (baseline + all previous weekly results).

**3. What preprocessing did the surrogate require?**

- **Kernel selection:** four candidate kernels were evaluated per function (Matern 5/2, Matern 3/2, RBF, RationalQuadratic). The kernel with the highest log-marginal likelihood was selected.
- **Noise modelling:** a white noise term was added to each kernel to account for potential observation noise.
- **Hyperparameter fitting:** all kernel parameters (length scales, noise level, amplitude) were learned from data using maximum likelihood with 15 random restarts to avoid poor local fits.
- **Query bounds:** the acquisition optimiser was constrained to [0.001, 0.999] to avoid proposing exact boundary points.

**4. Were outliers or unusual data points handled?**

F1 outputs were near machine zero (10⁻¹⁵ to 10⁻⁹⁸). No special handling was applied — the GP was trained on these values but the model fits were essentially meaningless. F1 queries were continued for completeness but no meaningful optimisation was expected or achieved.

---

## Weekly Iteration and Learning

**1. How did new data change understanding of the function landscapes?**

Each week revealed new structure. F5 appeared to have a smooth surface early on but W6's all-boundary query revealed a massive peak at the corner — a result that reshaped the entire F5 strategy. F4's W5 result showed the baseline data had completely missed a good region (0.696 vs baseline best of -4.026). F2 showed a clear x2=0.999 dependency that was only confirmed after deliberately varying x2 in W11.

**2. Were local optima encountered?**

Yes, particularly in F4 and F7. The F4 surrogate repeatedly proposed points within 0.01 of previous queries, unable to distinguish between them due to limited gradient signal around the good region. F7 saw EI overshoot the best region twice — once dropping from 1.536 to 1.022, and again at the final query (1.756 to 1.391).

**3. Which queried inputs were most informative?**

- F5 corner probes (W7–W10): pulling one dimension to 0.001 each week confirmed all four dimensions were critical for the 8585 peak.
- F2 W11 probe: setting x2=0.654 and watching the output drop to 0.468 (from 0.694) definitively confirmed x2=0.999 was essential.
- F8 W7 query: x5=0.999 pattern established here; all subsequent EI proposals kept x5 high and continued improving.

**4. What would be done differently on a restart?**

- Spend more early queries on F4 — the baseline was entirely in negative territory, meaning the good region was likely elsewhere from the start. A few random restarts in W1–W2 might have found it sooner.
- Avoid EI for F7 in the final week — the last query had no correction opportunity, and UCB's safer exploration would have been lower risk.
- Use a different approach for F1 entirely — with outputs near machine zero, the GP learns nothing useful. A simple random or manual search would have been equivalent at much lower cost.

---

## Performance and Results

**1. What is the best output value achieved per function?**

| Function | Best Value | Week Achieved |
|----------|-----------|--------------|
| F1 | 1.6e-15 | W4 |
| F2 | 0.694 | W10 |
| F3 | -0.035 | Baseline |
| F4 | 0.696 | W5 |
| F5 | 8585.3 | W6 |
| F6 | -0.328 | W7 |
| F7 | 1.756 | W12 |
| F8 | 9.972 | W9 |

**2. Which input vectors produced these values?**

| Function | Best Input Vector |
|----------|------------------|
| F1 | [0.008, 0.003] |
| F2 | [0.716599, 0.999000] |
| F3 | Baseline point (exact coordinates in baseline CSV) |
| F4 | [0.413, 0.410, 0.355, 0.423] |
| F5 | [0.999, 0.999, 0.999, 0.999] |
| F6 | [0.508, 0.417, 0.574, 0.696, 0.001] |
| F7 | [0.001, 0.248, 0.001, 0.209, 0.357, 0.713] |
| F8 | [0.170, 0.043, 0.189, 0.260, 0.999, 0.640, 0.358, 0.557] |

**3. How confident are we that these are near the global maximum?**

Confidence varies significantly by function:

- **F5:** High confidence. The corner map was systematically confirmed — every off-corner probe returned 4399 vs 8585 at the full corner. The peak gradient was also measured (4% drop per 0.019 shift), so we know the surface well near the optimum.
- **F7, F8:** Moderate confidence. Both improved steadily over many weeks but neither reached a stable plateau. There may be higher values nearby.
- **F2:** Moderate confidence. The x2=0.999 requirement was clearly established, and x1≈0.716 was identified as a sweet spot, but the region wasn't exhaustively mapped.
- **F4:** Low confidence. The best result (W5) was never reproduced, suggesting it may have been a narrow spike the model was unable to reliably locate.
- **F3:** Very low confidence for the weekly queries. The baseline best was never beaten. The true maximum may require a very specific input combination that was never sampled.
- **F1:** No meaningful confidence — the function outputs are near machine zero and appear to have no learnable structure.

**4. Did results align with expectations?**

Partly. F5's massive jump in W6 (from 3338 to 8585) was a genuine surprise — the all-boundary query hit a region no prior reasoning had suggested. F4's W5 breakthrough was similarly unexpected. F3's stubborn resistance to improvement was frustrating but consistent with what the baseline data suggested (very flat surface). F1's near-zero outputs were apparent from the baseline and confirmed throughout.

---

## Ethical, Practical and General Considerations

**1. How does this task relate to real-world applications?**

Black-box optimisation with limited query budgets appears in many high-stakes domains: drug discovery (each trial requires a lab experiment), materials science (each synthesis is expensive), hardware design (each simulation is computationally costly), and clinical trials (each patient cohort is a limited resource). The methodology here — surrogate models, acquisition functions, systematic probing — transfers directly to those problems.

**2. What limitations arise from the synthetic nature of the function?**

The functions are noiseless (or near-noiseless) and bounded. Real problems often have measurement error, constraints that aren't rectangular, non-stationarity, and input interactions that shift over time. The clean [0,1] domain and fixed dimensionality also simplify the problem considerably compared to real parameter spaces, which may have mixed types, correlated inputs, or infeasible regions.

**3. Would this strategy scale to more serious or more expensive problems?**

The core approach scales well — Gaussian Processes are standard practice for expensive black-box optimisation. However, GPs scale poorly with dimensionality (beyond ~20 dimensions the surrogate becomes unreliable) and with data size (fitting time grows cubically). For very expensive problems (e.g., clinical trials), the query budget would be even tighter and each acquisition decision would need more rigorous justification.

**4. What risks should a future user be aware of?**

- **Near-duplicate proposals:** the surrogate can get stuck proposing the same point repeatedly, especially near boundaries or in flat regions. A duplicate check before each submission is essential.
- **Boundary behaviour:** EI at a hard boundary (0.999 or 0.001) will keep returning the boundary point. Manual overrides or a boundary penalty are needed.
- **Kernel selection instability:** with very few data points, the "best" kernel can change dramatically from week to week. Decisions made on early kernel fits may not reflect the true function shape.
- **Degenerate functions:** some functions (like F1) are not meaningfully optimisable with a GP surrogate. Identifying these early and redirecting effort elsewhere is valuable.
