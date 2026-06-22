# Model Card

---

## Model Description

**Input:** A vector of continuous values in [0, 1], representing a query point for one of 8 unknown black-box functions. Dimensionality varies by function (2D to 8D).

**Output:** A predicted output value and an uncertainty estimate at any point in the input space. The model also produces a recommended next query point via an acquisition function.

**Model Architecture:** Gaussian Process (GP) surrogate with automatic kernel selection. For each function, the model tries four different kernel shapes and picks the one that fits the observed data best:

- Matern 5/2 — assumes a fairly smooth function
- Matern 3/2 — allows rougher, less predictable behaviour
- RBF (Squared Exponential) — assumes a very smooth function
- RationalQuadratic — handles functions that vary at multiple scales

The winning kernel is chosen using log-marginal likelihood (the model's own score for how well it explains the data). On top of the kernel, a white noise term accounts for observation noise, and a constant scaling factor adjusts the output amplitude. All internal parameters (length scales, noise level, amplitude) are learned from data with 15 random restarts to avoid bad fits.

The model feeds into an acquisition function (UCB or EI) that scores candidate query points by balancing predicted value against uncertainty, and a differential evolution optimiser finds the highest-scoring point.

---

## Performance

### Kernel selection results (W6)

| Function | Dims | Kernel Selected | Log-Marginal Likelihood |
|----------|------|----------------|------------------------|
| F1 | 2 | RBF | -19.26 |
| F2 | 2 | RBF | -10.98 |
| F3 | 3 | RBF | -20.26 |
| F4 | 4 | RationalQuadratic | -11.53 |
| F5 | 4 | RBF | -17.54 |
| F6 | 5 | Matern | -26.53 |
| F7 | 6 | RBF | -40.48 |
| F8 | 8 | Matern | 6.34 |

### Final best results (after all 13 weeks)

| Function | Best Value | Source | Baseline Best | Beats Baseline? |
|----------|-----------|--------|--------------|----------------|
| F1 | 1.6e-15 | W4 | 7.7e-16 | Yes |
| F2 | **0.694** | **W10** | 0.611 | Yes |
| F3 | -0.035 | Baseline | -0.035 | No |
| F4 | **0.696** | **W5** | -4.026 | Yes |
| F5 | **8585.3** | **W6** | 1088.9 | Yes |
| F6 | **-0.328** | **W7** | -0.714 | Yes |
| F7 | **1.756** | **W12** | 1.365 | Yes |
| F8 | **9.972** | **W9** | 9.598 | Yes |

7 out of 8 functions beat the baseline. F3 never surpassed the baseline (-0.035), though W12 came within 0.003.

### Model-driven improvements

- W5 was the first week using surrogate model queries
- F4: massive improvement (baseline -4.026 to 0.696) — model found a better region entirely
- F8: slight improvement (9.598 to 9.729)
- F5: dropped from 3338.8 (W4, heuristic) to 1600 (W5, model) — model explored a boundary point that didn't pay off

---

## Limitations

### Known limitations

- **Tiny data budget.** Each function has 15–45 total observations (baseline + 5 weekly). GPs can struggle with this few points, especially in higher dimensions (F7 has 6D, F8 has 8D).

- **Baseline won for 1 out of 8 functions.** F3 was the only function where the baseline best was never beaten across all 13 weeks. For all other functions, the surrogate model eventually found a better region.

- **F1 is effectively unsolvable with this approach.** Outputs span 10^-15 to 10^-291 — the GP can't meaningfully distinguish between these values. Any model fit here is basically noise.

- **Boundary behaviour.** The model sometimes pushes queries toward exact domain boundaries ([0.001, 0.999]). F5 W5 and F8 W6 both show this pattern. These extreme points may not always be good — F5 dropped significantly when the model suggested a boundary point.

- **No ground truth.** We never see the actual function — only point evaluations. We can't measure how close we are to the true optimum or whether the model's predicted surface is accurate globally.

- **Kernel selection is data-dependent.** With limited data, the "best" kernel might change as more observations arrive. A kernel that looks right with 20 points might not be the best choice with 30.

---

## Trade-offs

### Exploration vs exploitation (UCB vs EI)

- **UCB** explores more aggressively — it queries uncertain areas even if the predicted value isn't great. Risk: wastes queries on low-value regions. Benefit: might find hidden peaks.
- **EI** focuses on beating the current best — it won't bother with uncertain areas unless the prediction looks promising. Risk: might miss better regions it never checks. Benefit: more likely to improve incrementally.
- Final approach: per-function selection based on results, with an auto-selection mode (added W7) that runs both and picks the better proposal. Manual overrides were applied when the model got stuck in near-duplicate loops.

### Kernel smoothness assumptions

- **Smoother kernels (RBF)** produce gentler predicted surfaces. They're less likely to overfit but might miss sharp features.
- **Rougher kernels (Matern 3/2)** can capture sudden changes but are more prone to fitting noise in sparse data.
- Trade-off: with very few data points, smoother kernels are generally safer. As data grows, rougher kernels may become more appropriate.

### Model complexity vs data availability

- We have 4 candidate kernels but only 15–45 data points per function. There's a risk of overfitting the kernel selection itself — choosing a kernel that happens to score well on limited data but doesn't generalise.
- Mitigation: the log-marginal likelihood naturally penalises overly complex models (it's not just fit quality — it accounts for model complexity).

---

## Weekly Changelog

Track what changed in the model each week — architecture, performance, new limitations discovered, trade-off decisions made. This is the raw log that feeds the sections above.

### W1–W4 (no model)
- Queries chosen manually using geometric spread, cluster centres, and intuition
- No surrogate model — too little data
- Acquisition function logic existed (`src/acquisition.py`) but was rule-based, not model-driven
- **Limitation discovered:** F1 outputs are near-zero (10^-79 to 10^-291), unlikely to be learnable

### W5 (surrogate model introduced)
- **Architecture:** GP with Matern 5/2 kernel for all 8 functions (single kernel, no selection)
- **Acquisition:** UCB for all functions
- **Performance:**
  - F4 breakthrough: -4.026 to 0.696 — model found a completely different region
  - F8 new best: 9.598 to 9.729
  - F5 dropped: 3338.8 to 1600 — boundary point [0.001, 0.999, 0.999, 0.001] didn't work
  - F7 weekly best improved (0.377 to 0.654) but baseline 1.365 still ahead
- **Limitation discovered:** model pushed F5 to boundary and it backfired
- **Limitation discovered:** baseline still beats model for F2, F3, F6, F7
- **Trade-off noted:** model revealed F5's x4 dimension is unimportant — contradicted our manual assumption from W1-W4

### W6 (kernel selection + per-function acquisition)
- **Architecture change:** added automatic kernel selection — tries Matern 5/2, Matern 3/2, RBF, RationalQuadratic per function, picks best by log-marginal likelihood
- **Kernel results:** most functions preferred RBF (smoother), F4 picked RationalQuadratic, F6 and F8 kept Matern
- **Acquisition change:** switched from blanket UCB to per-function UCB/EI. F4, F7, F8 on EI (exploit promising regions), F1, F2 on UCB (still searching)
- **Performance:**
  - F5: new best 8585.3 (was 3338.8) — all-upper-boundary point paid off massively
  - F6: -0.464, first time beating baseline (-0.714)
  - F7: 1.149, new weekly best (was 0.654) but baseline 1.365 still ahead
  - F4: dropped to -1.07 (was 0.696 in W5) — EI found a nearby point but x3 shift was too large
  - F8: 9.522, slipped from W5 best of 9.729
- **Limitation confirmed:** boundary behaviour — F5 is now at the hard upper boundary [0.999,...,0.999], model may keep proposing the same point

### W7 (acquisition auto-selection)
- **Architecture change:** default behaviour when no strategy is specified now auto-selects between UCB and EI — runs both, picks whichever proposes a point with higher predicted mean. Explicit `acq_map` dict overrides this per function.
- **Rationale:** previous default (UCB for all) was a hidden assumption. Auto-selection is more principled — the model compares both strategies and picks the better proposal rather than relying on a blanket rule.
- **Acquisition for W7:** F1, F2, F3 → UCB (forced); F4, F5, F6, F7, F8 → EI (forced)
- **Performance:**
  - F8: new best 9.930 (was 9.729)
  - F6: new best -0.328 (was -0.464), consistently improving
  - F5: 4399 with x3=0.001 — confirms x3 must be high for the peak region
  - F4: 0.111, positive again but below W5 best of 0.696
  - F7: dropped to 0.123 — EI candidate moved into a poor region
- **Limitation confirmed:** surrogate getting stuck for F4 — both UCB and EI returned near-duplicates of W7 query, required manual override
- **Limitation confirmed:** F7 EI failed; switched back to UCB for W8

### W8 (boundary probing + manual overrides)
- **No architecture change**
- **Acquisition for W8:** F1, F2, F3 → UCB; F4 → manual override near W5 winner; F5 → manual override [0.999,0.999,0.999,0.001] to probe x4; F6, F8 → EI; F7 → UCB (switched back from EI)
- **Performance:**
  - F4: 0.549, recovering toward W5 peak of 0.696 — manual point near x3=0.354 working
  - F5: 4399 — same drop as W7 x3 probe. Confirms x4 is also critical; both x3 and x4 must be 0.999 for the 8585 peak
  - F6: -0.458, dropped from W7's -0.328 — EI overshot
  - F7: 1.140, UCB holding steady near W6's 1.149
  - F8: 9.874, slipped from 9.930 — x8=0.001 in EI candidate was counterproductive
- **Key finding:** F5 now has a complete corner map — x1=x2=x3=x4=0.999 required for peak. W9 probes x1.

### W9 (F5 x1 probe, F4 neighbourhood exploration)
- **No architecture change**
- **Acquisition for W9:** F1, F2, F3, F6, F7 → UCB; F4 → manual [0.440,0.430,0.354,0.410]; F5 → manual [0.001,0.999,0.999,0.999] (probe x1); F8 → EI
- **Rationale:** F6 switched back to UCB after EI dropped two weeks running. F4 manual keeps x3 at the W5 sweet spot while varying other dimensions to learn the neighbourhood. F5 systematic probing continues.

### W9 (F5 x1 probe results + two new bests)
- **No architecture change**
- **Performance:** F7 new best 1.536 (beats baseline 1.365 for first time); F8 new best 9.972; F5 x1 probe returned 4399 — corner map complete, all four dims need 0.999; F6 worst result yet (-0.634) with UCB; F4 drifting (0.393)
- **Key finding:** F5's optimum is definitively the full upper corner [0.999,0.999,0.999,0.999]

### W10 (F7/F8 exploitation, F5 x2 probe, F6 back to EI)
- **No architecture change**
- **Acquisition for W10:** F1, F2, F3 → UCB; F4, F6, F7, F8 → EI; F5 → manual [0.999,0.001,0.999,0.999] (probe x2)
- **Performance:** F2 new best 0.694 (beats baseline 0.611 for first time); F5 x2 probe returned 4399 — corner map complete; F7 dropped to 1.022 (EI overshot from 1.536); F8 slipped to 9.952; F4 drifted to 0.295 in near-dup loop; F3 worst ever at -0.435
- **Limitation confirmed:** F4 surrogate producing near-duplicate proposals — model sees little gradient signal around the good region and keeps returning the same neighbourhood

### W11 (F4 manual escape, F5 peak sharpness, F7 back to UCB)
- **No architecture change**
- **Acquisition for W11:** F1, F2, F3 → UCB (auto); F4 → manual [0.430,0.390,0.354,0.460]; F5 → manual [0.980,0.999,0.999,0.999]; F6, F8 → EI; F7 → UCB override
- **Performance:** F7 new best 1.685 (beats W9's 1.536); F5 returned 8233 at x1=0.980 — peak is gradual (~4% drop per 0.019 shift); F2 dropped to 0.468 — x2=0.654 confirmed x2=0.999 essential; F4 went negative at -0.259; F8 slipped to 9.903
- **Key finding:** F5 peak gradient confirmed — output changes smoothly near the boundary, not a step function

### W12 (F7 EI exploitation, F2 manual x2=0.999, F5 x2 sharpness probe)
- **No architecture change**
- **Acquisition for W12:** F1, F3 → UCB; F2 → manual [0.780,0.999]; F4, F6, F7, F8 → EI; F5 → manual [0.999,0.980,0.999,0.999]
- **Rationale:** F7 EI exploits new best 1.685. F2 model proposed x2=0.001 (both UCB and EI) despite confirmed x2=0.999 evidence — manual override forces back to the good region at higher x1. F5 probes x2 sharpness to mirror W11 x1 probe.

### W12 (F7 EI exploitation, F5 x2 sharpness, F2 x1=0.780 test)
- **No architecture change**
- **Performance:** F7 new best 1.756 (third consecutive); F5 x2=0.980 returned 8233 — gradient symmetric; F3 reached -0.038, nearest to baseline -0.035; F6 -0.353 (best since W7); F2 0.128 — x1=0.780 too far from sweet spot
- **Key finding:** F5 peak gradient is symmetric across x1 and x2; both drop ~4% per 0.019 shift from 0.999

### W13 (final round)
- **Acquisition for W13:** F1, F2 → UCB; F3, F6, F7, F8 → EI; F4 → manual [0.460,0.380,0.354,0.400]; F5 → manual [0.999,0.999,0.985,0.999]
- **Rationale:** Final query — all functions target best known regions. F5 uses nearest clean point to confirmed optimum (cannot resubmit W6 exact point). F4 manual keeps x3=0.354 but avoids near-dup loop.
- **Performance:**
  - F5: 8323.6 — best since W6, higher than W11/W12 x1/x2=0.980 probes (both 8233). x3 dimension slightly less sensitive.
  - F2: 0.665, close to W10 overall best 0.694. UCB near x1=0.700, x2=0.999 confirmed good.
  - F8: 9.965, close to W9 overall best 9.972. Consistent EI improvement in this region.
  - F7: 1.391, dropped from W12 best 1.756 — EI overshot the peak on the final attempt.
  - F6: -0.422, slight slip from W12's -0.353.
  - F3: -0.069, worse than W12's near-baseline -0.038.
  - F4: -0.448, negative again — manual point still couldn't hold the W5 region.
  - F1: -9.27e-16, near-zero as always.
- **Final outcome:** 7 out of 8 functions beat the baseline. Only F3 could not surpass its baseline best of -0.035 across all 13 weeks.
