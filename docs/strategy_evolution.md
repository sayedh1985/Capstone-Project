# Strategy Evaluation: Weekly Bayesian Optimisation Progress

This document evaluates how the optimisation strategy developed week by week across the Bayesian Black-Box Optimisation capstone project. The strategy gradually moved from manual exploration to a more systematic model-based approach using Gaussian Process surrogate models, EI/UCB acquisition functions, and function-specific judgement.

---

## Week 1: Initial Exploration

The first week focused mainly on understanding the search space. Since there was not enough information to build a reliable model, the strategy was based on broad exploration. The aim was to test different parts of the input domain and avoid concentrating too early on one small region.

This helped establish the first comparison points for each function and gave an early idea of which functions were smooth, flat, noisy, or highly sensitive.

**Evaluation:**  
This was a necessary starting point. The results were not always strong, but the week gave useful coverage of the search space and helped avoid early overfitting to weak assumptions.

---

## Week 2: Early Pattern Recognition

In Week 2, the strategy started to use the first signs of structure in the data. Some functions showed possible promising regions, while others still looked flat or difficult to interpret.

For functions with clearer patterns, the query choices moved slightly closer to exploitation. For uncertain functions, broader exploration was still preferred.

**Evaluation:**  
The strategy became more informed, but it was still mostly manual. This week helped identify which functions were likely to benefit from local refinement and which needed more exploration.

---

## Week 3: Function-Specific Direction

By Week 3, the same strategy was no longer suitable for every function. Lower-dimensional functions were easier to analyse, while higher-dimensional functions required more cautious exploration.

The strategy started to become function-specific. Promising functions received more focused query points, while difficult functions were still explored more broadly.

**Evaluation:**  
This was an important shift. Treating all functions the same would have wasted queries, especially because the functions had very different dimensionality and behaviour.

---

## Week 4: Targeted Manual Refinement

Week 4 continued to use manual decision-making, but the query points became more targeted. The focus was on testing areas that appeared promising from the earlier weeks.

Some functions began to show clearer regions of improvement. Others remained difficult because the output changed only slightly or behaved unpredictably.

**Evaluation:**  
This week improved the quality of the search, but the lack of a formal surrogate model limited how much could be learned from the data. A more systematic method was needed for the next stage.

---

## Week 5: Gaussian Process Strategy Introduced

Week 5 marked the beginning of the model-based strategy. Gaussian Process models were introduced to predict the output surface and estimate uncertainty.

Instead of choosing points only by intuition, the model generated predictions across candidate points. The acquisition logic helped balance predicted value and uncertainty.

The strategy used the available data to suggest points that were either likely to improve the result or likely to reveal useful new information.

**Evaluation:**  
This was a major improvement in the methodology. The model helped guide the search more systematically. However, some model suggestions were risky, especially near boundaries, so human review was still important.

---

## Week 6: Stronger Model Control and Better Acquisition Choices

In Week 6, the strategy became more refined. The model was no longer treated as a single fixed method for all functions. The approach started to consider how each function behaved.

Functions with promising regions were pushed toward exploitation, while uncertain functions kept more exploration. This better matched the purpose of EI and UCB.

Function 5 showed a major improvement after exploring an extreme high-value region. Function 6 also improved compared with its baseline.

**Evaluation:**  
This week showed the value of model-guided search. It also showed that boundary points can sometimes be extremely useful, but they can also be risky. The main lesson was that acquisition choices must match the function behaviour.

---

## Week 7: Balancing EI and UCB More Carefully

Week 7 continued the model-based approach and placed more attention on the balance between exploration and exploitation.

EI was useful for functions where good regions had already been found. UCB remained useful for functions where the model was still uncertain or where the surface looked unstable.

Some functions improved strongly, especially in higher-dimensional cases. However, other functions showed that EI can overshoot if the model becomes too confident around a weak region.

**Evaluation:**  
The strategy became more mature this week. The results confirmed that EI and UCB should not be applied blindly. The best acquisition function depends on the function's current learning stage.

---

## Week 8: Manual Overrides and Boundary Testing

In Week 8, manual overrides became an important part of the strategy. Some model suggestions were too similar to previous points or moved away from useful regions.

For Function 5, the strategy tested whether the high-value peak required all dimensions to stay near the boundary. This gave useful information about which dimensions were critical.

For Function 4, manual adjustment was used to stay close to the earlier good region while avoiding near-duplicate model suggestions.

**Evaluation:**  
This week showed that the model was helpful but not fully reliable on its own. Human interpretation improved the search by correcting model behaviour when it became too narrow or repetitive.

---

## Week 9: Confirming Important Dimensions

Week 9 focused on testing whether certain dimensions were truly important. Function 5 was explored systematically by changing one boundary dimension at a time.

This helped confirm that the strongest region required several dimensions to stay high. Function 7 and Function 8 also showed strong improvement, proving that continued exploration in high-dimensional spaces was still valuable.

**Evaluation:**  
This was one of the most informative weeks. The strategy was not only trying to improve results, but also trying to understand why certain regions worked. This made later decisions more confident.

---

## Week 10: Exploiting Strong Regions and Testing Assumptions

Week 10 used the information gained from earlier weeks to focus more strongly on promising regions.

Function 2 achieved a new best result by exploring a region where one coordinate appeared especially important. Function 5 confirmed that another dimension was also critical for the high-value peak.

However, some functions dropped when EI moved too aggressively away from the best neighbourhood.

**Evaluation:**  
This week showed both the strength and weakness of exploitation. When the model was correct, it improved results. When it overestimated a nearby region, the output dropped. The strategy needed continued adjustment.

---

## Week 11: Refining the Best Regions

Week 11 focused on refining confirmed regions rather than searching randomly. Function 5 tested how sharply the peak declined when one coordinate was moved slightly away from the boundary.

Function 2 tested whether the second coordinate needed to stay high. The result showed that moving it away caused a significant drop, confirming its importance.

Function 7 improved again after returning to a more exploratory strategy.

**Evaluation:**  
This week was useful because it converted earlier guesses into stronger evidence. The strategy became more analytical, using each query to test a specific hypothesis.

---

## Week 12: Focused Exploitation Before the Final Round

Week 12 used more focused exploitation for functions that had recently improved. Function 7 reached another strong result, showing that the good region was still productive.

Function 5 tested another small movement away from the peak and showed similar behaviour to the previous week. This suggested the high-value peak was smooth near the boundary rather than a sudden spike.

Function 3 came close to its best baseline result, but still did not clearly improve beyond it.

**Evaluation:**  
This week strengthened confidence in the best regions for several functions. It also confirmed that Function 3 was difficult to improve and may require a very specific input combination.

---

## Week 13: Final Query Selection

The final week focused on using the best-known information before the query budget ended. The strategy selected points close to the strongest known regions while avoiding exact duplicates.

Function 5 remained strong near the confirmed peak. Function 2 stayed close to its best region. Function 8 also remained consistent near its high-value area.

However, Function 7 dropped from its previous best, showing that final-round EI can be risky when there is no later opportunity to correct the move. Function 4 also remained difficult because the earlier good region could not be reliably reproduced.

**Evaluation:**  
The final week showed that the strategy had become much more informed, but uncertainty remained. For the last round, safer exploitation may have been better for some functions than aggressive EI.

---

## Overall Strategy Evaluation

Across the project, the strategy improved significantly. It began with broad manual exploration and developed into a structured Bayesian optimisation workflow.

The final approach combined:

- Gaussian Process surrogate models
- Matern kernels
- noise regularisation
- scaling for high-dimensional functions
- EI for exploitation
- UCB for exploration
- local candidate generation
- global Latin Hypercube Sampling
- manual review when the model behaved poorly

The most important lesson was that no single optimisation rule worked for all functions. Each function required its own balance of exploration and exploitation.

Low-dimensional functions benefited from local refinement once a good region was found. Higher-dimensional functions required broader exploration for longer. Some functions, such as Function 3 and Function 4, remained difficult because their landscapes were flat, narrow, or unstable.

Overall, the strategy successfully improved most functions and produced a more explainable optimisation process. The best results came when model predictions were combined with human interpretation rather than trusted blindly.
