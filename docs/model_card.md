# Model Card: Bayesian Black-Box Optimisation Surrogate Models

## Project Name

Bayesian Black-Box Optimisation Capstone

## Model Purpose

This project used surrogate models to guide the optimisation of eight unknown black-box functions.  
Each function could only be evaluated by submitting an input vector and receiving a scalar output.  
The aim was to choose new query points that maximise the output value while using a limited number of weekly evaluations.

The model was not designed to predict a real-world label. Instead, it was used as a decision-support tool for choosing the next best point to test.

---

## 1. Model Overview

### Model Type

The main model used was a **Gaussian Process Regression surrogate model**.

Each black-box function was modelled separately using all available observations collected up to that point.

### Main Strategy

The final strategy combined:

- Gaussian Process Regression
- Matern kernel
- WhiteKernel noise regularisation
- StandardScaler for higher-dimensional functions
- Expected Improvement acquisition function
- Upper Confidence Bound acquisition function
- Hybrid EI/UCB scoring
- Local search around the current best point
- Global exploration using Latin Hypercube Sampling

The model predicted both:

1. the expected output at candidate points, and  
2. the uncertainty of those predictions.

These two outputs were then used to decide the next query.

---

## 2. Input and Output

### Input

The input is a vector of continuous values in the range `[0, 1]`.

The dimensionality depends on the function:

| Function | Input Dimension |
|---|---:|
| F1 | 2D |
| F2 | 2D |
| F3 | 3D |
| F4 | 4D |
| F5 | 4D |
| F6 | 5D |
| F7 | 6D |
| F8 | 8D |

### Output

The model produces:

- a predicted output value,
- an uncertainty estimate,
- a recommended next query point,
- a submission-formatted input string.

The real black-box function output is a single scalar value.  
The goal is always to **maximise** that value.

---

## 3. Model Architecture

### Surrogate Model

The main surrogate model was:

```python
GaussianProcessRegressor
```

The kernel structure used in the final code was generally:

```python
ConstantKernel * Matern + WhiteKernel
```

### Why Gaussian Process?

Gaussian Processes were chosen because the project had a small number of observations for each function.  
This made GP models more suitable than neural networks, which usually require larger datasets.

Gaussian Processes were useful because they can:

- work with small data,
- estimate prediction uncertainty,
- guide exploration,
- support acquisition functions,
- adapt as new weekly results are added.

---

## 4. Kernel Choice

The main kernel used was the **Matern kernel**, usually with `nu=2.5`.

The Matern kernel was selected because it is flexible enough for black-box functions that are not perfectly smooth.  
It can model nonlinear behaviour while avoiding overly aggressive assumptions about the function surface.

A **WhiteKernel** term was added in the later strategy to handle possible noise and reduce overfitting.

---

## 5. Data Preprocessing

### Input Scaling

For low-dimensional functions, the inputs were already in `[0, 1]`, so direct modelling was acceptable.

For higher-dimensional functions, especially F6, F7, and F8, **StandardScaler** was used before training the GP model.

This helped the model learn more stable length scales.

### Output Scaling

For functions with wider or less stable output ranges, the output values were also scaled.

After prediction, the model converted outputs back to the original scale before calculating:

- Expected Improvement,
- Upper Confidence Bound,
- predicted output,
- uncertainty.

---

## 6. Acquisition Strategy

### Expected Improvement

Expected Improvement was used to focus on points likely to improve the current best result.

It was useful when a promising region had already been found.

### Upper Confidence Bound

Upper Confidence Bound was used to encourage exploration.

It was useful when the model was uncertain or when the function was high-dimensional.

### Hybrid EI/UCB Score

The final strategy combined both acquisition functions:

```text
Hybrid Score = EI weight x normalised EI + UCB weight x normalised UCB
```

This allowed the model to balance exploitation and exploration.

The balance changed by function:

| Function Type | Strategy |
|---|---|
| 2D functions | More local exploitation after a good region was found |
| 3D–5D functions | Balanced EI and UCB |
| 6D–8D functions | More global exploration |

---

## 7. Candidate Generation

Candidate points were generated in two ways.

### Local Candidates

Local candidates were sampled near the current best observed point.

This helped refine promising areas.

### Global Candidates

Global candidates were generated using Latin Hypercube Sampling.

This helped cover the full search space and avoid getting stuck in one region.

### Final Candidate Pool

The final candidate pool combined both local and global candidates.  
Each candidate was scored using the hybrid acquisition function, and the highest-scoring point was selected as the next query.

---

## 8. Function-Level Behaviour

### Function 1

Function 1 started with near-zero outputs.  
After exploration, a strong peak was found around:

```text
0.628540-0.628540
```

This became the main exploitation region.

### Function 2

Function 2 was smoother and easier to model.  
The GP model performed well, and the strategy focused on refining around the best region.

Best observed value:

```text
0.784369
```

### Function 3

Function 3 was difficult because most values were negative and close together.  
The surface appeared flat, so improvement was limited.

Best observed value:

```text
-0.033331
```

### Function 4

Function 4 contained a very large positive output compared with most negative values.  
This result was treated carefully because it may indicate either a sharp peak or a possible unusual data point.

Best observed value:

```text
32.625631
```

### Function 5

Function 5 showed a strong high-value region.  
The strategy used more exploitation because the promising area was clear.

Best observed value:

```text
1513.304903
```

### Function 6

Function 6 was 5D and had negative outputs.  
The goal was to maximise, so values closer to zero were better.

Best observed value:

```text
-0.289010
```

### Function 7

Function 7 was 6D and benefited from exploration.  
The model found stronger values over time, but the search space was still large.

Best observed value:

```text
2.402349
```

### Function 8

Function 8 was the highest-dimensional function.  
Because it was 8D, more global exploration was used.

One observed input had a negative coordinate, so the model kept it as training data but restricted all new candidate points to `[0, 1)`.

Best observed value:

```text
9.947740
```

---

## 9. Results Summary

| Function | Best Observed Value | Best Input |
|---|---:|---|
| F1 | 2.000000 | `0.628540-0.628540` |
| F2 | 0.784369 | `0.860214-0.472626` |
| F3 | -0.033331 | `0.493645-0.858708-0.508075` |
| F4 | 32.625631 | `0.948389-0.894513-0.851637-0.552196` |
| F5 | 1513.304903 | `0.385144-0.835222-0.875492-0.956764` |
| F6 | -0.289010 | `0.464984-0.369388-0.703268-0.908164-0.181040` |
| F7 | 2.402349 | `0.033969-0.126740-0.514117-0.133496-0.374088-0.621625` |
| F8 | 9.947740 | `0.039675-0.121395-0.150160-0.029602-0.752700-0.490718-0.085935-0.710564` |

---

## 10. Performance Notes

The model performed best when the function had a learnable structure and enough useful observations.

Stronger performance was seen in:

- F1 after the hidden peak was discovered,
- F2 due to smooth behaviour,
- F5 due to a clear high-value region,
- F7 and F8 through broader exploration.

The model was less reliable when:

- the function was flat,
- outputs were very close together,
- the search space was high-dimensional,
- a result looked like an outlier,
- the model repeatedly suggested near-duplicate points.

---

## 11. Limitations

### Small Data

Each function had a limited number of observations.  
This made modelling difficult, especially for functions with 5 or more dimensions.

### High Dimensionality

F6, F7, and F8 were more difficult because the search space grows quickly with each added dimension.

### Possible Overfitting

Gaussian Processes can overfit when data is small.  
The strategy used noise regularisation and scaling to reduce this risk.

### Boundary Behaviour

The model sometimes preferred points near boundaries.  
This can be useful if the optimum is near the edge, but it can also waste queries if the boundary is not actually good.

### No Ground Truth

The real mathematical functions were unknown.  
Therefore, the model could not prove that the best observed point was the true global maximum.

### Out-of-Domain Observation

Function 8 included one observed point with a negative input value.  
It was kept because it had a real output, but all future suggestions were restricted to the valid domain.

---

## 12. Why Neural Networks Were Not Used as the Main Model

Neural networks were tested as an optional comparison, but they were not used as the main optimiser.

The reason is that the dataset was too small.  
A neural network could fit the training data well but still generalise poorly to unseen points.

The Gaussian Process model was better suited because it provided uncertainty estimates, which are essential for Bayesian optimisation.

---

## 13. Risk and Responsible Use

The model should not be treated as a perfect representation of the true function.

A user should always check:

- whether the suggested point is valid,
- whether it is too close to a previous query,
- whether it is inside `[0, 1]`,
- whether the prediction uncertainty is very high,
- whether unusual values may be outliers.

The model is useful for guiding decisions, but final query choices should still include human review.

---

## 14. Recommended Future Improvements

Future versions could improve the optimisation strategy by adding:

1. duplicate detection before submitting a query,
2. boundary penalties,
3. automatic kernel comparison,
4. acquisition auto-selection by function,
5. separate local search radius per function,
6. stronger handling of outliers,
7. better visual tracking of weekly improvement,
8. ensemble models combining GP and tree-based surrogates.

---

## 15. Final Model Summary

The final model used Gaussian Process surrogate modelling with a hybrid EI/UCB acquisition strategy.  
This approach was suitable because the project had limited observations, unknown function formulas, and no gradient information.

The strategy improved over time by combining local exploitation with global exploration.  
Lower-dimensional functions benefited from local refinement, while higher-dimensional functions required broader search.

Overall, the model provided a systematic and explainable way to select new query points for black-box optimisation.
