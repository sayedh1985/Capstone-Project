# Datasheet: Bayesian Black-Box Optimisation Capstone

This datasheet documents the optimisation strategy, data handling, modelling decisions, weekly learning, and final results for the Bayesian Black-Box Optimisation capstone project. The project focused on maximising eight unknown black-box functions using a limited query budget.

---

## 1. Function Overview

### 1.1 Which functions does this datasheet describe?

This datasheet describes Functions 1 through 8. Each function is an unknown black-box optimisation problem with a different input dimensionality, output range, and difficulty level.

| Function | Dimensions | General behaviour observed | Main challenge |
|---|---:|---|---|
| F1 | 2D | Very sharp peak after near-zero initial outputs | Finding the hidden high-value region |
| F2 | 2D | Smooth and learnable | Refining around the best region |
| F3 | 3D | Mostly negative and relatively flat | Small improvements are difficult |
| F4 | 4D | Large variation in outputs | Possible outlier or very sharp peak |
| F5 | 4D | Strong high-output region | Exploiting the best cluster |
| F6 | 5D | Negative outputs with local improvements | Higher-dimensional search difficulty |
| F7 | 6D | Clear improvement through exploration | Multimodal high-dimensional landscape |
| F8 | 8D | High outputs clustered near good regions | Very high dimensionality and narrow search space |

The objective for all functions was to **maximise the output value**.

---

## 2. Nature of the Data

### 2.1 What does each dataset contain?

Each dataset contains input vectors and their corresponding output values. The inputs are continuous values mostly within the range `[0, 1]`. Each row represents one evaluation of the unknown black-box function.

The format is:

```text
x1, x2, ..., xd -> y
```

where `d` is the number of dimensions for that function, and `y` is the scalar output.

### 2.2 How did the dataset evolve?

The dataset started with the initial points provided for each function. After that, new query results were added from the weekly optimisation process. These added points were selected using a combination of:

- Gaussian Process surrogate modelling
- Expected Improvement
- Upper Confidence Bound
- Local search around the current best point
- Global exploration using Latin Hypercube Sampling

As more data was collected, the strategy moved from broader exploration toward more targeted exploitation around promising regions.

---

## 3. Optimisation Strategy

### 3.1 Which optimisation method was used?

The main strategy was **Bayesian Optimisation using Gaussian Process Regression**.

For each function, a separate Gaussian Process model was fitted using all available observations. The model was then used to predict promising new candidate points.

The final strategy used:

- Gaussian Process Regressor
- Matern kernel
- WhiteKernel noise regularisation
- Expected Improvement
- Upper Confidence Bound
- Hybrid acquisition scoring
- Latin Hypercube Sampling
- Local candidate generation around the current best point

### 3.2 Why was this method suitable?

The functions were black-box functions, meaning no formula, gradient, or internal structure was available. Only input-output evaluations were known.

Gaussian Processes were suitable because they can:

- learn from small datasets,
- estimate uncertainty,
- guide future sampling decisions,
- balance exploration and exploitation,
- work well with expensive or limited function evaluations.

Random search alone would waste many queries, especially in higher dimensions. Neural networks were considered, but they require more data and are more likely to overfit with small sample sizes.

---

## 4. Exploration and Exploitation

### 4.1 How was exploration handled?

Exploration was handled mainly through:

- **Upper Confidence Bound**
- **Latin Hypercube Sampling**
- global candidate generation across the full `[0, 1]` domain

UCB was useful because it gives higher scores to points where the model has high uncertainty. This helped avoid focusing only on the current best region too early.

### 4.2 How was exploitation handled?

Exploitation was handled through:

- **Expected Improvement**
- local candidate generation around the best observed point
- repeated refinement near high-performing regions

EI was used to identify points that were predicted to improve over the current best output.

### 4.3 What was the hybrid acquisition strategy?

Instead of using only EI or only UCB, both were combined into a hybrid score:

```text
Hybrid Score = EI weight x normalised EI + UCB weight x normalised UCB
```

The weights changed depending on dimensionality:

| Function type | Strategy |
|---|---|
| Low-dimensional functions | More exploitation |
| Medium-dimensional functions | Balanced EI and UCB |
| High-dimensional functions | More global exploration |

For example, Function 8 used more exploration because it is 8D and has a much larger search space.

---

## 5. Data Handling and Preprocessing

### 5.1 Were inputs scaled?

For lower-dimensional functions, the model could work directly with the input values because they were already in `[0, 1]`.

For higher-dimensional functions such as F6, F7, and F8, **StandardScaler** was used to scale the input data before training the Gaussian Process model. This helped stabilise training and made length-scale learning more reliable.

### 5.2 Were outputs scaled?

For functions with larger or more varied output ranges, outputs were also scaled using **StandardScaler**. Predictions were then converted back to the original output scale before calculating EI and UCB.

This was especially important for functions with high dimensionality or wide output variation.

### 5.3 How were candidate points generated?

Candidate points were generated in two ways:

1. **Local candidates**  
   These were sampled around the current best point using Gaussian noise.

2. **Global candidates**  
   These were generated using Latin Hypercube Sampling across the whole domain.

The final candidate pool combined both local and global candidates.

---

## 6. Function-Specific Strategy

### Function 1

Function 1 initially appeared almost impossible to optimise because many outputs were extremely close to zero. However, after exploring the space, a strong peak was found near:

```text
0.628540-0.628540
```

This became the main exploitation region. Later queries focused on refining around this point.

### Function 2

Function 2 was smoother and easier to model. The GP model identified useful regions, and the hybrid EI/UCB strategy helped refine points around the best observed values.

Because it is only 2D, more local exploitation was used after a promising region was found.

### Function 3

Function 3 was more difficult because most outputs were negative and close together. The function appeared relatively flat, so improvements were small.

The strategy used moderate exploration with EI/UCB to avoid getting stuck in weak local regions.

### Function 4

Function 4 showed large variation in output values. One observed value was much higher than the rest, so the code included a warning to check whether this value was a true output or a possible sign error.

The strategy used balanced local and global search because the function may contain sharp peaks.

### Function 5

Function 5 had a clear high-value region in the 4D space. The strategy gave more weight to exploitation because the best region was already visible from the data.

Local candidates were generated around the best point, while global Latin Hypercube points were still included to avoid missing another region.

### Function 6

Function 6 was 5D and had negative output values. The objective was still to maximise the function, meaning values closer to zero were better.

Scaling was important here because the function was higher-dimensional. The strategy used a balanced approach with both local exploitation and global exploration.

### Function 7

Function 7 was 6D and showed strong improvements through exploration. Because the search space was larger, the strategy used more global exploration than the lower-dimensional functions.

The hybrid EI/UCB acquisition helped identify both promising regions and uncertain regions.

### Function 8

Function 8 was the most difficult because it was 8D. The search space was large, so the strategy used more global candidates than local ones.

The final Function 8 code used:

- scaled input data,
- scaled output data,
- Matern kernel,
- EI/UCB hybrid acquisition,
- 30% local exploitation,
- 70% global exploration.

One observed input contained a negative value, so the code kept it as training data but restricted all new suggested candidates to `[0, 1)`.

---

## 7. Weekly Learning

### 7.1 How did the strategy change over time?

The strategy became more structured as more results were collected.

Early work focused on:

- testing different regions,
- understanding the output range,
- identifying whether each function was smooth, flat, or highly variable.

Later work focused on:

- fitting GP surrogate models,
- using EI and UCB acquisition functions,
- generating local/global candidate pools,
- refining around the best known points.

### 7.2 What was learned from the iterations?

The main learning was that each function required a different balance of exploration and exploitation.

Low-dimensional functions could be refined more aggressively, while high-dimensional functions required more global exploration.

For example:

- Function 1 needed repeated local refinement after the hidden peak was found.
- Function 5 benefited from exploitation around a strong high-value region.
- Function 7 and Function 8 required more exploration because of their larger dimensionality.

---

## 8. Performance and Results

### 8.1 What were the best observed values?

| Function | Best observed value | Best observed input |
|---|---:|---|
| F1 | 2.000000 | `0.628540-0.628540` |
| F2 | 0.784369 | `0.860214-0.472626` |
| F3 | -0.033331 | `0.493645-0.858708-0.508075` |
| F4 | 32.625631 | `0.948389-0.894513-0.851637-0.552196` |
| F5 | 1513.304903 | `0.385144-0.835222-0.875492-0.956764` |
| F6 | -0.289010 | `0.464984-0.369388-0.703268-0.908164-0.181040` |
| F7 | 2.402349 | `0.033969-0.126740-0.514117-0.133496-0.374088-0.621625` |
| F8 | 9.947740 | `0.039675-0.121395-0.150160-0.029602-0.752700-0.490718-0.085935-0.710564` |

### 8.2 How confident are we in these results?

Confidence varied by function.

| Function | Confidence level | Reason |
|---|---|---|
| F1 | High near the discovered peak | Repeated nearby points produced strong results |
| F2 | Moderate | Smooth behaviour made the model reliable |
| F3 | Low to moderate | Flat negative surface made improvement difficult |
| F4 | Low | One value was much higher than the others and should be checked |
| F5 | Moderate to high | Strong high-value cluster was identified |
| F6 | Moderate | Best values were still negative but improved |
| F7 | Moderate | Strong results were found, but 6D space may contain other peaks |
| F8 | Moderate | Very high-dimensional, so full confidence is difficult |

---

## 9. Limitations

### 9.1 What were the main limitations?

The main limitations were:

- small number of observations,
- high dimensionality for Functions 6, 7, and 8,
- possible overfitting in GP models,
- possible duplicate or near-duplicate suggestions,
- sensitivity to kernel length-scale estimation,
- possible outlier values in Function 4,
- one out-of-domain observed value in Function 8.

### 9.2 Why not rely fully on neural networks?

Neural networks were tested as an optional comparison, but they were not used as the main strategy because the dataset was too small. Neural networks can fit the training data very well but may generalise poorly when only a limited number of function evaluations are available.

Gaussian Processes were more suitable because they provide uncertainty estimates, which are essential for Bayesian optimisation.

---

## 10. Ethical and Practical Considerations

Black-box optimisation is relevant to real-world problems where experiments are expensive or time-consuming. Examples include:

- drug discovery,
- engineering design,
- materials testing,
- machine learning hyperparameter tuning,
- industrial process optimisation.

A responsible optimisation strategy should avoid blindly trusting model predictions. It should include sanity checks, uncertainty estimates, and careful review of unusual results.

In this project, the code included warnings for:

- overfitting,
- out-of-domain observations,
- unusual output values,
- very high uncertainty.

---

## 11. What Would Be Improved Next Time?

If restarting the project, the strategy could be improved by:

1. adding duplicate-checking before every query,
2. using more global exploration earlier for high-dimensional functions,
3. applying boundary penalties when the model repeatedly suggests edge points,
4. using multiple kernels and comparing them,
5. validating suspicious outputs such as the large Function 4 value,
6. using neural networks only as a secondary comparison, not as the main optimiser,
7. documenting every weekly query with the reason it was selected.

---

