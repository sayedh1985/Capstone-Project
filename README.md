# BBO Capstone Project

This project implements a Bayesian Optimization framework for black-box functions .



---

## Non-Technical Summary

This project explores how Bayesian Optimization can solve complex problems when the mathematical form of a function is unknown. Using optimization techniques, the system learned to identify the best input combinations for eight hidden “black-box” functions while working with very limited feedback.

Across the iterative rounds, the project combined Gaussian Process models, acquisition functions, clustering, support vector machines, neural-network-inspired reasoning, and pattern analysis to balance exploration and exploitation. Over time, the strategy became more adaptive, moving from broad discovery toward targeted refinement around promising regions.

Beyond finding strong outputs, the project demonstrates how optimization and machine learning tools can support decision-making under uncertainty. This is a challenge common in engineering, finance, scientific experimentation, business analytics, and real-world machine learning systems where testing is costly, slow, or limited.

---

## Section 1: Project Overview

This GitHub repository documents the Bayesian Black-Box Optimization (BBO) capstone project. The project is designed around a set of unknown objective functions, where the internal mathematical structure is not available to the user. The task is to select input values, observe the output, and gradually improve the search strategy over time.

The BBO capstone project has two main objectives. First, to implement an adaptive optimization strategy for black-box functions through an iterative process, seeking to identify the input vector X that maximizes each function. Second, to develop decision-making skills when working with limited information produced by an unknown process.

Professionally, this project helped me develop both technical and strategic thinking skills. The most valuable part of the project was learning how to make decisions with incomplete information, then update those decisions as new results became available.

---

## Section 2: Inputs and Outputs

The project includes eight synthetic black-box functions, each with a different input dimensionality ranging from 2D to 8D. Each function receives a vector of values X and returns one scalar output value. The main objective is to find the vector X that maximizes each black-box function.

### Input Format

All inputs are restricted to values between 0 and 1:

0 ≤ xi < 1

The input vector is written using hyphens and values are specified to six decimal places:

Function 6: [0.697910-0.990583-0.035622-0.974136-0.655285]

### Output Format

For each input X, the function returns a scalar output:

y = f(X)

The output can be positive or negative:

Function 6: -2.0390133791009304

---

## Section 3: Challenge Objectives

The main objective is to develop and implement a methodology to maximize the output of black-box functions by applying Bayesian Optimization models.

This type of challenge includes several constraints:

* Unknown function structure
* One query per function per iteration
* Limited evaluation budget
* Expensive or time-consuming testing
* Noisy and uncertain outputs
* Increasing dimensionality from 2D to 8D
* Risk of local optima
* Computational cost of surrogate modelling

Because of these constraints, the project required a careful balance between exploration and exploitation.

---

## Section 4: Technical Approach

The methodology is based on several complementary models and analysis techniques. Each method contributed to improving the decision-making process during the optimization rounds.

### Bayesian Optimization

Bayesian Optimization was the core technique used in this project. It allowed the search strategy to improve step by step by using previous observations to guide future query decisions.

Gaussian Process Regression was used as the main surrogate model because it provides both predicted output values and uncertainty estimates. This made it possible to identify regions that were likely to perform well and regions where the model still lacked confidence.

### Acquisition Functions

Two main acquisition functions were used:

* Expected Improvement (EI)
* Upper Confidence Bound (UCB)

Expected Improvement was used to focus on exploitation by selecting points that were predicted to improve on the best observed output. Upper Confidence Bound was used to encourage exploration by selecting points with high uncertainty.

By combining EI and UCB, the strategy avoided relying on only one decision rule. This helped the project balance short-term improvement with long-term learning.

### Local and Global Candidate Generation

Candidate points were generated both globally and locally. Global candidates helped explore new areas of the input space, while local candidates were generated around the current best-performing points to refine promising regions.

In early rounds, the strategy placed more weight on global exploration. As more data became available, the search gradually shifted toward local refinement. However, a small global exploration component was maintained, especially for the higher-dimensional functions.

### Clustering and Distance Analysis

Clustering and distance-based analysis were used to understand the structure of the observed data. These methods helped identify promising regions where nearby points showed stronger outputs.

Distance analysis also helped avoid selecting points that were too close to weak observations. This improved sample diversity and reduced wasted queries.

### Support Vector Machines

Support Vector Machines were considered as a complementary method rather than a replacement for Bayesian Optimization. By classifying observations into high-performing and low-performing regions, SVMs helped identify possible boundaries between promising and unpromising areas.

A soft-margin SVM was useful conceptually because the outputs were noisy and the boundary between good and bad regions was not always clear. Kernel SVMs were also useful for thinking about nonlinear and multimodal response surfaces.

### Neural Network Reasoning

Neural networks were used conceptually to support structural understanding of the search space. Unlike Gaussian Processes, neural networks do not naturally provide uncertainty estimates, but they are useful for thinking about nonlinear interactions and gradient-like behaviour.

Neural-network-inspired ideas helped interpret which dimensions appeared more influential and where small changes in input values could produce stronger output changes.

### PCA and Dimensional Thinking

PCA-style thinking helped identify which variables, regions, or search behaviours explained the most variation in performance. Instead of treating all dimensions equally, the strategy focused more attention on dimensions and regions that appeared to carry stronger information.

This helped reduce redundant exploration while still preserving enough flexibility to avoid missing important interactions.

### Reinforcement Learning Perspective

The optimization process also resembled reinforcement learning in a general sense. Each new query acted like feedback that updated the strategy for the next round.

Regions that produced strong outputs became more attractive for future sampling, while weak or flat regions were given lower priority. This created an adaptive process where the strategy improved through interaction with the black-box functions.

---

## Section 5: Documentation

This project includes structured documentation following Datasheets for Datasets and Model Cards best practices.

### Datasheet

The dataset datasheet describes:

* The structure of the dataset
* The input and output format
* The adaptive sampling process
* The limitations of the collected data
* The risk of exploitation-driven sampling bias

### Model Card

The model card provides transparency about the optimization framework, including:

* Gaussian Process surrogate modelling
* Expected Improvement and UCB acquisition functions
* Clustering and distance analysis
* SVM-based region classification
* Neural-network-inspired interpretation
* Intended use cases and limitations
* Risks, assumptions, and possible biases

---

## Section 6: Methodological Notes

The dataset and model should be interpreted as part of a sequential decision-making process. The data does not represent a random sample from the full input space. Instead, it reflects an optimization trajectory shaped by previous decisions.

The strategy evolved based on:

* Best observed outputs
* GP posterior mean
* GP uncertainty
* Expected Improvement values
* UCB scores
* Neighbourhood analysis
* Clustering structure
* Learned length-scales
* Local curvature and sensitivity
* Dimensionality of each function

### Limitations

The main limitations include:

* Strong bias toward local exploitation in later rounds
* Possible missed global optima
* Sparse coverage in higher-dimensional functions
* Dependence on initial exploration quality
* Risk of Gaussian Process overconfidence
* Limited number of evaluations
* Possible noise in observed outputs

---

## Section 7: Weekly Search Strategies

### Early Rounds

The early strategy focused on broad exploration. The main objective was to map the search space, reduce uncertainty, and avoid committing too early to one local region.

A hybrid EI and UCB strategy was used, with both global candidate generation and local sampling around the best observed points.

### Middle Rounds

As more data became available, the strategy shifted toward targeted refinement. The model began to identify stronger basins, especially in lower-dimensional functions.

At this stage, local sampling became more important, while a smaller amount of global exploration was kept to protect against model overconfidence and multimodal behaviour.

### Later Rounds

In later rounds, the strategy became more confidence-driven. The Gaussian Process posterior mean, uncertainty, and learned length-scales played a stronger role in selecting query points.

For lower-dimensional functions, the model was more stable, so exploitation around promising regions was prioritized. For higher-dimensional functions, especially 6D and 8D, exploration was still maintained because the search space remained sparse and uncertain.

### Final Strategy Direction

The final strategy focused on disciplined refinement. Strong-performing basins were exploited more directly, while a small exploration reserve was kept for uncertain regions, boundary areas, and under-sampled subspaces.

This approach aimed to maximize expected value per query while reducing the risk of becoming trapped in a local optimum.

---

## Section 8: Key Learning Outcomes

This project developed practical understanding of:

* Bayesian Optimization
* Gaussian Process Regression
* Expected Improvement
* Upper Confidence Bound
* Exploration and exploitation trade-offs
* Query strategy design
* Clustering and distance analysis
* SVM-based classification
* Neural-network-inspired interpretation
* PCA-style dimensional reasoning
* Decision-making under uncertainty

The most important learning outcome was that good optimization is not simply about choosing the next point. It is about managing uncertainty, interpreting feedback, adapting the strategy, and making the best possible decision with limited information.

---

## Section 9: Conclusion

The BBO Capstone Project demonstrates how Bayesian Optimization can be used to solve difficult black-box optimization problems under strict evaluation limits.

By combining Gaussian Process models, acquisition functions, local and global candidate generation, clustering, SVM analysis, neural-network reasoning, and adaptive strategy refinement, the project developed a practical framework for decision-making under uncertainty.

The project shows that optimization is not only a technical modelling task, but also a process of learning, adapting, and making informed decisions when the true system is unknown.
