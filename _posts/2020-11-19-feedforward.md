---
toc: true
layout: post
description: Notes for Chapter 2.6 in the book Deep Learning (Goodfellow et al.).
categories: [deep-learning-book, part-2, feedforward, neural-network]
title: "Deep Learning Book Notes: Deep Feedforward Networks"
---

Link to chapter: [2.6 Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)

*(Note: I've only noted down chunks that were interesting / I haven't already internalised. Fairly large chunks from the book are therefore not covered)*

## Gradient-Based Learning

Note that "stochastic gradient descent applied to *nonconvex* loss functions has no [...] convergence guarantee and is sensitive to the values of the initial parameters". In other words, SGD-based methods simply find strong local optima, which depend on where we start looking.

### Cost Functions

A cost function is what connects a model to a performance measure.

A cost function is can be thought of as a **functional**, which is a mapping from functions (in this case, input models) to real numbers. It takes in a model, and outputs some performance measure, typically with the help of some *fixed* training data.

Why is this useful? Well, first assume that there is some *desirable property* we wish our model to have. If we can derive a cost functional that has a minimum at the point where the model (i.e. input function) satisfies this property, then we can optimise to find this model (i.e. by changing its parameters).

This leaves us with two questions:

1. What is this desirable property?
2. How do we derive a cost function that has a minimum at this point?

One common answer to the first of these questions is: **the maximum likelihood principle**. This simply states that we desire the model that maximises the probability of some set of test data occurring.

If we have i.i.d. data, the likelihood of the data (in the SL setting) $p_{model}(\textbf{y}_0, \dots \textbf{y}_m \mid \textbf{x}_0, \dots \textbf{x}_m ; \theta)$ is equal to  $\Pi^{m}_{i=1}p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)$. We want the minimum so we negate, and then to turn the multiplication into addition (to make life easier) we move into log-space. This gives us the most popular loss function, the **negative log-likelihood**:
$$
J(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim p_{data}}\left[-\log p_{model}( \mathbf{y} \mid \textbf{x} ; \theta)\right]
$$
Given that the $\theta$ parameters of the model are fixed while the cost function is evaluated, this formulation is exactly the same as something we have seen before: the **cross-entropy**. This leads us to a neat conclusion: 

*Minimising the NLL is the same as minimising the cross-entropy of the model's distribution relative to the data distribution*.

Neat!

Note that this can also be framed as minimising the KL divergence between the two distributions, as the KL is simply $H(p_{data}, p_{model}) - H(p_{data})$ and the entropy term here is irrelevant for the optimisation.

In a sense, this gives us three inter-related ways of motivating minimising the NLL:

1. It satisfies the maximum likelihood principle.
2. It minimises the cross-entropy of the model's distribution relative to the data distribution.
3. It minimises the KL divergence from the model's distribution to the data distribution.

As we have seen in Chapter 1.3, the best model from the space of *functions* for minimising the NLL is the Empirical Distribution, which will not generalise to new data at all.

However, we optimise in *parameter* space, relative to some hand-picked likelihood model. The problem of model selection then becomes one of finding a model such that the parameters that minimise the NLL do lead to strong generalisation. Accounting for stochastic noise in our model can help to address this.