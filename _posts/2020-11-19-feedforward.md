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

*(Note: much of this section is also based on the previous chapter in the book)*

A cost function enables us to calculate a scalar performance metric evaluating our model's predictions over a dataset. We aim to optimise with respect to this metric, so we want to choose a cost function with a minimum point that represents the "best" performance for the model.

We typically think of the model's functional form as fixed, but parameterised by some learned $\theta$. This reduces our optimisation problem to finding the right solution in parameter space, rather than function space. Thus we can frame the cost function as an evaluation of our model's parameters.

We can think of our model as either outputting a single prediction, or as defining a conditional probability distribution: $p_{model}( \mathbf{y} \mid \textbf{x} ; \theta)$. We will consider the latter case first. We desire a cost function with a minimum at the point where our model is "best", but how do we define this?

One common answer to this question is the **maximum likelihood principle**. This simply states that given a dataset (inputs and labels), the "best" model is the one that allocates the highest probability to the correct labels given the inputs.

The following steps show how we can describe this criterion using a cost function, denoted $J(\theta)$:

1. We define the likelihood of the data as:  $p_{model}(\textbf{y}_0, \dots \textbf{y}_m \mid \textbf{x}_0, \dots \textbf{x}_m ; \theta)$.
2. Assuming the data is i.i.d., we can factorise the joint distribution as:  $\Pi^{m}_{i=1}p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)$.
3. Our criterion states that the "best" set of parameters should give the most probability to the training data. In mathematical terms, this means:  $\theta_{best} = \arg \max_{\theta}{\Pi^{m}_{i=1}p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)}$.
4. As specified by our definition of the cost function, we need  $\arg \min_\theta J(\theta) = \theta_{best}$.
5. One form for $J(\theta)$ which satisfies this is:  $J(\theta) = \Pi^{m}_{i=1}{-p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)}$.
6. Long chains of multiplication can lead to problems such as numerical instability. Moving into log space solves this problem without changing $\theta_{best}$. This gives us our final form for $J(\theta)$, the **negative log-likelihood**:

$$
J(\theta) = \sum^{m}_{i=1}{-\log p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)}
$$

We now have a criterion for our model's parameters! This gives us something to tune the parameters with respect to, *regardless of the choice of model*.

There is another observation we can make to further justify our use of maximum likelihood. We can re-frame our formula for the NLL as an expectation in the following way:
$$
J(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim p_{data}}\left[-\log p_{model}( \mathbf{y} \mid \textbf{x} ; \theta)\right]
$$
This formulation is exactly the same as something we have seen before: the **cross-entropy**. This leads us to a neat conclusion: 

*Minimising the NLL is the same as minimising the cross-entropy of the model's distribution relative to the data distribution*.

Neat!

Note that this can also be framed as minimising the KL divergence between the two distributions, as the KL is simply $H(p_{data}, p_{model}) - H(p_{data})$ and the entropy term here is irrelevant for the optimisation.

This gives us three inter-related ways of motivating our decision to minimise the NLL:

1. It satisfies the maximum likelihood principle.
2. It minimises the cross-entropy of the model's distribution relative to the data distribution.
3. It minimises the KL divergence from the model's distribution to the data distribution.

### A Note on the Sigmoid Function

Often in ML the sigmoid unit is used when predicting the output of a binary variable (i.e. a Bernoulli distribution). The sigmoid is defined as follows:
$$
\sigma(x) = \frac{e^x}{e^x + 1} = \frac{1}{1 + e^{-x}}
$$
When this is used to minimise cross-entropy this becomes the *softplus* function:???
$$
J(\theta) = \frac{1}{m}\sum_{i=1}^m-\log P(y_i|x_i)
$$
