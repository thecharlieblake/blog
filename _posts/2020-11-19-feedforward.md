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

### Sigmoid Units for Bernoulli Output Distributions

For tasks where the prediction is of a binary label, we can use our model to define a Bernoulli distribution over $y$ conditioned on $x$. The task of our network is to learn a conditional value for the distribution's (single) parameter $\theta$, which we can then use for prediction.

We have a particular requirement for this parameter:  $\theta \in [0, 1]$. To satisfy this, we must add a layer to the end of our network to bound the output $z$ (note: this value is sometimes called a **logit**). One common choice is the sigmoid function[^1].

[^1]: Wikipedia defines the sigmoid function as a general family of S-shaped curves, and refers to this particular function as the *logistic function*.

The sigmoid function is defined as follows:
$$
\sigma(z) = \frac{e^z}{e^z + 1} = \frac{1}{1 + e^{-z}}
$$
We can use this to model $P(y = 1 \mid x) = \theta$ (recalling that $z$ is a function of $x$), and then in accordance with the laws of probability we can take $P(y = 0 \mid x) = 1 - P(y = 1 \mid x)$ to give us our full distribution over labels.

Three interesting properties of this function are:


$$
1 - \sigma(z) = \sigma(-z) \\
\sigma^\prime(z) = \sigma(z)(1-\sigma(z)) = \sigma(z)\sigma(-z)\\
\int\sigma(z)dz = \log(1+e^z) = \zeta(z) \quad \text{(softplus)}
$$


But why use this particular bounding function over any other form? Well, it turns out that if we assume a very simple linear model for the probability, this is what results.

We begin by modelling the unnormalised log probability, $\log\tilde{P}(y \mid x)$. This is a good place to start, as whereas  $P(y \mid x) \in [0, 1]$,   $\log\tilde{P}(y \mid x) \in \mathbb{R}$.  The most simple model for our final layer is the linear model[^2]:


$$
\log\tilde{P}(y \mid x) = yz = \begin{cases}
z, & y=1\\
0, & y=0
\end{cases}
$$


[^2]: One useful feature of this form is that one case is constant. We will see when we normalise how this translates into a single output controlling the probabilities of both cases, as required for a Bernoulli distribution.

To convert this into an expression for the probability distribution we take the following steps:


$$
\tilde{P}(y \mid x) = e^{yz} = \begin{cases}
e^z, & y=1\\
1, & y=0
\end{cases} \\
P(y \mid x) = \begin{cases}
\frac{e^{z}}{1 + e^{z}} = \sigma(z), & y=1\\
\frac{1}{1 + e^{z}} = \sigma(-z) = 1-\sigma(z), & y=0
\end{cases}
$$


Thus we have shown that the sigmoid activation function is the natural consequence of a linear model for our log probabilities.

We can use this form with the NLL defined in the previous section:


$$
J(\theta) = -\log P(y \mid x) = \begin{cases}
\log(1 + e^{-z}) = \zeta(-z), & y=1\\
\log(1 + e^{z}) = \zeta(z), & y=0
\end{cases}
$$


Visualising this function, its curve looks like a smooth version of the ReLU function. To minimise these two cases (which is our objective for the cost function) we must therefore move to the positive and negative extremes for our respective cases.

Thus the consequence of using the sigmoid activation in combination with maximum likelihood is that our learning objective for the logits $z$ is to make the predictions for our 1 labels as positive as possible, and for our 0 labels as negative as possible.

This is pretty much what we would intuitively expect! It's promising to see that ML in combination with a linear log probability model (i.e. sigmoid) leads to such a natural objective for our network.

One practical consideration we also shouldn't overlook here is how amenable this combination of cost function and final layer (/distribution) are to gradient-based optimisation.

What we really care about here is the degree to which the size of the gradient shrinks when the outputs of the layer are towards the extremes. We call this phenomenon *saturation*, and it leads to very slow learning in cases where we have predictions that are incorrect by a large amount[^3].

[^3]: Sigmoid activation combined with MSE has *exactly* this problem. See how the left extreme of [this graph](https://www.wolframalpha.com/input/?i=d%2Fdx+%28sigmoid%28x%29-1%29%5E2) (the derivative of the MSE) tends to 0, whereas in our case it tends to -1.

The derivative of the cost function with respect to $z$ are simply:


$$
\frac{d}{dz} J(\theta) = \begin{cases}
-\sigma(-z) = \sigma(z)-1, & y=1\\
\sigma(z), & y=0
\end{cases}
$$


In the case of a very wrong input for a positive label ($y=1, z \to -\infty$), we have $\sigma(z) = 0$  so the derivative tends to $-1$; for a very wrong negative label the derivative tends to $1$.

This is exactly the behaviour we want: large gradients for very wrong predictions (although not too large). Conversely, the gradient for good predictions tends to zero in both cases. Learning will only slow down for this layer when we get close to the right answer!



