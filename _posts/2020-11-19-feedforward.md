---
toc: true
layout: post
description: Notes for Chapter 6 in the book Deep Learning (Goodfellow et al.).
categories: [deep-learning-book, part-2, feedforward, neural-network]
title: "Deep Learning Book Notes: Deep Feedforward Networks"
---

Link to chapter: [6 Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)

*(Note: I've only noted down sections that were interesting / I haven't already internalised. Fairly large chunks from the book are therefore not covered)*

## Gradient-Based Learning

One thing worth noting before we begin is that, in the words of the authors:

stochastic gradient descent applied to *non-convex* loss functions has no [...] convergence guarantee and is sensitive to the values of the initial parameters

In other words, SGD-based methods simply find strong local optima, which depend on where we start looking. These methods are not perfect and have a limited theoretical justification for the kinds of onn-convex problems we typically use them for (Geoff Hinton thinks we should start again from scratch), although empirically they seem to work well.

### Cost Functions

*(Note: much of this section is based on the previous chapter in the book)*

A cost function enables us to calculate a scalar performance metric evaluating our model's predictions over a dataset. We aim to optimise with respect to this metric, so we want to choose a cost function with a minimum point that represents the "best" performance for the model.

We typically think of the model's functional form as fixed, but parameterised by some learned $\theta$. This reduces our optimisation problem to finding the right solution in parameter space, rather than function space. Thus we can frame the cost function as an evaluation of our model's parameters.

We can think of our model as either outputting a single prediction, or as defining a conditional probability distribution: $p_{model}( \mathbf{y} \mid \textbf{x} ; \theta)$. We will consider the latter case first. We desire a cost function with a minimum at the point where our model is "best", but how do we define this?

One common answer to this question is the **maximum likelihood principle**. This simply states that given a dataset (inputs and labels), the "best" model is the one that allocates the highest probability to the correct labels given the inputs.

The following steps show how we can describe this criterion using a cost function, denoted $J(\theta)$:

1. We define the likelihood of the data as:  $p_{model}(\textbf{y}_0, \dots \textbf{y}_m \mid \textbf{x}_0, \dots \textbf{x}_m ; \theta)$.
2. Assuming the data is i.i.d., we can factorise the joint distribution as:  $\Pi_{i=1}^m p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)$.
3. Our criterion states that the "best" set of parameters should give the most probability to the training data. In mathematical terms, this means:  $\theta_{best} = \arg \max_{\theta}{\Pi_{i=1}^m p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)}$.
4. As specified by our definition of the cost function, we need  $\arg \min_\theta J(\theta) = \theta_{best}$.
5. One form for $J(\theta)$ which satisfies this is:  $J(\theta) = \Pi_{i=1}^m {-p_{model}(\textbf{y}_i \mid \textbf{x}_i ; \theta)}$.
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

*Some good notes on this section can be found at: [peterroelants.github.io/posts/cross-entropy-logistic](https://peterroelants.github.io/posts/cross-entropy-logistic/)*

For tasks where the prediction is of a binary label, we can use our model to define a Bernoulli distribution over $y$ conditioned on $x$. The task of our network is to learn a conditional value for the distribution's parameter $a$ (the final activation), which we can then use for prediction.

We have a particular requirement for this parameter:  $a \in [0, 1]$. To satisfy this, we must add a layer to the end of our network to bound the output $z$ (note: this value is sometimes called a **logit**). One common choice is the sigmoid function[^1].

[^1]: Wikipedia defines the sigmoid function as a general family of S-shaped curves, and refers to this particular function as the *logistic function*.

The sigmoid function is defined as follows:


$$
a = \sigma(z) = \frac{e^z}{e^z + 1} = \frac{1}{1 + e^{-z}}
$$


We can use this to model $P(y = 1 \mid x)$ (recalling that $z$ is a function of $x$), and then in accordance with the laws of probability we can take $P(y = 0 \mid x) = 1 - P(y = 1 \mid x)$ to give us our full distribution over labels.

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
\frac{e^{z}}{1 + e^{z}}, & y=1\\
\frac{1}{1 + e^{z}}, & y=0
\end{cases}\\

P(y \mid x) = \begin{cases}
\sigma(z), & y=1\\
\sigma(-z) = 1-\sigma(z), & y=0
\end{cases}
$$


Thus we have shown that the sigmoid activation function is the natural consequence of a linear model for our log probabilities.

We can use this form with the NLL defined in the previous section:


$$
J(\theta)
	=  -\sum_{j=1}^m \log P(y_j \mid x)\\
J(\theta)    =  \sum_{j=1}^m \begin{cases}
        	\log(1 + e^{-z_j}) = \zeta(-z_j), & y=1\\
        	\log(1 + e^{z_j}) = \zeta(z_j), & y=0
		\end{cases}
$$

Visualising the above softmax function, its curve looks like a smooth version of the ReLU function. To minimise these two cases (which is our objective for the cost function) we must therefore move to the positive and negative extremes for our respective cases.

Thus the consequence of using the sigmoid activation in combination with maximum likelihood is that our learning objective for the logits $z$ is to make the predictions for our 1 labels as positive as possible, and for our 0 labels as negative as possible.

This is pretty much what we would intuitively expect! It's promising to see that ML in combination with a linear log probability model (i.e. sigmoid) leads to such a natural objective for our network.

Note that the above equation is not the form one typically sees the NLL/CE of a binary variable written in. More common (although perhaps less insightful) is the form:


$$
J(\theta) =  -\sum_{j=1}^m \left[ y_j \log(a_j) + (1-y_j)\log(1-a_j)\right]
$$


One practical consideration we also shouldn't overlook here is how amenable this combination of cost function and final layer (/distribution) are to gradient-based optimisation.

What we really care about here is the degree to which the size of the gradient shrinks when the outputs of the layer are towards the extremes. We call this phenomenon *saturation*, and it leads to very slow learning in cases where we have predictions that are incorrect by a large amount[^3].

[^3]: Sigmoid activation combined with MSE has *exactly* this problem. See how the left extreme of [this graph](https://www.wolframalpha.com/input/?i=d%2Fdx+%28sigmoid%28x%29-1%29%5E2) (the derivative of the MSE) tends to 0, whereas in our case it tends to -1.

The derivative of the cost function with respect to $z$ are simply:


$$
    \frac{d}{dz} J(\theta) = \begin{cases}
    -\sigma(-z) = \sigma(z)-1, & y=1\\
    \sigma(z), & y=0
    \end{cases}\\
     \frac{d}{dz} J(\theta) = a - y
$$

Take a moment to appreciate how wonderfully simple this is!

For the purpose of learning, the specific gradient values are ideal. In the case of a very wrong input for a positive label ($y=1, z \to -\infty$), we have $a = 0$  so the derivative tends to $-1$; for a very wrong negative label the derivative tends to $1$.

This is exactly the behaviour we want: large gradients for very wrong predictions (although not too large). Conversely, the gradient for good predictions tends to zero in both cases. Learning will only slow down for this layer when we get close to the right answer!

### Softmax Units for Multinoulli Output Distributions

*Some good notes on this section can also be found at: [peterroelants.github.io/posts/cross-entropy-softmax](https://peterroelants.github.io/posts/cross-entropy-softmax/)*

When we have $n$ output classes we instead use a Multinoulli distribution with $n$ parameters. The labels here are now  $y \in \{1, \dots, n\}$.

To avoid an implicit assumption about numerically close classes being more similar, we model this using a network with $n$ outputs,  $z_i, \; i \in \{1, \dots, n\}$.  We add a final layer to convert each of these into a probability distribution over the associated class being the value of the label:  $P(y=i \mid x) = a_i = f(z_i)$.

We can think of our model's final output as a vector that represents a probability distribution over labels. Note that this means we must also make sure to normalise the output values so that they sum to 1.

We use the same approach as in the binary-class case to model $f(z_i)$. We begin with a linear model for the log probability at each output:


$$
z_i = \log \tilde{P}(y=i \mid x)
$$


Exponentiating and normalising gives:


$$
P(y=i \mid x) = a_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^ne^{z_k}}
$$


We can think of this $\text{softmax}(z)_i$ function as a "soft" version of the $\arg\max$ function; in fact, some suggest that it should more properly be named $\text{softargmax}.$ Softmax is the generalisation of the sigmoid over a vector. It's derivative is also similar (in fact, for the first case it's the same). We find this using the quotient rule:


$$
\text{if} \;\; i = j: \frac{da_i}{dz_j} = \frac{e^{z_i}\sum_{k=1}^ne^{z_k} - e^{z_i}e^{z_j}}{\sum_{k=1}^ne^{z_k}} = \frac{e^{z_i}}{\sum_{k=1}^ne^{z_k}}\left(1-\frac{e^{z_i}}{\sum_{k=1}^ne^{z_k}}\right) = a_i(1-a_i) \\
\text{if} \;\; i \ne j: \frac{da_i}{dz_j} = \frac{0 - e^{z_i}e^{z_j}}{\sum_{k=1}^ne^{z_k}} = -\frac{e^{z_i}}{\sum_{k=1}^ne^{z_k}}\frac{e^{z_j}}{\sum_{k=1}^ne^{z_k}} = -a_i a_j
$$


We can plug the softmax into the NLL to give:


$$
J(\theta) = -\sum_{j=1}^m \log P(y = y_j \mid x)\\
J(\theta) = -\sum_{j=1}^m \log a_{y_j} \quad \text{(we just ignore the other } a_{i\ne y_j} \text{)}\\
J(\theta) = -\sum_{j=1}^m \log \text{softmax}(z_{y_j})\\
J(\theta) = -\sum_{j=1}^m \left[z_{y_j} - \log\sum_{k=1}^n e^{z_k} \right]\\
$$


One point worth noting is that this looks a bit simpler than the binary case - shouldn't it be at least as complex? The reason for this is due to us having one parameter for each class here, whereas because of the probabilities summing to 1 we only really need $n-1$ parameters. We do this for the binary case, which makes the reasoning a little more complex.

Intuitively, we can understand this cost function as incentivising the correct output to increase and the rest to decrease. The second term also punishes the largest incorrect output the most, which is also desirable.

We can see exactly how learning progresses by calculating the gradient of the NLL. We do this over a single label $y$ for a single $z_i$:


$$
\frac{d}{dz_i} J(\theta) = \frac{d}{dz_i}\left(-\log a_y \right)\\
\frac{d}{dz_i} J(\theta) = -\frac{1}{a_y}\begin{cases}
			a_i(1-a_i), & i = y\\
			-a_ia_y, & i \ne y\\
		\end{cases}\\
\frac{d}{dz_i} J(\theta) = \begin{cases}
			a_i - 1, & i = y\\
			a_i, & i \ne y\\
		\end{cases}\\
\frac{d}{dz_i} J(\theta) = a_i - y^{(hot)}_i \quad \text{(where } y^{(hot)} \text{ is the 1-hot encoded label vector)}
$$


Fantastic! This is exactly the same as the gradient in the binary-case, but we have it over each item of the output vector instead of the scalar we had before. Everything we deduced for that scenario works just the same here.

We won't prove it here, but if we create a regression model which outputs the mean of a Gaussian, it turns out that the partial derivatives at for the final layer are also of the form $a_i - y_i$ . It turns out that's a really simple and desirable gradient for the final layer!

### Numerically Stable Softmax

We should instinctively be wary of the exponential and log terms in our softmax function. As we saw in chapter 4, these terms have the potential to drive our values into ranges that can't be precisely represented by floating-point numbers if we're not careful.

Specifically, we want to avoid extremely large or extremely negative inputs to the softmax, that will drive the exponentials to infinity or zero respectively.

Fortunately, there is a trick which can help in this regard. It can be easily shown that  $\text{softmax}(\mathbb{z}) = \text{softmax}(\mathbb{z} + c)$. From this we can derive our numerically stable variant of softmax:


$$
\text{softmax}(\mathbb{z}) = \text{softmax}(\mathbb{z} - \max_i z_i)
$$


This means that our inputs to the function are simply the differences between each input and the largest input. Thus, if the scale of all the inputs becomes very large or negative, the computation will still be stable so long as the relative values are not extremely different across the inputs[^4].

[^4]: I believe that the gradient of the softmax should discourage these relative values getting too large, but I'm not sure. If this is the case, then we are protected against numerical instability from all directions.



## General Back-Propagation

The purpose-crafted output layers and cost functions above are designed to reduce to simple derivatives with useful properties.

In the general case though, neural architectures are not guaranteed to have nice mathematical expressions representing the derivatives. We require a method for calculating these automatically. This is the domain of the **back-propagation** algorithm.

The premise of back-propagation is the use of the chain rule to calculate derivatives of parameters with respect to the cost function. Doing this in the naive way for each node in the network requires exponentially many computational steps. The insight for back-propagation is that this can be reduced substantially by computing and retaining the derivatives, computed front-to-back[^5]. 

[^5]: The chain rule can be computed from the left or right, the latter of which enables back-propagation. This is typically preferred as it build up gradients from front to back, which is quicker in the (standard) case where the number of neurons reduces as we proceed (with a single output neuron). I think there are some cases though in which it's actually more efficient to go forward.

Below is the most general form of the back-propagation algorithm, which can be applied to any model comprising differentiable operations.

### General Approach

Our objective here is to create a second computational graph that flows backwards to compute gradients.

This enables us to do the forward and backward pass in exactly the same way. In the textbook it is suggested that these graphs are combined, although they could equally be managed separately.

### Preliminaries

#### Tensor Notation

We assume a computation graph where each node represents an operation parameterised by a tensor $\mathsf{X}$.

Tensors are typically indexed with a coordinate per rank, which can make notation challenging. To simplify this, we index using a single coordinate representing the tuple of original coordinates. This allows us to essentially treat the tensor (and its gradient) like a standard 2D matrix in our equations.

Based on this, we can write the chain rule as it applies to tensors. Letting $\mathsf{Y} = g(\mathsf{X})$ and $z = f(\mathsf{Y})$:[^6]


$$
\nabla_{\mathsf{X}}z = \sum_j \: \nabla_{\mathsf{X}}\mathsf{Y}_j \; \frac{\delta z}{\delta \mathsf{Y}_j}
$$

[^6]: My instinct for this equation is to think that we can represent this as one inner product. But $\nabla_{\mathsf{X}}\mathsf{Y}_j$ is already a matrix, so unless we want to get into some fairly scary tensor notation (not necessary, potentially less useful!) it's better to leave it like this. Recall as well that our standard assumption is that the derivative of the cost with respect to a parameter tensor has the same shape as that tensor.

### Requirements

1. A computation graph $\mathcal{G}$ where nodes represent variables and edges their dependencies. The function $\mathtt{get children(\mathsf{V})}$ is assumed to exist.
2. A set of target variables $\mathbb{T}$ whose gradients must be computed.
3. The variable to be differentiated $z$.
4. A back-propagation operation for each variable $\mathsf{V}$:  $\; \mathtt{bprop(\mathsf{V}, \mathsf{W}, \mathsf{D})}$.

For each fundamental operation used in $\mathcal{G}$, the framework/programmer must define a corresponding back-propagation operation, which should satisfy the following equation:


$$
\mathtt{bprop(\mathsf{V}, \mathsf{W}, \mathsf{D})} = \sum_j \: (\nabla_{\mathsf{V}}\mathsf{W}_j) \; \mathsf{D}_j
$$


where $\mathsf{D}$ is expected to represent $\frac{\delta z}{\delta \mathsf{W}}$. This is just a step of the chain rule we're familiar with for a given operation.

The final thing to note is the existence of a pre-processing step which prunes all nodes that are not on any path from $\mathbb{T}$ to $\mathcal{G}$. When adding the new backwards nodes we do this to the full graph, but for all other purposes we consider the pruned graph.

### Algorithm

```python
def general_back_propagation(T, z):
    grad_table = {z: 1}  // mutable
    for V in T:
        build_grad(V, grad_table)
    return {v: g for v, g in grad_table if v in V}

def build_grad(V, grad_table):
    if V in grad_table:
        return grad_table[V]
    grad_v = 0
    for W in get_children(V):
        D = build_grad(W, grad_table)
        grad_v += bprop(V, W, D)
    grad_table[V] = grad_v
    # + function to insert grad_table[V] into graph, along with ops creating it
    return grad_v
```

For an example of this in practice, see my notebook outlining [how to implement back-propagation for a vanilla neural network](https://thecharlieblake.co.uk/neural-networks/backpropagation/2020/12/23/simple-backprop.html)!
