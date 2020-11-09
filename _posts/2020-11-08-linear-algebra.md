---
toc: true
layout: post
description: Notes for Chapter 1.2 in the book Deep Learning (Goodfellow et al.).
categories: [deep-learning-book, part-1, linear-algebra]
title: "Deep Learning Book Notes: Linear Algebra"
---

Link to chapter: [1.2 Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)

## Matrix Multiplication

### Key Terms

**Span** (of a matrix / set of vectors): the set of all vectors obtainable via linear combination

**Column / row space**: span of a matrix's columns / rows

**Linear independence**: no vectors are in the span of any other group of vectors

**Invertible matrix**: square matrix with linearly independent columns

### Motivation

Consider the "basic" matrix-vector multiplication equation: $Ax = b$, where $A$ is a fixed matrix and $x$ is a variable vector to be found.

For a fixed $b$, we care about the possible solutions (i.e. values of $x$): how many are there and what are they?

Considering the case where $b$ is arbitrary (i.e. considering what is true for *all* $b$) is perhaps more interesting, and can tell us a lot about $A$ and its properties. The key question is: does the equation have a solution for all $b$?

### Solutions

In the case of a fixed $b$ the basic equation either has 0, 1 or $\infty$ many solutions. The authors provide a useful way of thinking about this:

> To analyze how many solutions the equation has, think of the columns of $A$ as specifying diﬀerent directions we can travel in from the origin ... then determine how many ways there are of reaching $b$.

In the case of an arbitrary $b$:

- There is **>= 1** solution for all $b$ iff $A$ has **a set** of $m$ linearly independent columns.
  - This is due to the column space of $A$ being all of $\mathbb{R}^{m}$.
  - A *necessary* (but not sufficient) condition here is $n \gt m$ (at least as many columns as rows), otherwise the column space can't span $\mathbb{R}^{m}$.

- There is **= 1** solution for all $b$ iff $A$ has **exactly** $m$ linearly independent columns.
  - A *necessary* condition is therefore that m = n (i.e. $A$ is square).
  - Thus **=1** solution iff $A$ is **invertible**.
  - Note that a square matrix that is *not* invertible is called **singular** or **degenerate**.

Why is this all useful? Consider...

### Inverse Matrices

Think of $A$ as applying a transformation to a vector or matrix.

If the basic equation has one solution (i.e. $A$ is invertible) then this transformation can be reversed. This is often really useful!

This reversal can be expressed as the matrix inverse $A^{-1}$.

In practice computing $A^{-1}$ directly is often avoided as it can be numerically unstable, but this property is still very important.

## Norms

### Motivation

A norm is a function that gives us some **measure of the distance from the origin** to a vector or matrix.

Clearly this is a useful concept!

### Definition

A norm is any function which satisfies the following three properties (for all $\alpha, x, y$):

**point-separating:** $f(x) = 0 \implies x = 0$

**absolutely scalable:** $f(\alpha x) = \|\alpha\|f(x)$

**triangle inequality:** $f(x + y) <= f(x) + f(y)$

### Vector Norms

The $L^p$ norm of $x$, often denoted by $\|\|x\|\|_p$ , is defined as:


$$
||x||_p = \left(\sum_i |x_i|^p\right)^{\frac{1}{p}} \quad
$$


where ($p \in \mathbb{R}, p \ge 1$) .

The $L^1$ norm is called the **Manhattan norm**.

The $L^2$ norm is called the **Euclidean norm**. This is the standard norm and is commonly referred to without the subscript as simply $\|\|x\|\|$. The squared $L^2$ norm is also used in some contexts, which is simply $x^Tx$.

The $L^\infty$ norm is called the **max norm**. It is defined as $\|\|x\|\|_\infty = \max_i{\|x_i\|}$.

### Matrix Norms

*(This is a much more [complex](https://en.wikipedia.org/wiki/Matrix_norm) field that we only touch on briefly here!)*

We consider two analogous matrix norms for the $L^2$ vector norm.

In wider mathematics the **spectral norm** is often used. It can be useful in ML for analysing (among other things) exploding/vanishing gradients. It is defined as $A$'s largest singular value: $\|\|A\|\|_2 = \sigma_{\max}{(A)}$

However, most in most ML applications it is assumed the **Frobenius norm** is used. This is defined as:


$$
\|\|A\|\|_F = \sqrt{\sum_{i,j}{A^2_{i,j}}}
$$


## Eigendecomposition

### Key Terms

**Unit vector**: the $L^2$ norm = 1

**Orthogonal vectors**: $x^Ty = 0$

**Orthonormal vectors**: orthogonal unit vectors

**Orthogonal matrix**: rows & columns are mutually *orthonormal*

### Motivation

Decomposing matrices can help us learn about a matrix by breaking it down into its constituent parts. This can reveal useful properties about the matrix.

Eigendecomposition decomposes a matrix into **eigenvectors** and **eigenvalues**.

These tell us something about the directions and sizes of the transformation created when multiplying by the matrix.

### Eigenvectors & eigenvalues

Vector $v$ and scalar $\lambda$ are a eigenvector-eigenvalue pair for *square* matrix $A$ iff:

1. $v \neq \mathbf{0}$
2. $Av = \lambda v$

*(Strictly speaking, here $v$ is a right eigenvector. A left eigenvector is such that $v^TA = \lambda v^T$. We care primarily about right eigenvectors.)*

If $v$ is an eigenvector it follows that any rescaled version of $v$ is also an eigenvector with the same eigenvalue. We typically use a scale such that we have a unit eigenvector.

If $A$ has $n$ independent eigenvectors we can create a matrix of them, $V$, such that $AV = V diag(\lambda)$.

The eigendecomposition of $A$ is then defined as: $A = V diag(\lambda) V^{-1}$.

We are only guaranteed an eigendecomposition if $A$ is symmetric (and real-valued). In this case it is often denoted:


$$
A = Q \Lambda Q^T
$$


Here the decomposition is guaranteed to be real-valued and $Q$ is orthogonal.

The decomposition may not be unique if two (independent) eigenvectors have the same eigenvalues.

Zero-valued eigenvalues exist iff $A$ is singular.

## Singular value decomposition

...