---
toc: true
layout: post
description: Notes for Chapter 1.2 in the book Deep Learning (Goodfellow et al.).
categories: [deep-learning-book, part-1, linear-algebra]
title: "Deep Learning Book Notes: Linear Algebra"
---

Link to chapter: [1.2 Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)

**Recall:**

- Core equation: $Ax = b$    ($A$ & $b$ fixed, $x$ variable / to be found)
- $A$ is in $\mathbb{R}^{m \times n}$, which means $m$ up (rows) and $n$ across (columns).

***

## Inverse Matrices

- For $A^{-1}$ to exist, the core equation must have exactly one solution. (Note that here we're just considering the matrix $A$, so we're also treating $b$ as unknown - i.e. must have one solution *for all possible* $x$ & $b$.)
- Computing $A^{-1}$ is often avoided as it can be numerically unstable

## Linear Dependence & Span

- The core equation either has 0, 1 or infinitely many solutions for a given $b$.

  - The following quote gives a useful way of thinking about this:

  - > To analyze how many solutions the equation has, think of the columns of $A$ as specifying diﬀerent directions we can travel in from the origin ... then determine how many ways there are of reaching $b$.

- **span** (of a matrix / set of vectors) = the set of all vectors obtainable via linear combination
  - **column / row space** = span of a matrix's columns / rows
- For a solution to exist (for all $b$), the column space of $A$ must be all of $\mathbb{R}^{m}$.
- For the above to be the case, it is necessary (not sufficient) for $n \gt m$ (at least as many columns as rows).
- Sufficiency requires **linear independence** = no vectors are in the span of any other group of vectors.