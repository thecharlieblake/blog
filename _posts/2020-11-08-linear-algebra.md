---
toc: true
layout: post
description: Notes for Chapter 1.2 in the book Deep Learning (Goodfellow et al.).
categories: [deep-learning-book, part-1, linear-algebra]
title: "Deep Learning Book Notes: Linear Algebra"
---

Link to chapter: [1.2 Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)

## Matrix Multiplication

### Definitions

**span** (of a matrix / set of vectors): the set of all vectors obtainable via linear combination

**column / row space**: span of a matrix's columns / rows

**linear independence**: no vectors are in the span of any other group of vectors

**Invertible matrix**: square matrix with linearly independent columns

### Motivation

Consider the "basic" matrix-vector multiplication equation: $Ax = b$, where $A$ is a fixed matrix and $x$ is a variable vector to be found.

For a fixed $b$, we care about the possible solutions (i.e. values of $x$): how many are there and what are they?

Considering the case where $b$ is arbitrary (i.e. considering what is true for *all* $b$) is perhaps more interesting, and can tell us a lot about $A$ and its properties. The key question is: does the equation have a solution for all $b$?

### Solutions

In the case of a fixed $b$ the basic equation either has 0, 1 or infinitely many solutions. The authors provide a useful way of thinking about this:

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