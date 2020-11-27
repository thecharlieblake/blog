---
toc: true
layout: post
description: Estimating the difficulty of a Klondike deal using a computerised solver
categories: [markdown]
title: How to Measure the Difficulty of a Klondike Deal
---

## Preamble

In 2018 [Ian Gent](https://ipg.host.cs.st-andrews.ac.uk/) and I worked on a project to solve solitaire card games. We wrote [a paper](https://arxiv.org/abs/1906.12314) together and produced a rather neat solver called [Solvitaire](https://github.com/thecharlieblake/Solvitaire). We managed to prove the win rate for the most popular solitaire game, [Klondike](https://en.wikipedia.org/wiki/Klondike_(solitaire)), to within a very tight bound (81.956% ± 0.096% to be precise!) and found a lot of other win rates for many other games.

Several people have asked me recently how one might measure the *difficulty* of a particular deal/layout in standard Klondike. I.e. given some starting setup for the game, is there any way I can automatically get a good estimate for how long it will take the average player to solve, without having to actually play the game themselves?

The answer to this is, of course, *sort of!*

Someone who I've been in correspondence with recently posted this question on the board games stack exchange and I decided to reply. Because my post was rather long I've decided to reproduce it here too. Here's what I replied:

---

From a human perspective I suppose one's interpretation of difficulty is quite subjective: what one person might find difficult another may  find easy. If, for our purposes, we use a rough definition along the  lines of, "how much time the average player spends on a layout", then  there are a few interesting things we can say about the relationship  between a layout's difficulty and its general features.

As your first point suggests, the solvability of the layout (i.e.  does a sequence of legal moves exist that results in a win) is obviously key. Let's consider this case first:

## Unsolvable Layouts

If a layout is not solvable then in one sense it's infinitely difficult. However, even for unsolvable layouts, the depth of the [search tree](https://en.wikipedia.org/wiki/Search_tree) can reflect a kind of difficulty.

In the extreme case, if to begin with there are no legal moves  available in the tableau (main cards) or the stock, then although the  layout isn't solvable, in a sense it is easy because you can immediately give up. Alternatively, a layout may have a huge number of promising  moves, only to turn out to be unsolvable in the end.

Using a tool like [Solvitaire](https://github.com/thecharlieblake/Solvitaire) (full disclosure: I am one of the authors of Solvitaire) can identify  unsolvable layouts and record the number of unique states that needed to be searched to prove that there's no solution (often a very large  number of states!). Another good solver is [Klondike-Solver](https://github.com/ShootMe/Klondike-Solver), although I'm not aware of what metrics it reports.

This isn't a perfect measure by any means. Humans aren't computers,  and our brains probably aren't using depth-first search in the way a  solver might. Without looking at human gameplay data though, this is  probably the best heuristic of difficulty for unsolvable layouts.

## Solvable Layouts

As was the case for unsolvable layouts, if one has access to human  gameplay data then that's almost certainly going to be the best source  of understanding how difficult humans will find a layout.

Let's assume though, that we don't have this data available, but we  do have a computerised solver. Just looking at the static cards in the  starting layout is unlikely to tell us much, but by running the solver  we can deduce more about a deal's difficulty (for a human).

Here are some heuristics one could use instead:

### Size of Search Tree

For a computer this is important, but in the solvable case it may not tell us much about human play. For instance, the search tree for a  layout could be huge, but have one solution that's very obvious to a  human. We would not want to classify such a layout as difficult just  because it has a large search tree.

### Depth of Shallowest / Best Solution

Based on the above logic, the depth of the shallowest (i.e. "best")  solution in the search tree is perhaps a better measure. However, this  still may not be ideal. We might label a layout as easy because it has a shallow solution, but if this solution is extremely hard for a human to spot then our heuristic will be misleading.

### Number of different solutions

This heuristic could be quite accurate, although it's hard to say.  Having a lot of different solutions in the search tree would suggest a  human player is likely to come across a solution sooner, but there is no guarantee. This would certainly be something worth exploring further.

### Number of states searched until first solution

This is probably the most obvious heuristic: how many states did it  take before the solver came across the first solution? However, this  measure can be deeply flawed. There is no guarantee that computerised  solvers search in anything like the kind of order a human would search  in (Solvitaire certainly doesn't).

Say move "A" obviously leads to a solution in a few moves, whereas  move "B" leads in a very different direction. There is no guarantee the  solver picks "A" first; it might instead pick "B" and try thousands or  even millions more moves before it backtracks to the point where it  thinks to try "A".

To make this kind of method work one needs a policy for which moves to try. For Solvitaire, which just wants to explore *all possible* moves, this doesn't matter, but for determining difficulty this is very important! If one could come up with a  policy which reflected closely  how an average human would play, then the number of states searched  until the first solution would be an excellent metric. But coming up  with such a metric is hard, complex work.

## What Is A Move: Dominances and K+

In all of these discussions we've talked about moves and states. But how we define a move in Klondike is actually not trivial.

Firstly, consider the stock. The standard rules require we turn over 3 cards in one go. A consequence of this is that at different times there are different groups of cards that are effectively "available" to us.

[Bjarnason et al.](http://web.engr.oregonstate.edu/~afern/papers/solitaire.pdf) use a solver which can move any of the available stock cards into the tableau in a single move (they call this the **K+ representation**). For a human, this is like saying "I remember there's a 5H in there, so  I'll just loop back over the stock to get it". From the point of  considering difficulty, one may wish to consider this sort of thing to  be a single move, rather than several separate moves.

Finally, in [our paper on Solvitaire](https://arxiv.org/pdf/1906.12314.pdf) we consider something called **dominances**. These are cases in which we can prove that one of the available moves is *guaranteed* to be a good move. For instance, sometimes it can clearly be shown that "putting up" say an ace, is always the right thing to do. This is often obvious to a human player and one might not wish to count this as a  move (Solvitaire doesn't count this as an extra move).

## In summary:

- One must first consider solvability
- Then if a layout is solvable, one can begin with easier, but  probably less accurate methods like the number of possible solutions, or the depth of the best solution
- Finally, and *only* if one can craft a good (human-like)  policy for moves to try at each search step, a better approach may be to measure the number of states the solver searches until the fist  solution



