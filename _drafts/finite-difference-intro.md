---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference Theory"
---

_Intro._ Programs = Algorithms + Data Structures + Platform

In my previous post, Pricing Derivatives on a Budget, I presented benchmarks for pricing American
Options on CPU vs GPU. For pricing I used the finite-difference method, which is widely-used in the
production for various tasks. In this post, I will go into some details of implementing the
final-difference in GPU in C++ / CUDA. For people with no GPU experience this might seem like a
complicated topic. However, in fact once you learn a few basic concepts the whole picture become
clear and simple.

## Finite-Difference Algorithm

Let me start with a _brief sketch_ of the finite-difference algorithm. (It is a huge topic even if
limited to the quantitative finance, with entire books dedicated to describe it. If mastered, you
can make a decent career as a quant.) To master, this topic deserves much more attention. For now,
however, we focus on some basic concepts in order to better understand the algorithm and necessary
computational step.

**Feynman-Kac Theorem.** To price a derivative contract we need three components:

1. <u>Payoff Function</u>. This is a legal document that defines a payoff function -- what and when
   is paid. Such documents vary in size from short (e.g., for listed options) to long (e.g., for
   exotic derivatives). What is important for us is that such a document is easily translated into a
   formula for numerical valuation.

2. <u>Underlying Model</u>. In the literature, quantitative finance models are defined as stochastic
   differential equations. Before translated into a more computationally-friendly computer language,
   such a concise mathematical form requires some massage

3. <u>Derivative Price</u>. Is defined an expectation value of the payoff function under the
   underlying mode. This definition is rather general and is not trivially translated into the
   computational language.

Feynman-Kac Theorem give the derivative price a more computationally friendly look. Instead of the
definition (3) it allows to express the derivative price as a differential equation. From this point
we just need to solve PDE which is very understood problem in the applied mathematics.

For example,

- turns a stochastic differential equation into a regular differential equation: show example for
  BS.
- You can write and solve such equations for many popular models, like BS, SABR, Heston, LocalVol,
  and many others.
- Solution method is universal and model-independent.
- Finite-Difference is used for low-dim problems: 1- or 2-dim + time.

![Partial Differential Equation](/assets/img/pde.png)

**PDE Discretization**

- Illustration of PDE discretization with colors

![Partial Differential Equation](/assets/img/demo.png)

**Algorithm + Data Structures**

- The finite-difference method contains of 3 steps.

1. Initializations Step

   - Allocate grid
   - Initial condition

2. Backward Step

   - build linear system
   - solve linear system

3. Adjustment Step

4. If (T != T0) Goto 2
