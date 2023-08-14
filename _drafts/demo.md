---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference method (backup)"
---

# **Quant Corner**: Finite-Difference method

Finally, let's take a quick look why it's so easy to adopt finite-difference for GPU computing (at
least in 1D case). In general, the price of a derivative instrument can be found as a solution of an
ordinary differential equation (see
[Feynman–Kac formula](https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula)).

**Pricing Equation.** For one-asset derivatives, such as american options we saw above, the pricing
differentail equation is

\\[ - \frac{\partial V}{\partial t} = - r(x,t) V(x,t) + \mu(x,t) \frac{\partial V}{\partial x} (x,t)

- \frac{1}{2}\sigma^2(x,t) \frac{\partial^2 V}{\partial x^2}(x,t) \stackrel{\text{def}}{=} \widehat
  {A}\_ {xx} V(x,t). \\]

**Discretization.** Next, we apply
[Crank-Nicolson discretization](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method) to this
equation and get a system of linear equations for the unknown \\( V(t- \delta t) \\) given by

\\[ \big(1 - \theta \, \delta t \, \mathbb{A}\_{xx}\big) V(t - \delta t, x) = \big(1 + (1 - \theta)
\, \delta t \, \mathbb{A}\_{xx} \big) V(t,x), \\]

where \\( \mathbb{A}\_{xx} \\) is a descrete version of the \\(\widehat {A}\_ {xx} \\) operator.

**Back Propagation.** To find the current price function, \\( V(0,x) \\), we propagate back from the
maturity time \\(t=T \\) (when \\( V(T,x) \\) boundary condition is known) to the present time
\\(t=0\\).

A pseudo-code for the final valuation looks as following:

```
θ = 1/2
for t in (T, T-dt, .., dt)
    U = 1 - θ dt Axx                    //  3 x N matrix (tridiagonal)
    Y = (1 + (1 - θ) dt Axx) V(t)       //  1 x N matrix (vector)

    V(t-dt) = SolveTridiagonal(U, Y)    //  O(N), see Thomas algorithm

    V(t-dt) = max(V(t-dt), payoff)      //  apply early-exercise constraint
```

For more details, see a series of lectures on "Finite Difference Methods for Financial Partial
Differential Equations" by Andreasen & Huge at <https://github.com/brnohu/CompFin>.
