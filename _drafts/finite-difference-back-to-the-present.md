---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference – Back to the Present"
---

- Option Chain for AAPL / AMD / TSLA

**In this post** I'll show how to implement a
[Finite-Difference Pricer](https://en.wikipedia.org/wiki/Finite_difference_methods_for_option_pricing)
for American options. It should be accessible for people without C++ experience, but some basic
knowledge of the options pricing would be helpful, although also not required.

It's neither required to have hands-on experience with a finite-difference method. You'll find an
overview with all necessary details below, since for this post we need only key calculation steps,
without deep details of the algorithm. It's neither scarry nor difficult, believe me.

Let me know if you want those details in the next post (like, react, subscribe, etc.). For now, see
books for more details.

**We will use C++** on Linux and Windows. My current setup is Ubuntu 22.04 and Windows 11 with
Visual Studio 2022. To compile cross-platform I use CMake, which is deeply integrated with Visual
Studio and gives as smooth development experience as native VS project.

**Pricing for the Market** ...

![AMD Option Chain on 2023-11-17 15:05](/assets/img/202311171505-AMD-retro.png)

## Pricing Equation

**Derivative price** is defined by the Black-Scholes equation. It is a partial-differential equation
which defines dynamics of the derivative price as a function of time and spot (or several spot
variables in exotic cases).

![Pricing PDE](/assets/img/fd-black-scholes.jpg)

Below is a brief overview of two methods to derive this equation, in order of my personal
preferences.

**Delta Hedging.** This is a very intuitive yet powerful way to derive a pricing equation, not only
for vanilla options, but for more exotic multi-asset instruments as well.

**Feynman-Kac Theorem.** This approach is very theoretical and so far I don't find it useful in
practice.

Please, share in the comments you have different experience.

## Finite-difference grid

Differential equations without a closed-form solution are usually solved numerically, with a
finite-difference method being one of the most popular used in practice (others popular methods are
Monte-Carlo and Fourier transformation), derivatives calculated on a grid with small steps between
valuation points.

Fortunately, the diffusion equation is one of the easiest to solve numerically.

**Discretization.** Let's introduce a rectangular grid on the domain of independent variables (t,s)
defined by x[i] = x[i-1] + dx_i and t[j] = t[j-1] + dt and labeled by i and j indices.

Forward difference (backward difference):

Central difference:

![Discretization](/assets/img/fd-difference.png)

Finite-Difference Equation(Discrete PDE) step-by-step generates solution from the initial values:

![Pricing PDE](/assets/img/fd-difference-equation.png)

**Crank-Nicolson** is probably on of the most popular discretization schemes for the Black-Scholes
equation.

Explicit:

- Euler forward
- Euler backward

Implicit:

- Crank-Nicolson

![Finite-Difference Schemes](/assets/img/fd-crank-nicolson.png)

Depending on how we expand the time derivative, we get forward or backward equation (?). These
equations however are O(dt + dx^2) error. We can do better than that by introducing a continuos
parameter \theta that defines a mixed scheme which is strictly forward for \theta = 1, backward for
\theta = 0, and mixed for everything in between. For \theta = 1/2 it's called Crank-Nicolson scheme
and has O(dt^2 + dx^2) error.

Finally, we get a linear system of equations that defines evolution of the price function over the
time axis.

## Backward Evolution

We don't know function V(t=0, s), obviously this is what we are looking for. Hence, forward
evolution is no go, as we don't know the initial condition is unknown. However, we know the initial
condition for the backward evolution, as V(t=T, s) is a payoff at expiry. It is (K-s)+ for CALL and
(s-K)+ for PUT options.

**Thomas Algorithm** is [link](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm) ...

**For American options** we should taken into account the right to early exercise. If continue like
that we will get a European option price. For the American exercise, we should ensure that option
price is not less than its intrinsic value, otherwise we'll get an arbitrage situation when one can
buy an option for the lower price than its exercise value.

In other words:

![Early Exercise Condition](/assets/img/fd-early-exercise.png)

## Boundary Conditions

The last topic left for the discussion is how to choose steps of the finite-difference grid. The
rather obvious choice of the uniform step leads to a poor convergence. Also limits ...

**X-grid step** is more tricky as it affects accuracy of the method. What we want is to draw more
points around the spot price and less points at the edges of the x-grid.

## Finite-Difference Grid

**The t-Axis** is divided uniformly with a step dt = T / N between points. It doesn't seem to use
some non-uniform step here, at least not something I observed in practice.

**The x-Axis** is divided in a more tricky way. ...

![Asinh Plot](/assets/img/fd-asinh.png)

```cpp
/// Init X-Grid

const f64 density = 0.1;
const f64 scale = 10;

const f64 xMid = log(asset.s);
const f64 xMin = xMid - scale * asset.z * sqrt(asset.t);
const f64 xMax = xMid + scale * asset.z * sqrt(asset.t);

const f64 yMin = std::asinh((xMin - xMid) / density);
const f64 yMax = std::asinh((xMax - xMid) / density);

const f64 dy = 1. / (xDim - 1);
for (auto j = 0; j < xDim; ++j) {
    const f64 y = j * dy;
    xGrid(j) = xMid + density * std::sinh(yMin * (1.0 - y) + yMax * y);
}

/// Inspired by https://github.com/lballabio/QuantLib files:
///   - fdmblackscholesmesher.cpp
///   - fdblackscholesvanillaengine.cpp
```

## Conclusion

Finite-Difference is a very powerful method since it's not limited to the problems with constant
market factors or exotic contract features, like early exercise.
