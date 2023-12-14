---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference"
---

- Option Chain for AAPL / AMD / TSLA

**In this post** I'll show how to price American options using [Finite-Difference
Method](https://en.wikipedia.org/wiki/Finite_difference_methods_for_option_pricing). I code in C++,
however it should be accessible for people even with minimum C++ experience. It is more important to
have basic knowledge of options and linear algebra, although not required either.

It's neither required to have hands-on experience with a finite-difference method. You'll find an
overview with all necessary details below, without deep details of the algorithm. It's neither
scarry nor difficult, believe me.

**We will use C++** on Linux and Windows. My current setup is Ubuntu 22.04 and Windows 11 with
Visual Studio 2022. To compile cross-platform I use CMake, which is deeply integrated with Visual
Studio and gives as smooth development experience as native VS project.

**Pricing for the Market** ...

![AMD Option Chain on 2023-11-17 15:05](/assets/img/202311171505-AMD-retro.png)

**Don't worry** if some concepts seem complicated at first. There is nothing difficult in what
follows. Trust me. Give it another chance and consult references for more details and alternative
perspective, which usually helps to connect the dots.

Let's start!

## Pricing Equation

- Black-Scholes formula / equation
- Black-Scholes family of equations

It's not a surprise that we start with the [Black-Scholes
equation](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_equation), since it's this equation
that we should solve to find the price of the American option. I'm not going to dwell much on its
foundation and properties here, however there is one thing which is important to keep in mind.

**The European Option** price, the famous Black-Scholes formula, is just one example of the
Black-Scholes equation. In general, we can define similar pricing equations for many other
derivative types and flavours. All these equations have very similar mathematical structure and can
be solved using the finite-difference method discussed below. This makes the
method we discuss in this post universal and worth to be aware of.

Below is a brief overview of two methods to derive pricing equations for various derivatives,
similar in spirit to the Black-Scholes equation for the European option price. in order of my
personal preferences.

**Delta Hedging.** This is a very intuitive yet powerful way to derive a pricing equation, not only
for vanilla options, but for more exotic multi-asset instruments as well.

**Feynman-Kac Theorem.** This approach is very theoretical and so far I don't find it useful in
practice.

Please, share in the comments you have different experience.

![Pricing PDE](/assets/img/fd-pde.png)

![Pricing PDE](/assets/img/fd-pde-x.png)

![Pricing PDE](/assets/img/fd-x.png)

## Numerical Solution

**The Black-Scholes family** of equations, in mathematics, fit under the umbrella of the [diffusion
equations](), which in general case has no closed-form solution. Fortunately, this is one of the
easiest differential equations to solve numerically. Some popular methods for that are
[Monte-Carlo]() and [Fourier transformation]().

This post, however, is dedicated to the [finite-difference method](https://en.wikipedia.org/wiki/Finite_difference_method), which is as popular in the
quantitative finance as the famous Monte-Carlo method.

**The Unknown.** It's important to explicitly formulate a problem we are solving. Hence, let's
define the unknown price as a function `V(t,s)`, which we look for for an arbitrary value of `s` and
a fixed time `t=0`, which corresponds to today. Here, `t` is the time from today (to expiry) and `s`
is the spot price of the option's underlying asset.

**1) Grid.** We start with introducing an `N x M` rectangular grid on the domain of
independent variables `(t,s)` defined by `t[i] = t[i-1] + dt[i]` and `x[j] = x[j-1] + dx[j]`, such
that `i=0..N-1` and `j=0..M-1`.

This naturally leads to the following C++ definitions:

**2) Difference Operator.** Next, let's introduce the difference operator, so that we can
approximate continuous derivatives with a [finite
difference](https://en.wikipedia.org/wiki/Finite_difference#Basic_types), on the grid we just
defined.

Forward difference (backward difference):

![Discretization](/assets/img/fd-dVdt.png)

Central difference:

![Discretization](/assets/img/fd-dVdx.png)

**3) Finite-Difference Equation.** With the help of the difference operators we just defined, let's
write down a finite-difference version of the differential pricing equation.

Where it's convenient to introduce the A operator, which consists of the difference operators over
the x-axis only.

![Pricing PDE](/assets/img/fd-pde2.png)

**4) Solution Scheme.**

**Crank-Nicolson** is probably on of the most popular discretization schemes for the Black-Scholes
equation.

Explicit:

- Euler forward
- Euler backward

Implicit:

- Crank-Nicolson

![Pricing PDE](/assets/img/fd-schemes.png)

Depending on how we expand the time derivative, we get forward or backward equation (?). These
equations however are O(dt + dx^2) error. We can do better than that by introducing a continuos
parameter \theta that defines a mixed scheme which is strictly forward for \theta = 1, backward for
\theta = 0, and mixed for everything in between. For \theta = 1/2 it's called Crank-Nicolson scheme
and has O(dt^2 + dx^2) error.

Finally, we get a linear system of equations that defines evolution of the price function over the
time axis.

**5) Backward Evolution**

We don't know function V(t=0, s), obviously this is what we are looking for. Hence, forward
evolution is no go, as we don't know the initial condition is unknown. However, we know the initial
condition for the backward evolution, as V(t=T, s) is a payoff at expiry. It is (K-s)+ for CALL and
(s-K)+ for PUT options.

**Thomas Algorithm** is [link](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm) ...

**6) Early Exercise.** For American options we should taken into account the right to early
exercise. If continue like that we will get a European option price. For the American exercise, we
should ensure that option price is not less than its intrinsic value, otherwise we'll get an
arbitrage situation when one can buy an option for the lower price than its exercise value.

In other words:

![Early Exercise Condition](/assets/img/fd-early-exercise.png)

## Boundary Conditions

You probably noticed one inconsistency with difference operators that deserve extra attention.
Namely, the x-axis difference is not well defined at the grid's boundaries.

**At boundaries**, the x-axis difference operators are not well defined as values outside of the
grid are missing. For example when calculating dV_0 according to the definition (xx), we need V[-1]
value which is undefined.

**The solution** is to account for the asymptotic behavior of the price function `V(t,x)` at
boundaries of `s`, when `s -> 0` and `s -> +oo`.

We know that `delta` is constant at boundaries, either 0 or 1, depending on the parity (PUT or
CALL). However, more universal relation is that `gamma` is zero at boundaries. This gives the
following relation:

## Finite-Difference Grid

Finally, it's time to discuss how grid points are distributed of `x`- and `t`-axes. So far we just
said that there are `N` and `M` points over the each axis, but said nothing about the limits and
distribution of those points. In other words, what are the values `x[0]` / `x[M-1]` and gaps `dt[i]
= t[i+1] - t[i]` and `dx[i] = x[i+1] - x[i]`

**The t-Axis** is divided uniformly with a step dt = T / N between points. It doesn't seem to use
some non-uniform step here, at least not something I observed in practice.

**The x-Axis** is divided in a more tricky way. ...

![Asinh Plot](/assets/img/asinh.png)

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

Finite-Difference is a powerful numerical method, widely used by the quantitative finance
practitioners. It can treat problems with exotic derivatives with non-constant coefficients and is
easy to program, as we saw in this post.
