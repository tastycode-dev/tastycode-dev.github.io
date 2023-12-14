---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference"
---

- Option Chain for AAPL / AMD / TSLA

**Pricing American options** is an open problem in the quantitative finance. Currently, it has no
closed form solution similar to the Black-Scholes formula for European options. Therefore, various
_numerical methods_ are used to solve this problem in practice. This post is about one of such
methods -- [Finite-Difference
Method](https://en.wikipedia.org/wiki/Finite_difference_methods_for_option_pricing).

I code in C++,
however it should be accessible for people even with minimum C++ experience. It is more important to
have basic knowledge of options and linear algebra, although not required either.

You don't need to have hands-on experience with a finite-difference method. All necessary details
will appear as we go. It's neither scarry nor difficult, believe me.

**C++ is a great language** to implement a finite-difference pricer. My current setup is Ubuntu
22.04 and Windows 11 with Visual Studio 2022. To compile cross-platform I use CMake, which is deeply
integrated with Visual Studio and gives excellent development experience, similar to a native VS
project.

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
foundation and properties here, however there is one thing I'd like to stress.

**The European Option's** price, defined by the famous [Black-Scholes formula](), is just a
particular example of the Black-Scholes equation. In general, we can define similar pricing
equations for many other derivative types and flavours.

 <!-- Equations of this family have very similar
mathematical structure and can be solved using the finite-difference method we discuss here. This
makes it a very robust method for pricing derivatives that is worth to be familiar with. -->

**The pricing equation** is usually derived using one of the following methods (in order of my
personal preference):

- [Delta Hedging]() argument is a very intuitive yet powerful way to derive a pricing equation, not
  only for vanilla options, but for more exotic multi-asset derivatives as well. See ... for more
  details.

- [Feynman-Kac Theorem]() is very theoretical and I don't find it useful in practice.

Please, share in the comments you have different experience.

![Pricing PDE](/assets/img/fd-pde.png)

![Pricing PDE](/assets/img/fd-pde-x.png)

![Pricing PDE](/assets/img/fd-x.png)

## Numerical Solution

**The Black-Scholes equations** belong to the family of [diffusion equations](), which in general
case have no closed-form solution. For now we should remember only that it's one of the easiest
differential equations to solve numerically. Usually, apart from the
[Finite-Difference method](https://en.wikipedia.org/wiki/Finite_difference_method), they are also treated
with [Monte-Carlo]() or [Fourier transformation]() methods.

**The Unknown.** Let's start by formulating a problem we solve. Ultimately, we're looking for the
**option price** as a function `V(t,s)`, for _arbitrary spot_ price `s` at a _fixed time_ `t=0`,
where

- `t` is the future time from today;
- `s` is the spot price of the option's underlying asset.

Next, we continue with particular steps of the method:

**1) Finite Grid** is defined an `N x M` rectangular grid on the domain of independent variables
`(t,s)` defined as

- `t[i] = t[i-1] + dt[i] ` for `i=0..N-1`
- `x[j] = x[j-1] + dx[j] ` for `j=0..M-1`.

This naturally leads to the following C++ definitions:

**2) Difference Operator** is used to approximate continuous derivatives with a [finite
difference](https://en.wikipedia.org/wiki/Finite_difference#Basic_types) operation on the grid that
we have defined above.

Forward difference (backward difference):

![Discretization](/assets/img/fd-dVdt.png)

Central difference:

![Discretization](/assets/img/fd-dVdx.png)

**3) Finite-Difference Equation**, a discrete version of the Black-Scholes equation, is derived from
the pricing equation by replacing continuous derivatives with difference operators defined above.

It's convenient to introduce the A operator, which contains difference operators over the x-axis
only.

![Pricing PDE](/assets/img/fd-pde2.png)

**4) Solution Scheme.** The equation above is not defined completely yet, as we still have freedom
of choice for the difference operators.

**\delta_x and \delta_xx** operators are generally chosen according to the central difference
definition above.

**\delta_t** operator, on the other hand, might be chosen as _Forward_ or _Backward_ difference,
which lead to the [explicit
scheme](https://en.wikipedia.org/wiki/Finite_difference_method#Explicit_method) solution. However,
the numerical error is O(dt) + O(dx^2), which is not the best we can achieve.

**[Crank-Nicolson](https://en.wikipedia.org/wiki/Finite_difference_method#Crank%E2%80%93Nicolson_method)
scheme** is an implicit scheme, which is a better alternative to the explicit scheme, because the
numerical error is O(dt^2) + O(dx^2). It's slightly more complicated, since requires to solve a
liner system of equation, however offers better convergence. In short, Crank-Nicolson scheme is is a
fine mix of forward and backward schemes that reduces the numerical error.

- Euler forward for `\theta = 1`
- Euler backward for `\theta = 0`
- Crank-Nicolson for `\theta = 1/2`

![Pricing PDE](/assets/img/fd-schemes.png)

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
