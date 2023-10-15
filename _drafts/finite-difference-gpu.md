---
layout: post
author: Oleksandr Gituliar
title: "Tasty Quant – Finite-Difference at 60fps"
---

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

## Crossheck

Obviously, it doesn't make sense to benchmark the wrong code. To ensure that my code is correct, I
compare the results by pricing a portfolio of 4'495 American put options against a highly-accurate
algorithm of Andersen et al. Its implementation by Klaus Spanderen is available in QuantLib, see his
blog for more details \[2\]. Thank you Klaus for your contribution!

In fact, this is the same portfolio used in \[1\], constructed of options by permuting all
combinations of the following parameters (with filtering out options cheaper than 0.5):

| Parameter                   | Range                                        |
| --------------------------- | -------------------------------------------- |
| **k** -- strike             | 100                                          |
| **s** -- spot               | 25, 50, 80, 90, 100, 110, 120, 150, 175, 200 |
| **t** -- time to maturity   | 1/12, 0.25, 0.5, 0.75, 1.0                   |
| **z** -- implied volatility | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6                 |
| **r** -- interest rate      | 2%, 4%, 6%, 8%, 10%                          |
| **q** -- dividend rate      | 0%, 4%, 8%, 12%                              |
| **w** -- parity             | PUT                                          |

In the table below are the crosscheck results, which contain root-mean-square (RMSE / RRMSE) and
maximum (MAE / MRE) absolute / relative errors.

|       | CPU x32 | CPU x64 | GPU x32 | GPU x64 |
| ----- | ------- | ------- | ------- | ------- |
| RMSE  | 20.7e-4 | 5.4e-4  | 15.8e-4 | 5.4e-4  |
| RRMSE | 9.9e-5  | 8.1e-5  | 9.1e-5  | 8.1e-5  |
| MAE   | 23.7e-3 | 4.3e-3  | 25.1e-3 | 4.3e-3  |
| MRE   | 1.1e-3  | 1.1e-3  | 1.1e-3  | 1.1e-3  |

See Andersen et al where they compare the same portfolio with various other methods.

## References

<https://hpcquantlib.wordpress.com/2022/10/09/high-performance-american-option-pricing> by Klaus
Spanderen
