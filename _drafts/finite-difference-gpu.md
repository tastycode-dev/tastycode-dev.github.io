---
layout: post
author: Oleksandr Gituliar
title: "Tasty C++ – Finite-Difference method with GPU"
---

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
