---
layout: post
author: Oleksandr Gituliar
title: "Tasty Quant – Finite Difference with GPU"
---

When it comes to pricing derivatives, there are two universal methods adopted by the industry
capable to deal with a wide range of products. The first one is the famous **Monte-Carlo** method.
Another one is the **Finite-Difference** method, which we will focus on in this post today.

What I'd like to share with you are the results of my experiment of porting a finite-difference
pricer from CPU to GPU. Since benchmarks look interesting and overall this sort of projects are not
very common on the web, I think it would be nice to share it with a wider audience.

**Source code** is here: <https://github.com/gituliar/kwinto-cuda>. It should be easy to compile on
Linux or Windows and run your own benchmark if you have Nvidia GPU.

## CPU vs GPU

At first, let's take a look what theoretical benefits we can expect by switching to GPU.

- First of all, $$$, a **GPU is cheap**.

  In Apr'23, on a secondary market in Denmark I paid $250 for the AMD Ryzen 9 X5900 and only $120
  for Nvidia GTX 1070. Remeber, that you can run multiple GPUs on a single motherboard (up to 20 on
  crypto-mining motherboards). Finally, keep in mind that to run an extra CPU it requires an another
  motherboard, RAM, HDD -- essentially a whole new machine (not counting for devops work to connect
  all them into a network, keep it up, manage software, etc.).

  All this easily gives extra 3-5x advantage in favour of GPU to start with.

- On average, a **GPU has ~100x more cores** than a CPU.

  For example, my Nvidia GTX 1070 has 1920 cores vs 12 cores on my AMD Ryzen 9 X5900, this is 160x
  more cores. Of course, GPU cores are less powerful and run at lower frequency. But still, a
  perspective of at least 10x speedup seems realistic.

- On average, a **GPU has 32x more single-precision units** than double-precision units, also known
  as Arithmetic Logic Units (ALU).

  In theory, this means 32x more operations per second by simply switching from `double` to `float`.
  Too good to be true for such a trivial change, hence worth to check on practice.

It's time to run some benchmarks to see how these theoretical arguments relate with practice.

## Benchmark

For benchmarking, we'll price american options in batches. This is usually how things are run in
practice when risk managing real trading books. Every batch (or portfolio) contains from 256 to
16'384 options, all different.

American options are good candidates for this exercise for two reasons. First, there is no
analytical formula for the price of American options, similar to the Black-Scholes-Merton formula
for European options. Second, since recently there is a highly-accurate (and fast) method available
for pricing American options, developed by Andersen et al. \[1\], which we'll employ as a
crosscheck.

<!-- <figure>
  <img src="/img/fd1d-gpu-z800.png"/>
  <figcaption>This is my caption text.</figcaption>
</figure> -->

![CPU vs GPU performance plot](/static/img/fd1d-gpu-b550.png)

In the plot above, every bin is an average over 8 consequitive batch runs, such that

- Every **CPU batch** is run in a single thread (so, account for _theoretical speedup_ of 12x for
  Ryzen 9 X5900 when run on multiple cores).
- Every **GPU batch** is run on all available GPU cores.

Let's postpone to discuss these results for the summary. Now, we'd better get more confidence in the
option prices we've generated for benchmarking.

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

## Summary

As we saw at the beginning, a single GPU card is about **4x cheaper** to run than CPU. This is
already a big benefit on the start. The question is whether it's faster or at least not that much
slower.

Next, **10x speedup** comes from the number of cores GPU contains. This is not 100x as we expected
from theory, but still a considerable gain.

Finally, instead of expected 32x speedup by switching from `double` to `float` we get only **2x
gain**. This is very likely due to that main bottleneck is not computation itself but data transfer
(as float-grid is 2x smaller than double-grid).

Final verdict: GPU is at least **5x faster / cheaper** than CPU to price with finite-difference.

## References

<https://hpcquantlib.wordpress.com/2022/10/09/high-performance-american-option-pricing> by Klaus
Spanderen

American options Pricing: "High-Performance American Option Pricing" by Andersen, Lake, Offengenden
