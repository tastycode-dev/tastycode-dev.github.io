---
layout: post
author: Oleksandr Gituliar
title: "Tasty Quant – Benefits of Pricing Derivatives with GPU"
---

After five years working as a quant, I can tell that the wast majority of derivative pricing in the
financial industry is done on CPU. This is easily explained by two facts: (1) no GPU was available
when banks started developing their pricing analytics in 90's; and (2) banking is a conservative
business/sector, slow to upgrade to a new stack when main business works as usual (hence Cobol and
mainfraims are still very common).

This post is my benchmark of pricing American Options with a finite-difference method on CPU vs GPU.
My main focus is on two things:

- How much _faster_ is GPU vs CPU? <br/>
  Speed is a convenient metric to compare performance, as faster usually means better (given all
  other factors equal).
- How much _cheaper_ is GPU vs CPU? <br/>
  Budget is always essential when making decisions in a bank. Speed is important, as fast code eats
  less CPU hours, but there are also other factors worth to discuss.

<!-- ![image](/assets/img/2023-08-22/og-image.png) -->

When it comes to pricing derivatives, there are two methods, widely-adopted by the industry,
that are capable to solve a wide range of problems. The first one is the famous **Monte-Carlo**
method. Another one is the **Finite-Difference** method, which we will focus on in this post today.

**Source code**: <https://github.com/gituliar/kwinto-cuda>. You should be able to run it on Linux or
Windows (Nvidia GPU is required).

## Benchmark

My approach is to price american options in batches. This is usually how things are run in
production, when risk-managing trading books. Every batch (or portfolio) contains from 256 to 16'384
american options.

The total pool of options is constructed by permuting all combinations of the following parameters
(with filtering out options cheaper than 0.5):

| Parameter     | Range                                        |
| ------------- | -------------------------------------------- |
| Parity        | PUT                                          |
| Strike        | 100                                          |
| Spot          | 25, 50, 80, 90, 100, 110, 120, 150, 175, 200 |
| Maturity      | 1/12, 0.25, 0.5, 0.75, 1.0                   |
| Volatility    | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6                 |
| Interest rate | 2%, 4%, 6%, 8%, 10%                          |
| Dividend rate | 0%, 4%, 8%, 12%                              |

For this project, American options are good candidates for several reasons:

1.  No analytical formula exist to price American options (similar to the Black-Scholes-Merton
    formula for European options), which doesn't make the code artificial / for-benchmark-only.
2.  Since recently, a highly-accurate and fast method to price American options became available,
    (see Andersen et al. \[1\]). We use its implementation from the QuantLib for a crosscheck.
3.  Thirdly, the code can be extended to price exotic options for which no fast method exist.

<!-- <figure>
  <img src="/img/fd1d-gpu-z800.png"/>
  <figcaption>This is my caption text.</figcaption>
</figure> -->

In the plot below, every bin is an average over 8 consequitive batch runs, such that

![Benchmark CPU vs GPU](/assets/img/2023-08-22/bench-512-cpu-gpu.png)

Let's postpone to discuss these results for the summary. Now, we'd better get more confidence in the
option prices we've generated for benchmarking.

## Other Factors

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

### Old CPU + Same GPU

![Benchmark CPU vs GPU](/assets/img/2023-08-22/bench-z800.png)

On another image you can see how the same algorithm performs on a much older hardware from 20xx.
What immediately catches the eye is that the new Ryzen 9 outperforms the old Xeon. This is something
we expect and is not a surprise. However, surprising is that GPU performs equally well on both
machines. In practice, this reduces operational costs by eliminating the need to replace CPU platform every N
years.

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

Next, **2x speedup** comes from the number of cores GPU contains. This is not 100x as we expected
from theory, but still a considerable gain.

Finally, instead of expected 32x speedup by switching from `double` to `float` we get only **2x
gain**. This is very likely due to that main bottleneck is not computation itself but data transfer
(as float-grid is 2x smaller than double-grid).

Final verdict: GPU is **20x cheaper** to price with finite-difference than CPU.

## References

<https://hpcquantlib.wordpress.com/2022/10/09/high-performance-american-option-pricing> by Klaus
Spanderen

American options Pricing: "High-Performance American Option Pricing" by Andersen, Lake, Offengenden
