---
layout: post
author: Oleksandr Gituliar
title: "Tasty GPU – Cheaper Prices of American Options"
---

After five years working as a quant, I can tell that the wast majority of derivative pricing in the
financial industry is done on CPU. This is easily explained by two facts: (1) no GPU was available
when banks started developing their pricing analytics in 90's; and (2) banking is a conservative
business/sector, slow to upgrade to a new stack when main business works as usual (hence Cobol and
mainfraims are still very common).

**American Options.** In this post, I benchmark pricing of American Options on GPU. Since no
analytical formula exist to price American options (similar to the Black-Scholes formula for
European options), people in banks use numerical methods to solve this sort of problems. Such
methods are computationally greedy and require lots of hardware to risk-manage thousands of trades
in trading books.

**Finite Difference.** For the benchmark, I use my own implementations of the [finite-difference
method](https://en.wikipedia.org/wiki/Finite_difference_method) for CPU and GPU.
_[mention MC and Andersen]_
American options Pricing: "High-Performance American Option Pricing" by Andersen, Lake, Offengenden

**Source Code.** C++ / CUDA code is available on Github: <https://github.com/gituliar/kwinto-cuda>.
You should be able to run it on a Linux or Windows machine (with Nvidia GPU). _[, which deserves a
dedicated post]_

My main focus is on two things:

- How much _faster_ is GPU vs CPU? <br/>
  Speed is a convenient metric to compare performance, as faster usually means better (given all
  other factors equal).
- How much _cheaper_ is GPU vs CPU? <br/>
  Budget is always essential when making decisions in a company. Speed is important, as fast code
  means less CPU hours, but there are also other factors worth to discuss.

<!-- ![image](/assets/img/2023-08-22/og-image.png) -->

When it comes to pricing derivatives, there are two methods, widely-adopted by the industry, that
are capable to solve a wide range of pricing problems. The first one is the famous **Monte-Carlo**
method. Another one is the **Finite-Difference** method, which we will focus on in this post today.

## Benchmark

My approach is to price american options in batches. This is usually how things are run in banks,
for risk-managing trading books. Every batch contains from 256 to 16'384 american options, which are
priced in parallel utilizing all available cores on a CPU or GPU.

The total pool of 378'000 options for the benchmark is constructed by permuting all combinations of
the following parameters (with options cheaper than 0.5 rejected). Reference prices are calculated
using the [high-performance pricing
algorithm](https://hpcquantlib.wordpress.com/2022/10/09/high-performance-american-option-pricing/)
implemented by Klaus Spanderen in QuantLib, see
[portfolio.py](https://github.com/gituliar/kwinto-cuda/blob/main/test/portfolio.py) on GitHub.

| Parameter     | Range                                                                     |
| ------------- | ------------------------------------------------------------------------- |
| Parity        | PUT                                                                       |
| Strike        | 100                                                                       |
| Spot          | 25, 50, 80, 90, 100, 110, 120, 150, 175, 200                              |
| Maturity      | 1/12, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0                          |
| Volatility    | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 |
| Interest rate | 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%, 10%                                   |
| Dividend rate | 0%, 2%, 4%, 6%, 8%, 10%, 12%                                              |

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

**Results.** Below are the main results. Each bin shows how many options are priced per second,
so that higher is better.

![Benchmark CPU vs GPU](/assets/img/2023-08-22/bench-512-cpu-gpu.png)

**US Options Market.**
To give you some idea how the US option market look like, here are some numbers from
[OPRA](https://www.opraplan.com):

- 5'800 stocks
- 680'000 options (delta in 5%-95% range)

In other words, a $100 GPU, can **price entire US Options Market in just 2 min.**
Obviously, this assumes that models are calibrated. And remember, the algorithm of Andersen et al.
is much faster for this exercise.

Few things to keep in mind when analyzing the plot:

1. **GPU is 2x faster** in a single-precision FP32 mode only (with a good enough accuracy for
   production). This is given that Nvidia GTX 1070 has 1920 cores vs 12 cores on AMD Ryzen 9 X5900.
2. **GPU is 4 years older.** This is a considerable gap in the hardware world:

   - [Nvidia GTX 1070](https://www.techpowerup.com/gpu-specs/geforce-gtx-1070.c2840) was released in
     Jun'16.
   - [AMD Ryzen 9 X5900](https://www.techpowerup.com/cpu-specs/ryzen-9-5900x.c2363) was released in
     Nov'20.

3. **GPU loves big batches**, while CPU is most efficient for small jobs.

## Budget

In the enterprise environment, _faster_ is almost always means _cheaper_. For example, an overnight
job that takes 4 hours to calculate a risk report, might still take the same time, but half of the
hardware resources if optimized to run 2x faster.

Speed is not the only factor that affects operational costs. Let's take a look at other factors that
appeal in favour of GPU:

**Cheap to buy.** In Apr'23, on a secondary market in Denmark I paid $250 for the AMD Ryzen 9 X5900
and only $120 for Nvidia GTX 1070. To run an extra CPU it requires another motherboard, RAM, HDD --
essentially a whole new machine. For a GPU you need an extra PCI-E slot only.

**Cheap to run.** A single motherboard can run multiple GPUs (up to 20 on crypto-mining
motherboards). Less hardware = lower operational cost and less staff. No need to run a network,
manage software updates on every machine, and hire an army of devops to automate all that.

**Cheap to upgrade.** PCI-E architecture is backward compatible and is upgraded not as often as CPU
sockets. Below is the same benchmark run on a much older [HP
Z800](https://en.wikipedia.org/wiki/HP_Z) machine from 2009:

![Benchmark CPU vs GPU](/assets/img/2023-08-22/bench-z800.png)

What immediately catches the eye is that Ryzen 9 outperforms the dual-Xeon setup. This is something
we expect and is not a surprise. However, surprising is that **GPU performs equally well on the
machine from 2009 and 2020**. In practice, this reduces operational costs by eliminating the need to
replace CPU platform every 10 years.

All this easily gives extra 3-5x advantage in favour of GPU.

## Summary

As we saw at the beginning, a single GPU card is about **4x cheaper** to run than CPU. This is
already a big benefit on the start. The question is whether it's faster or at least not that much
slower.

Next, **2x speedup** comes from the number of cores GPU contains. This is not 100x as we expected
from theory, but still a considerable gain.

Finally, instead of expected 32x speedup by switching from `double` to `float` we get only **2x
gain**. This is very likely due to that main bottleneck is not computation itself but data transfer
(as float-grid is 2x smaller than double-grid).

**Final verdict.** GPU is **20x cheaper** to price with finite-difference than CPU.
