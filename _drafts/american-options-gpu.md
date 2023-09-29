---
layout: post
author: Oleksandr Gituliar
title: "Tasty GPU – Pricing Derivatives on a Budget"
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

**Finite Difference.** For the benchmark, I use my own implementations of the
[finite-difference method](https://en.wikipedia.org/wiki/Finite_difference_method) for CPU and GPU.
_[mention MC and Andersen]_ American options Pricing: "High-Performance American Option Pricing" by
Andersen, Lake, Offengenden. When it comes to pricing derivatives, there are two methods,
widely-adopted by the industry, that are capable to solve a wide range of pricing problems. The
first one is the famous **Monte-Carlo** method. Another one is the **Finite-Difference** method,
which we will focus on in this post today.

**Source Code.** C++ / CUDA code is available on Github: <https://github.com/gituliar/kwinto-cuda>.
You should be able to run it on a Linux or Windows machine (with Nvidia GPU). _[, which deserves a
dedicated post]_

**Main focus** is on two things:

- How much <u>faster</u> is GPU vs CPU? <br/> Speed is a convenient metric to compare performance,
  as faster usually means better (given all other factors equal).
- How much <u>cheaper</u> is GPU vs CPU? <br/> Budget is always important when making decisions.
  Speed matters because fast code means less CPU time, but there are other essential factors worth
  discussing that imact your budget.

<!-- ![image](/assets/img/2023-08-22/og-image.png) -->

## Benchmark

My approach is to price american options <u>in batches</u>. This is usually how things are run in
banks, when risk-managing trading books. Every batch contains from 256 to 16'384 american options,
which are priced in parallel utilizing <u>all CPU or GPU cores</u>.

The total pool of 378'000 options for the benchmark is constructed by permuting all combinations of
the following parameters (with options cheaper than 0.5 rejected). <u>Reference prices</u> are
calculated with [portfolio.py](https://github.com/gituliar/kwinto-cuda/blob/main/test/portfolio.py)
in QuantLib, using the
[Spanderen implementation](https://hpcquantlib.wordpress.com/2022/10/09/high-performance-american-option-pricing/)
of a high-performance
[Andersen-Lake-Offengenden algorithm](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2547027)
for pricing american options.

| Parameter     | Range                                                                     |
| ------------- | ------------------------------------------------------------------------- |
| Parity        | PUT                                                                       |
| Strike        | 100                                                                       |
| Spot          | 25, 50, 80, 90, 100, 110, 120, 150, 175, 200                              |
| Maturity      | 1/12, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0                          |
| Volatility    | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 |
| Interest rate | 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%, 10%                                   |
| Dividend rate | 0%, 2%, 4%, 6%, 8%, 10%, 12%                                              |

<!-- For this project, American options are good candidates for several reasons:

1.  No analytical formula exist to price American options (similar to the Black-Scholes-Merton
    formula for European options), which doesn't make the code artificial / for-benchmark-only.
2.  Since recently, a highly-accurate and fast method to price American options became available,
    (see Andersen et al. \[1\]). We use its implementation from the QuantLib for a crosscheck.
3.  Thirdly, the code can be extended to price exotic options for which no fast method exist. -->

<!-- <figure>
  <img src="/img/fd1d-gpu-z800.png"/>
  <figcaption>This is my caption text.</figcaption>
</figure> -->

**Results.** A plot below depicts the main results. Each bin shows how many options are priced per
second (higher is better):

![Benchmark CPU vs GPU](/assets/img/2023-08-22/bench-512-cpu-gpu.png)

**US Options Market.** To get some idea about how these results are useful in practice, let's have a
look at a size of the US options market. The data from [OPRA](https://www.opraplan.com) tells that
there are:

- 5'800 stocks
- 680'000 options (with 5%-95% delta)

In other words, it takes **2 min to price** an entire US Options Market on a $100 GPU. It should
take 10x longer for calibration, which is a more challenging task.

Few things to keep in mind for the plot above:

1. **GPU is 2x faster** in a <u>single-precision</u> mode (gray bin) vs double-precision (yellow
   bin). Meantime, CPU performs more or less the same in both modes (blue and orange bins).

   This is a clear sign that the GPU is limited by <u>data throughput</u>. In other words, with its
   1'920 cores, the GPU processes data faster than loads it from the GPU memory.

2. **GPU is 4 years older**, which is a big gap for hardware. Nevertheless, the oldish GPU is still
   faster than the modern CPU.

   - [Nvidia GTX 1070](https://www.techpowerup.com/gpu-specs/geforce-gtx-1070.c2840), 16nm --
     released in 2016.
   - [AMD Ryzen 9 X5900](https://www.techpowerup.com/cpu-specs/ryzen-9-5900x.c2363), 7nm -- released
     in 2020.

<!-- 3. **GPU loves big batches**, while CPU is most efficient for small jobs. -->

## Budget

In the production environment, <u>faster</u> is almost always means <u>cheaper</u>. Speed is not the
only factor that affects operational costs. Let's take a look at other factors that appeal in favour
of GPU:

<!-- **Cheap to setup.** In Apr'23, on a secondary market in Denmark I paid $250 for the AMD Ryzen 9
X5900 and only $120 for Nvidia GTX 1070. Obviously, <u>CPU requires</u> a motherboard, RAM, HDD -- a
whole machine -- so final price is 3x higher than that. <u>GPU requires</u> only a PCI-E slot. -->

**Cheap to scale.** To run an <u>extra CPU</u> it requires a motherboard, RAM, HDD -- a whole new
machine, which quickly becomes pricy at scale.

An <u>extra GPU</u>, however, requires only a PCI-E slot. Some motherboards offer a dozen PCI-E
ports, like [ASRock Q270 Pro BTC+](https://www.asrock.com/mb/Intel/Q270%20Pro%20BTC+/index.asp),
which is especially popular among crypto miners. Such a motherboard can handle <u>17 GPUs on a
single machine</u>. In addition, there is no need to setup a network, manage software on various
machines, and hire an army of devops to automate all that.

**Cheap to upgrade.** PCI-E standard is backward compatible, so that new GPU cards are still run on
old motherboards. Below is the same benchmark run on a much older machine with dual
[Xeon X5675](https://www.techpowerup.com/cpu-specs/xeon-x5675.c949) from 2011:

![Benchmark CPU vs GPU](/assets/img/2023-08-22/bench-z800.png)

What immediately catches the eye is that Ryzen 9 outperforms the dual-Xeon setup (both have equal
number of physical cores, btw). This is not a surprise, given a 10-year gap.

However, surprising is that a newer **GPU performs equally well** on a much older machine. In
practice, this means that at some point in the future when GPU cards deserve an upgrade there is no
need to upgrade other components, like CPU, motherboard, etc.

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
