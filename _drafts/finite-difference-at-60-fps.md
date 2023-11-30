---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference at 60fps"
---

**GPU computing** is a buzzword today more than ever. More often I can't believe what incredible
projects people are building with GPU, which in a nutshell is just a box for crunching
floating-point numbers. _LLM, image generation models, but also HPC on GPU_

I was curious about GPU programming for a couple of years now, but never had a chance to write
anything at a low level myself. Until recently, when I started coding a finite-difference pricer for
american options and realized that it might be a good candidate to run on GPU as well. Take a look
at my previous post, [Pricing Derivatives on a Budget](), with benchmarks and some performance
analysis.

**In this post** I show how to implement a
[Finite-Difference Pricer](https://en.wikipedia.org/wiki/Finite_difference_methods_for_option_pricing#External_links)
on GPU. It is for people familiar with C++, but with no programming experience for GPU.

It's neither required to have hands-on experience with a finite-difference method. You'll find an
overview with all necessary details below, since for this post we need only key calculation steps,
without deep details of the algorithm. It's neither scarry nor difficult, believe me.

Let me know if you want those details in the next post (like, react, subscribe, etc.). For now, see
books for more details.

**We will use C++** and [CUDA SDK](https://developer.nvidia.com/cuda-toolkit), which is available
for Linux and Windows. My current setup is Ubuntu 22.04 and Windows 11 with Visual Studio 2022. To
compile cross-platform I use CMake, which is deeply integrated with Visual Studio and gives as
smooth development experience as native VS project.

## Finite-Difference Pricer

As promised, let me start with a brief description of the finite-difference pricer and what
computational challenges should we expect. You don't need to know any complicated math, basics of
linear algebra will be enough.

**The problem we solve** is to find a price of the american option today given that we know it at
some future point in time. This option price is a function of the spot for the fixed strike and
maturity. The future point is convenient to fix at expiry date as the option price there is simply a
payoff function, for example `max(0, S-K)` for a call option.

This problem is easy to generalize to other derivatives that's why the finite-difference is so
popular for pricing a wide range of instruments.

**Backward Evolution.** The overall idea is to evolve the known option price backwards to the
present moment. This backward evolution is driven by the _evolution matrix_ which is a
[tridiagonal matrix](https://en.wikipedia.org/wiki/Tridiagonal_matrix) that depends on volatility,
interest and dividend rates.

In short, the evolution matrix is a _discrete form_ of the continuous partial-differential equation
for the option price. The discretization also defines a finite grid, points on the spot-time plane,
where we will calculate the option price function. The practical size of the grid is usually 500
(spot) x 1000 (time) points. See Wilmott for more details.

**Tridiagonal System.** For every backward step on the grid we have to solve a tridiagonal system.
At the end of the calculation, this results in solving 1000 tridiagonal systems in total, each of
the size 500.

The solution is usually found with the
[Thomas algorithm](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm) which is a simplified
version of Gaussian elimination. Its implementation take 10 lines of C++ code, so not very
complicated.

**Parallelization.** Apart from a couple of matrix multiplications, the Thomas algorithm in the
bottleneck of the overall computation. Unfortunately, it has no parallel version. There are some
multi-threading alternatives, which are complex and don't optimally utilize all available cores.
Also, we can't solve all 1000 systems at once as every system depends on the solution of the
previous one.

However, we can solve multiple grids in parallel, effectively pricing multiple american options
simultaneously. Of course, this gives no speedup for pricing an individual option. However, when
pricing a portfolio of multiple options or calibrating an option chain, this can give a noticeable
speedup.

I made an attempt to visualize the solution process in the following animation:

<!-- ![CPU](/assets/img/fd-cpu-comics.gif) -->

![CPU](/assets/img/fd-video.gif)

## GPU Memory

The above algorithm for a finite-difference pricer is a low-hanging fruit even for a mediocre
programmer. It requires to update an array of doubles in a loop, which contains the Thomas solver
and a couple of matrix multiplications. Basically, nothing more difficult than a regular sorting
algorithm. To program this pricer for GPU, however, we need to perform a few extra steps.

Let's uncover what exactly makes GPU a more complicated beast.

**GPU has its own RAM.** Not a surprise, as you likely know that every GPU has "Memory Size" in its
specification. The memory chip is soldered directly to the card and is not upgradable, like CPU
memory. This physical separation is what makes GPU memory a special resource from the CPU's
perspective. Hence, we have to use special GPU-specific functions from CUDA library:

**`cudaMalloc`** function from `cuda.h` which allocates a new block of GPU memory and returns a
pointer to it. This pointer is an address that points -- important -- to the GPU
[memory space](https://en.wikipedia.org/wiki/Address_space). This behavior is similar to `malloc`
function that also returns a pointer, but to the CPU space.

Therefore, the first obstacle is to remember which memory space, CPU or GPU, every pointer belongs
to. Unfortunately, compiles don't keep track of this, as for them any pointer is just a 64-bit
number.

If misused, this leads to a logical error, when GPU memory is accessed by CPU code, like in the
example below.

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));

gpuArray[0] = 12.3;       //  <-- LOGICAL ERROR:
                          //      GPU memory access from CPU code
```

**`cudaMemcpy`** function from `cuda.h` copies data between CPU and GPU. It is a standard way to
initialize GPU data: by copying from CPU memory. This is also how you can transfer GPU results back
to the CPU memory space. This function is much slower than direct memory access from GPU code, hence
use with care. For example:

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));
double cpuArray = std::malloc(n * sizeof(double));

for (auto i = 0; i < n; i++)
    cpuArray[i] = i*i;    //  <-- OK

//  CPU -> GPU (init GPU array)
cudaMemcpy(cpuArray, gpuArray, n, HostToDevice);
```

**STL in CUDA** deserves a special comment. For example, think about an internal representation of
some STL container, like `std::map`, `std::list` or `std::set`. They heavily rely on pointers in
their internal representation, and all those are CPU pointers. It is not an easy task to migrate an
STL container from CPU to GPU pointers. Hence, STL usage is very limited within GPU code.

The [`Thrust` library](https://thrust.github.io/) is designed to partially fill this gap. Its main
focus, however is on the parallel implementation of the STL algorithm, rather than STL containers,
but it provides useful abstractions to work with CPU and GPU memory (for example, see
`thrust::host_vector` and `thrust::device_vector` to get a better idea of what I mean).

## GPU Code

Now that we understand how to deal with GPU memory (roughly), let's talk about actually writing GPU
code.

**CUDA Kernels.** To run on GPU, C++ code should be implemented in the form of a kernel. These CUDA
kernels are very much are like regular C++ functions with some extra info available to them at
runtime, like a GPU thread number and total number of threads to run. This allows to select which
part of the data this particular function is going to process, as many of such functions are running
in parallel.

Source code for CUDA kernels is located in `.cu` files and isn't much different from regular C++
code. However, they should be compiled with a proprietary
[Nvidia CUDA Compiler](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler) (NVCC). In principle, you
can compile your whole codebase with NVCC, however it lacks many features from the C++ Standard.

**Modern C++.** In practice, I limit use NVCC only with GPU code, which I wrap into a regular C++
functions and compile with NVCC to a separate static library. This library is then linked with a
main code dedicated for CPU, compiled with MSVC or GCC. This way I'm able to use the latest features
of C++ in combination with GPU kernels. This setup is possible with CMake, see how I do it in
[kwinto-cuda](https://github.com/gituliar/kwinto-cuda) repo on GitHub.

**CUDA Libraries.** CUDA SDK offers a broad range of libraries fine-tuned for GPU. One of such
libraries is [cuSparse](https://developer.nvidia.com/cusparse) for sparse matrix operations, which
is handy to use for the finite-different pricer.

The cuSparse library provides GPU version of the
[Thomas solver](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm) that solves multiple
tridiagonal systems in parallel on GPU:
[`cusparseDgtsvInterleavedBatch`](https://docs.nvidia.com/cuda/cusparse/#gtsvInterleavedBatch). This
is what I use for the efficient implementation of the finite-difference pricer.

## Final Word

Finally, let me focus on a more general ideas than GPU programming, but nevertheless very often
forgotten in the ocean of technical details and sunshine of the modern hardware.

**GPU is an excellent device** to crunch decimal numbers. This is crucial for numerical methods used
in many derivative pricers, however, in order to achieve considerable gain, GPU code should be
tailored to a specific problem and GPU model. This is tedious, time-consuming, and requires
exclusive knowledge of the particular GPU architecture. However, even a non-expert can achieve
decent speedup with a minimum effort by translating CPU code to GPU.

**The algorithm is crucial** and when chosen wisely even a mediocre CPU can beat the fastest GPU. A
good example is the
[Andersen-Lake-Offengenden](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2547027) algorithm
for pricing american options, which outperforms the finite-difference pricer 1000x.
