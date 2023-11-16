---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference at 60fps"
---

_Intro._ Programs = Algorithms + Data Structures + Platform

**Intro.** I hear about GPU computing nowadays more often than ever. Sometimes it's hard to believe
what an incredible things people are building with a tiny box just for crunching floating-point
numbers that we call GPU.

I was curious about GPU programming for a couple of years now, but never had a chance to write
anything it myself. Until recently, when I started coding a finite-difference pricer for american
options and realized that it might be a good candidate to run on GPU as well. In my previous
post, [Pricing Derivatives on a Budget](), I presented my benchmarks for CPU and GPU.

I'm going to share with you everything I learned while coding my pricer for GPU. After reading this
post, you will be ready to start your own journey with GPU programming in C++. It's neither scarry
nor difficult, believe me.

**Tools.** We will use C++ and CUDA SDK from Nvidia, which you can install on Windows or Linux. In
my case, it was Windows 11 and Ubuntu 22.04. I also use Visual Studio 2022 with CMake projects,
which allow me to smoothly compile my code on both platforms.

## Finite-Difference Pricer

At first, let me briefly describe what is inside of the finite-difference pricer and how it works.
You don't need to know any complicated math, basics of linear algebra will be enough.

**Backward Evolution.** The overall idea is simple: we evolve an option price backwards in time to
the present moment from the future moment where it's known. The future moment is maturity time,
where option price is simply a payoff.

The backward evolution is driven by the _evolution matrix_ which is a tridiagonal matrix that
depends on volatility, interest and dividend rates. The evolution matrix is a discrete version of
partial-differential equation for the option price, see Wilmott for more details.

**Tridiagonal System.** To make a backward step on the grid we should solve a tridiagonal system of
linear equations. This is usually done using Thomas algorithm which is a simplified version of Gaussian
elimination. Its implementation take 10 lines of C++ code, so not very complicated.

<!-- ![CPU](/assets/img/fd-cpu-comics.png) -->

<!-- ![CPU](/assets/img/fd-cpu.png) -->

**Parallelization.** What is complicated however, is to run Thomas algorithm in parallel. There are
some parallel implementation, which are complex and don't utilize all available cores at maximum
performance.

I made an attempt to visualize all said above in the following animation:

![CPU](/assets/img/fd-cpu-comics.gif)

## GPU Programming

A seasoned programmer should have no difficulties to program the finite-difference pricer for CPU,
as it requires a couple of matrix multiplications and a solution of the tridiagonal linear system,
all wrapped in a loop. It might be not so straightforard for GPU though, so let's figure out why.

**GPU has its own RAM.** This shouldn't be a surprise to you, as anyone who has been looking for a
GPU card knows that "Memory Size" is in the specs of every GPU. The memory chip is soldered directly
to the card and is not upgradable, unlike CPU memory. From now on, let's focus more on the soft part
of the business and answer the following question:

_How to access GPU memory from C++ code ?_

Indeed, as C++ developers we are used to have access to the memory for granted. All we need is to
call `new` or `std::malloc` and, voila, we have a new block of CPU RAM for read and write.

**`cudaMalloc`.** GPU memory is similar to that. All we need is to use `cudaMalloc` from `cuda.h` to
allocate a new block of GPU RAM and of course not to forget about `cudaFree` when we are done.

However, we can't use `operator[]` to access a GPU block as we do with a CPU block. It's because
this code runs on CPU, which has no direct access to GPU memory:

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));

gpuArray[12] = 34.56;    //  <-- ERROR
```

**`cudaMemcpy`.** To work with GPU memory we should use `cudaMemcpy` from `cuda.h` that transfers
data from CPU to GPU and back, like this:

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));
double cpuArray = std::malloc(n * sizeof(double));

cpuArray[12] = 34.56;    //  <-- OK

//  CPU -> GPU
cudaMemcpy(cpuArray, gpuArray, n, HostToDevice);
```

The `HostToDevice` flag indicates that we copy data from CPU (the host) to GPU (the device). You
will guess what `DeviceToHost` flag is used for.

**STL in CUDA.** Like a modern C++ programmer you're likely feeling that using `cudaMalloc` and
`cudaMemcpy` is too low-level especially when you need to manually manage memory with `cudaMalloc`
and`cudaFree` without relying on smart pointers. Unfortunately you can't use STL in CUDA (there is
Thrust, but I did not try it)...

**Workflow.** Overall, the common workflow with GPU is to:

1. Initialize input data in CPU memory.
2. Transfer input data to GPU with `cudaMemcpy` and `HostToDevice`.
3. Calculate with GPU kernels. (See below about GPU kernels.)
4. Transfer output data to CPU with `cudaMemcpy` and `DeviceToHost`.

## GPU Code

**CUDA Kernels.** Now let's talk about CUDA kernels, which are C++ functions that run on GPU cores
with a direct access to GPU memory. GPU cores are not as fast as CPU cores, but running thousands of
them in parallel feels very fast.

Source code for CUDA kernels is located in `.cu` files and isn't much different from regular C++
code. GPU code shuld be compiled by a proprietory [Nvidia CUDA
Compiler](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler) (NVCC). In principle, you can compile
your whole codebase with NVCC, however it still lacks some modern features from the C++ Standard.

**Modern C++.** In practice, I limit usa NVCC only with GPU code, which I wrap into a regular C++
functions and compile with NVCC to a separate static library. This library is then linked with a
main code dedicated for CPU, compiled with MSVC or GCC. This way I'm able to use the latest features
of C++ in combination with GPU kernels. This setup is possible with CMake, see how I do it in
[kwinto-cuda](https://github.com/gituliar/kwinto-cuda) repo on GitHub.

**CUDA Libraries** contain a broad range of algorithms fine-tuned for GPU. One of such libraries is
[cuSparse](https://developer.nvidia.com/cusparse) for sparse matrix operations, which is handy for
the finite-different pricer.

It provides GPU version of the [Thomas
solver](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm) that solves multiple tridiagonal
systems in parallel on GPU:
[`cusparseDgtsvInterleavedBatch`](https://docs.nvidia.com/cuda/cusparse/#gtsvInterleavedBatch). This
is what I use for the efficient implementation of the finite-difference pricer.

## Final Word

At the end, let me just say two thoughts that are important, but nevertheless very often forgetten
in the ocean of technical details and sunshine of the modern hardware.

**GPUs excel** at crunching floating-point numbers. This is crucial for many classical pricers,
however in order to achive considerable gain GPU code should be tailored to a specific pricer and
GPU generation, which is tedious and time-consuming. Ensure that the gain is worth the investment,
as many problems are just good enough to solve on CPU.

**The algorithm is crucial** and when choosen wisely even a medicore CPU can beat the fastest GPU. A
good example is the
[Andersen-Lake-Offengenden](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2547027) algorithm
for pricing american options, which outperforms the finite-difference pricer 1000x.
