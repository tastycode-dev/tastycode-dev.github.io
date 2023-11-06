---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference at 60fps"
---

_Intro._ Programs = Algorithms + Data Structures + Platform

**Intro.** I hear about GPU computing nowadays more often than ever. Sometimes it's hard to believe
what an incredible things people are building with a tiny box for crunching floating-point numbers
that we call GPU.

I was curious about GPU programming for a couple of years, but never had a chance to do it myself.
Until recently, when I started coding a finite-difference method for pricing derivatives and
realized that it might be a good candidate to run on GPU as well. In my previous post,
[Pricing Derivatives on a Budget](), I presented benchmarks for pricing American Options on CPU vs
GPU.

I hope, that after reading this post, you will be ready to start your own journey with GPU
programming in C++. It's neither scarry nor difficult, believe me. I'm going to share with you
everything I learned while porting the finite-difference pricer for American options to GPU.

**Tools.** We will use C++ and CUDA SDK from Nvidia, which you can install on Windows or
Linux. In my case, it was Windows 11 and Ubuntu 22.04. I also use Visual Studio 2022 with CMake
projects, which allow me to smoothly compile my code on both platforms.

## Finite-Difference Pricer

At first, let's look inside of the finite-difference pricer, so that we
better understand computational steps, their complexity, and potential to run in parallel on GPU.

**Backward Evolution.** The overall idea is to start with an option price at the maturity and
backward propagate it to the current point in time. The propagation happens on the 2D grid with
stock price and time to maturity along the axes. The backward evolution is governed by a discrete
version of the partial-differential equation, which is a difference equation, hence the name of the
method.

**Tridiagonal System.** Every backward step requires to solve a tridiagonal system of linear
equations. For this we can use the Thomas algorithm which is a simplified version of Gaussian
elimination and takes 20 lines of C++ code.

![CPU](/assets/img/fd-cpu.gif)

**Parallelization.** There are parallel versions of the Thomas method, however they are complex and
don't utilize all available cores at maximum performance.

## GPU Programming

A seasoned C++ programmer should have no difficulties to sketch CPU code for the animated algorithm.
Let's see why it might be not so straightforard on GPU and what we should do instead.

**GPU has its own RAM.** This shouldn't be a surprise to you, as anyone who has been looking for a
GPU card knows that "Memory Size" is in the specs of every GPU. The memory chip is soldered directly
to the card and is not upgradable, unlike CPU memory. From now on, let's focus more on the soft part
of the business and answer the following question:

_How to access GPU memory from C++ code ?_

Indeed, as C++ developers we are used to have access to the memory for granted. All we need is to
call `new` or `std::malloc` and, voila, we have a new block of CPU RAM for read and write.

**`cudaMalloc`.** GPU memory is similar to that. All we need is to use `cudaMalloc` from `cuda.h` to
allocate a new block of GPU RAM and of course not to forget about `cudaFree` when we are done.

However, we are unable to work with GPU block as we used to work with CPU block:

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));

gpuArray[12] = 34.56;    //  <-- ERROR
```

The issue is that this code runs on CPU, which has no direct access to GPU memory.

**`cudaMemcpy`.** To work with GPU memory we should use `cudaMemcpy` from `cuda.h` that allows to
transfer data from CPU to GPU and back, like this:

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));
double cpuArray = std::malloc(n * sizeof(double));

cpuArray[12] = 34.56;    //  <-- OK

//  CPU -> GPU
cudaMemcpy(cpuArray, gpuArray, n, HostToDevice);
```

The `HostToDevice` flag indicates that we copy data from CPU (the host) to GPU (the device). Guess
what another `DeviceToHost` flag is used for.

**Workflow.** Overall, the common workflow is:

1. Initialize input data in CPU memory.
2. Transfer input data to GPU with `cudaMemcpy` and `HostToDevice`.
3. Run GPU kernels to calculate the output. (See below about GPU kernels.)
4. Transfer output data to CPU with `cudaMemcpy` and `DeviceToHost`.

**STL in CUDA.** Like a modern C++ programmer you're likely feeling that using `cudaMalloc` and
`cudaMemcpy` is too low-level especially when you need to manually manage memory with `cudaMalloc`
and`cudaFree` without relying on smart pointers. Unfortunately you can't use STL in CUDA (there is
Thrust, but I did not try it)...

## GPU Code

**CUDA Kernels.** Now let's talk about CUDA kernels, which are C++ functions that run on GPU cores
with a direct access to GPU memory. GPU cores are not as fast as CPU cores, but running thousands of
them in parallel feels very fast.

<!-- ![CPU](/assets/img/demo.gif) -->

![CPU](/assets/img/fd-cpu.png)

![CPU](/assets/img/fd-cpu.gif)

Source code for CUDA kernels is located in `.cu` files and isn't much different from regular C++
code:

```cpp
/// kernel.cu
__global__
void cudaKernel() {
}

/// main.cpp
```

which must be compiled by a proprietory
[Nvidia CUDA Compiler](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler) (NVCC). In principle, you
can compile your whole codebase with NVCC, however it still lacks some modern features from the C++
Standard.

**Modern C++.** In my case, I decided to limit usage of NVCC to minimum and use it to compile only
GPU-related code. For the remaining code I keep using a more modern compilers, like GCC or MSVC.
This is possible to setup with CMake, see ... for how its done.

The idea is to wrap CUDA kernels into a regular C++ functions and compile those with NVCC into a
separate library. Then I link this library with a main part of the code, compiled with MSVC. This
way I'm able to use the latest features of C++ and get rid of many compilation errors.

## Nvidia Libraries

**cuSPARSE.** Nvidia offers a big set of libraries tuned for high-performance computation. One of
such libraries is cuSparse for sparse matrix operations. The function we need from this lib is a
tridiagonal matrix solver. The version in cuSparse can solve many of such matrices in parallel. This
is exactly what we need.

## Final Word

When doing HPC you should know your platform:

- **Memory.** GPU memory is not flat, there are many flavors of GPU RAM shared at different levels
  with various access rights (for example, read-only usually used to store textures)

- **Code.** Should be optimized for a specific GPU generation.

- **Hardware.** Physical architecture...
