---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference at 60fps"
---

_Intro._ Programs = Algorithms + Data Structures + Platform

I hear about GPU computing nowadays more often than ever. Sometimes it's hard to believe what an
incredible things people are building with a tiny box for crunching floating-point numbers that we
call GPU.

I was curious about GPU programming for a couple of years, but never had a chance to do it myself.
Until recently, when I started coding a finite-difference method for pricing derivatives and
realized that it might be a good candidate to run on GPU as well. In my previous post, [Pricing
Derivatives on a Budget](), I presented benchmarks for pricing American Options on CPU vs GPU.

After reading this post, you will be ready to start your own journey with GPU programming in C++.
It's neither scarry nor difficult, believe me. I'm going to share with you everything I learned
while porting my finite-difference pricer for American options from CPU to GPU.

We will use C++ and CUDA SDK from Nvidia. You have freedom to install CUDA on Windows or Linux. In
my case, it was Windows 11 and Ubuntu 22.04. I also use Visual Studio 2022 with CMake-based projects
which allow me to comfortably compile my code on both platforms.

## GPU Memory

![CPU](/assets/img/cpu-gpu.png)

**GPU has its own RAM.** This shouldn't be a secret to you, as anyone who has been looking to buy a
GPU cardjj knows that "Memory Size" is in the specs of every GPU. The memory chip is soldered directly
to the card and is not upgradable, unlike CPU memory. Let's focus on the soft part of the business
and answer the following question:

_How to access GPU memory from C++ code ?_

Indeed, as C++ developers we are used to have access to CPU memory for granted. All we need is to
call `new` or `std::malloc` and a new block of memory is ready for read and write.

GPU memory is very similar in that sense. All we need is to use `cudaMalloc` from `cuda_xxx.h` to
allocate a new block of GPU memory and of course not to forget about `cudaFree` when we are done.

However, we are unable to work with GPU block as we used to work with CPU block:

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));

gpuArray[12] = 34.56;    //  <-- ERROR
```

The issue is that this code runs on CPU, which has no direct access to GPU memory.

**Access by Copying.** To work with GPU memory we should use `cudaMemcpy` from `cuda_xxx.h` that
allows to transfer data from CPU to GPU and back, like this:

```cpp
size_t n = 1024;
double gpuArray = cudaMalloc(n * sizeof(double));
double cpuArray = std::malloc(n * sizeof(double));

cpuArray[12] = 34.56;    //  <-- OK

//  CPU -> GPU
cudaMemcpy(cpuArray, gpuArray, n, HostToDevice);
```

The `HostToDevice` flag indicates that we copy data from CPU (the host) to GPU (the device).
Guess what another `DeviceToHost` flag is used for.

Overall, the common workflow is:

1. Initialize input data in CPU memory.
2. Transfer input data to GPU with `cudaMemcpy` and `HostToDevice`.
3. Run GPU kernels to calculate the output. (See below about GPU kernels.)
4. Transfer output data to CPU with `cudaMemcpy` and `DeviceToHost`.

**No STL in CUDA.** Like a modern C++ programmer you're likely feeling that using `cudaMalloc` and
`cudaMemcpy` is too low-level especially when you need to manually manage memory with `cudaMalloc`
and`cudaFree` without relying on smart pointers. Unfortunately you can't use STL in CUDA (there is
Thrust, but I did not try it)...

**Abstraction.** In my case, I wrote a `Vector2d` class to abstract low-level operations

- template + constexpr

```c++
template<u64 Options>
class Vector2d
{
private:
  f64* m_buf;

public:
  __host__
  Vector2d(u64 nCol, u64 nRow)
  {
    if constexpr (traits<Vector2d>::isCpu)
      m_buf = malloc(nCol * nRow * sizeof(f64));
    if constexpr (traits<Vector2d>::isGpu)
      cudaMalloc(&m_buf, nCol * nRow * sizeof(f64));
    else
      static_assert("Unknown device")
  }
}
```

See how `traits<Vector2d>` is done in ... on GitHub.

```c++
template<typename RhsVector2d>
__host__
Vector2d&
    operator=(const RhsVector2d& src)
{
  // CPU -> GPU
  if constexpr (traits<Vector2d>::isGpu &&
                traits<RhsVector2d>::isCpu)
    cudaMemcpy(m_buf, src.buf(), src.size(), cudaMemcpyHostToDevice);

  // GPU -> CPU
  else if constexpr (traits<Vector2d>::isGpu &&
                     traits<RhsVector2d>::isCpu)
    cudaMemcpy(m_buf, src.buf(), src.size(), cudaMemcpyDeviceToHost);

  // GPU -> GPU
  else if constexpr (traits<Vector2d>::isGpu &&
                     traits<RhsVector2d>::isGpu)
    cudaMemcpy(m_buf, src.buf(), src.size(), cudaMemcpyDeviceToDevice);

  // CPU -> CPU
  else
    std::memcpy(m_buf, src.buf(), src.sizeInBytes());

  return *this;
}
```

## GPU Code

**CUDA Kernels.** Now let's talk about CUDA kernels, which are C++ functions that run on GPU with a
direct access to GPU memory. A special feature of the kernel is that GPU runs it in parallel
on thousands of GPU cores. GPU cores are not as fast as CPU cores, but running thousands of them in
parallel feels very fast.

Source code for CUDA kernels is located in `.cu` files and isn't much different from regular C++
code:

```cpp
/// kernel.cu
__global__
void cudaKernel() {
}

/// main.cpp
```

which
must be compiled by a proprietory [Nvidia CUDA
Compiler](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler) (NVCC). In principle, you can compile
your whole codebase with NVCC, however it still lacks some modern features from the C++ Standard.

**Modern C++.** In my case, I decided to limit usage of NVCC to minimum and use it to compile
only GPU-related code. For the remaining code I keep using a more modern compilers, like GCC or
MSVC. This is possible to setup with CMake, see ... for how its done.

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