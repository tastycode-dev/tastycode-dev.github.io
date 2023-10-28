---
layout: post
author: Oleksandr Gituliar
title: "Finite-Difference at 60fps"
---

_Intro._ Programs = Algorithms + Data Structures + Platform

In my previous post, Pricing Derivatives on a Budget, I presented benchmarks for pricing American
Options on CPU vs GPU. For pricing I used the finite-difference method, which is widely-used in the
production for various tasks. In this post, I will go into some details of implementing the
final-difference in GPU in C++ / CUDA. For people with no GPU experience this might seem like a
complicated topic. However, in fact once you learn a few basic concepts the whole picture become
clear and simple.

## GPU Memory

![CPU](/assets/img/cpu-gpu.png)

**GPU has its own RAM.** It's soldered directly on the GPU card, not upgradable, and is part of the
GPU specification.

**GPU has no access to CPU RAM**.

At a low-level, the workflow is as following:

1. Allocate data

   `cudaMalloc`, `cudaFree`

2. Copy data from CPU to GPU

   `cudaMemcpy`

3. Process it with GPU code

   `doWork(const f32* src, u64 n, f32* dst)`

4. Copy the result back to CPU.

**High-Level Abstraction.**

In my case, I wrote a `Vector2d` class to abstract low-level operations

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

Let's talk now about GPU code that works with GPU memory space. This code is also known as **CUDA
kernels** an is located in `.cu` files, which must be compiled by a proprietory
[Nvidia CUDA Compiler](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler) (NVCC). In principle, you
can compile your whole codebase with NVCC, however it still lacks some modern features from the C++
Standard.

In my case, I decided to limit usage of NVCC to minimum and use it to compile only GPU-related code.
For the remaining code I keep using a more modern compilers, like GCC or MSVC. This is possible to
setup with CMake, see ... for how its done.

The idea is to wrap CUDA kernels into a regular C++ functions and compile those with NVCC into a
separate library. Then I link this library with a main part of the code, compiled with MSVC. This
way I'm able to use the latest features of C++ and get rid of many compilation errors.

Finally, to make CPU and GPU version more convenient to use, let's create a base class with a
factory, that creates an appropriate implementation depending on the user config.

Abstract into Factory / Strategy

```c++
class PriceEngine
{
public:
  virtual Error
    init(const Config& config) = 0;

  virtual Error
    price(const vector<Option>& assets, vector<f64>& prices) = 0;
};
```

## Nvidia Libraries

Nvidia offers a big set of libraries tuned for high-performance computation. One of such libraries
is cuSparse for sparse matrix operations. The function we need from this lib is a tridiagonal matrix
solver. The version in cuSparse can solve many of such matrices in parallel. This is exactly what we
need.

## Final Word

When doing HPC you should know your platform:

- **Memory.** GPU memory is not flat, there are many flavors of GPU RAM shared at different levels
  with various access rights (for example, read-only usually used to store textures)

- **Code.** Should be optimized for a specific GPU generation.

- **Hardware.** Physical architecture...
