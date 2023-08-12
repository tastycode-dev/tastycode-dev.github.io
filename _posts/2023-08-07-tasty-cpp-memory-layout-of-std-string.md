---
author: Oleksandr Gituliar
date: 2023-08-07
layout: post
title: "Tasty C++ – Memory Layout of std::string"
description: "Learn about memory layout of std::string in c++ standard libraries: MSVC STL, libstdc++, libc++."
---

For a professional C++ developer, it's important to understand memory organization of the data
structures, especially when it comes to the containers from the C++ Standard Library. In this post
of Tasty C++ series we'll look inside of `std::string`, so that you can more effectively work with
C++ strings and take advantage (or avoid pitfalls) of the C++ Standard Library you are using.

In C++ Standard Library, `std::string` is one of the three [contiguous
containers](https://en.cppreference.com/w/cpp/named_req/ContiguousContainer) (the other two are
`std::array` and `std::vector`). This means that a sequence of characters is stored in a
_contiguous_ area of the memory and an individual character can be efficiently accessed by its index
at O(1) time. The C++ Standard imposes more requirements on the complexity of string operations,
which we will briefly focus on later this post.

What is important to remember is that the C++ Standard doesn't impose exact implementation of
`std::string`, nor does it specify how much memory should `std::string` allocate. In practice, as
we'll see, the most popular implementations of the C++ Standard Library allocate 24 or 32 bytes for
the same `std::string` object. In addition, the techniques to organize internal memory layout are
also different, which results in a tradeoff between optimal memory or CPU utilization.

## Long Strings

When people start using `std::string`, it is associated with three data members living somewhere in
memory:

- **Buffer** – the buffer where string characters are stored.
- **Size** – the current number of characters in the string.
- **Capacity** – the max number of character the buffer can fit.

Talking C++ language, this picture could be expressed as the following class:

```cpp
class TastyString {
    char *    m_buffer;     //  string characters
    size_t    m_size;       //  number of characters
    size_t    m_capacity;   //  m_buffer size
}
```

`TastyString` occupies 24 bytes, which is only 3x more than **fundamental types** such as `void *`,
`size_t`, or `double`. This means that `TastyString` is cheap to copy or pass by value as a function
argument. What is not cheap, however, is (1) copying the buffer, especially when the string is long,
and (2) allocating a buffer for a new, even small, copy of the string.

Let's look how it actually looks like with `std::string`. In the _most popular implementations_ of
the C++ Standard Library the size of `std::string` object is the following:

| C++ Standard Library | Size of std::string() |
| -------------------- | --------------------- |
| MSVC STL             | 32 bytes              |
| GCC libstdc++        | 32 bytes              |
| LLVM libc++          | 24 bytes              |

To our surprise, only **LLVM** allocates expected **24 bytes** for `std::string`. The other two,
**MSVC** and **GCC**, allocate **32 bytes** for the same string. (Numbers in the table are for -O3
optimization. Note that MSVC allocates 40 bytes for `std::string` in the _debug mode_.)

Is this information optimal to represent a string ?

In fact, the _capacity_ is not required. We can use _size_ and _buffer_ only, but when the string
grows, a new buffer should be allocated on the heap (because we can't tell how many extra characters
the current buffer can fit). Since heap allocation is slow, such allocations are avoided by tracking
the buffer capacity.

The _buffer_ is a [null terminated string](https://en.wikipedia.org/wiki/Null-terminated_string)
well known in C.

## Small Strings

Let's get some intuition about why various implementation allocate different amount of memory for
the same object.

The members of `TastyString` contain only auxiliary data, while the actual data is stored in the
buffer. It seems inefficient to reserve 24 or 32 bytes for auxiliary data (and allocate extra
dynamic memory) when the actual data is smaller than that, isn't it?

**Small String Optimization.** This optimization, also known as SSO, is to keep the actual data in
the auxiliary region (when it is small enough). This way `std::string` objects become cheap to copy
and construct (almost like fundamental types ...) as we don't allocate dynamic memory. This
technique is popular among various implementations, however is not a part of the C++ Standard.

It makes sense now why some implementations increase the auxiliary region to 32 bytes --- to store
bigger strings in the auxiliary region before switching into the regular mode which dynamically
allocated buffer.

**How big are small strings?** Let's see how many characters the auxiliary region fits in practice.
This is what `std::string().capacity()` will tell us:

| C++ Standard Library | Small String Capacity |
| -------------------- | --------------------- |
| MSVC STL             | 15 chars              |
| GCC libstdc++        | 15 chars              |
| LLVM libc++          | 22 chars              |

What a surprise! LLVM with its 24 bytes for `std::string` fits more characters than MSVC or GCC with
their 32 bytes. (In fact, it's possible to fully utilize the auxiliary region, so that n-byte area
fits n-1 chars and `'\0'`.)

**How fast are small strings?** As with many things in programming, there is a tradeoff between
memory utilization and code complexity. In other words, the more characters we want to fit into the
auxiliary memory, the more complex logic we should introduce. This results not only in more assembly
operations, but also into branching that is not good for CPU pipelines to evaluate.

To illustrate this point, let's see what the most commonly used `size()` method compiles to in
various standard libraries:

**GCC stdlibc++**. The function directly copies `m_size` field into the output register (see
<https://godbolt.org/z/7nYe9rWdE>):

| Example                                                  | GCC libstdc++                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| ![string size C++ code](/static/img/string-size-src.png) | ![string size GCC assembler](/static/img/string-size-gcc.png) |

**LLVM libc++**. The function at first checks if the string is short and then calculates its size
(see <https://godbolt.org/z/xM349cG5P>).

| Example                                                  | LLVM libc++                                                     |
| -------------------------------------------------------- | --------------------------------------------------------------- |
| ![string size C++ code](/static/img/string-size-src.png) | ![string size LLVM assembler](/static/img/string-size-llvm.png) |

LLVM code is more complex for other string methods too. It's hard to say how badly this impacts the
overall performance, so experiment with various implementations and benchmark your particular use
case.

## Memory Allocation Policy

Finally, let's come back to long strings and see how `m_buffer` grows when it's time to allocate
more memory. Some
[comments](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/basic_string.tcc#L142)
in the GCC source code, refer to _exponential growth policy_. It's not clear if this is an internal GCC
decision or part of the C++ Standard. In any case, all three implementations use exponential growth,
so that **MSVC** has **1.5x factor** growth, while **GCC** and **LLVM** use **2x factor**.

The code below illustrates the growth algorithm in each implementation. The capacity examples show
how the capacity changes as the string grows in a loop one character at a time:

- **MSVC STL**

  ```cpp
  size_t newCapacity(size_t newSize, size_t oldCap) {
      return max(newSize, oldCap + oldCap / 2);
  }
  ```

  Capacity growth: 15, 31, 47, 70, 105, 157, 235, 352, 528, 792, 1'188, 1'782.

- **GCC libstdc++**

  ```cpp
  size_t newCapacity(size_t newSize, size_t oldCap) {
      return max(newSize + 1, 2 * oldCap);
  }
  ```

  Capacity growth: 15, 30, 60, 120, 240, 480, 960, 1'920, 3'840, 7'680, 15'360.

- **LLVM libc++**

  ```cpp
  size_t newCapacity(size_t newSize, size_t oldCap) {
      return max(newSize, 2 * oldCap) + 1;
  }
  ```

  Capacity growth: 22, 47, 95, 191, 383, 767, 1'535, 3'071, 6'143, 12'287.

## Summary

The actual implementation of `std::string` varies among the most popular implementations of the C++
Standard Library. The main difference is in the implementation of the Small String Optimization,
which is not explicitly specified by the C++ Standard. In the following table we list some of the
main differences:

| C++ Standard Library | String Size | Small String Capacity | Growth Factor |
| -------------------- | ----------- | --------------------- | ------------- |
| MSVC STL             | 32 bytes    | 15 chars              | 1.5x          |
| GCC libstdc++        | 32 bytes    | 15 chars              | 2x            |
| LLVM libc++          | 24 bytes    | 22 chars              | 2x            |

In the most cases the standard implementation will be okay for your task. In some cases, you will
need something slightly different. Rarely, a completely bespoke implementation of a string class
will be necessary.

Thanks for reading this far.

**Recommended Links:**

- [The strange details of std::string at Facebook](https://www.youtube.com/watch?v=kPR8h4-qZdk), CppCon 2016 talk
  by Nicholas Ormrod.
- [libc++'s implementation of std::string](https://joellaity.com/2020/01/31/string.html) by Joel
  Laity with the [discussion on HN](https://news.ycombinator.com/item?id=22198158).

TastyCode by Oleksandr Gituliar.
