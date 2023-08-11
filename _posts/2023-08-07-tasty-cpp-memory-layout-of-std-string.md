---
author: Oleksandr Gituliar
date: 2023-08-07
layout: post
title: "Tasty C++ – Memory Layout of std::string"
description: "Learn about memory layout of std::string in c++ standard libraries: MSVC STL, libstdc++, libc++."
---

For a professional C++ developer, it's important to understand memory organization of the data
structures, especially when working with containers from the C++ Standard Library. In this post of
Tasty C++ series we'll look inside of `std::string`, so that you can more effectively work with C++
strings and take advantage (or avoid pitfalls) of the C++ Standard Library implementation you are
currently using.

In C++ Standard Library, `std::string` is one of the three [contiguous
containers](https://en.cppreference.com/w/cpp/named_req/ContiguousContainer) (the other two are
`std::array` and `std::vector`). This means that a sequence of characters is stored in a
_contiguous_ area of the memory and an individual character can be efficiently accessed by its index
at O(1) time. The C++ Standard imposes more requirements on the complexity of string operations,
which we will briefly focus on later this post.

What is important to remember is that the C++ Standard doesn't impose exact implementation of
`std::string`, nor does it specify how much memory should it allocate. In practice, as we'll see,
the most popular implementations of the C++ Standard Library allocate various amount of memory for
the same `std::string` object, so that the result of `sizeof(std::string)` might be `24` or `32`
bytes.

## Long Strings

Usually, to fully represent its internal state, `std::string` needs three pieces of information:

- **Size** – the current number of characters in the string.
- **Buffer** – the pointer to the memory buffer where characters are stored.
- **Capacity** – the max number of character the buffer can fit.

In fact, the _capacity_ is not required. We can use _size_ and _buffer_ only, but when the string
grows, a new buffer should be allocated on the heap (because we can't tell how many extra characters
the current buffer can fit). Since heap allocation is slow, such allocations are avoided by tracking
the buffer capacity.

Following this logic, we could implement a C++ string as:

```cpp
class TastyString {
    size_t    m_size;
    char *    m_buffer;
    size_t    m_capacity;
}
```

`MyString` occupies 24 bytes, which is only 3x more than **fundamental types** such as `void *`,
`size_t`, or `double`.

Let's see how things look in reality. In the _most popular implementations_ of the C++ Standard
Library the size of `std::string` object is the following:

| C++ Standard Library | Size of std::string() |
| -------------------- | --------------------- |
| MSVC STL             | 32 bytes              |
| GCC libstdc++        | 32 bytes              |
| LLVM libc++          | 24 bytes              |

To our surprise, only **LLVM** allocates expected **24 bytes** for `std::string`. The other two,
**MSVC** and **GCC**, allocate **32 bytes** for the same string. (For completeness, note that in the
_debug mode_ MSVC allocates 40 bytes for `std::string`.)

## Short Strings

Let's get some intuition about why various implementation allocate different amount of memory for
the same object. In fact, 24 or 32 bytes is already enough to fit a relatively big string, with no
need to allocate dynamic memory (and free it afterwards, which is costly as well). The trick, called
**Small String Optimization** (aka SSO), is to store string characters in the memory dedicated for
the size, capacity, and data pointer fields. Not sure this technique is part of the C++ Standard,
but for sure it's popular among various implementations.

Without going into much technicalities of SSO, let's mention two points worth to remember.

**How big are short strings?** It seems obvious that every implementation is free to extend
internal buffer for a small string far beyond required 24 bytes. This is why `std::string` in
MSVC and GCC is 32 bytes. However, the result of **`std::string().capacity()`** is:

| C++ Standard Library | Capacity of std::string() |
| -------------------- | ------------------------- |
| MSVC STL             | 15 chars                  |
| GCC libstdc++        | 15 chars                  |
| LLVM libc++          | 22 chars                  |

Again, LLVM version seems to beat MSVC and GCC, since for a smaller memory usage (24 bytes) it's
able to store longer strings (22 chars). (In fact, it's possible to fully utilize the memory and
fit 23 chars + `'\0'`.)

**How fast are short strings?** In this particular case, utilizing more space is not for
free. The more characters we pack into a string's memory area, the more CPU operations we have to
run. For LLVM, with its superior memory efficiency, even such a simple call as `size()` requires
to check if the string is short or long. This sort of conditions might slow down a calculation
pipeline.

A simple example of `size()` method clearly demonstrates this point. (BTW, this is one of
the most commonly used method of `std::string`.)

**GCC stdlibc++** code (see https://godbolt.org/z/7nYe9rWdE) directly copies string's size into
the output register:

| Example                                                  | GCC libstdc++                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| ![string size C++ code](/static/img/string-size-src.png) | ![string size GCC assembler](/static/img/string-size-gcc.png) |

**LLVM libc++** code (see https://godbolt.org/z/xM349cG5P) at first checks if the string is short
and then calculates its size.

| Example                                                  | LLVM libc++                                                     |
| -------------------------------------------------------- | --------------------------------------------------------------- |
| ![string size C++ code](/static/img/string-size-src.png) | ![string size LLVM assembler](/static/img/string-size-llvm.png) |

Eventually, it's hard to say which approach is more efficient. Now, that you know the difference,
the best advice here is to experiment with various implementations and benchmark your particular use
case.

## Memory Allocation Policy

Finally, let's see how `std::string` grows its internal buffer when it's time to allocate more
memory. Some [comments in the GCC
sources](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/basic_string.tcc#L142),
mentioned _amortized linear time requirement_ and _exponential growth policy_. Not clear if this is
internal GCC decision or part of the C++ Standard. In any case, all three implementations use
exponential growth, so that **MSVC** has **1.5x factor** growth, while **GCC** and **LLVM** use **2x
factor**. Below are some examples with more explicit (but simplified) code:

**MSVC STL**

```cpp
size_t newCapacity(size_t newSize, size_t oldCap) {
    return max(newSize, oldCap + oldCap / 2);
}
```

Example: 15, 31, 47, 70, 105, 157, 235, 352, 528, 792, 1'188, 1'782, 2'673, 4'009.

**GCC libstdc++**

```cpp
size_t newCapacity(size_t newSize, size_t oldCap) {
    return max(newSize + 1, 2 * oldCap);
}
```

Example: 15, 30, 60, 120, 240, 480, 960, 1'920, 3'840, 7'680, 15'360, 30'720.

**LLVM libc++**

```cpp
size_t newCapacity(size_t newSize, size_t oldCap) {
    return max(newSize, 2 * oldCap) + 1;
}
```

Example: 22, 47, 95, 191, 383, 767, 1'535, 3'071, 6'143, 12'287, 24'575, 49'151.

## Summary

Because the C++ Standard doesn't provide specific implementation details for `std::string`, there
are a couple of tradeoffs for the developer of the C++ Standard Library to consider:

- **Size**: _24 bytes_ (LLVM) vs _32 bytes_ (GCC, MSVC)
- **Capacity**: _15 chars + Simple Code_ (GCC, MSVC) vs _22 chars + Complex Code_ (LLVM)
- **Growth Policy**: Exponential with _1.5x factor_ (MSVC) vs _2x factor_ (GCC, LLVM)

In some cases they might be the nice features provided directly by the C++ Standard Library. In
other situations they might be the limitations, which require extra attention from your side or even
completely new implementation.

Hopefully, these details will make you a better programmer, help write more efficient C++ code, and
design better data structures.

**Recommended Links:**

- "libc++'s implementation of std::string" by Joel Laity:\\
  <https://joellaity.com/2020/01/31/string.html>\\
  Discussion on Hacker News:\\
  <https://news.ycombinator.com/item?id=22198158>
- CppCon 2016: “The strange details of std::string at Facebook" by Nicholas Ormrod:\\
  <https://www.youtube.com/watch?v=kPR8h4-qZdk>

TastyCode by Oleksandr Gituliar.
