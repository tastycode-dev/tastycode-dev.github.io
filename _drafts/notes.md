---
layout: post
author: Oleksandr Gituliar
title: "Notes"
---

**Statistical Arbitrage.** Exploit market inefficiencies (arbitrage) with statistical tools.

- Media Hub: blog post + YouTube video. Draw / animate memory, data structure operations. Memory as
  table, cache-line wide (64 bytes).

- How costly is risk managing options in various models (PnL explain accuracy) ?

- Cloud computing in finance is worth it (for batch jobs) ? Origin: start small and scale as needed.
  Memory management = slow serialization. What is core business and competence (hint, not software
  development or hardware ops).

## Tasty C++

- Inter-Process Communication using Shared Code

- `DateTime` + `Duration` classes

- Parser for Option Symbols (based on small user-defined strings ?)

- Inside of `std::string` / `std::map` / `std::shared_ptr` / `std::set` / `std::vector`. Explain
  internals with step-by-step examples of various operations (and differences among most common
  implementations, if any).

## Quant Corner

- What goes wrong when pricing Americans with Black-Scholes ? How to tell if the model works ? PnL
  explain.

- Finite Difference: Multi-center grid

- Local Volatility with FD PDE. Solve Dupire equation. Calibrate to American option prices. Switch
  to moneyness.

- Interview Quants. Start with CS colleagues.

- Study correlation.
