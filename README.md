# MPL — Minimal MLP in C

A compact multi-layer perceptron for binary image classification, written in ~100 lines of C.

**Network:** 20×20 input → 20 hidden neurons (ReLU) → 1 output (Sigmoid + BCE)

**Task:** rectangle (class 0) vs circle (class 1) on a 20×20 grid

## Files

```
mpl.h     — macros, types, inline utilities
main.c    — layer ops, data generation, training
Makefile  — build & run
```

## Build

```sh
make          # compile & run
make gif      # generate training visualization
```

## Notes

- Weights initialized with He initialization
- BCE + Sigmoid output gradient simplified to `ŷ - y`
- Single-header dependency: just include `mpl.h`

  ∂L/∂z_H1  =  ∂L/∂z_out  ×  ∂z_out/∂z_H1
  └─┬──┘        └──┬───┘     └────┬────┘
   本层δ         上游δ          局部偏导数
