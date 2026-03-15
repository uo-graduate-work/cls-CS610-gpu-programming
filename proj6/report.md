# CS 610 GPU Programming - Project 6 Report

## Overview
Implementation of three CUDA parallel algorithms: Histogram, Scan (Prefix Sum), and Merge Sort.

## Hardware
- GPU: NVIDIA A100

## Results

| Task | Algorithm | Execution Time | Status |
|------|-----------|---------------|--------|
| Task 1 | Parallel Histogram (privatization + shared memory) | 0.213 ms | PASSED |
| Task 2 | Parallel Scan (Kogge-Stone) | 21.26 ms | PASSED |
| Task 3 | Parallel Merge (tiled + co-ranks) | ~0.009 ms | PASSED |

## Task Details

**Task 1 - Histogram:** 2048 elements, 10 bins. Verified all bins match CPU reference implementation.

**Task 2 - Scan:** 2048 elements. Full scan verified with correct block-level offset propagation across 8 blocks.

**Task 3 - Merge:** All edge cases verified:
- Equal-sized arrays (20+20)
- Single element in A (1+50)
- Single element in B (50+1)
- Empty A (0+50)
- Empty B (50+0)

## Conclusion
All three implementations pass verification and execute correctly on the A100 GPU.
