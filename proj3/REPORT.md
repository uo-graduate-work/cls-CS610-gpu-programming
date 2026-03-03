# Project 3 Report

## 1) Final Validation Status

All tasks are validated against the provided references in `answers/` on the local environment.

- **Task 01**: no reference output file provided; internal correctness checks pass:
  - CPU vs GPU check on 256x256: **PASSED**
  - prefetch vs non-prefetch full-output equality: **PASSED**
  - large 10k sampled verification (100 points): **PASSED**
- **Task 02**: `output.ppm` is now **byte-for-byte identical** to `answers/task02_correct_output.ppm`.
- **Task 03**: outputs for radii 1, 2, 4, 8 are each **byte-for-byte identical** to their matching files in `answers/`.

## 2) Performance Results

### Task 01: Batched Matrix Multiplication
Workload: `K=10`, `m=k=n=10000`.

| GPU | Without Prefetch | With Prefetch | Speedup |
| :-- | --: | --: | --: |
| RTX 4090 (local) | 6136.63 ms | 5813.86 ms | 1.06x |
| A100 (cloud) | 8818.04 ms | 8564.88 ms | 1.03x |

### Task 02: Grayscale (AoS vs SoA)
Workload: `2048x2048` image.

| GPU | AoS | SoA | AoS->SoA Speedup |
| :-- | --: | --: | --: |
| RTX 4090 (local) | 0.056320 ms | 0.082688 ms | 0.68x |
| A100 (cloud) | 0.027360 ms | 0.045984 ms | 0.59x |

### Task 03: Blur (Basic vs Tiled), Radius 8
Workload: `2048x2048` image, `17x17` filter.

| GPU | Basic | Tiled | Speedup |
| :-- | --: | --: | --: |
| RTX 4090 (local) | 0.769024 ms | 0.757760 ms | 1.01x |
| A100 (cloud) | 2.059712 ms | 1.959840 ms | 1.05x |

## 3) Notes on Final Fixes

- `task02.cu`: adjusted grayscale arithmetic precision so output exactly matches the provided reference bytes.
- `task03.cu`: enabled per-radius output mode and matched output header formatting used by reference files.
- `task01.cu`: robust verification and pipelined stream execution retained.

## 4) Reproduction Commands

```bash
cp og_input.ppm input.ppm

nvcc -o task01 task01.cu -lm
nvcc -o task02 task02.cu
nvcc -o task03 task03.cu

./task01

./task02
cmp output.ppm answers/task02_correct_output.ppm

./task03 --mode basic --radius 1 --output out_r1.ppm
./task03 --mode basic --radius 2 --output out_r2.ppm
./task03 --mode basic --radius 4 --output out_r4.ppm
./task03 --mode basic --radius 8 --output out_r8.ppm

cmp out_r1.ppm answers/task03_correct_output_radius_1.ppm
cmp out_r2.ppm answers/task03_correct_output_radius_2.ppm
cmp out_r4.ppm answers/task03_correct_output_radius_4.ppm
cmp out_r8.ppm answers/task03_correct_output_radius_8.ppm
```
