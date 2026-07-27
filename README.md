# CS 610 — GPU Programming

Coursework archive for CS 610 (GPU Programming): six CUDA coding assignments covering the
progression from basic kernel launches through memory-hierarchy optimization, warp-level
primitives, stream concurrency, and parallel algorithm patterns.

Each project directory contains the CUDA sources, the original assignment PDF, a written
report with measured results, and captured run output.

## Toolchain and hardware

- **Compiler:** `nvcc` (CUDA Toolkit). Project 4's report records `-O2 -arch=sm_80`; the
  other reports record a plain `nvcc` invocation.
- **GPUs used for the reported measurements:**
  - NVIDIA A100 80GB PCIe (Ampere, CC 8.0, 108 SMs, HBM2e) — cloud
  - NVIDIA RTX 4090 (Ada Lovelace, CC 8.9, 128 SMs, GDDR6X) — local
- Projects 3, 4, and 5 report both GPUs side by side; projects 2 and 6 report the A100 only.
- Project 2 links OpenMP (`-Xcompiler -fopenmp`) for its parallel CPU baseline.

Nothing here is built by a makefile — every task is a single translation unit compiled
directly with `nvcc`. Build commands are listed per project below.

## Repository layout

```
proj1/  Affine cipher decryption, vector add, matrix add
proj2/  Naive vs. tiled shared-memory matrix multiply
proj3/  Batched matmul with stream prefetch; AoS vs SoA; image blur
proj4/  Four reduction strategies for a max-value search
proj5/  Static device arrays + bandwidth; sphere ray tracer
proj6/  Histogram, Kogge-Stone scan, tiled merge
```

---

## Project 1 — CUDA fundamentals

First exposure to kernel launches, thread indexing, and host/device memory transfer.

| File | Topic |
| :-- | :-- |
| `task01.cu` | Affine cipher decryption; single-block and multi-block indexing variants |
| `task02.cu` | Vector addition, `N = 2050` (deliberately not a multiple of block size, to force a bounds check) |
| `task03.cu` | Matrix addition with 2D grid/block indexing, `2048 x 1024` |

`task01` reads `encrypted.bin` from the working directory and decrypts with the affine
inverse `A_INV = 111`, `B = 27`, `M = 128`. `decrypted_correct.txt` is the expected plaintext.

```bash
cd proj1
nvcc -o task01 task01.cu && ./task01     # must run from proj1/ to find encrypted.bin
nvcc -o task02 task02.cu && ./task02
nvcc -o task03 task03.cu && ./task03
```

Tasks 2 and 3 validate the GPU result against a host implementation and print an error count.

---

## Project 2 — Matrix multiplication and the memory hierarchy

Compares a naive global-memory matmul against a tiled shared-memory version at
`m = k = n = 10,000` (~2 x 10^12 integer ops), sweeping block size.

- `task1.cu` — naive; block dimensions passed at **runtime** via argv
- `task2.cu` — tiled shared-memory; `BLOCK_SIZE` is a **compile-time** constant, so each
  block size needs its own binary

Both sources use OpenMP for the CPU baseline, so `-Xcompiler -fopenmp` is required to link.

```bash
cd proj2
nvcc -Xcompiler -fopenmp -o task1 task1.cu
./task1 16 16                                     # block x, block y

nvcc -Xcompiler -fopenmp -DBLOCK_SIZE=16 -o task2 task2.cu
./task2
```

Measured on the A100 (kernel-only time; captured output in `results/`):

| Block size | Task 1 naive | Task 1 GFLOPS | Task 2 tiled | Task 2 GFLOPS |
| --: | --: | --: | --: | --: |
| 8  | 1517.42 ms | 1318.03 | 687.22 ms | 2910.28 |
| 16 |  837.82 ms | 2387.14 | 453.06 ms | 4414.44 |
| 32 |  668.56 ms | 2991.50 | 416.13 ms | 4806.17 |

Tiling gives roughly a 1.6x kernel speedup at every block size, and larger blocks help both
versions. Full write-up in `report.tex` / `report.pdf`.

---

## Project 3 — Streams, data layout, and stencils

| File | Topic |
| :-- | :-- |
| `task01.cu` | Batched matrix multiply, `K = 10` batches of `10000^3`, with and without a CUDA-stream prefetch pipeline that overlaps H2D/D2H copies with compute |
| `task02.cu` | RGB-to-grayscale, comparing array-of-structs against struct-of-arrays layout |
| `task03.cu` | 2D box blur, naive vs. shared-memory tiled, at radii 1/2/4/8 |

`validate_outputs.py` diffs generated images against the references in `answers/`.
`assignment.txt` holds the task specification (there is no assignment PDF for this one).

```bash
cd proj3
cp og_input.ppm input.ppm          # tasks read input.ppm; og_input.ppm is the pristine copy

nvcc -o task01 task01.cu -lm && ./task01
nvcc -o task02 task02.cu && ./task02
cmp output.ppm answers/task02_correct_output.ppm

nvcc -o task03 task03.cu
for r in 1 2 4 8; do
  ./task03 --mode basic --radius $r --output out_r$r.ppm
  cmp out_r$r.ppm answers/task03_correct_output_radius_$r.ppm
done
```

Results (see `REPORT.md`, `report.pdf`) — all outputs byte-for-byte identical to the references:

| Task | RTX 4090 | A100 |
| :-- | :-- | :-- |
| Batched matmul, no prefetch → prefetch | 6136.63 → 5813.86 ms (1.06x) | 8818.04 → 8564.88 ms (1.03x) |
| Grayscale, AoS → SoA | 0.0563 → 0.0827 ms (0.68x) | 0.0274 → 0.0460 ms (0.59x) |
| Blur r=8, basic → tiled | 0.7690 → 0.7578 ms (1.01x) | 2.0597 → 1.9598 ms (1.05x) |

Two of these are negative results worth noting: SoA was *slower* than AoS here, and tiling
bought almost nothing for the blur. The prefetch pipeline's modest gain reflects a kernel
that is compute-bound relative to its transfers.

---

## Project 4 — Reduction strategies

Finds the maximum student mark (and the owning student ID) across a record set, implemented
four ways to contrast reduction techniques: global atomics, recursive multi-pass kernels,
shared-memory reduction, and warp shuffle intrinsics.

Two build variants differ only in dataset size, set by `NUM_RECORDS` in the paired header:

- `task.cu` + `kernels.cuh` → reads `Student.dat`, 2048 records
- `task5.cu` + `task5_kernels.cuh` → reads `Student_large.dat`, 2^20 records

```bash
cd proj4
nvcc -O2 -arch=sm_80 -o task task.cu   && ./task
nvcc -O2 -arch=sm_80 -o task5 task5.cu && ./task5
```

A100 timings (`report.pdf`):

| Method | 2048 records | 2^20 records |
| :-- | --: | --: |
| Global atomics | 4.95 ms | ~208,953 ms |
| Recursive | 0.063 ms | 6.58 ms |
| Shared memory | 0.034 ms | 2.08 ms |
| Warp shuffle | 0.036 ms | 2.11 ms |

The atomics version collapses at scale — roughly 100,000x slower than shared memory on the
large set — from contention on a single global address. All four find the correct maximum;
on the large dataset atomics returns a different student ID because ties break
non-deterministically. Curiously the RTX 4090 beat the A100 on every method despite lower
memory bandwidth.

---

## Project 5 — Bandwidth and ray tracing

| File | Topic |
| :-- | :-- |
| `task01.cu` | Vector add over `N = 4,194,304` using statically declared `__device__` arrays; measures achieved vs. theoretical bandwidth with CUDA events |
| `task02.cu` | Ray tracer casting one ray per pixel over a `2048 x 2048` image, sweeping sphere counts 16 → 2048 by powers of two |

```bash
cd proj5
nvcc -o task01 task01.cu && ./task01
nvcc -o task02 task02.cu && ./task02     # writes output_<count>.ppm for each sphere count
```

Ray tracer kernel time (ms):

| Spheres | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| RTX 4090 | 0.077 | 0.105 | 0.187 | 0.345 | 0.664 | 1.305 | 2.573 | 5.124 |
| A100 | 0.104 | 0.148 | 0.268 | 0.486 | 0.942 | 1.845 | 3.654 | 7.272 |

Scaling is cleanly linear in sphere count, as expected when every pixel tests every sphere.
The vector-add measurement reports 2409 GB/s on the 4090, above its theoretical DRAM
bandwidth — an artifact of timing granularity and cache effects on a kernel that runs for
only 19 microseconds, discussed in `performance_report.pdf`.

---

## Project 6 — Parallel algorithm patterns

| File | Topic | A100 time | Status |
| :-- | :-- | --: | :-- |
| `task01.cu` | Histogram with per-block privatization in shared memory | 0.213 ms | PASSED |
| `task02.cu` | Kogge-Stone inclusive scan with block-offset propagation | 21.26 ms | PASSED |
| `task03.cu` | Tiled merge using co-rank search | ~0.009 ms | PASSED |

```bash
cd proj6
for t in task01 task02 task03; do nvcc -o $t $t.cu && ./$t; done
```

Captured output is in `task0*.out`. The merge implementation is verified against equal-sized
inputs, single-element inputs on either side, and empty inputs on either side. Write-up in
`report.md` / `report.pdf`.

---

## Notes on this archive

**Generated files are not committed.** Compiled binaries and the ~430 MB of generated `.ppm`
images are excluded via `.gitignore`; rebuild and regenerate them with the commands above.

Two categories of image *are* committed, because they cannot be regenerated from this
repository:

- `proj3/og_input.ppm` — the source image every proj3 task reads
- `proj3/answers/*.ppm` — instructor reference outputs, needed for the byte-exact `cmp` checks

Report PDFs are committed alongside their LaTeX sources; the `.aux`/`.log` build artifacts
are not. Assignment PDFs are retained for context, except for project 3, whose specification
is `proj3/assignment.txt`.
