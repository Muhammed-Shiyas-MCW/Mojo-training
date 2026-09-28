# GPU Matmul Benchmark: Naive vs Tiled vs Register-Tiled

This compares three Mojo GPU matmul kernels on square `float32` matrices from 64×64 to 4096×4096.
Speedup is measured against the naive GPU kernel.

- Kernels: [kernels/matmul.mojo](kernels/matmul.mojo)
- Benchmark: [benchmark_matmul.py](benchmark_matmul.py)

## Kernels

| Kernel | Idea |
|---|---|
| GPU naive | Each thread computes one output, reading A and B straight from global memory |
| GPU tiled | Each block loads 16×16 tiles of A and B into shared memory and reuses them |
| GPU regtiled | Each block computes a 64×64 tile, and each thread keeps a 4×4 block of results in registers |

## How to run

From the repo root:

```bash
pixi run python GPU/example/benchmark_matmul.py
```

## Results

GPU: NVIDIA GeForce RTX 2080 Ti

| N | Naive (ms) | Tiled (ms) | Tiled speedup | Regtiled (ms) | Regtiled speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.0796 | 0.0783 | 1.02x | 0.0807 | 0.99x |
| 128 | 0.0826 | 0.0798 | 1.04x | 0.0844 | 0.98x |
| 256 | 0.1000 | 0.0920 | 1.09x | 0.0911 | 1.10x |
| 512 | 0.2854 | 0.2556 | 1.12x | 0.1070 | 2.67x |
| 1024 | 2.0876 | 1.6204 | 1.29x | 0.4395 | 4.75x |
| 2048 | 17.4623 | 12.7598 | 1.37x | 2.5229 | 6.92x |
| 4096 | 140.2055 | 101.4923 | 1.38x | 19.5233 | 7.18x |

## Observations

- **Small sizes (64–256):** all three run at about the same speed. The matrices are too small to keep the GPU busy.
- **Tiled:** only a small gain, at most 1.38x.
- **Regtiled:** the clear winner. It is 2.7x faster at 512 and 7.2x faster at 4096, because each thread
  computes 16 outputs and reuses data from registers.
