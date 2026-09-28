import time
from pathlib import Path

import numpy as np

from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
KERNELS = [
    ("GPU naive", "matmul_naive"),
    ("GPU tiled", "matmul_tiled"),
    ("GPU regtiled", "matmul_regtiled"),
]
# matmul_regtiled has no bounds checks, so every dim must be a multiple of BM/BN (64).
REGTILED_MULTIPLE = 64
COL_W = 26


def build_model(device, op_name, m, k, n):
    mojo_kernels_dir = Path(__file__).parent / "kernels"
    dtype = DType.float32
    device_ref = DeviceRef.from_device(device)

    with Graph(
        f"{op_name}_bench",
        input_types=[
            TensorType(dtype, shape=[m, k], device=device_ref),
            TensorType(dtype, shape=[k, n], device=device_ref),
        ],
        custom_extensions=[mojo_kernels_dir],
    ) as graph:
        a, b = graph.inputs
        c = ops.custom(
            name=op_name,
            device=device_ref,
            values=[a, b],
            out_types=[TensorType(dtype=dtype, shape=[m, n], device=device_ref)],
        )[0].tensor
        graph.output(c)

    session = InferenceSession(devices=[device])
    return session.load(graph)


def bench(device, op_name, m, k, n, warmup, iters):
    model = build_model(device, op_name, m, k, n)
    rng = np.random.default_rng(0)
    a_np = rng.uniform(size=(m, k)).astype(np.float32)
    b_np = rng.uniform(size=(k, n)).astype(np.float32)
    a_buf = Buffer.from_numpy(a_np).to(device)
    b_buf = Buffer.from_numpy(b_np).to(device)

    for _ in range(warmup):
        model.execute(a_buf, b_buf)
    device.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        model.execute(a_buf, b_buf)
    device.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters


def gpu_repeats(size):
    if size <= 512:
        return 5, 50
    if size <= 1024:
        return 3, 20
    if size <= 2048:
        return 2, 10
    return 1, 5


def format_cell(seconds, baseline):
    if seconds is None:
        return "n/a".rjust(COL_W)
    return f"{seconds * 1e3:9.4f} ms ({baseline / seconds:8.2f}x)".rjust(COL_W)


def main():
    if accelerator_count() == 0:
        print("No GPU detected, nothing to benchmark.")
        return
    gpu = Accelerator()

    print("Speedup is relative to GPU naive.")
    header = f"{'Size':>6} | " + " | ".join(name.rjust(COL_W) for name, _ in KERNELS)
    print(header)
    print("-" * len(header))

    for size in SIZES:
        warmup, iters = gpu_repeats(size)
        timings = []
        for _, op_name in KERNELS:
            if op_name == "matmul_regtiled" and size % REGTILED_MULTIPLE != 0:
                timings.append(None)
                continue
            timings.append(bench(gpu, op_name, size, size, size, warmup, iters))

        baseline = timings[0]
        cells = " | ".join(format_cell(t, baseline) for t in timings)
        print(f"{size:>6} | {cells}", flush=True)


if __name__ == "__main__":
    main()
