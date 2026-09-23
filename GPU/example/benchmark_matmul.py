import time
from pathlib import Path

import numpy as np

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


SIZES = [64, 128, 256, 512, 1024]
COLUMNS = ["CPU naive", "GPU naive", "GPU tiled"]
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


def cpu_repeats(size):
    if size <= 128:
        return 3, 20
    if size <= 256:
        return 2, 10
    if size <= 512:
        return 1, 3
    return 1, 2


def format_cell(seconds, baseline):
    if seconds is None:
        return "n/a".rjust(COL_W)
    return f"{seconds * 1e3:9.4f} ms ({baseline / seconds:10.4f}x)"


def main():
    cpu = CPU()
    gpu = Accelerator() if accelerator_count() > 0 else None
    if gpu is None:
        print("No GPU detected, the GPU columns will be empty.")

    header = f"{'Size':>6} | " + " | ".join(name.rjust(COL_W) for name in COLUMNS)
    print(header)
    print("-" * len(header))

    for size in SIZES:
        warmup, iters = cpu_repeats(size)
        baseline = bench(cpu, "matmul_naive", size, size, size, warmup, iters)

        if gpu is None:
            timings = [baseline, None, None]
        else:
            timings = [
                baseline,
                bench(gpu, "matmul_naive", size, size, size, 5, 30),
                bench(gpu, "matmul_tiled", size, size, size, 5, 30),
            ]

        cells = " | ".join(format_cell(t, baseline) for t in timings)
        print(f"{size:>6} | {cells}", flush=True)


if __name__ == "__main__":
    main()
