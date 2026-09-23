from pathlib import Path
import numpy as np

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def main():
    mojo_kernels_dir = Path(__file__).parent / "kernels"

    m, k, n = 4, 6, 5
    dtype = DType.float32

    device = CPU() if accelerator_count() == 0 else Accelerator()
    device_ref = DeviceRef.from_device(device)
    print("Device:", device)

    with Graph(
        "matmul_graph",
        input_types=[TensorType(dtype, shape=[m,k], device=device_ref),
                     TensorType(dtype, shape=[k,n], device=device_ref)],
        custom_extensions=[mojo_kernels_dir],
    ) as graph:
        A,B = graph.inputs
        C = ops.custom(
            name="matmul_tiled",
            device=device_ref,
            values=[A,B],
            out_types=[TensorType(dtype=A.tensor.dtype, shape=[m,n], device=device_ref)],
        )[0].tensor
        graph.output(C)

    # print("Here")

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    # print("Here2")

    matA = np.arange(m*k, dtype=np.float32).reshape(m, k)
    a_buf = Buffer.from_numpy(matA).to(device)

    matB = np.arange(k*n, dtype=np.float32).reshape(k, n)
    b_buf = Buffer.from_numpy(matB).to(device)


    result = model.execute(a_buf,b_buf)[0]
    # print("Here3")

    result_np = result.to(CPU()).to_numpy()
    print("A: ", matA)
    print("B:",matB)
    print("Output:", result_np)
    print("Expected:", matA @ matB)
    print("Match:", np.allclose(result_np, matA @ matB))


if __name__ == "__main__":
    main()