from pathlib import Path
import numpy as np

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def main():
    mojo_kernels_dir = Path(__file__).parent / "kernels"

    rows = 4
    cols = 8
    dtype = DType.float32




    device = CPU() if accelerator_count() == 0 else Accelerator()
    device_ref = DeviceRef.from_device(device)
    print(device)




    graph = Graph(
        "add_one_pipeline",
        forward=lambda x: (
            ops.custom(
                name="add-one",
                device=device_ref,
                values=[x],
                out_types=[
                    TensorType(
                        dtype=x.dtype,
                        shape=x.tensor.shape,
                        device=device_ref,
                    )
                ],
            )[0].tensor
        ),
        input_types=[
            TensorType(
                dtype=dtype,
                shape=[rows, cols],
                device=device_ref,
            ),
        ],
        custom_extensions=[mojo_kernels_dir],
    )

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    x_numpy = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    x_device = Buffer.from_numpy(x_numpy).to(device)

    result_device = model.execute(x_device)[0]

    assert isinstance(result_device, Buffer)
    result_numpy = result_device.to(CPU()).to_numpy()

    print("\nInput Data:")
    print(x_numpy)
    print("\nOutput Data (Kernel Result):")
    print(result_numpy)
    print("\nVerification (output == input + 1):", np.allclose(result_numpy, x_numpy + 1.0))


if __name__ == "__main__":
    main()