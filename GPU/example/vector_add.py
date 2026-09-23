import numpy as np
from pathlib import Path
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

def main():
    mojo_kernels_dir=Path(__file__).parent / "kernels"

    vector_width = 10
    dtype = DType.float32

    device=CPU() if accelerator_count()==0 else Accelerator()

    device_ref=DeviceRef.from_device(device)
    print(f"Device:{device}")


    with Graph(
        "vector_addition_pipeline",
        input_types=[
            TensorType(dtype,shape=[vector_width],device=device_ref),
            TensorType(dtype,shape=[vector_width],device=device_ref)
        ],
        custom_extensions=[mojo_kernels_dir]

    )as graph:
        lhs,rhs=graph.inputs
        output=ops.custom(
            name="vector-add",
            device=device_ref,
            values=[lhs,rhs],
            out_types=[
                TensorType(
                    dtype=lhs.tensor.dtype,
                    shape=lhs.tensor.shape,
                    device=device_ref
                )
            ]
        )[0].tensor
        graph.output(output)

    session = InferenceSession(
        devices=[device],
    )

    compiled = session.compile(graph)
    model = session.init(compiled)

    lhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)
    rhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)

    lhs_tensor = Buffer.from_numpy(lhs_values).to(device)
    rhs_tensor = Buffer.from_numpy(rhs_values).to(device)

    print("hi")

    result = model.execute(lhs_tensor, rhs_tensor)[0]

    assert isinstance(result, Buffer)
    result = result.to(CPU())

    print("Left-hand-side values:")
    print(lhs_values)
    print()

    print("Right-hand-side values:")
    print(rhs_values)
    print()

    print("Graph result:")
    print(result.to_numpy())
    print()

    print("Expected result:")
    print(lhs_values + rhs_values)


if __name__=="__main__":
    main()