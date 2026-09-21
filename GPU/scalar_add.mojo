from std.sys import exit, has_accelerator
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx

def scalar_add(vector: Pointer[Float32, MutAnyOrigin], size: Int32, scalar: Float32):
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < Int(size):
        vector[unsafe_offset=idx] += scalar

def main() raises:

    comptime if not has_accelerator():
        print("No GPUs detected")
        exit(0)
    else:


        var num=DeviceContext.number_of_devices()

        print(num)

        # var ctx = DeviceContext(device_id=0,api="hip")

        var ctx=DeviceContext()

        # This ctx manages memory allocation on GPU , Data Transfers , Launching Kernels , Synchronization  

        var data:List[Float32]=[1.0, 2.0, 3.0, 4.0, 5.0]
        var size = len(data)

        var host_buffer = ctx.enqueue_create_host_buffer[DType.float32](size)

        ctx.synchronize()

        for i in range(size):
            host_buffer[i] = data[i]


        print(host_buffer)

        var device_buffer = ctx.enqueue_create_buffer[DType.float32](size)


        ctx.enqueue_copy(src_buf=host_buffer,dst_buf=device_buffer)

        var kernel = ctx.compile_function[scalar_add]()

        ctx.enqueue_function(kernel, device_buffer, Int32(size), Float32(20.0),grid_dim=1, block_dim=size)

        # ctx.enqueue_function[scalar_add](device_buffer, Int32(size), Float32(20.0),grid_dim=1, block_dim=n)


        ctx.enqueue_copy(src_buf=device_buffer,dst_buf=host_buffer)

        ctx.synchronize()

        print(device_buffer)









