from max.gpu.host import DeviceContext
from max.gpu import thread_idx, lane_id
from max.gpu.sync import barrier, syncwarp
from max.gpu.primitives import warp, block
from layout import TileTensor, stack_allocation
from layout.tile_layout import row_major


comptime SIZE = 64
comptime DTYPE = DType.float32
comptime IN_LAYOUT = row_major[SIZE]()
comptime OUT_LAYOUT = row_major[1]()



def gpu_sync(
    input_tensor: TileTensor[DTYPE, type_of(IN_LAYOUT), MutAnyOrigin],
    output_tensor: TileTensor[DTYPE, type_of(OUT_LAYOUT), MutAnyOrigin],
):
    var tid = thread_idx.x
    var lane = lane_id()

    comptime smem_layout = row_major[SIZE]()
    var shared_buf = stack_allocation[DTYPE, address_space=.SHARED](smem_layout)

    shared_buf[tid] = input_tensor[tid]

    barrier()

    if lane < 16:
        shared_buf[tid] += 10.0
    else:
        shared_buf[tid] += 20.0

    syncwarp()

    var my_val = shared_buf[tid]

    var warp_sum = warp.sum(my_val)
    var block_sum = block.sum[block_size=SIZE](my_val)

    if tid == 0:
        output_tensor[0] = block_sum


def main() raises:

    var ctx = DeviceContext()

    var host_in = ctx.enqueue_create_host_buffer[DTYPE](SIZE)
    var host_out = ctx.enqueue_create_host_buffer[DTYPE](1)
    var dev_in = ctx.enqueue_create_buffer[DTYPE](SIZE)
    var dev_out = ctx.enqueue_create_buffer[DTYPE](1)


    ctx.synchronize()

    for i in range(SIZE):
        host_in[i] = Float32(i)



    ctx.enqueue_copy(dev_in, host_in)

    var input_tensor = TileTensor(dev_in, IN_LAYOUT)
    var output_tensor = TileTensor(dev_out, OUT_LAYOUT)

    ctx.enqueue_function[gpu_sync](
        input_tensor,
        output_tensor,
        grid_dim=(1,),
        block_dim=(SIZE,),
    )

    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()

    # print(host_out)

    print(host_out[0])