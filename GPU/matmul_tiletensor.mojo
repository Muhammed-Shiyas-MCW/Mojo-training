from std.math import ceildiv
from std.sys import has_accelerator
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu import thread_idx, block_idx
from layout import TileTensor, stack_allocation
from layout.tile_layout import row_major

comptime dtype = DType.float32

comptime M = 64  
comptime K = 64
comptime N = 64
comptime tile = 16
comptime layout_A = row_major[M, K]()
comptime layout_B = row_major[K, N]()
comptime layout_C = row_major[M, N]()

def matmul_kernel(
    A: TileTensor[dtype, type_of(layout_A), MutAnyOrigin],
    B: TileTensor[dtype, type_of(layout_B), MutAnyOrigin],
    C: TileTensor[dtype, type_of(layout_C), MutAnyOrigin],
):

    var tx = thread_idx.x
    var ty = thread_idx.y

    var row = block_idx.y * tile + ty
    var col = block_idx.x * tile + tx

    var tile_a = stack_allocation[dtype, AddressSpace.SHARED](
        row_major[tile, tile]()
    )
    var tile_b = stack_allocation[dtype, AddressSpace.SHARED](
        row_major[tile, tile]()
    )
    




    var acc: C.ElementType = 0.0
    
    comptime for k_start in range(0, K, tile):
        tile_a[ty,tx]=A[row, k_start + tx]

        tile_b[ty,tx] = B[k_start + ty, col]

        # barrier()




        comptime for k in range(tile):
            acc += tile_a[ty, k] * tile_b[k, tx]

        barrier()

    
    C[row, col] = acc


def main() raises:
    comptime if not has_accelerator():
        print("No GPU found")
    else:

        var ctx = DeviceContext()

        var buf_A = ctx.enqueue_create_buffer[dtype](M * K)
        var buf_B = ctx.enqueue_create_buffer[dtype](K * N)
        var buf_C = ctx.enqueue_create_buffer[dtype](M * N)
        var host_A = ctx.enqueue_create_host_buffer[dtype](M * K)
        var host_B = ctx.enqueue_create_host_buffer[dtype](K * N)
        var host_C = ctx.enqueue_create_host_buffer[dtype](M * N)
        ctx.synchronize()

        
        var view_A = TileTensor(host_A, layout_A)
        var view_B = TileTensor(host_B, layout_B)

        for i in range(M):
            for j in range(K):
                view_A[i, j] = Float32(i + 1)

        for i in range(K):
            for j in range(N):
                view_B[i, j] = Float32(j + 1)

        ctx.enqueue_copy(dst_buf=buf_A, src_buf=host_A)
        ctx.enqueue_copy(dst_buf=buf_B, src_buf=host_B)

        var dev_A = TileTensor(buf_A, layout_A)
        var dev_B = TileTensor(buf_B, layout_B)
        var dev_C = TileTensor(buf_C, layout_C)

        
        comptime col_blocks = ceildiv(N, tile)
        comptime row_blocks = ceildiv(M, tile)

        ctx.enqueue_function[matmul_kernel](
            dev_A, dev_B, dev_C,
            grid_dim=(col_blocks, row_blocks),
            block_dim=(tile, tile),
        )

        
        
        ctx.enqueue_copy(dst_buf=host_C, src_buf=buf_C)
        ctx.synchronize()

        var result = TileTensor(host_C, layout_C)

        


        print(result)
