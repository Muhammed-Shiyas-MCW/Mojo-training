import extensibility
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu import block_dim, block_idx, thread_idx
from extensibility import InputTensor, OutputTensor, ManagedTensorSlice
from std.utils.index import IndexList


def cpu_matmul_naive(
    c: ManagedTensorSlice[mut=True, ...],
    a: ManagedTensorSlice[rank=c.rank, dtype=c.dtype, ...],
    b: ManagedTensorSlice[rank=c.rank, dtype=c.dtype, ...],
    ctx: DeviceContext,
):
    var m = c.dim_size(0)
    var n = c.dim_size(1)
    var k = a.dim_size(1)




    for i in range(m):
        for j in range(n):
            var acc: Scalar[c.dtype] = 0.0
            for p in range(k):
                var a_val = a.load[1](IndexList[2](i,p))
                var b_val = b.load[1](IndexList[2](p,j))
                acc += a_val*b_val
            c.store[1](IndexList[2](i,j), acc)






def gpu_matmul_naive(
    c: ManagedTensorSlice[mut=True, ...],
    a: ManagedTensorSlice[rank=c.rank, dtype=c.dtype, ...],
    b: ManagedTensorSlice[rank=c.rank, dtype=c.dtype, ...],
    ctx: DeviceContext,
) raises:




    comptime block_size = 16
    var m = c.dim_size(0)
    var n = c.dim_size(1)
    var k = a.dim_size(1)
    var gpu_ctx = ctx






    @__parameter
    def matmul_naive_gpu_kernel(m_dev:Int32,n_dev:Int32,k_dev: Int32):



        var row = Int(block_dim.y*block_idx.y+thread_idx.y)
        var col = Int(block_dim.x*block_idx.x+thread_idx.x)



        if row < Int(m_dev) and col < Int(n_dev):


            var acc: Scalar[c.dtype] = 0
            for p in range(Int(k_dev)):
                var a_val = a.load[1](IndexList[2](row,p))
                var b_val = b.load[1](IndexList[2](p,col))
                acc += a_val * b_val
            c.store[1](IndexList[2](row,col), acc)






    var grid_cols = ceildiv(n,block_size)
    var grid_rows = ceildiv(m,block_size)






    gpu_ctx.enqueue_function[matmul_naive_gpu_kernel](
        Int32(m),Int32(n),Int32(k),
        grid_dim=(grid_cols, grid_rows),
        block_dim=(block_size, block_size),
    )




@extensibility.register("matmul_naive")
struct MatMul_naive:

    @staticmethod
    def execute[target: StaticString](
        c: OutputTensor[rank=2, ...],
        a: InputTensor[dtype=c.dtype, rank=c.rank, ...],
        b: InputTensor[dtype=c.dtype, rank=c.rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "cpu":
            cpu_matmul_naive(c, a, b, ctx)
        elif target == "gpu":
            gpu_matmul_naive(c, a, b, ctx)
        else:
            raise Error("No target device")











#matmul tiled





comptime tile = 16


def gpu_matmul_tiled[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    CLayout: TensorLayout,
](
    c: TileTensor[dtype, CLayout, MutAnyOrigin],
    a: TileTensor[dtype, ALayout, MutAnyOrigin],
    b: TileTensor[dtype, BLayout, MutAnyOrigin],
    ctx: DeviceContext,
)raises:



    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())




    @__parameter
    def matmul_tiled_kernel(m_arg:Int32,n_arg:Int32,k_arg:Int32):
        comptime assert a.flat_rank == 2, "a must be 2D"
        comptime assert b.flat_rank == 2, "b must be 2D"
        comptime assert c.flat_rank == 2, "c must be 2D"

        var i = Int(m_arg)
        var j = Int(n_arg)
        var k = Int(k_arg)

        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)

        var row = Int(block_idx.y) * tile + local_row
        var col = Int(block_idx.x) * tile + local_col

        var a_tile = stack_allocation[
            dtype, address_space=AddressSpace.SHARED](row_major[tile,tile]())
        var b_tile = stack_allocation[
            dtype, address_space=AddressSpace.SHARED](row_major[tile,tile]())

        
        var total: c.ElementType = 0.0

        for k_start in range(0, k, tile):
            if row < i and k_start+local_col < k:
                a_tile[local_row,local_col] = a[row,k_start+local_col]
            else:
                a_tile[local_row,local_col] = 0

            if k_start + local_row<k and col<j:
                b_tile[local_row,local_col] = b[k_start + local_row, col]
            else:
                b_tile[local_row,local_col] = 0

            
            barrier()

            
            comptime for i in range(tile):
                total += a_tile[local_row,i] * b_tile[i,local_col]

            
            barrier()

        
        if row<i and col<j:
            c[row, col] = total







            

    var grid_rows = ceildiv(n, tile)
    var grid_cols = ceildiv(m, tile)

    ctx.enqueue_function[matmul_tiled_kernel](
        Int32(m),Int32(n),Int32(k),
        grid_dim=(grid_rows, grid_cols),
        block_dim=(tile, tile),
    )


@extensibility.register("matmul_tiled")
struct MatMul_tiled:
    @staticmethod
    def execute[target: StaticString](
        c: OutputTensor[rank=2, ...],
        a: InputTensor[dtype=c.dtype, rank=c.rank, ...],
        b: InputTensor[dtype=c.dtype, rank=c.rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            
        
            var a_tensor = a.to_tile_tensor()
            var b_tensor = b.to_tile_tensor()
            var c_tensor = c.to_tile_tensor()



            gpu_matmul_tiled(c_tensor, a_tensor, b_tensor, ctx)


        else:
            raise Error("Not implemented")










#matmul register tiled




comptime BM = 64
comptime BN = 64
comptime BK = 16
comptime TM = 4
comptime TN = 4
comptime NUM_THREADS = (BM * BN)//(TM * TN)


def gpu_matmul_regtiled[
    dtype: DType, ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout,
](
    c: TileTensor[dtype, CLayout, MutAnyOrigin],
    a: TileTensor[dtype, ALayout, MutAnyOrigin],
    b: TileTensor[dtype, BLayout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    @__parameter
    def kernel(k_dim: Int32):
        comptime assert a.flat_rank == 2 and b.flat_rank == 2 and c.flat_rank == 2

        var tid = Int(thread_idx.x)
        var block_row = Int(block_idx.y) *BM
        var block_col = Int(block_idx.x) *BN
        var thread_row = (tid // (BN//TN)) *TM
        var thread_col = (tid % (BN // TN)) *TN

        var a_tile = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[BM, BK]())
        var b_tile = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[BK, BN]())
        var acc = stack_allocation[dtype](row_major[TM, TN]()).fill(0)

        for k_start in range(0, Int(k_dim), BK):
            comptime for step in range(BM*BK//NUM_THREADS):
                var idx = tid+step*NUM_THREADS
                a_tile[idx//BK,idx %BK] = a[block_row + idx // BK, k_start + idx % BK]
                b_tile[idx// BN, idx % BN] = b[k_start + idx // BN, block_col + idx % BN]




            barrier()

            

            comptime for kk in range(BK):
                comptime for i in range(TM):
                    comptime for j in range(TN):
                        acc[i, j] += rebind[acc.ElementType](
                            a_tile[thread_row + i, kk] * b_tile[kk, thread_col + j])




            barrier()

        comptime for i in range(TM):
            comptime for j in range(TN):
                c[block_row+thread_row+i, block_col+thread_col+j] = rebind[c.ElementType](acc[i, j])

    ctx.enqueue_function[kernel](
        Int32(k),
        grid_dim=(n//BN, m//BM),
        block_dim=NUM_THREADS,
    )


@extensibility.register("matmul_regtiled")
struct MatMul_regtiled:
    @staticmethod
    def execute[target: StaticString](
        c: OutputTensor[rank=2, ...],
        a: InputTensor[dtype=c.dtype, rank=c.rank, ...],
        b: InputTensor[dtype=c.dtype, rank=c.rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            gpu_matmul_regtiled(c.to_tile_tensor(), a.to_tile_tensor(), b.to_tile_tensor(), ctx)
        else:
            raise Error("Not implemented")
