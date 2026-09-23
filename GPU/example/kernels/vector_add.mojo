import extensibility
from max.gpu.host import DeviceContext
from std.math import ceildiv
from max.gpu import block_dim,block_idx,thread_idx
from extensibility import InputTensor,OutputTensor,ManagedTensorSlice
from std.utils.index import IndexList


def cpu_vector_add(
    output:ManagedTensorSlice[mut=True, ...],
    lhs:ManagedTensorSlice[rank=output.rank,dtype=output.dtype,...],
    rhs:ManagedTensorSlice[rank=output.rank,dtype=output.dtype,...],
    ctx:DeviceContext
    ):
        var vector_length = output.dim_size(0)
        for i in range(vector_length):
            var index=IndexList[output.rank](i)
            var result=lhs.load[1](index)+rhs.load[1](index)
            output.store[1](index,result)


def gpu_vector_add(
    output:ManagedTensorSlice[mut=True, ...],
    lhs:ManagedTensorSlice[rank=output.rank,dtype=output.dtype,...],
    rhs:ManagedTensorSlice[rank=output.rank,dtype=output.dtype,...],
    ctx:DeviceContext
) raises:

    comptime block_size=16
    var vector_length=output.dim_size(0)
    var gpu_ctx=ctx

    @__parameter
    def vector_addition_gpu_kernel(length_dev:Int32):
        var length=Int(length_dev)
        var tid=block_dim.x*block_idx.x+thread_idx.x
        if tid<length:
            var idx=IndexList[output.rank](tid)
            var result=lhs.load[1](idx)+rhs.load[1](idx)
            output.store[1](idx,result)

    var num_blocks=ceildiv(vector_length,block_size)

    gpu_ctx.enqueue_function[vector_addition_gpu_kernel](
        Int32(vector_length),grid_dim=num_blocks,block_dim=block_size
    )




@extensibility.register("vector-add")
struct VectorAddition:

    @staticmethod
    def execute[target:StaticString](
        output:OutputTensor[rank=1,...],
        lhs:InputTensor[rank=output.rank,dtype=output.dtype,...],
        rhs:InputTensor[rank=output.rank,dtype=output.dtype,...],
        ctx:DeviceContext
    )raises:
        comptime if target=="cpu":
            print("here")
            cpu_vector_add(output,lhs,rhs,ctx)
        elif target=="gpu":
            gpu_vector_add(output,lhs,rhs,ctx)

        else:
            raise Error("No target device")

