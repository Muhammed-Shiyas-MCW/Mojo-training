import extensibility
from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor , foreach
from layout import Coord


@extensibility.register("add-one")
struct AddOne:

    @staticmethod
    def execute[target:StaticString](output:OutputTensor, x:InputTensor[dtype=output.dtype, rank=output.rank, ...],ctx:DeviceContext)raises:
        @always_inline
        def elementwise_add_one[width:Int](idx:Coord) {imm} -> SIMD[x.dtype,width]:
            return x.load[width](idx)+1

        
        foreach[elementwise_add_one,target=target](output , ctx)

