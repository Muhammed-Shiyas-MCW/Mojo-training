from std.sys.info import simd_width_of
from std.memory.alloc import unsafe_alloc

comptime simd_w = simd_width_of[DType.int]()
comptime unroll = 4
comptime M_tile = 64
comptime K_tile = 64
comptime N_tile = 64


# struct Matrix:
#     var data:List[List[Int]]
#     var r:Int
#     var c:Int

struct Matrix:
    var data: Pointer[Int,MutUntrackedOrigin]
    var n: Int

    def __init__(out self, n: Int):
        self.n = n
        self.data = unsafe_alloc[Int](n*n)

    def free(self):
        self.data.unsafe_free()

    def ptr(self, r: Int) -> Pointer[Int, MutUntrackedOrigin]:
        return self.data.unsafe_offset(r * self.n)

    def set(self, r: Int, c: Int, val: Int):
        self.data.unsafe_offset(r*self.n + c).unsafe_store(val)

    def get(self, r: Int, c: Int) -> Int:
        return self.data.unsafe_offset(r *self.n+ c).unsafe_load()

    def zero(self):
        for i in range(self.n):
            var row = self.ptr(i)
            for j in range(self.n):
                row.unsafe_store(j, 0)
