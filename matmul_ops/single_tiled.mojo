from std.math import min
from matmul_ops.matrix import Matrix, simd_w, unroll, K_tile, N_tile

def matmul_single_level_tiled(a: Matrix, b: Matrix, mut c: Matrix, n: Int):
    c.zero()
    for k_tile in range(0, n, K_tile):
        var k_end = min(k_tile + K_tile, n)
        for j_tile in range(0, n, N_tile):
            var j_end = min(j_tile + N_tile, n)
            for i in range(n):
                var a_ptr = a.ptr(i)
                var c_ptr = c.ptr(i)
                for k in range(k_tile, k_end):
                    var a_val = a_ptr.unsafe_load(k)
                    var b_ptr = b.ptr(k)
                    var j = j_tile
                    while j + (simd_w * unroll) <= j_end:
                        comptime for step in range(unroll):
                            var offset = j + step * simd_w
                            var b_vec = b_ptr.unsafe_load[width=simd_w](offset)
                            var c_vec = c_ptr.unsafe_load[width=simd_w](offset)
                            c_vec += a_val * b_vec
                            c_ptr.unsafe_store(offset, c_vec)
                        j += simd_w * unroll
                    while j < j_end:
                        var curr = c_ptr.unsafe_load(j)
                        c_ptr.unsafe_store(j, curr + a_val * b_ptr.unsafe_load(j))
                        j += 1

def main():
    var n = 128
    var a = Matrix(n)
    var b = Matrix(n)
    var c = Matrix(n)
    for i in range(n):
        for j in range(n):
            a.set(i, j, 1)
            b.set(i, j, 1)
    matmul_single_level_tiled(a, b, c, n)
    print(c.get(0, 0))
    a.free()
    b.free()
    c.free()
