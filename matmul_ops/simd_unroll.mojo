from matmul_ops.matrix import Matrix, simd_w, unroll

def matmul_simd_unrolled(a: Matrix, b: Matrix, mut c: Matrix, n: Int):
    


    c.zero()
    for i in range(n):
        var a_ptr = a.ptr(i)
        var c_ptr = c.ptr(i)
        for k in range(n):
            var a_val = a_ptr.unsafe_load(k)
            var b_ptr = b.ptr(k)
            var j = 0
            while j + (simd_w * unroll) <= n:
                comptime for step in range(unroll):
                    var offset = j + step * simd_w
                    var b_vec = b_ptr.unsafe_load[width=simd_w](offset)
                    var c_vec = c_ptr.unsafe_load[width=simd_w](offset)
                    c_vec += a_val * b_vec
                    c_ptr.unsafe_store(offset, c_vec)
                j += simd_w * unroll
            while j < n:
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
    matmul_simd_unrolled(a, b, c, n)
    print(c.get(0, 0))
    a.free()
    b.free()
    c.free()
