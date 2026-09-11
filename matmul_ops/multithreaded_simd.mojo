from max.algorithm import parallelize
from matmul_ops.matrix import Matrix, simd_w, unroll

def matmul_multithreaded_simd(a: Matrix, b: Matrix, mut c: Matrix, n: Int):
    c.zero()
    def worker(i: Int) {imm a, imm b, mut c, imm n}:
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

    parallelize(worker, n)

def main():
    var n = 4096
    var a = Matrix(n)
    var b = Matrix(n)
    var c = Matrix(n)
    for i in range(n):
        for j in range(n):
            a.set(i, j, 1)
            b.set(i, j, 1)
    matmul_multithreaded_simd(a, b, c, n)
    print(c.get(0, 0))
    print(c.get(n-1,n-1))
    a.free()
    b.free()
    c.free()
