from matmul_ops.matrix import Matrix

def matmul_loop_interchanged(a: Matrix, b: Matrix, mut c: Matrix, n: Int):
    c.zero()
    for i in range(n):
        var a_ptr = a.ptr(i)
        var c_ptr = c.ptr(i)
        for k in range(n):
            var a_val = a_ptr.unsafe_load(k)
            var b_ptr = b.ptr(k)
            for j in range(n):
                var curr = c_ptr.unsafe_load(j)
                c_ptr.unsafe_store(j, curr + a_val * b_ptr.unsafe_load(j))

def main():
    var n = 128
    var a = Matrix(n)
    var b = Matrix(n)
    var c = Matrix(n)
    for i in range(n):
        for j in range(n):
            a.set(i, j, 1)
            b.set(i, j, 1)
    matmul_loop_interchanged(a, b, c, n)
    print("opt02_loop_interchange (N=128) finished, c[0,0]:", c.get(0, 0))
    a.free()
    b.free()
    c.free()
