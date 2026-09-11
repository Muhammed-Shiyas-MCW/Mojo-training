from matmul_ops.matrix import Matrix

def matmul_naive(a: Matrix, b: Matrix, mut c: Matrix, n: Int):
    c.zero()
    for i in range(n):
        for j in range(n):
            var sum = 0
            for k in range(n):
                sum += a.get(i, k) * b.get(k, j)
            c.set(i, j, sum)

def main():
    var n = 128
    var a = Matrix(n)
    var b = Matrix(n)
    var c = Matrix(n)
    for i in range(n):
        for j in range(n):
            a.set(i, j, 1)
            b.set(i, j, 1)
    matmul_naive(a, b, c, n)
    print(c.get(0, 0))
    a.free()
    b.free()
    c.free()
