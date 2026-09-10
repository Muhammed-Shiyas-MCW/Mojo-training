from std.benchmark import run, keep, Unit
from std.sys.info import simd_width_of
from std.math import min
from max.algorithm import parallelize,sync_parallelize

comptime simd_w = simd_width_of[DType.int]()
comptime unroll = 4
comptime tile = 64

def matrix_multiplication(a: List[List[Int]], b: List[List[Int]], mut c: List[List[Int]], n: Int):
    for i in range(n):
        for j in range(n):
            c[i][j] = 0


    


    def worker(i: Int) {imm a, imm b, mut c, imm n}:
        var c_ptr = c[i].unsafe_ptr()


        for k_tile in range(0, n, tile):
            var k_end = min(k_tile + tile, n)
            for j_tile in range(0, n, tile):
                var j_end = min(j_tile + tile, n)

                for k in range(k_tile, k_end):
                    var a_val = a[i][k]
                    var b_ptr = b[k].unsafe_ptr()

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
                        c[i][j] += a_val * b[k][j]
                        j += 1

    parallelize(worker,n)

    

def main() raises:
    var n = 2048

    var a: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]
    var b: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]
    var c: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            a[i][j] = i * j + n
            b[i][j] = i * j + n

    def benchmark() {imm a, imm b, mut c, imm n}:
        matrix_multiplication(a, b, c, n)
        keep(c[0][0])

    var report = run(benchmark, max_runtime_secs=0.5)
    report.print(Unit.ms)

    print("c[0][0]:", c[0][0])
    print("c[n-1][n-1]:", c[n-1][n-1])
