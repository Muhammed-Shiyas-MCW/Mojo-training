from std.benchmark import run, keep, Unit
from std.sys.info import simd_width_of
from std.math import min

comptime simd_w = simd_width_of[DType.int]()
comptime unroll = 4
comptime tile = 64

def matrix_multiplication(a: List[List[Int]], b: List[List[Int]], mut c: List[List[Int]], n: Int):
    for i in range(n):
        for j in range(n):
            c[i][j] = 0


    # for i in range(0, n, tile):
    #     var i_end = min(i_tile +tile, n)
    #     for k in range(0, n, tile):
    #         var k_end = min(k_tile +tile, n)
    #         for j in range(0, n, tile):
    #             var j_end = min(j_tile +tile, n)


    for i_tile in range(0, n, tile):
        var i_end = min(i_tile +tile, n)
        for k_tile in range(0, n, tile):
            var k_end = min(k_tile +tile, n)
            for j_tile in range(0, n, tile):
                var j_end = min(j_tile +tile, n)




                for i in range(i_tile, i_end):
                    var c_ptr = c[i].unsafe_ptr()
                    for k in range(k_tile, k_end):
                        var a_val = a[i][k]
                        var b_ptr = b[k].unsafe_ptr()

                        var j = j_tile
                        while j + (simd_w*unroll) <= j_end:
                            comptime for step in range(unroll):
                                var offset = j+step*simd_w
                                var b_vec = b_ptr.unsafe_load[width=simd_w](offset)
                                var c_vec = c_ptr.unsafe_load[width=simd_w](offset)
                                c_vec += a_val * b_vec
                                c_ptr.unsafe_store(offset, c_vec)
                            j += simd_w * unroll

                    


                        while j < j_end:
                            c[i][j] += a_val * b[k][j]
                            j += 1

def loop(n: Int) raises -> Float64:
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

    var report = run(benchmark, min_runtime_secs=0.05, max_runtime_secs=0.2)
    var mean_ms = report.mean(Unit.ms)
    return mean_ms


def main() raises:
    var sizes: List[Int] = [
        8, 9, 16, 17, 32, 33, 64, 65,
        128, 129, 256, 257, 512, 513,
        1024, 1025, 2048, 2049
    ]

    print("Size (N x N)      Mean Time (ms)")
    print("--------------------------------")

    for i in range(len(sizes)):
        var n = sizes[i]
        var mean_ms = loop(n)
        print(n, "           ", mean_ms, "ms")
