from std.benchmark import run, keep, Unit
from std.math import round

import kernel.matmul as naive_mod
import kernel.matmul_optimized as loop_interchange
import kernel.matmul_simd as simd_mod
import kernel.matmul_multilevel_tiled as tiled_mod
import kernel.matmul_multithread as thread_mod
# import kernel.matmul_multilevel_tiled_multithread as m_tiled_thread


def fmt5(val: Float64) -> String:
    var r = round(val, 5)
    var s = String(r)
    var dot = s.find(".")
    if dot == -1:
        return s + ".00000"
    var needed = dot + 1 + 5
    if s.byte_length() > needed:
        return String(s[byte=0:needed])
    var res = s
    while res.byte_length() < needed:
        res += "0"
    return res


def pad(s: String, width: Int) -> String:
    var res = s
    while res.byte_length() < width:
        res += " "
    return res


def benchmark_size(n: Int) raises:
    var a: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]
    var b: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]
    var c: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            a[i][j] = i * j + n
            b[i][j] = i * j + n

    var t_naive = "-"
    var t_interchange ="-"
    if n <= 1024:
        def bench_naive() {imm a, imm b, mut c, imm n}:
            naive_mod.matrix_multiplication(a, b, c, n)
            keep(c[0][0])
        var r_naive = run(bench_naive, min_runtime_secs=0.02, max_runtime_secs=0.15)
        t_naive = fmt5(r_naive.mean(Unit.ms)) + " ms"
    # if n <= 2048:
        def bench_loopinterchange(){imm a, imm b, mut c, imm n}:
            loop_interchange.matrix_multiplication(a,b,c,n)
            keep(c[0][0])
        var r_loopchange=run(bench_loopinterchange,min_runtime_secs=0.02, max_runtime_secs=0.15)
        t_interchange = fmt5(r_loopchange.mean(Unit.ms)) + " ms"


    def bench_simd() {imm a, imm b, mut c, imm n}:
        simd_mod.matrix_multiplication(a, b, c, n)
        keep(c[0][0])
    var r_simd = run(bench_simd, min_runtime_secs=0.02, max_runtime_secs=0.15)
    var t_simd = fmt5(r_simd.mean(Unit.ms)) + " ms"

    def bench_tiled() {imm a, imm b, mut c, imm n}:
        tiled_mod.matrix_multiplication(a, b, c, n)
        keep(c[0][0])
    var r_tiled = run(bench_tiled, min_runtime_secs=0.02, max_runtime_secs=0.15)
    var t_tiled = fmt5(r_tiled.mean(Unit.ms)) + " ms"

    def bench_threaded() {imm a, imm b, mut c, imm n}:
        thread_mod.matrix_multiplication(a, b, c, n)
        keep(c[0][0])
    var r_threaded = run(bench_threaded, min_runtime_secs=0.02, max_runtime_secs=0.15)
    var t_threaded = fmt5(r_threaded.mean(Unit.ms)) + " ms"

    # def bench_multilevel_tiled_threaded() {imm a, imm b, mut c, imm n}:
    #     m_tiled_thread.matrix_multiplication(a,b,c,n)
    #     keep(c[0][0])
    # var r_m_tiled_threaded=run(bench_multilevel_tiled_threaded,  min_runtime_secs=0.02, max_runtime_secs=0.15)
    # var t_m_tiled_threaded=fmt5(r_m_tiled_threaded.mean(Unit.ms)) + " ms"


    print(
        pad(String(n), 10),
        pad(t_naive, 16),
        pad(t_interchange,16),
        pad(t_simd, 16),
        pad(t_tiled, 18),
        pad(t_threaded, 18),
        # pad(t_m_tiled_threaded,16),
    )


def main() raises:
    
    print(
        pad("Input Size", 10),
        pad("Naive", 16),
        pad("Loop Interchange",16),
        pad("SIMD", 16),
        pad("Multilevel Tiled", 18),
        pad("Multithreaded", 18),
        # pad("Multi Tiled + Threaded",16),
    )
    print("-------------------------------------------------------------------------------------------------------")

    var test_sizes: List[Int] = [32, 64, 128, 256, 512, 1024, 2048]

    for i in range(len(test_sizes)):
        benchmark_size(test_sizes[i])

    
