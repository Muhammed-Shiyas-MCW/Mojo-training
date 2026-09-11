from std.benchmark import run, keep, Unit
from matmul_ops.matrix import Matrix
from matmul_ops.naive import matmul_naive
from matmul_ops.loop_interchange import matmul_loop_interchanged
from matmul_ops.simd_unroll import matmul_simd_unrolled
from matmul_ops.single_tiled import matmul_single_level_tiled
from matmul_ops.multilevel_tiled import matmul_multilevel_tiled
from matmul_ops.multithreaded_simd import matmul_multithreaded_simd
from matmul_ops.multithreaded_block_tiled import matmul_multithreaded_block_tiled

def fmt(val: Float64) -> String:
    var i_part = Int(val)
    var f_part = Int((val - Float64(i_part)) * 1000.0)
    if f_part < 0:
        f_part = -f_part
    var f_str = String(f_part)
    while f_str.byte_length() < 3:
        f_str = "0" + f_str
    return String(i_part) + "." + f_str + " ms"

def pad(s: String, width: Int) -> String:
    var res = s
    while res.byte_length() < width:
        res += " "
    return res

def main() raises:

    print()
    print("--------------------------------------------------------------------------------------------------------------------------")
    
    var header = "| " + pad("N", 6) + " | " + pad("Naive", 12) + " | " + pad("Interchanged", 13) + " | " + pad("SIMD Unroll", 12) + " | " + pad("Single Tiled", 13) + " | " + pad("Multi Tiled", 13) + " | " + pad("MT SIMD", 12) + " | " + pad("M Til Th", 15) + " |"
    print(header)
    print("--------------------------------------------------------------------------------------------------------------------------")

    var test_sizes: List[Int] = [32, 64, 128, 256, 512, 1024, 2048]

    for idx in range(len(test_sizes)):
        var n = test_sizes[idx]

        var a = Matrix(n)
        var b = Matrix(n)
        var c = Matrix(n)

        for i in range(n):
            for j in range(n):
                a.set(i, j, (i + j) % 17)
                b.set(i, j, (i * 2 + j) % 19)

        var t1_str:String
        if n <= 1024:
            def bench_naive() {imm a, imm b, mut c, imm n}:
                matmul_naive(a, b, c, n)
                keep(c.get(0, 0))
            var r1 = run(bench_naive, min_runtime_secs=0.01, max_runtime_secs=0.1)
            t1_str = fmt(r1.mean(Unit.ms))
        else:
            t1_str = "-"

        def bench_interchanged() {imm a, imm b, mut c, imm n}:
            matmul_loop_interchanged(a, b, c, n)
            keep(c.get(0, 0))
        var r2 = run(bench_interchanged, min_runtime_secs=0.01, max_runtime_secs=0.1)
        var t2_str = fmt(r2.mean(Unit.ms))

        def bench_simd() {imm a, imm b, mut c, imm n}:
            matmul_simd_unrolled(a, b, c, n)
            keep(c.get(0, 0))
        var r3 = run(bench_simd, min_runtime_secs=0.01, max_runtime_secs=0.1)
        var t3_str = fmt(r3.mean(Unit.ms))

        def bench_single_tiled() {imm a, imm b, mut c, imm n}:
            matmul_single_level_tiled(a, b, c, n)
            keep(c.get(0, 0))
        var r4 = run(bench_single_tiled, min_runtime_secs=0.01, max_runtime_secs=0.1)
        var t4_str = fmt(r4.mean(Unit.ms))

        def bench_multi_tiled() {imm a, imm b, mut c, imm n}:
            matmul_multilevel_tiled(a, b, c, n)
            keep(c.get(0, 0))
        var r5 = run(bench_multi_tiled, min_runtime_secs=0.01, max_runtime_secs=0.1)
        var t5_str = fmt(r5.mean(Unit.ms))

        def bench_mt_simd() {imm a, imm b, mut c, imm n}:
            matmul_multithreaded_simd(a, b, c, n)
            keep(c.get(0, 0))
        var r6 = run(bench_mt_simd, min_runtime_secs=0.01, max_runtime_secs=0.1)
        var t6_str = fmt(r6.mean(Unit.ms))

        def bench_mt_block() {imm a, imm b, mut c, imm n}:
            matmul_multithreaded_block_tiled(a, b, c, n)
            keep(c.get(0, 0))
        var r7 = run(bench_mt_block, min_runtime_secs=0.01, max_runtime_secs=0.1)
        var t7_str = fmt(r7.mean(Unit.ms))

        var row = "| " + pad(String(n), 6) + " | " + pad(t1_str, 12) + " | " + pad(t2_str, 13) + " | " + pad(t3_str, 12) + " | " + pad(t4_str, 13) + " | " + pad(t5_str, 13) + " | " + pad(t6_str, 12) + " | " + pad(t7_str, 15) + " |"
        print(row)

        a.free()
        b.free()
        c.free()

