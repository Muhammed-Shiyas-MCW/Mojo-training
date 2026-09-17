from std.benchmark import run, keep, Unit

import convolution_naive
import convolution_im2Col
import convolution_tiled
import convolution_SIMD


def round3(val: Float64) -> Float64:
    return Float64(Int(val * 1000.0)) / 1000.0


def round2(val: Float64) -> Float64:
    return Float64(Int(val * 100.0)) / 100.0

def pad(s: String, width: Int) -> String:
    var res = s
    while res.byte_length() < width:
        res += " "
    return res



def run_benchmark_case(
    name: String,H: Int,W: Int,
    Cin: Int,Cout: Int,
    Kh: Int = 3,Kw: Int = 3,
    batch: Int = 1,
) raises:
    var input = List[Float32]()
    input.resize(batch * Cin * H * W, 0.0)
    for i in range(len(input)):
        input[i] = Float32((i % 20) + 1) * 0.1

    var weights = List[Float32]()
    weights.resize(Cout * Cin * Kh * Kw, 0.0)
    for i in range(len(weights)):
        weights[i] = Float32((i % 10) + 1) * 0.05

    var out_naive = List[Float32]()
    var out_im2col = List[Float32]()
    var out_tiled = List[Float32]()
    var out_simd = List[Float32]()

    def bench_naive() {
        imm input, imm weights, mut out_naive,
        imm batch, imm Cin, imm H, imm W, imm Cout, imm Kh, imm Kw
    }:
        convolution_naive.conv2d(
            input, weights, out_naive,
            batch, Cin, H, W, Cout, Kh, Kw, 1, 0
        )
        keep(out_naive[0])

    def bench_im2col() {
        imm input, imm weights, mut out_im2col,
        imm batch, imm Cin, imm H, imm W, imm Cout, imm Kh, imm Kw
    }:
        convolution_im2Col.convolution_im2col(
            input, weights, out_im2col,
            Cin, H, W, Cout, Kh, Kw, 1, 0, batch
        )
        keep(out_im2col[0])

    def bench_tiled() {
        imm input, imm weights, mut out_tiled,
        imm batch, imm Cin, imm H, imm W, imm Cout, imm Kh, imm Kw
    }:
        convolution_tiled.conv2d_tiled(
            input, weights, out_tiled,
            batch, Cin, H, W, Cout, Kh, Kw, 1, 0
        )
        keep(out_tiled[0])

    def bench_simd() {
        imm input, imm weights, mut out_simd,
        imm batch, imm Cin, imm H, imm W, imm Cout, imm Kh, imm Kw
    }:
        convolution_SIMD.conv2d_simd(
            input, weights, out_simd,
            batch, Cin, H, W, Cout, Kh, Kw
        )
        keep(out_simd[0])

    var r_naive = run(bench_naive, min_runtime_secs=0.03, max_runtime_secs=0.08)
    var r_im2col = run(bench_im2col, min_runtime_secs=0.03, max_runtime_secs=0.08)
    var r_tiled = run(bench_tiled, min_runtime_secs=0.03, max_runtime_secs=0.08)
    var r_simd = run(bench_simd, min_runtime_secs=0.03, max_runtime_secs=0.08)

    var t_naive = r_naive.mean("ms")
    var t_im2col = r_im2col.mean("ms")
    var t_tiled = r_tiled.mean("ms")
    var t_simd = r_simd.mean("ms")

    var sp_im2col = r_naive.mean() / r_im2col.mean()
    var sp_tiled = r_naive.mean() / r_tiled.mean()
    var sp_simd = r_naive.mean() / r_simd.mean()

    var col_im2col = String(round3(t_im2col)) + " ms (" + String(round2(sp_im2col)) + "x)"
    var col_tiled  = String(round3(t_tiled))  + " ms (" + String(round2(sp_tiled))  + "x)"
    var col_simd   = String(round3(t_simd))   + " ms (" + String(round2(sp_simd))   + "x)"

    var row = "| " + pad(name, 10)
    row += " | " + pad(String(Cin) + "->" + String(Cout), 8)
    row += " | " + pad(String(round3(t_naive)) + " ms", 10)
    row += " | " + pad(col_im2col, 18)
    row += " | " + pad(col_tiled, 18)
    row += " | " + pad(col_simd, 18) + " |"
    print(row)


def main() raises:
    print("-----------------------------------------------------------------------------------------------------")
    print("| Resolution | Channels | Naive Conv | im2col             | Direct Tiled       | SIMD Vectorized    |")
    print("|------------|----------|------------|--------------------|--------------------|--------------------|")

    # run_benchmark_case("4 x 4",4,4,1,16)
    run_benchmark_case("16 x 16", 16, 16, 2, 4)
    run_benchmark_case("32 x 32", 32, 32, 3, 8)
    run_benchmark_case("64 x 64", 64, 64, 3, 16)
    run_benchmark_case("128 x 128", 128, 128, 4, 16)

    print("----------------------------------------------------------------------------------------------------")
