from std.sys.info import simd_width_of

comptime simd_w = simd_width_of[DType.float32]()


def conv2d_simd(
    input: List[Float32],weights: List[Float32],mut output: List[Float32],
    batch: Int,Cin: Int,H_in: Int,W_in: Int,
    Cout: Int,Kh: Int,Kw: Int,
):
    var H_out = H_in-Kh + 1
    var W_out = W_in-Kw + 1
    output.resize(batch*Cout*H_out*W_out, 0.0)

    var in_ptr = input.unsafe_ptr()
    var w_ptr = weights.unsafe_ptr()
    var out_ptr = output.unsafe_ptr()

    for b in range(batch):
        for oc in range(Cout):
            for h in range(H_out):
                var out_row = ((b*Cout+oc)*H_out+h) * W_out




                var w = 0
                while w + simd_w <= W_out:
                    var acc: SIMD[DType.float32, simd_w] = 0.0

                    for ic in range(Cin):
                        var in_channel = (b*Cin+ic) * H_in
                        var w_channel = (oc*Cin+ic) * Kh



                        for kh in range(Kh):
                            var in_row = (in_channel+ h + kh) * W_in
                            var w_row = (w_channel + kh) * Kw

                            for kw in range(Kw):
                                var w_val = w_ptr.unsafe_load(w_row+kw)
                                # print(w_val)
                                var in_vec = in_ptr.unsafe_load[width=simd_w](in_row+w+kw)
                                # print(in_vec)
                                acc += in_vec * w_val

                    out_ptr.unsafe_store[width=simd_w](out_row+w,acc)


                    w+=simd_w

                while w < W_out:
                    var scalar: Float32 = 0.0
                    for ic in range(Cin):
                        for kh in range(Kh):
                            for kw in range(Kw):
                                var in_idx = ((b*Cin+ic)*H_in+h+kh)*W_in+w + kw
                                var w_idx = ((oc*Cin+ic)*Kh+kh)*Kw + kw
                                scalar += in_ptr.unsafe_load(in_idx) * w_ptr.unsafe_load(w_idx)

                    out_ptr.unsafe_store(out_row+w,scalar)
                    w += 1

def main():

    # var input: List[Float32] = [
    #     1.0,  2.0,  3.0,  4.0,
    #     5.0,  6.0,  7.0,  8.0,
    #     9.0, 10.0, 11.0, 12.0,
    #     13.0, 14.0, 15.0, 16.0,
    # ]

    # var weights: List[Float32] = [
    #     1.0, 0.0, 1.0,
    #     0.0, 1.0, 0.0,
    #     1.0, 0.0, 1.0,
    # ]

    # var output = List[Float32]()

    # conv2d_simd(
    #     input, weights, output,
    #     1, 1, 4, 4, 1, 3, 3
    # )

    # print("Input:  ", input)
    # print("Weight: ", weights)
    # print("Output: ", output)

    var H_in = 10
    var W_in = 10
    var Kh = 3
    var Kw = 3
    var H_out = H_in - Kh + 1 
    var W_out = W_in - Kw + 1

    var input = List[Float32]()
    for i in range(H_in * W_in):
        input.append(Float32(i + 1))

    for r in range(H_in):
        var row_str = String("[ ")
        for c in range(W_in):
            var val = Int(input[r * W_in + c])
            row_str += String(val) + "\t"
        row_str += "]"
        print(row_str)

    var weights: List[Float32] = [
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0,
    ]

    var output = List[Float32]()

    conv2d_simd(
        input, weights, output,
        1, 1, H_in, W_in, 1, Kh, Kw
    )

    print(output)