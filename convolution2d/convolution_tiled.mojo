from std.math import min


def conv2d_tiled(
    input: List[Float32],weights: List[Float32],mut output: List[Float32],
    batch: Int,Cin: Int,H_in: Int,W_in: Int,
    Cout: Int,Kh: Int,Kw: Int,
    stride: Int = 1,padding: Int = 0,
    tile_h:Int = 16,tile_w:Int = 16,tile_c:Int = 8,
):
    var H_out = (H_in-Kh+2*padding)//stride + 1
    var W_out = (W_in-Kw+2*padding)//stride + 1
    output.resize(batch*Cout*H_out*W_out, 0.0)

    for b in range(batch):
        for oc_tile in range(0,Cout,tile_c):
            var oc_end = min(oc_tile+tile_c, Cout)

            for ht in range(0,H_out,tile_h):
                var ht_end = min(ht+tile_h,H_out)

                for wt in range(0,W_out,tile_w):
                    var wt_end = min(wt+tile_w, W_out)

                    for oc in range(oc_tile, oc_end):
                        for h in range(ht, ht_end):
                            var h_start = h*stride - padding

                            for w in range(wt, wt_end):
                                var w_start = w*stride - padding
                                var acc: Float32 = 0.0

                                for ic in range(Cin):
                                    for kh in range(Kh):
                                        var cur_h = h_start + kh
                                        if cur_h >= 0 and cur_h < H_in:
                                            for kw in range(Kw):
                                                var cur_w = w_start + kw
                                                if cur_w >= 0 and cur_w < W_in:
                                                    var in_idx = ((b*Cin+ic)*H_in+cur_h)*W_in + cur_w
                                                    var w_idx = ((oc*Cin+ic)*Kh+ kh) * Kw + kw
                                                    acc += input[in_idx] * weights[w_idx]

                                var out_idx = ((b*Cout+oc)*H_out+h)*W_out + w
                                output[out_idx] = acc


def main():
    var input: List[Float32] = [
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]

    var weights: List[Float32] = [
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0,
    ]

    var output = List[Float32]()

    conv2d_tiled(
        input, weights, output,
        1, 1, 4, 4, 1, 3, 3,
        tile_h=2, tile_w=2, tile_c=1
    )

    print("Input:  ", input)
    print("Weight: ", weights)
    print("Output: ", output)
