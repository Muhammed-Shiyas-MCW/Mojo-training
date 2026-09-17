def matmul(weights: List[Float32],patches: List[Float32],mut output: List[Float32],
    offset: Int,Cout: Int,K: Int,N: Int,
):

    # print("weights:",weights)

    

    # print("patches:",patches)
    # print("output:",output)

    



    

    for oc in range(Cout):
        for k in range(K):
            var w = weights[oc * K + k]
            var p_offset = k * N
            var out_base = offset + oc * N
            for j in range(N):
                output[out_base+j] += w * patches[p_offset + j]


def convolution_im2col(input: List[Float32],weights: List[Float32], mut output: List[Float32],
    Cin: Int,H_in: Int, W_in: Int,
    Cout: Int, Kh: Int,Kw: Int,
    stride: Int = 1,
    padding: Int = 0,
    batch: Int = 1,
):
    var H_out = (H_in-Kh+2*padding)//stride + 1
    var W_out = (W_in-Kw+2*padding)//stride + 1

    
    var K = Cin*Kh*Kw
    var N = H_out*W_out

    output.resize(batch*Cout*H_out*W_out, 0.0)

    var patches = List[Float32]()
    patches.resize(K * N, 0.0)

    for b in range(batch):
        

        for h in range(H_out):
            for w in range(W_out):
                var col = h * W_out + w
                # print(col)

                for Ic in range(Cin):
                    for kh in range(Kh):
                        var curr_h = h*stride-padding+kh
                        if curr_h >= 0 and curr_h < H_in:
                            for kw in range(Kw):
                                var curr_w = w*stride-padding+kw
                                if curr_w >= 0 and curr_w < W_in:
                                    var in_idx = ((b*Cin+Ic)*H_in+curr_h)*W_in+curr_w
                                    
                                    var row = (Ic*Kh+kh) * Kw + kw


                                    patches[row*N+col] = input[in_idx]


        var offset = b * Cout * N
        matmul(weights, patches, output, offset, Cout, K, N)


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

    convolution_im2col(
        input, weights, output,
        1, 4, 4, 1, 3, 3,
    )

    print("Input:  ", input)
    print("Weight: ", weights)
    print("Output: ", output)
