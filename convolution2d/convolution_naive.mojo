from std.benchmark import run, keep, Unit


#  input shape [B,C,H,W] // [B,H,W,C]

# kernel shape [Cout,C,Kh,Kw]



# output shape [B,Cout,Hout,Wout]



def conv2d(
    input:List[Float32],weights:List[Float32],mut output:List[Float32],
    batch:Int,Cin:Int,H_in:Int,W_in:Int,Cout:Int,
    Kh:Int,Kw:Int,stride:Int,padding:Int,
):
    var H_out=(H_in-Kh+2*padding)//stride + 1
    var W_out=(W_in-Kw+2*padding)//stride + 1

    var total_out = batch*Cout*H_out*W_out
    output.resize(total_out,0.0)

    for n in range(batch):



        for Oc in range(Cout):
            for h in range(H_out):
                # print("row:",h)
                # var h_start=h*stride-padding
                # print(h_start)

                for w in range(W_out):
                    # print(w)
                    # var w_start=w*stride-padding
                    # print(w_start)

                    var element:Float32=0.0


                    
                    
                    
                    for Ic in range(Cin):
                        for kh in range(Kh):
                            var cur_h=h+kh
                            # print(cur_h)

                            if cur_h>=0 and cur_h < H_in:
                                for kw in range(Kw):
                                    var cur_w=w+kw
                                    # print(cur_w)

                                    if cur_w>=0 and cur_w<W_in:
                                        #input[B,Cin,H_in,W_in]
                                        #(n,Ic,cur_h,cur_w)
                                        var in_idx=((n*Cin+Ic)*H_in+cur_h)*W_in+cur_w


                                        #weight[Cout,Cin,Kh,Kw]
                                        #(Oc,Ic,kh,kw)
                                        var w_idx=((Oc*Cin+Ic)*Kh+kh)*Kw+kw

                                        element+=input[in_idx]*weights[w_idx]
                

                    #output[B,Cout,H_out,W_out]
                    #(n,Oc,h,w) 
                    var out_idx=((n*Cout+Oc)*H_out+h)*W_out+w
                    output[out_idx]=element















def main():
    # var batch=1
    # var Cin=3
    # var H_in=4
    # var W_in=4

    # var Cout=16
    # var Kh=3
    # var Kw=3

    # var stride=1
    # var pad=0

    # var input = List[Float32]()
    # for i in range(batch * Cin * H_in * W_in):
    #     input.append(Float32(i + 1))

    # var weights = List[Float32]()
    # for _ in range(Cout * Cin * Kh * Kw):
    #     weights.append(1.0)

    # var output=List[Float32]()

    # conv2d(input,weights,output,batch,Cin,H_in,W_in,Cout,Kh,Kw,stride,pad)


    var input:List[Float32]=[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    ]

    var weights:List[Float32]=[
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0
    ]

    var output = List[Float32]()

    conv2d(
        input, weights, output,
        1, 1, 4, 4, 1, 3, 3, 1, 0
    )

    print("Input: ",input)
    print("Weight: ",weights)
    print("Output:",output)

