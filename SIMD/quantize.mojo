from std.math import round,clamp
from std.sys.info import simd_width_of

def qunatize(s:List[Float32], mut d:List[Int8], scale:Float32):
    comptime w= simd_width_of[DType.float32]()

    var ptr_s=s.unsafe_ptr()
    var ptr_d=d.unsafe_ptr()

    var i=0
    while i+w <= len(s):
        var data = ptr_s.unsafe_load[width=w](i)
        var q=clamp(round(data * (1.0/scale)), -127.0, 127.0).cast[DType.int8]()
        ptr_d.unsafe_store[width=w](i, q)
        i+=w

    while i< len(s):
        d[i]=Int8(clamp(round(s[i]*(1.0/scale)),-127.0, 127.0))
        i+=1

def dequantize(s:List[Int8],mut d:List[Float32], scale:Float32):
    comptime w=simd_width_of[DType.float32]()
    var ptr_s=s.unsafe_ptr()
    var ptr_d=d.unsafe_ptr()

    var i=0
    while i+w <= len(s):
        var data=ptr_s.unsafe_load[width=w](i).cast[DType.float32]()
        ptr_d.unsafe_store[width=w](i, data * scale)
        i+=w

    while i< len(s):
        d[i]=Float32(s[i]) * scale
        i+=1

def main():
    # comptime N=10
    var scale:Float32 = 0.2

    var original:List[Float32] = [-10.0, -5.0, -1.0, 0.0, 1.0, 2.5, 5.0, 7.0, 10.0, 12.0, 20.0]
    var quantized:List[Int8]= [0,0,0,0,0,0,0,0,0,0,0]
    var dequantized:List[Float32]= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    qunatize(original , quantized , scale )
    dequantize(quantized , dequantized , scale)

    print("--Original--")
    print(original)
    print()
    print("--Quantized--")
    print(quantized)
    print()
    print("--Dequantized--")
    print(dequantized)

