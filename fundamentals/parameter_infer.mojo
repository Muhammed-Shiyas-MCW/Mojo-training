from std.math import sqrt
from std.builtin.simd_length import SIMDLength

def rsqrt[dt: DType, size: SIMDLength](x: SIMD[dt, size]) -> SIMD[dt, size]:
        return 1 / sqrt(x)

def main():
    var v= SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var result = rsqrt(v)
    print("Input  vector:", v)
    print("rsqrt  result:", result)