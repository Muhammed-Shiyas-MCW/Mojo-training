def repeat[count: Int](msg: String):
    comptime for _ in range(count):
        print(msg)


def rsqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    from std.math import sqrt
    return 1 / sqrt(x)

def main():
    repeat[3]("Mojo")
    var val = Scalar[DType.float32](16.0)
    print(rsqrt(val))