from std.benchmark import run,keep,Unit

def matrix_multiplication(a:List[List[Int]], b:List[List[Int]], mut c:List[List[Int]], n:Int):
    for i in range(n):
        for j in range(n):
            c[i][j] = 0

    for i in range(n):
        for j in range(n):
            var x=0
            for k in range(n):
                x+=a[i][k]*b[k][j]
            c[i][j]=x



def loop(n: Int) raises -> Float64:
    var a: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]

    var b: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]
    var c: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            a[i][j] = i * j + n
            b[i][j] = i * j + n

    def benchmark() {imm a, imm b, mut c, imm n}:
        matrix_multiplication(a, b, c, n)
        keep(c[0][0])

    var report = run(benchmark, min_runtime_secs=0.05, max_runtime_secs=0.2)
    var mean_ms = report.mean(Unit.ms)
    return mean_ms


def main() raises:
    var sizes: List[Int] = [
        8, 9, 16, 17, 32, 33, 64, 65,
        128, 129, 256, 257, 512, 513,
        1024, 1025, 2048, 2049
    ]

    print("Size (N x N)      Mean Time (ms)")
    print("--------------------------------")

    for i in range(len(sizes)):
        var n = sizes[i]
        var mean_ms = loop(n)
        print(n, "           ", mean_ms, "ms")
