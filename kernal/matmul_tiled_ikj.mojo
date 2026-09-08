from std.benchmark import run, keep, Unit
from std.math import min

def matmul_tiled(a: List[List[Int]],b: List[List[Int]],mut c: List[List[Int]],n: Int,tile_size: Int = 32):
    for i in range(n):
        for j in range(n):
            c[i][j] = 0

    for i_tile in range(0,n, tile_size):
        for k_tile in range(0, n,tile_size):
            for j_tile in range(0, n,tile_size):

                for i in range(i_tile, min(i_tile + tile_size, n)):
                    for k in range(k_tile, min(k_tile + tile_size, n)):
                        var a_val = a[i][k]
                        for j in range(j_tile, min(j_tile + tile_size, n)):
                            c[i][j] += a_val * b[k][j]

def main() raises:
    var n = 128
    

    var a: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]
    var b: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]
    var c: List[List[Int]] = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            a[i][j] = i*j + n
            b[i][j] = i*j + n

    def benchmark() {imm a, imm b, mut c, imm n}:
        matmul_tiled(a, b, c, n)
        keep(c[0][0])

    


    var report = run(benchmark, max_runtime_secs=0.5)
    report.print(Unit.ms)
