from std.benchmark import run,keep,Unit

def matrix_multiplication(a:List[List[Int]], b:List[List[Int]], mut c:List[List[Int]], n:Int):
    for i in range(n):
        for j in range(n):
            c[i][j] = 0

    for i in range(n):
        for k in range(n):
            var a_val = a[i][k]
            for j in range(n):
                c[i][j] += a_val * b[k][j]
            



def main() raises:
    var n=1000

    var a:List[List[Int]]=[[0 for _ in range(n)] for _ in range(n)]
    var b:List[List[Int]]=[[0 for _ in range(n)] for _ in range(n)]
    var c:List[List[Int]]=[[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            a[i][j]=i*j+n
            b[i][j]=i*j+n

    

    def benchmark(){imm a,imm b, mut c, imm n}:
        matrix_multiplication(a,b,c,n)
        keep(c[0][0])
    

    var report=run(benchmark,max_runtime_secs=0.5)
    report.print(Unit.ms)
