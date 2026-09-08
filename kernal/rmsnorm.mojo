from std.benchmark import run,keep,Unit
from std.math import sqrt



comptime N=1000000

def rmsnorm(x:List[Float32],w:List[Float32], mut out:List[Float32]):
    var n=len(x)
    var eps:Float32=1e-6

    var sum:Float32=0.0
    for i in range(n):
        sum+=x[i]*x[i]
    
    var inv=1.0 / sqrt((sum/Float32(n))+eps)

    for i in range(n):
        out[i]=x[i]* inv * w[i]



def main() raises:
    var x=List[Float32]()
    var w=List[Float32]()
    var out=List[Float32]()

    x.resize(N,2.5)
    w.resize(N, 1.0)
    out.resize(N, 0.0)

    def benchmark(){imm x, imm w, mut out}:
        rmsnorm(x, w, out)
        keep(out[0])

    var report = run(benchmark,max_runtime_secs=0.5)
    report.print(Unit.ms)