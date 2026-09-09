from std.benchmark import run,keep,Unit
from std.math import sqrt
from std.sys.info import simd_width_of

comptime N=1000000
def rmsnorm(x:List[Float32],w:List[Float32],mut out:List[Float32]):
    
    comptime simd_w= simd_width_of[DType.float32]()
    var n=len(x)
    var eps:Float32=1e-6
    var x_ptr=x.unsafe_ptr()
    var w_ptr=w.unsafe_ptr()
    var out_ptr=out.unsafe_ptr()

    var sum=SIMD[DType.float32,simd_w](0.0)
    var i=0
    while i+simd_w <=n:
        var v=x_ptr.unsafe_load[width=simd_w](i)
        sum+=v*v
        i+=simd_w
    
    var rem_sum:Float32 = 0.0
    while i < n:
        var v = x_ptr[unsafe_offset=i]
        rem_sum+= v * v
        i += 1

    var total_sum=sum.reduce_add()+rem_sum
    var inv=1.0/sqrt((total_sum/Float32(n))+eps)

    i=0
    while i+simd_w<=n:
        var v=x_ptr.unsafe_load[width=simd_w](i)
        var weight=w_ptr.unsafe_load[width=simd_w](i)
        out_ptr.unsafe_store(i,v*inv*weight)
        i+=simd_w
    
    while i < n:
        out_ptr[unsafe_offset=i]= x_ptr[unsafe_offset=i]* inv *w_ptr[unsafe_offset=i]
        i += 1

def main() raises:
    var x=List[Float32]()
    var w=List[Float32]()
    var out=List[Float32]()

    x.resize(N,2.5)
    w.resize(N,1.0)
    out.resize(N,0.0)

    def benchmark(){imm x, imm w, mut out}:
        rmsnorm(x, w, out)
        keep(out[0])

    var report = run(benchmark,max_runtime_secs=0.5)
    report.print(Unit.ms)