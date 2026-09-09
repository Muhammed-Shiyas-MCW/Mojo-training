from std.benchmark import run,keep,Unit
from std.sys.info import simd_width_of

comptime simd_w=simd_width_of[DType.int]()

def matrix_multiplication(a:List[List[Int]], b:List[List[Int]], mut c:List[List[Int]], n:Int):
    for i in range(n):
        for j in range(n):
            c[i][j] = 0

    for i in range(n):
        var c_ptr=c[i].unsafe_ptr()
        for k in range(n):
            var a_val = a[i][k]
            var b_ptr=b[k].unsafe_ptr()

            # for j in range(0,n,simd_w):
                # var b_vec1=b_ptr.unsafe_load[width=simd_w](j)
                # var c_vec1=c_ptr.unsafe_load[width=simd_w](j)
                # c_vec1+=a_val*b_vec1
                # c_ptr.unsafe_store(j,c_vec1)

            var j=0
            while j +(simd_w*4) <=  n:

                var b_vec1=b_ptr.unsafe_load[width=simd_w](j)
                var b_vec2=b_ptr.unsafe_load[width=simd_w](j+simd_w)
                var b_vec3=b_ptr.unsafe_load[width=simd_w](j+simd_w*2)
                var b_vec4=b_ptr.unsafe_load[width=simd_w](j+simd_w*3)
                var c_vec1=c_ptr.unsafe_load[width=simd_w](j)
                var c_vec2=c_ptr.unsafe_load[width=simd_w](j+simd_w)
                var c_vec3=c_ptr.unsafe_load[width=simd_w](j+simd_w*2)
                var c_vec4=c_ptr.unsafe_load[width=simd_w](j+simd_w*3)
                
                c_vec1+=a_val*b_vec1
                c_vec2+=a_val*b_vec2
                c_vec3+=a_val*b_vec3
                c_vec4+=a_val*b_vec4

                c_ptr.unsafe_store(j,c_vec1)
                c_ptr.unsafe_store(j+simd_w,c_vec2)
                c_ptr.unsafe_store(j+simd_w*2,c_vec3)
                c_ptr.unsafe_store(j+simd_w*3,c_vec4)

                j+=simd_w*4

            while j<n:
                c[i][j]+=a_val*b[k][j]
                j+=1

            

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
