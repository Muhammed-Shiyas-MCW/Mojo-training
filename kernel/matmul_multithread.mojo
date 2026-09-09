from std.benchmark import run,keep,Unit
from std.sys.info import simd_width_of, num_physical_cores
from std.math import min
from max.algorithm import parallelize,sync_parallelize

comptime simd_w=simd_width_of[DType.int]()

def matrix_multiplication(a:List[List[Int]], b:List[List[Int]], mut c:List[List[Int]], n:Int):

    # var threads=num_physical_cores()
    # var row_per_thread=n//threads

    for i in range(n):
        for j in range(n):
            c[i][j] = 0


    def worker(i: Int) {imm a, imm b, mut c, imm n}:
        var c_ptr = c[i].unsafe_ptr()

        for k in range(n):
            var a_val = a[i][k]
            var b_ptr = b[k].unsafe_ptr()



            var j = 0
            while j + simd_w <= n:
                var b_vec = b_ptr.unsafe_load[width=simd_w](j)
                var c_vec = c_ptr.unsafe_load[width=simd_w](j)
                c_vec += a_val * b_vec
                c_ptr.unsafe_store(j,c_vec)
                j += simd_w

            while j < n:
                c[i][j] += a_val * b[k][j]
                j += 1

    parallelize(worker,n)


    # def worker(id:Int){imm a,imm b,mut c,imm n,imm threads,imm row_per_thread}:
    #     var start=id * row_per_thread
    #     # var end=min(start+row_per_thread , n)
    #     var end = n if (id == threads-1) else start + row_per_thread

        

    #     for i in range(start,end):
    #         var c_ptr=c[i].unsafe_ptr()
    #         for k in range(n):
    #             var a_val = a[i][k]
    #             var b_ptr=b[k].unsafe_ptr()

    #             var j=0
    #             # for j in range(0,n,simd_w):
    #             while j +simd_w<=n:

    #                 var b_vec1=b_ptr.unsafe_load[width=simd_w](j)
    #                 var c_vec1=c_ptr.unsafe_load[width=simd_w](j)
    #                 c_vec1+=a_val*b_vec1
    #                 c_ptr.unsafe_store(j,c_vec1)

    #                 j+=simd_w


    #             while j<n:
    #                 c[i][j]+=a_val*b[k][j]
    #                 j+=1

    # sync_parallelize(worker,threads)
            

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


    print(c[0][0])
    print(c[n-1][n-1])