from layout.tile_layout import row_major, col_major
from layout import Coord,TileTensor,Idx
from std.collections import Array
from std.sys.info import simd_width_of

def main() raises:

    comptime simd_w=simd_width_of[DType.float32]()


    comptime rows = 4
    comptime cols = 4
    comptime layout = row_major[rows, cols]()



    var rows2:Int32=3
    var cols2:Int32=5
    var dynamic_layout=row_major(Coord(rows2,cols2))



    var mixed_layout=row_major((rows2,Idx[cols]))




    var storage = Array[Float32, rows * cols](uninitialized=True)
    for i in range(rows * cols):
        storage[i] = Float32(i)

    var tensor = TileTensor(storage, layout)





    print("Value at row 1, col 2:", tensor[1, 2])
    tensor[1, 2] = 99.0
    # print(tensor[1, 2])




    print(tensor)
    var vec = tensor.load[simd_w]((0, 0))
    vec = vec * 2.0
    tensor.store((0, 0), vec)
    print(tensor)




    var sub_tile = tensor.tile[2, 2](0, 1)

    print(sub_tile[0, 0])
    print(sub_tile[0, 1])

   
    sub_tile[0, 0] = 777.0
    print("tensor[0, 2]", tensor[0, 2])

    var vec_tensor = tensor.vectorize[1, 2]()
    print(vec_tensor)


    var dst_storage = Array[Float32, 16](fill=0.0)
    var dst_tensor = TileTensor(dst_storage, col_major[4, 4]())
    dst_tensor.copy_from(tensor)


