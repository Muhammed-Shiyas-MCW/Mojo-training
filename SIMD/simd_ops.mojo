def main():
  var a = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
  print("a: ",a)
  var b = SIMD[DType.float32, 8](2.0)
  print("b: ",b)
  var zeros = SIMD[DType.float32, 8]()
  print("zeros: ",zeros)

  var math_add = a + b
  print("addition: ",math_add)
  var math_mul = a * b
  print("multiplication: ",math_mul)
  var math_div = a / b
  print("division: ",math_div)
  var math_sub = a - b
  print("subtraction: ",math_sub)

  var fma_result = a.fma(b, SIMD[DType.float32, 8](10.0))
  print("fma_result: ",fma_result)

  var clamped = a.clamp(SIMD[DType.float32, 8](3.0), SIMD[DType.float32, 8](6.0))
  print("clamped: ",clamped)



  var data = SIMD[DType.float32, 8](-5.0, 12.0, -1.5, 0.0, 8.5, -9.0, 4.0, -0.1)
  print("data: ",data)

  var is_positive = data.gt(0.0)
  print("is_positive: ",is_positive)

  var relu_out=is_positive.select(data,SIMD[DType.float32,8](0.0))
  print("relu_out: ",relu_out)

  var leaky_relu = is_positive.select(data, data * 0.1)
  print("leaky_relu :", leaky_relu)


  var data2 = SIMD[DType.float32, 4](10.0, 25.0, 5.0, 40.0)
  print("data2: ",data2)
  print("reduce_add():  ", data2.reduce_add())
  print("reduce_mul():  ", data2.reduce_mul())
  print("reduce_max():  ", data2.reduce_max())
  print("reduce_min():  ", data2.reduce_min())


  var any_gt_20 = data2.gt(20.0).reduce_or()
  var all_gt_0  = data2.gt(0.0).reduce_and()

  print("any_gt_20: ",any_gt_20)
  print("all_gt_0: ",all_gt_0)


  var v_a = SIMD[DType.int32, 4](1, 2, 3, 4)
  var v_b = SIMD[DType.int32, 4](10, 20, 30, 40)
  print("v_a: ", v_a)
  print("v_b: ", v_b)

  var v_slice = v_a.slice[2, offset=1]()
  print("v_slice: ", v_slice)

  var joined = v_a.join(v_b)
  print("joined: ", joined)

  var interleaved = v_a.interleave(v_b)
  print("interleaved: ", interleaved)