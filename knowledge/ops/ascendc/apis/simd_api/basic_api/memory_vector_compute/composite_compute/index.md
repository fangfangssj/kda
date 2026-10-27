# 复合计算

- [AbsSub](abs_sub_isasi.md)：将src0与src1相减再求绝对值，并将计算结果写入dst。
- [AddDeqRelu](add_deq_relu.md)：依次计算按元素求和、结果进行deq量化后再进行relu计算（结果和0对比取较大值）。需与SetDeqScale接口配合使用。
- [AddRelu](add_relu.md)：按元素求和，再进行Relu计算（结果和0对比取较大值）。
- [AddReluCast](add_relu_cast.md)：按元素求和，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。
- [Axpy](axpy.md)：计算向量和标量的乘积，并将乘积结果逐元素加到的输出向量上。
- [CastDequant](cast_dequant.md)：对输入做量化并进行精度转换，对输出结果的溢出采用饱和处理。需与SetDeqScale接口配合使用。
- [ExpSub](exp_sub_isasi.md)：将src0与src1相减，以差值作为指数计算自然常数e的幂次，并将结果写入dst。
- [FusedMulAdd](fused_mul_add.md)：按元素将src0和dst相乘并加上src1，最终结果存放入dst。
- [MulAddDst](mul_add_dst.md)：按元素将src0和src1相乘并和dst相加，将最终结果存放进dst中。
- [MulAddRelu](mul_add_relu.md)：按元素将src0和dst相乘并加上src1，再进行Relu计算（结果和0对比取较大值），最终结果存放进dst中。
- [MulCast](mul_cast.md)：按元素求积，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。
- [MulsCast](muls_cast_isasi.md)：将矢量源操作数前count个数据与标量相乘再按CAST_ROUND模式转换成half类型，结果写入dst，支持标量在前和标量在后两种场景。
- [复合计算概述](overview.md)：复合计算通常指的是在数学或编程中对复合函数或运算的处理。
- [SubRelu](sub_relu.md)：按元素求差，再进行Relu计算（结果和0对比取较大值）。
- [SubReluCast](sub_relu_cast.md)：按元素求差，再进行Relu计算，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。
