# 复合计算

- [asc_add_relu](asc_add_relu.md)：按元素求和，再进行Relu计算（结果和0对比取较大值），并提供转换最终结果的数据类型的功能(s162s8、f322f16、f162s8)。计算公式如下：
- [asc_axpy](asc_axpy.md)：源操作数src中每个元素与标量value求积后和目的操作数dst中的对应元素相加计算公式如下：
- [asc_deq_int162b8](asc_deq_int162b8.md)：将int16_t类型转换为int8_t或uint8_t类型，并将数据存放在每个DataBlock的高半块或低半块。使用该接口前需要调用asc_set_deq_scale接口设置量化参数。
- [asc_deq_int322half](asc_deq_int322half.md)：对输入的int32_t类型的数据按元素做量化并转换为half类型。计算公式如下： $$ dst_i = Cast(src_i * scale) $$ 其中的scale可通过asc_set_deq_scale设置。
- [asc_fma](asc_fma.md)：按元素将src0和src1相乘并和dst相加，将最终结果存放进dst中，计算公式如下： $$ dst_i = (src0_i * src1_i) + dst_i $$
- [asc_leakyrelu](asc_leakyrelu.md)：执行矢量Leaky Relu运算。计算公式如下：
- [asc_mul_add](asc_mul_add.md)：执行矢量的乘加运算。计算公式如下：
- [asc_mul_add_relu](asc_mul_add_relu.md)：按元素将src0和dst相乘并加上src1，再进行Relu计算（结果和0对比取较大值），最终结果存放进dst中。计算公式如下：
- [asc_mul_cast_half2int8](asc_mul_cast_half2int8.md)：按元素求积，并将结果转换为int8_t类型。计算公式如下： $$ dst_i = Cast(src0_i * src1_i) $$
- [asc_mul_cast_half2uint8](asc_mul_cast_half2uint8.md)：按元素求积，并将结果转换为uint8_t类型。计算公式如下： $$ dst_i = Cast(src0_i * src1_i) $$
- [asc_relu](asc_relu.md)：按元素做线性整流Relu，计算公式如下： $$ dst_i = relu(src_i) $$
- [asc_set_deq_scale](asc_set_deq_scale.md)：设置DEQSCALE（量化参数）寄存器的值。
- [asc_sub_relu](asc_sub_relu.md)：按元素求差，再进行Relu计算（结果和0对比取较大值），并提供转换最终结果的数据类型的功能(s162s8、f322f16、f162s8)。计算公式如下：
- [asc_vdeq_int162b8](asc_vdeq_int162b8.md)：将int16_t类型转化为int8_t或uint8_t类型，并将数据存放在每个DataBlock的上半块或下半块。使用该接口前需要调用asc_set_deq_scale设置量化参数。
