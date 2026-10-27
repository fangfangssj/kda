# bfloat16x2类型算术函数

- [__habsx2](_habsx2.md)：计算输入bfloat16x2_t类型数据各分量的绝对值。
- [__haddx2](_haddx2.md)：计算两个bfloat16x2_t类型数据各分量的相加结果，并遵循CAST_RINT模式对结果进行舍入。
- [__hcmadd](_hcmadd.md)：将三个bfloat16x2_t输入视为复数，第一个分量为实部，第二个分量为虚部，执行复数乘加运算x*y+z。
- [__hdivx2](_hdivx2.md)：计算两个bfloat16x2_t类型数据各分量的相除结果，并遵循CAST_RINT模式对结果进行舍入。
- [__hfmax2](_hfmax2.md)：计算输入bfloat16x2_t类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST_RINT模式对结果进行舍入。
- [__hfmax2_relu](_hfmax2_relu.md)：计算输入bfloat16x2_t类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST_RINT模式对结果进行舍入。负数结果置为0。
- [__hmulx2](_hmulx2.md)：计算两个bfloat16x2_t类型数据各分量的相乘结果，并遵循CAST_RINT模式对结果进行舍入。
- [__hnegx2](_hnegx2.md)：获取输入bfloat16x2_t类型数据各分量的负值。
- [__hsubx2](_hsubx2.md)：计算两个bfloat16x2_t类型数据各分量的相减结果，并遵循CAST_RINT模式对结果进行舍入。
