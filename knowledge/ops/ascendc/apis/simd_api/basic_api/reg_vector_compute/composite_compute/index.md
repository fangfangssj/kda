# 复合计算

- [AbsSub](abs_sub.md)：将两个源操作数逐元素相减再求绝对值，结果写入目的操作数。
- [Axpy](axpy.md)：对标量与源操作数执行乘加操作，将标量乘积加到目的操作数上。
- [ExpSub](exp_sub.md)：将两个源操作数相减的差值作为e的指数进行计算，结果写入目的操作数。
- [MulAddDst](mul_add_dst.md)：将两个源操作数逐元素相乘再加到目的操作数上，结果写回目的操作数。
- [MulDstAdd](mul_dst_add.md)：将目的操作数与源操作数逐元素相乘再加另一个源操作数，结果写回目的操作数。
- [MulsCast](muls_cast.md)：将源操作数与标量相乘后按CAST_ROUND模式转换为half类型，结果写入目的操作数。
- [复合计算概述](overview.md)：复合计算支持以下接口：
