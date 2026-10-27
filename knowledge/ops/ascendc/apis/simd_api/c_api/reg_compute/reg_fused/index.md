# 复合计算

- [asc_abs_sub](asc_abs_sub.md)：根据mask对源操作数src0、src1按元素相减后取绝对值，得到计算结果。计算公式如下：
- [asc_axpy](asc_axpy.md)：根据mask将源操作数src中的每个元素与标量value求积后，与目的操作数dst中的对应元素相加，将结果写入dst。其中，dst既作为输入累加器参与计算，又用于存储最终结果。计算公式如下：
- [asc_exp_sub](asc_exp_sub.md)：根据mask将src0与src1按元素相减，并计算以差值为指数的自然指数函数。
- [asc_fma](asc_fma.md)：根据mask对源操作数src0、src1按元素相乘后，与源操作数src2中的对应元素相加，将计算结果作为返回值返回。计算公式如下：
- [asc_leakyrelu](asc_leakyrelu.md)：根据掩码对源矢量数据寄存器中的元素执行Leaky ReLU操作，并将结果通过函数返回值返回或写入目的矢量数据寄存器。源操作数中大于0的元素直接写入输出结果，小于或等于0的元素乘以标量立即数后写入输出结果。掩码对应位置为1的元素参与计算，为0的元素在输出结果中置零。计算公式如下：
- [asc_madd](asc_madd.md)：根据mask对目的操作数dst与源操作数src0按元素相乘后，与源操作数src1中的对应元素相加，将结果写入dst。其中，dst既作为输入参与计算，又用于存储最终结果。计算公式如下：
- [asc_mul_scalar_float2half_rn](asc_mul_scalar_float2half_rn.md)：根据mask将源操作数src（float类型）按元素与标量value相乘后，按照RINT舍入模式转换为half类型，得到计算结果。计算公式如下：
- [asc_mula](asc_mula.md)：根据mask对源操作数src0、src1按元素相乘后，与目的操作数dst中的对应元素相加，将结果写入dst。其中，dst既作为输入累加器参与计算，又用于存储最终结果。计算公式如下：
- [asc_prelu](asc_prelu.md)：根据掩码对源操作数矢量数据寄存器src0中的元素执行PReLU（Parametric ReLU）运算，负半轴斜率由矢量数据寄存器alpha中的对应元素指定，并将结果通过函数返回值返回或写入目的操作数矢量数据寄存器。src0大于0时直接写入输出结果，否则将src0与alpha相乘后写入输出结果。掩码对应位置为1的元素参与计算，为0的元素在输出结果中置零。计算公式如下：
- [asc_relu](asc_relu.md)：根据mask对源操作数src按元素执行ReLU操作，得到计算结果。计算公式如下：
