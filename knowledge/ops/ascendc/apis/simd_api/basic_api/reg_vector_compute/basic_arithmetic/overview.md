---
type: API
title: Reg矢量计算概述
description: 寄存器矢量基本算术运算接口概述，涵盖Abs、Exp、Add、Mul等单目、双目及标量双目运算。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/overview.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# 概述

**表 1**  单目运算

| 接口 | 描述 |
| --- | --- |
| [Abs](abs.md) | &bull; 对实数类型<br>对srcReg中的有效元素逐个取绝对值，并将结果写入dstReg对应位置，计算公式如下：$dstReg_i = \lvert srcReg_i \rvert$<br>&bull; 对复数类型<br>对srcReg中有效元素逐个取模，并将结果写入dstReg对应位置，计算公式如下：$dstReg_i = \lvert srcReg_i \rvert = (\alpha^2 + \beta^2)^{1/2}$（其中$srcReg_i = \alpha + \beta i$，α为复数的实部，β为复数的虚部） |
| [Exp](exp.md) | $dstReg_i = e^{srcReg_i}$ |
| [Ln](ln.md) | $dstReg_i = \ln(srcReg_i)$ |
| [Log](log.md) | $dstReg_i = \log(srcReg_i)$ |
| [Log2](log2.md) | $dstReg_i = \log_2(srcReg_i)$ |
| [Log10](log10.md) | $dstReg_i = \log_{10}(srcReg_i)$ |
| [Neg](neg.md) | $dstReg_i = -srcReg_i$ |
| [Relu](relu.md) | $dstReg_i = \max(0, srcReg_i)$ |
| [Sqrt](sqrt.md) | $dstReg_i = (srcReg_i)^{1/2}$ |

**表 2**  双目运算

| 接口 | 描述 |
| --- | --- |
| [Add](add.md) | $dstReg_i = srcReg0_i + srcReg1_i$ <br>$\{carry_i, dstReg_i\} = srcReg0_i + srcReg1_i$ |
| [AddC](add_c.md) | $\{carry_i, dstReg_i\} = srcReg0_i + srcReg1_i + carrySrc_i$ |
| [Div](div.md) | $dstReg_i = srcReg0_i / srcReg1_i$ |
| [Max](max.md) | $dstReg_i = \max(srcReg0_i, srcReg1_i)$ |
| [Min](min.md) | $dstReg_i = \min(srcReg0_i, srcReg1_i)$ |
| [Mul](mul.md) | $dstReg_i = srcReg0_i \times srcReg1_i$ |
| [Mull](mull.md) | $dstReg0_i = (srcReg0_i \times srcReg1_i) \bmod 2^{bit}$<br>$dstReg1_i = (srcReg0_i \times srcReg1_i) / 2^{bit}$<br>其中，bit表示操作数的位宽bit数。 |
| [Prelu](prelu.md) | $dstReg_i = \begin{cases} srcReg0_i & srcReg0_i > 0 \\ srcReg0_i \times srcReg1_i & srcReg0_i \leq 0 \end{cases}$ |
| [Sub](sub.md) | $dstReg_i = srcReg0_i - srcReg1_i$ <br>$\{carry_i, dstReg_i\} = \{1'b0, srcReg0_i\} + \{1'b0, \sim srcReg1_i\} + 33'b1$ |
| [SubC](sub_c.md) | $\{carry_i, dstReg_i\} = \{1'b0, srcReg0_i\} + \{1'b0, \sim srcReg1_i\} + \{32'b0,carrySrc_i\}$ |

**表 3**  标量双目运算

| 接口 | 描述 |
| --- | --- |
| [Adds](adds.md) | $dstReg_i = srcReg_i + scalarValue$ |
| [LeakyRelu](leaky_relu.md) | $dstReg_i = \begin{cases} srcReg_i & srcReg_i > 0 \\ srcReg_i \times scalarValue & srcReg_i \leq 0 \end{cases}$ |
| [Maxs](maxs.md) | $dstReg_i = \max(srcReg_i, scalarValue)$ |
| [Mins](mins.md) | $dstReg_i = \min(srcReg_i, scalarValue)$ |
| [Muls](muls.md) | $dstReg_i = srcReg_i \times scalarValue$ |
