---
type: API
title: SetQuantScalar
description: 对输出矩阵所有值采用同一系数进行量化或反量化，量化参数shape为[1]，需在Iterate前调用。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetQuantScalar.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
---

# SetQuantScalar

## 功能说明

本接口提供对输出矩阵的所有值采用同一系数进行量化或反量化的功能，即整个C矩阵对应一个量化参数，量化参数的shape为\[1\]。

Matmul反量化场景：在Matmul计算时，左、右矩阵的输入为int8\_t或int4b\_t类型，输出为half类型；或者左、右矩阵的输入为int8\_t类型，输出为int8\_t类型。该场景下，输出C矩阵的数据从L0C Buffer（CO1）搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half或int8\_t类型。

Matmul量化场景：在Matmul计算时，左、右矩阵的输入为half或bfloat16\_t类型，输出为int8\_t类型。该场景下，输出C矩阵的数据从L0C Buffer（CO1）搬出到Global Memory时，会执行量化操作，将最终结果量化为int8\_t类型。

## 函数原型

```
__aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| quantScalar | 输入 | 量化或反量化系数。 |

将float数据类型的量化计算参数scale、offset转换为uint64类型的入参的计算公式如下：

1.  quantScalar为64位格式，初始为0。
2.  scale按bit位取高19位截断，存储于quantScalar的bit位32位处，并将46位修改为1。

    quantScalar = quantScalar ∣ \(scale & 0xFFFFE000\) ∣ \(1 << 46\)

3.  根据offset取值进行后续计算：
    -   若offset不存在，不再进行后续计算。
    -   若offset存在：
        1.  将offset值处理为int，范围为\[-256, 255\]。

            offset = Max\(Min\(INT\(Round\(offset\)\), 255\), 256\)

        2.  再将offset按bit位保留9位并存储于quantScalar的37到45位。

            quantScalar = \(quantScalar & 0x4000FFFFFFFF\) ∣ \(\(offset & 0x1FF\) << 37\)

## 返回值说明

无

## 约束说明

量化参数不能为inf/nan和非规格化数。

需与[SetDequantType](../matmul_tiling/set_dequant_type.md)保持一致。

本接口必须在[Iterate](iterate.md)或者[IterateAll](iterate_all.md)前调用。

## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
float tmp = 0.1; // 输出gm时会乘以0.1
// 将浮点值的量化或反量化系数，转换为uint64_t类型
uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
mm.SetQuantScalar(ans);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(biasGlobal);
}
mm.IterateAll(gm_c);
mm.End();
```

- [InplaceQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_quant_matmul_all_reduce_add_rms_norm.md)
- [QuantGroupedMatMulAlltoAllv 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_grouped_mat_mul_allto_allv_950.md)
