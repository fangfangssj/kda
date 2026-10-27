---
type: API
title: CeilDivision
description: 计算两个整数num1和num2相除后向上取整结果。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/math_functions/CeilDivision.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
---

# CeilDivision

## 功能说明

计算两个整数num1和num2相除后向上取整结果。

## 函数原型

```cpp
__aicore__ constexpr inline int32_t CeilDivision(int32_t num1, int32_t num2)
```

## 参数说明

**表1** 参数说明

| 参数名 | 描述 |
| --- | --- |
| num1 | 参数1，被除数。 |
| num2 | 参数2，除数。 |

## 返回值说明

两个整数相除的向上取整结果。

## 约束说明

- 当num2为0时，结果为0。
- 该接口仅支持在num1和num2为正数场景下使用。

## 调用示例

本示例中使用CeilDivision计算迭代次数repeatTimes，通过对数据量count与单次处理数据量进行向上取整除法，确保所有数据（包括尾块）均被完整处理。

```cpp
template <typename T>
__aicore__ inline void AddCustomImpl(__local_mem__ T *dst, __local_mem__ T *src0, __local_mem__ T *src1,
    uint32_t count)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(T);
    uint16_t repeatTime = AscendC::CeilDivision(count, oneRepeatSize);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0 + i * oneRepeatSize );
        AscendC::Reg::LoadAlign(srcReg1, src1 + i * oneRepeatSize );
        AscendC::Reg::Add(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dst + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [Xlog1py 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/xlog1py_950.md)
- [BlockSparseAttention 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/block_sparse_attention_950.md)
