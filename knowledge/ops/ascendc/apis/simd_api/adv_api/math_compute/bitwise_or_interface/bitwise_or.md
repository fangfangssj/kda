---
type: API
title: BitwiseOr
description: 逐比特对两个输入进行或操作。 !
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/BitwiseOr_interface/BitwiseOr.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/adv_api/math/bitwise_or.h
---

# BitwiseOr

## 功能说明

逐比特对两个输入进行或操作。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002442050326.png)

## 函数原型

```
template <const BitwiseOrConfig& config = DEFAULT_BITWISE_OR_CONFIG, typename T>
__aicore__ inline void BitwiseOr(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const uint32_t count)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| config | BitwiseOr算法的相关配置。此参数可选配，BitwiseOrConfig类型，具体定义如下方代码所示，其中参数的含义为：<br><br>isReuseSource：该参数预留，传入默认值false即可。 |
| T | 源操作数、目的操作数数据类型。支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。 |

```
struct BitwiseOrConfig {
    bool isReuseSource;
};
```

**表2**  参数说明

| 参数名称 | 类型 | 说明 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0、src1 | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>数据类型需要与目的操作数保持一致。 |
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址偏移对齐要求请参见[通用说明和约束](../../../general_description_and_constraints.md)。

## 调用示例

```
AscendC::LocalTensor<uint32_t> src0, src1;
AscendC::LocalTensor<uint32_t> dst;
uint32_t count = 512; // 参与计算的元素个数
AscendC::BitwiseOr(dst, src0, src1, count);
```

结果示例如下：

```
输入数据（src0）:
[-111, -83, 95, -52, ... -102]
输入数据（src1）:
[64, -44, 45, -117, ... -107]
输入数据（dst）:
[-47, -3, 127, -49, ... -97]
```
