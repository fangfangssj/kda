---
type: API
title: BitwiseAnd
description: 逐比特对两个输入进行与操作。 !
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/BitwiseAnd_interface/BitwiseAnd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/adv_api/math/bitwise_and.h
---

# BitwiseAnd

## 功能说明

逐比特对两个输入进行与操作。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002386108598.png)

## 函数原型

```
template <const BitwiseAndConfig& config = DEFAULT_BITWISE_AND_CONFIG, typename T>
__aicore__ inline void BitwiseAnd(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const uint32_t count)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| config | BitwiseAnd算法的相关配置。此参数可选配，BitwiseAndConfig类型，具体定义如下方代码所示，其中参数的含义为：<br><br>isReuseSource：该参数预留，传入默认值false即可。 |
| T | 操作数的数据类型。支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。 |

```
struct BitwiseAndConfig {
    bool isReuseSource;
};
```

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0、src1 | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型与目的操作数保持一致。 |
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

## 调用示例

```
AscendC::LocalTensor<uint16_t> dst, src0, src1;
uint32_t count = 512; // 参与计算的元素个数
AscendC::BitwiseAnd(dst, src0, src1, count);
```

结果示例如下：

```
输入数据（src0）:
[93, 87, 99, 1, 87, 58, 6, 16, 85, 66, 56, 65, 24, 98, 96, 50, 18, 37, 0, ... 66]
输入数据（src1):
[81, 12, 33, 47, 47, 18, 82, 44, 91, 34, 17, 99, 7, 82, 42, 85, 88, 99, 61, ... 81]
输出数据（dst）:
[81, 4, 33, 1, 7, 18, 2, 0, 81, 2, 16, 65, 0, 66, 32, 16, 16, 33, 0, ... 64]
```
