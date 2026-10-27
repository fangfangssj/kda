---
type: API
title: BitwiseNot
description: 逐比特对输入进行取反。 !
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/BitwiseNot_interface/BitwiseNot.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/adv_api/math/bitwise_not.h
---

# BitwiseNot

## 功能说明

逐比特对输入进行取反。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002477048097.png)

## 函数原型

```
template <const BitwiseNotConfig& config = DEFAULT_BITWISE_NOT_CONFIG, typename T>
__aicore__ inline void BitwiseNot(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| config | BitwiseNot算法的相关配置。此参数可选配，BitwiseNotConfig类型，具体定义如下方代码所示，其中参数的含义为：<br><br>isReuseSource：该参数预留，传入默认值false即可。 |
| T | 源操作数、目的操作数数据类型。支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。 |

```
struct BitwiseNotConfig {
    bool isReuseSource;
};
```

**表2**  参数说明

| 参数名称 | 类型 | 说明 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>数据类型需要与目的操作数保持一致。 |
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址偏移对齐要求请参见[通用说明和约束](../../../general_description_and_constraints.md)。

## 调用示例

```
AscendC::LocalTensor<uint32_t> dst, src;
uint32_t count = 512; // 参与计算的元素个数
AscendC::BitwiseNot(dst, src, count);
```

结果示例如下：

```
输入数据（src）:
[ 91, 48, -71, 112, 64, -107, 74, -121, -103, -50, 112, -23, 15, ... -74]
输出数据（dst）:
[-92, -49, 70, -113, -65, 106, -75, 120, 102, 49, -113, 22, -16, ... 73]
```
