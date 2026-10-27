---
type: API
title: LogicalXor
description: 按元素进行逻辑异或操作。当输入的数据类型不是bool时，零被视为False，非零数据被视为True。 !
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/math_compute/LogicalXor_interface/LogicalXor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/adv_api/math/logical_xor.h
---

# LogicalXor

## 功能说明

按元素进行逻辑异或操作。当输入的数据类型不是bool时，零被视为False，非零数据被视为True。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002475452485.png)

## 函数原型

```
template <const LogicalXorConfig& config = DEFAULT_LOGICAL_XOR_CONFIG, typename T, typename U>
__aicore__ inline void LogicalXor(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1, const uint32_t count)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| config | LogicalXor算法的相关配置。此参数可选配，LogicalXorConfig类型，具体定义如下方代码所示，其中参数的含义是：<br><br>isReuseSource：该参数预留，传入默认值false即可。 |
| T | 目的操作数的数据类型。支持的数据类型为：bool。 |
| U | 源操作数的数据类型。支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。 |

```
struct LogicalXorConfig {
    bool isReuseSource;
};
```

**表2**  参数说明

| 参数名称 | 类型 | 说明 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0、src1 | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 约束说明

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址偏移对齐要求请参见[通用说明和约束](../../../general_description_and_constraints.md)。

## 调用示例

```
AscendC::LocalTensor<uint32_t> src0, src1;
AscendC::LocalTensor<bool> dst;
uint32_t count = 512; // 参与计算的元素个数
AscendC::LogicalXor(dst, src0, src1, count);
```

结果示例如下：

```
输入数据（src0）:
[-4, 4, 1, -8, 0, ... 3]
输入数据（src1）:
[-1, 3, 1, 5, 1, ... -3]
输出数据（dst）:
[False, False, False, False, True, ... False]
```
