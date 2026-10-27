---
type: API
title: Extract（排序操作）
description: 处理Sort的结果数据，输出排序后的value和index。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.ordering
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/sort_operations/Extract.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
---

# Extract

## 功能说明

处理Sort的结果数据，输出排序后的value和index。

## 函数原型

```
template <typename T>
__aicore__ inline void Extract(const LocalTensor<T>& dstValue, const LocalTensor<uint32_t>& dstIndex, const LocalTensor<T>& sorted, const int32_t repeatTime)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| dstValue | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>LocalTensor的起始地址需要32字节对齐。 |
| dstIndex | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>LocalTensor的起始地址需要32字节对齐。<br><br>此源操作数固定为uint32_t数据类型。 |
| sorted | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>LocalTensor的起始地址需要32字节对齐。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| repeatTime | 输入 | 重复迭代次数，int32_t类型。<br><br>取值范围：repeatTime∈[0,255]。 |

## 返回值说明

无

## 约束说明

-   操作数地址对齐要求请参见[通用地址对齐约束](../../general_description_and_constraints.md#section796754519912)。

## 调用示例

请参见[MrgSort](mrg_sort.md)的调用示例。
