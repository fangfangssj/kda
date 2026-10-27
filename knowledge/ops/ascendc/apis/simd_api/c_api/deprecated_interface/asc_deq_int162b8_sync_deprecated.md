---
type: API
title: asc_deq_int162b8_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_deq_int162b8_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_deq_int162b8_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_deq_int162b8_h_sync`和`asc_deq_int162b8_l_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_fused/asc_deq_int162b8.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

将int16_t类型转换为int8_t或uint8_t类型，并将数据存放在每个DataBlock的高半块或低半块。使用该接口前需要调用[asc_set_deq_scale](../vector_compute/vector_fused/asc_set_deq_scale.md)接口设置量化参数。

- asc_deq_int162b8_h：将数据存放在每个DataBlock的高半块。
- asc_deq_int162b8_l：将数据存放在每个DataBlock的低半块。

如下图所示：
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/c_api/figures/asc_deq_int162b8_halfblock.png)

计算公式如下：

$$
dst_i = (src_i * scale) + offset
$$

## 函数原型

- 同步计算

    ```cpp
    __aicore__ inline void asc_deq_int162b8_h_sync(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    __aicore__ inline void asc_deq_int162b8_h_sync(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    __aicore__ inline void asc_deq_int162b8_l_sync(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    __aicore__ inline void asc_deq_int162b8_l_sync(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, uint32_t count)
    ```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。

## 调用示例

```cpp
constexpr uint64_t total_length = 128;    // total_length指参与计算的数据长度
__ubuf__ int16_t src[total_length];
__ubuf__ int8_t dst[total_length];
float scale = 1.0;        // 量化参数为1
int16_t offset = 0;       // 不带偏移
bool sign_mode = true;    // 量化结果带符号（dst为int8_t类型）
asc_set_deq_scale(scale, offset, sign_mode);    // 计算公式为dst = src
asc_deq_int162b8_h_sync(dst, src, total_length);    // 将src转换为int8_t类型并存放在dst每个DataBlock的高半块
```
