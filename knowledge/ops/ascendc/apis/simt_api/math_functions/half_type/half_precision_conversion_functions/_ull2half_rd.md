---
type: API
title: __ull2half_rd
description: 遵循CAST_FLOOR模式，将uint64类型数据转换为half类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__ull2half_rd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_ull2half\_rd

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_FLOOR模式，将uint64类型数据转换为half类型数据，返回转换后的值。

## 函数原型

```cpp
inline half __ull2half_rd(const unsigned long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_FLOOR模式转换成的half类型数据。本接口受全局饱和寄存器影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 2049 | 2048 | 2048 |
| 4097 | 4096 | 4096 |
| 8193 | 8192 | 8192 |
| ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 |
| 65505 | ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 |
| x≥65536 | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| 18446744073709551615（unsigned long long最大值） | inf | ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__ull2half_rd(half* dst, uint64_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __ull2half_rd(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ull2half_rd(__gm__ half* dst, __gm__ uint64_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __ull2half_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 1, 2049, 4097, 65535
result: 0, 1, 2048, 4096, 65504
```
