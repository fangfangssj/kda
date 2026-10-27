---
type: API
title: __half2uint_rna
description: 遵循CAST_ROUND模式，将half类型数据转换为unsigned int类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__half2uint_rna.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_half2uint\_rna

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_ROUND模式，将half类型数据转换为unsigned int类型数据，返回转换后的值。

## 函数原型

```cpp
inline unsigned int __half2uint_rna(const half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_ROUND模式转换成的无符号整数。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0.5 | 1 |
| ±0 | 0 |
| nan | 0 |
| inf | 4294967295（uint32_t最大值） |
| -inf | 0 |
| ASCRT\_MAX\_NORMAL\_FP16 | 65504 |
| -ASCRT\_MAX\_NORMAL\_FP16 | 0 |
| ASCRT\_MIN\_DENORM\_FP16 | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__half2uint_rna( uint32_t* dst, half* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __half2uint_rna(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__half2uint_rna(__gm__ uint32_t* dst, __gm__ half* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __half2uint_rna(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -2.5, -1.5, 0.5, 1.5, 2.5, inf, -inf, nan
result: 0, 0, 1, 2, 3, 4294967295, 0, 0
```
