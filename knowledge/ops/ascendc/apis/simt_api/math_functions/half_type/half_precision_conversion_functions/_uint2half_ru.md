---
type: API
title: __uint2half_ru
description: 遵循CAST_CEIL模式，将uint32类型数据转换为half类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__uint2half_ru.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_uint2half\_ru

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_CEIL模式，将uint32类型数据转换为half类型数据，返回转换后的值。

## 函数原型

```cpp
inline half __uint2half_ru(const unsigned int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_CEIL模式转换成的half类型数据。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 0 |
| x>ASCRT\_MAX\_NORMAL\_FP16 | inf |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__uint2half_ru(half* dst, uint32_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __uint2half_ru(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__uint2half_ru(__gm__ half* dst, __gm__ uint32_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __uint2half_ru(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 1, 2049, 4097, 65535
result: 0, 1, 2050, 4100, inf
```
