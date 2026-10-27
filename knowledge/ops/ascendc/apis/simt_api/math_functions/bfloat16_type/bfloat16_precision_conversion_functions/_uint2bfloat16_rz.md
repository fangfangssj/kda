---
type: API
title: __uint2bfloat16_rz
description: 遵循CAST_TRUNC模式，将uint32类型数据转换为bfloat16类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__uint2bfloat16_rz.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_uint2bfloat16\_rz

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_TRUNC模式，将uint32类型数据转换为bfloat16类型数据，返回转换后的值。

## 函数原型

```cpp
inline bfloat16_t __uint2bfloat16_rz(const unsigned int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_TRUNC模式转换成的bfloat16类型数据。特殊值如下：

| 输入 | 返回值 |
|---|---|
| 0 | 0 |
| 257 | 256 |
| 514 | 512 |
| 1002 | 1000 |
| 4294967295（UINT32_MAX） | 4278190080 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__uint2bfloat16_rz(bfloat16_t* dst, uint32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __uint2bfloat16_rz(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__uint2bfloat16_rz(__gm__ bfloat16_t* dst, __gm__ uint32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __uint2bfloat16_rz(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 129, 257, 258, 259, 100000, 1000000, 4294967295, 0
result: 129.0, 256.0, 258.0, 258.0, 99840.0, 999424.0, 4278190080.0, 0.0
```
