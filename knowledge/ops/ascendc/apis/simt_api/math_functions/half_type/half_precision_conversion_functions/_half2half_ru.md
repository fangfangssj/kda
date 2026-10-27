---
type: API
title: __half2half_ru
description: 将half类型数据遵循CAST_CEIL模式取整。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__half2half_ru.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_half2half\_ru

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将half类型数据遵循CAST\_CEIL模式取整。

## 函数原型

```cpp
inline half __half2half_ru(const half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_CEIL模式取整后的half类型数据。本接口受全局饱和寄存器的影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| ±0 | 0 | 0 |
| ASCRT\_MIN\_DENORM\_FP16 | 1 | 1 |
| nan | nan | 0 |
| inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__half2half_ru(half* dst, half* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __half2half_ru(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__half2half_ru(__gm__ half* dst, __gm__ half* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __half2half_ru(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -2, -0.5, 0.5, 2, nan
result: -2, 0, 1, 2, nan
```
