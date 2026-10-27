---
type: API
title: __ll2half_rz
description: 遵循CAST_TRUNC模式，将int64类型数据转换为half类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__ll2half_rz.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_ll2half\_rz

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_TRUNC模式，将int64类型数据转换为half类型数据，返回转换后的值。

## 函数原型

```cpp
inline half __ll2half_rz(const long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_TRUNC模式转换成的half类型数据。本接口受全局饱和寄存器影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 2049 | 2048 | 2048 |
| -2049 | -2048 | -2048 |
| ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 |
| 65505 | ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 |
| -65504 | -ASCRT\_MAX\_NORMAL\_FP16 | -ASCRT\_MAX\_NORMAL\_FP16 |
| -ASCRT\_MAX\_NORMAL\_FP16 | -ASCRT\_MAX\_NORMAL\_FP16 | -ASCRT\_MAX\_NORMAL\_FP16 |
| 9223372036854775807（long long最大值） | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -9223372036854775808（long long最小值） | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__ll2half_rz(half* dst, int64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ll2half_rz(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ll2half_rz(__gm__ half* dst, __gm__ int64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ll2half_rz(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 1, -1, 2049, 4097, -4097
result: 0, 1, -1, 2048, 4096, -4096
```
