---
type: API
title: Hrint
description: 获取与输入bfloat16_t数据最接近的整数，若存在两个同样接近的整数则取偶数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_math_functions/hrint.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# hrint

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。

## 函数原型

```cpp
inline bfloat16_t hrint(bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

与输入浮点数最接近的整数值。特别场景说明如下：本接口受全局饱和模式影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0.5 | 0 | 0 |
| 1.5 | 2 | 2 |
| -100.5 | -100 | -100 |
| inf | inf | ASCRT_MAX_NORMAL_BF16 |
| -inf | -inf | -ASCRT_MAX_NORMAL_BF16 |
| nan | nan | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelRint(bfloat16_t* dst, bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hrint(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelRint(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hrint(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.5
result: 0.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 2.0
```
