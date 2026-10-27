---
type: API
title: __ull2half_rna
description: 遵循CAST_ROUND模式，将uint64类型数据转换为half类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__ull2half_rna.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_ull2half\_rna

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_ROUND模式，将uint64类型数据转换为half类型数据，返回转换后的值。

## 函数原型

```cpp
inline half __ull2half_rna(const unsigned long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_ROUND模式转换成的half类型数据。本接口受全局饱和寄存器影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| 0 | 0 | 0 |
| ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 |
| x>ASCRT\_MAX\_NORMAL\_FP16 | inf | ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__ull2half_rna(half* dst, uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2half_rna(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ull2half_rna(__gm__ half* dst, __gm__ uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2half_rna(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 1, 2049, 4097, 65535
result: 0, 1, 2050, 4096, inf
```
