---
type: API
title: __ull2float_rna
description: 遵循CAST_ROUND模式，将uint64类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__ull2float_rna.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_ull2float\_rna

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_ROUND模式，将uint64类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __ull2float_rna(const unsigned long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_ROUND模式转换成的浮点数。特殊值如下：

| 输入 | 返回值 |
|---|---|
| 0 | 0 |
| 16777217（2^24+1） | 16777218 |
| 33554431（2^25-1） | 33554432 |
| 33554434（2^25+2） | 33554436 |
| 18446744073709551615（ULLONG_MAX） | 1.84467441e+19 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__ull2float_rna(float* dst, uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2float_rna(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ull2float_rna(__gm__ float* dst, __gm__ uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2float_rna(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 4, 5, 16777220, 16777222, 4294967293, 9007199254740991, 9223372036854775805, 18446744073709551613
result: 4.0, 5.0, 16777220.0, 16777222.0, 4294967296.0, 9007199254740992.0, 9.22337204e+18, 1.84467441e+19
```
