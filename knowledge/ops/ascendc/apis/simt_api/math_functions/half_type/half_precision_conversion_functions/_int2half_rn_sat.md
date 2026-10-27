---
type: API
title: __int2half_rn_sat
description: 将int32类型数据按照CAST_RINT模式转换为half类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__int2half_rn_sat.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_int2half\_rn\_sat

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将int32类型数据按照CAST\_RINT模式转换为half类型数据，返回转换后的值。

## 函数原型

```cpp
inline half __int2half_rn_sat(const int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

遵循CAST\_RINT模式，将输入int32数据转换成的half数据。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 0 |
| x≥ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 |
| x≤-ASCRT\_MAX\_NORMAL\_FP16 | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

SIMT编程场景由于无法设置CTRL寄存器，本接口的饱和模式不生效。

## 调用示例

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__int2half_rn_sat(__gm__ half* dst, __gm__ int32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __int2half_rn_sat(x[idx]);
    }
    ```

    输入输出示例如下：

    ```text
    x: 0, 1, -1, 65504, 65535, 2147483647, -2147483648
    result: 0, 1, -1, 65504, 65504, 65504, -65504
    ```
