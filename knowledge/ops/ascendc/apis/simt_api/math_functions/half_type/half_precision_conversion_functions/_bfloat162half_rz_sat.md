---
type: API
title: __bfloat162half_rz_sat
description: 将bfloat16类型数据转换为half类型数据，并遵循CAST_TRUNC模式，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__bfloat162half_rz_sat.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_bfloat162half\_rz\_sat

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

将bfloat16类型数据转换为half类型数据，并遵循CAST\_TRUNC模式，返回转换后的值。

## 函数原型

```cpp
inline half __bfloat162half_rz_sat(const bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

将输入遵循CAST\_TRUNC模式转换成的half类型数据。本接口受全局饱和寄存器的影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| 0 | 0 | 0 |
| nan | nan | 0 |
| inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |
| x>ASCRT\_MAX\_NORMAL\_FP16 | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| x<-ASCRT\_MAX\_NORMAL\_FP16 | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

使用此接口前需将CTRL\[60\]寄存器设置为0，否则饱和模式不生效。设置方式请参见[控制饱和行为的方式](../../data_type_conversion/overview.md#section1194916101549)。

SIMT编程场景由于无法设置CTRL寄存器，本接口的饱和模式不生效。

## 调用示例

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__bfloat162half_rz_sat(__gm__ half* dst, __gm__ bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __bfloat162half_rz_sat(x[idx]);
    }
    ```

    输入输出示例如下：

    ```text
    x: 0, -0, 65536, 65536, 9.96920997e+37, inf, -inf, nan
    result: 0, -0, 65504, 65504, 65504, 65504, -65504, 0
    ```
