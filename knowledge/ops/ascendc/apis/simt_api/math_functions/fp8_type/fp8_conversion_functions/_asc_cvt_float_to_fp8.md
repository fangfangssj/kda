---
type: API
title: __asc_cvt_float_to_fp8
description: 将float类型数据按照CAST_RINT模式转换为指定类型（float8_e4m3_t和float8_e5m2_t）的8位浮点数，并根据指定的饱和模式（饱和或非饱和）进行溢出处理。转换结果以位级形式存储为__asc_fp8_storage_t类型，该类型为unsigned
  char，用于存储float8_e4m3_t或float8_e5m2_t类型的数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/fp8_type/fp8_conversion_functions/__asc_cvt_float_to_fp8.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp8.h
---

# \_\_asc\_cvt\_float\_to\_fp8

## 功能说明

头文件路径为：`"simt_api/asc_fp8.h"`。

将float类型数据按照CAST\_RINT模式转换为指定类型（`float8_e4m3_t`和`float8_e5m2_t`）的8位浮点数，并根据指定的饱和模式（饱和或非饱和）进行溢出处理。转换结果以位级形式存储为`__asc_fp8_storage_t`类型，该类型为unsigned char，用于存储`float8_e4m3_t`或`float8_e5m2_t`类型的数据。

## 函数原型

```cpp
inline __asc_fp8_storage_t __asc_cvt_float_to_fp8(const float x, const __asc_saturation_t saturation, const __asc_fp8_interpretation_t fp8_interpretation)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述                                                                                                                 |
| --- | --- |--------------------------------------------------------------------------------------------------------------------|
| x | 输入 | 源操作数。                                                                                                              |
| saturation | 输入 | 控制饱和行为，支持的取值为：__ASC_NOSAT、__ASC_SATFINITE。<br><br>__ASC_NOSAT表示使用非饱和模式，__ASC_SATFINITE表示使用饱和模式。                    |
| fp8_interpretation | 输入 | 指定转换类型，支持的取值为：__ASC_E4M3、__ASC_E5M2。<br><br>__ASC_E4M3表示转换为`float8_e4m3_t`格式的浮点数，__ASC_E5M2表示转换为`float8_e5m2_t`格式的浮点数。 |

## 返回值说明

输入遵循CAST\_RINT模式，根据指定的8位浮点数类型和指定的饱和模式，转换成的`__asc_fp8_storage_t`类型数据。本接口受全局饱和模式影响，特殊值如下：

- fp8_interpretation参数为__ASC_E4M3时：

    | 输入 | 非饱和模式返回值 | 饱和模式返回值 |
    |---|---|---|
    | 0 | 0 | 0 |
    | nan | nan | +0 |
    | ±inf | nan | ±448 |
    | 超出float8_e4m3_t范围的正值 | nan | 448 |
    | 超出float8_e4m3_t范围的负值 | nan | -448 |

- fp8_interpretation的参数为__ASC_E5M2时：

    | 输入 | 非饱和模式返回值 | 饱和模式返回值 |
    |---|---|---|
    | ±0 | ±0 | ±0 |
    | nan | nan | +0 |
    | ±inf | ±inf | ±57344 |
    | 超出float8_e5m2_t范围的正值 | inf | 57344 |
    | 超出float8_e5m2_t范围的负值 | -inf | -57344 |

## 约束说明

使用此接口前需将CTRL\[60\]寄存器设置为0，否则饱和模式不生效。设置方式请参见[控制饱和行为的方式](../../data_type_conversion/overview.md#section1194916101549)。

SIMT编程场景当前不支持使用该接口。

## 调用示例

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp8.h"

    __simt_vf__ __launch_bounds__(1024) inline void simt_asc_cvt_float_to_fp8(__gm__ float* input, __gm__ uint8_t* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= input_total_length) {
            return;
        }
        output[idx] = __asc_cvt_float_to_fp8(input[idx], __ASC_NOSAT, __ASC_E4M3);
    }
    __global__ __vector__ void cast_kernel(__gm__ float* input, __gm__ uint8_t* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_asc_cvt_float_to_fp8>(dim3(1024), input, output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x(float)：-1.5, 0.5, 600.0, inf
result(float8_e4m3_t)：-1.5, 0.5, nan, nan
```
