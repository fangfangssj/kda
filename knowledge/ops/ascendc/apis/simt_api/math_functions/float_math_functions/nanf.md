---
type: API
title: nanf
description: 根据输入的字符串生成一个float类型的安静NaN值（浮点尾数最高位为1），并将tagp解析值的低22位写入该NaN值的低22位。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/nanf.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# nanf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

根据输入的字符串生成一个float类型的安静NaN值（浮点尾数最高位为1），并将`tagp`解析值的低22位写入该NaN值的低22位。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/nanf.png)

## 函数原型

```cpp
inline float nanf(const char* tagp)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tagp | 输入 | 源操作字符串首地址。支持十进制、八进制和十六进制格式；字符串无法按支持的进制解析时按0处理。 |

## 返回值说明

返回float类型的安静NaN值。返回值的低22位为`tagp`解析值的低22位；`tagp`字符串无法按支持的进制解析时，低22位为0；`tagp`解析溢出时，低22位为全1。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(1024) void kernel_nanf(float* dst)
    {
        dst[0] = nanf("0x1A");
        dst[1] = nanf("123");
        dst[2] = nanf("abc");
        dst[3] = nanf("0");
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_nanf(__gm__ float* dst)
    {
        dst[0] = nanf("0x1A");
        dst[1] = nanf("123");
        dst[2] = nanf("abc");
        dst[3] = nanf("0");
    }
    ```

输入输出示例如下：

```text
tagp: "0x1A", "123", "abc", "0"
result: nan(0x7fc0001a), nan(0x7fc0007b), nan(0x7fc00000), nan(0x7fc00000)
```
