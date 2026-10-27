---
type: API
title: min（整数数学函数）
description: 获取两个输入数据中的最小值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/min.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# min

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取两个输入数据中的最小值，当入参为一个有符号整数、一个无符号整数时，有符号整数会先被转换为对应的无符号类型，再进行大小比较。

## 函数原型

```cpp
long long min(long long x, long long y)
```

```cpp
long min(long x, long y)
```

```cpp
int min(int x, int y)
```

```cpp
short min(short x, short y)
```

```cpp
char min(char x, char y)
```

```cpp
unsigned long long min(unsigned long long x, unsigned long long y)
```

```cpp
unsigned long min(unsigned long x, unsigned long y)
```

```cpp
unsigned int min(unsigned int x, unsigned int y)
```

```cpp
unsigned short min(unsigned short x, unsigned short y)
```

```cpp
unsigned char min(unsigned char x, unsigned char y)
```

```cpp
unsigned int min(unsigned int x, int y)
```

```cpp
unsigned int min(int x, unsigned int y)
```

```cpp
unsigned long int min(long int x, unsigned long int y)
```

```cpp
unsigned long int min(unsigned long int x, long int y)
```

```cpp
unsigned long long int min(long long int x, unsigned long long int y)
```

```cpp
unsigned long long int min(unsigned long long int x, long long int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据中的最小值。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(1024) void KernelMin(long long* dst, long long* x, long long* y, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = min(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelMin(__gm__ long long* dst, __gm__ long long* x, __gm__ long long* y, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = min(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -12, 0, 23, -7
y: -20, 0, 15, 7
result: -20, 0, 15, -7
```
