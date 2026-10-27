---
type: API
title: __cvta_global_to_generic
description: 将Global Memory内存空间的地址值转换为对应的指针并返回。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.data_movement.layout_transform
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/address_space_conversion_functions/__cvta_global_to_generic.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_cvta\_global\_to\_generic

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将Global Memory内存空间的地址值转换为对应的指针并返回。

## 函数原型

```cpp
void* __cvta_global_to_generic(size_t rawbits)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| rawbits | 输入 | 源操作数。 |

## 返回值说明

指向Global Memory内存空间的指针。该接口不校验输入是否为有效Global Memory地址。异常场景示例说明如下：

| 输入场景 | 返回值 |
| --- | --- |
| rawbits为0 | 返回指针使用__isGlobal()、__isUbuf()、__isLocal()均判断为0。 |
| rawbits为全1 | 返回指针使用__isGlobal()判断为1，但该指针不是可安全访问的Global Memory地址。 |

## 约束说明

SIMD与SIMT混合编程场景不支持使用该接口。

## 调用示例

SIMT编程场景：

```cpp
#include "simt_api/device_functions.h"

__global__ __launch_bounds__(1024) void kernel__cvta_global_to_generic(uint32_t* dst, uint64_t addr_val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t* ptr = static_cast<uint32_t*>(__cvta_global_to_generic(addr_val));
    dst[idx] = ptr[0];
}
```
