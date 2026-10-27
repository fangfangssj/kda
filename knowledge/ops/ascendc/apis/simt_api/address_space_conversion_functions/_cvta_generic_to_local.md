---
type: API
title: __cvta_generic_to_local
description: 将输入的指针转换为其指向的栈空间的地址值并返回。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.data_movement.layout_transform
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/address_space_conversion_functions/__cvta_generic_to_local.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_cvta\_generic\_to\_local

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将输入的指针转换为其指向的栈空间的地址值并返回。

## 函数原型

```cpp
size_t __cvta_generic_to_local(const void* ptr)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ptr | 输入 | 源操作数。 |

## 返回值说明

输入指针指向栈空间的地址值。
该接口不校验输入地址是否为可安全访问的有效地址。只有当`ptr`实际指向栈空间时，返回值才是有效的栈空间地址值，特殊场景说明如下：

| 输入场景 | 返回值 |
| --- | --- |
| `ptr`为Global Memory指针 | 未定义行为，返回值不是有效栈空间地址。 |
| `ptr`为`nullptr` | 返回`0x00000000fff00000`。 |
| `ptr`为UB指针 | 编译阶段报错。 |

## 约束说明

SIMD与SIMT混合编程场景不支持使用该接口。

## 调用示例

SIMT编程场景：

```cpp
#include "simt_api/device_functions.h"

__global__ __launch_bounds__(1024) void kernel__cvta_generic_to_local(uint32_t* src)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t local_data[2] = {src[idx], src[idx]};
    size_t addr_val = __cvta_generic_to_local(local_data);
    if (addr_val > 1152) {  // 1152:SIMT栈空间大小
        printf("栈(local_data)溢出，偏移量为：%lu\n", addr_val);
    } else {
        printf("栈(local_data)的偏移量为：%lu\n", addr_val);
    }
}
```
