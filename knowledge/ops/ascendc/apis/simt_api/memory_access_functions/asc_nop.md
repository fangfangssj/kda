---
type: API
title: asc_nop（内存访问函数）
description: 本接口的功能为生成一条空操作指令，占用当前线程15个时钟周期，不执行任何实际计算与访存操作。
tags:
- task.lookup
- topic.data_movement.memory_access
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/memory_access_functions/asc_nop.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# asc\_nop

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

本接口的功能为生成一条空操作指令，占用当前线程15个时钟周期，不执行任何实际计算与访存操作。

## 函数原型

```cpp
inline void asc_nop()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

在生产者-消费者同步场景中，消费者通过轮询标记位来判断生成者是否完成数据的写入，`asc_nop`接口可用于降低轮询GM的频率，从而减少带宽压力。

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel_nop(uint32_t *data, uint32_t* flag)
    {
        ...
        while (asc_ldcg(flag) != 0) { // 轮询标记位
            asc_nop(); // 15个cycle后再发送下一个load指令
        }
        uint32_t x = *data; // 读取数据
        ...
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_nop(__gm__ uint32_t *data, __gm__ uint32_t* flag)
    {
        ...
        while (asc_ldcg(flag) != 0) { // 轮询标记位
            asc_nop(); // 15个cycle后再发送下一个load指令
        }
        uint32_t x = *data; // 读取数据
        ...
    }
    ```
