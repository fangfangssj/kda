---
type: API
title: AscThreadfenceBlock
description: Block范围内存栅栏接口，保证调用前的内存读写操作对同一线程块内其他线程可见，不阻塞线程，仅约束Block范围可见性。
tags:
- task.lookup
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/sync_and_memory_fence/memory_fence/asc_threadfence_block.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_sync_functions.h
---

# asc\_threadfence\_block

## 功能说明

头文件路径为：`"simt_api/device_sync_functions.h"`。

在同一Thread Block内，多个线程共享256KB的Unified Buffer（UB），线程间也会存在数据竞争问题。

与[asc\_threadfence](asc_threadfence.md)接口类似，`asc_threadfence_block()`函数用于保证当前线程的内存读写操作在同一Thread Block范围内的可见性顺序，确保某一线程在调用`asc_threadfence_block()`之前的所有内存读写操作对同一线程块内的其他线程可见。该函数不会阻塞当前线程的执行，仅建立Block范围内的内存操作可见性顺序约束。`asc_threadfence_block()`提供Block范围内的可见性保证，其实现仅需要确保Block内共享的缓存层级一致性，无需进行全局内存屏障操作。

## 函数原型

```cpp
inline void asc_threadfence_block()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_sync_functions.h"

    __global__ __launch_bounds__(1024) void kernel_thread_fence_block(float* dst, float* src)
    {
        if (threadIdx.x == 0) {
            src[0] = src[0] + 1;
            asc_threadfence_block();
        }
        if (threadIdx.x == 1) {
            dst[0] = src[0];
        }
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_sync_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_thread_fence_block(__gm__ float* dst, __gm__ float* src)
    {
        src[0] = src[0] + 1;
        asc_threadfence_block();
        dst[0] = src[0];
    }
    ```
