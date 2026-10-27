---
type: API
title: ReadGmBypassDCache(ISASI)
description: 头文件路径为："basic_api/kernel_operator_scalar_intf.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/scalar_compute/ReadGmBypassDCache_ISASI.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_scalar_intf.h
---

# ReadGmBypassDCache\(ISASI\)

## 功能说明

头文件路径为：`"basic_api/kernel_operator_scalar_intf.h"`。

不经过DCache从GM地址上读数据。

## 函数原型

```cpp
template <typename T>
__aicore__ inline T ReadGmBypassDCache(__gm__ T* addr)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| addr | 输入 | 源GM地址。 |

## 数据类型

支持的数据类型为int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。

## 返回值说明

源GM地址上的数据。

## 约束说明

无

## 调用示例

```cpp
if (blockIdx == 0) {
    // 先写入被依赖数据，再通过DataSyncBarrier等待DDR访问完成。
    AscendC::WriteGmBypassDCache<T>(reinterpret_cast<__gm__ T *>(srcGm) + 1, DATA_VALUE);
    // DataSyncBarrier<DDR>阻塞后续GM写，确保上一条GM写对其他核可见。
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    // 最后写入同步标记，block 1读到该标记后即可安全读取srcGm[1]。
    AscendC::WriteGmBypassDCache<T>(reinterpret_cast<__gm__ T *>(srcGm), SYNC_FLAG);
}

if (blockIdx == 1) {
    while (true) {
        __gm__ T *addr = const_cast<__gm__ T *>(srcGlobal.GetPhyAddr());
        // 轮询GM第0个元素，等待block 0写入同步标记。
        T flagValue = AscendC::ReadGmBypassDCache<T>(addr);
        if (flagValue == SYNC_FLAG) {
            // DataSyncBarrier保证同步标记之前的srcGm[1]写入已完成。
            T dataValue = AscendC::ReadGmBypassDCache<T>(addr + 1);
            AscendC::WriteGmBypassDCache<T>(reinterpret_cast<__gm__ T *>(dstGm), 2 * dataValue);
            return;
        }
    }
}
```

完整样例请参考[DataSyncBarrier样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/05_sync_control/data_sync_barrier)。
