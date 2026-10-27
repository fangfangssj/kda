---
type: API
title: DisableDmaAtomic
description: 关闭数据搬运随路原子操作，后续搬运时GM原始数据将被新数据完全覆盖。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.atomic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/DisableDmaAtomic.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_operator_set_atomic_intf.h
---

# DisableDmaAtomic<a name="ZH-CN_TOPIC_0000001839732381"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_set_atomic_intf.h"`。

关闭数据搬运随路原子操作功能，后续执行数据搬运时，GM中原始数据将被新搬运数据完全覆盖。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void DisableDmaAtomic()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

[SetAtomicAdd](set_atomic_add.md)、[SetAtomicMax](set_atomic_max_isasi.md)、[SetAtomicMin](set_atomic_min_isasi.md)使用完成后，通过DisableDmaAtomic清空原子操作的状态，以免影响后续相关指令功能。

## 调用示例<a name="section177231425115410"></a>

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
AscendC::LocalTensor<T> srcLocal = ubAllocator.Alloc<T, SIZE>();

AscendC::DisableDmaAtomic();
AscendC::DataCopy(srcLocal, srcGlobal, SIZE);
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
AscendC::SyncAll();

// 开启原子累加，将Unified Buffer（UB）数据原子累加到GM
AscendC::SetAtomicAdd<T>();
AscendC::DataCopy(dstGlobal, srcLocal, SIZE);

// 关闭原子累加
AscendC::DisableDmaAtomic();
```

完整样例请参考[DataMovementWithAtomicOperations样例](https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/06_atomic/data_movement_with_atomic_operations/data_movement_with_atomic_operations.asc)。
