---
type: API
title: DataSyncBarrier
description: 阻塞后续指令执行，直到所有之前的内存访问指令执行结束，可通过模板参数控制需要等待的内存位置，用于解决Scalar的内存依赖。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/DataSyncBarrier_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_block_sync_intf.h
---

# DataSyncBarrier\(ISASI\)<a name="ZH-CN_TOPIC_0000001787873298"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_block_sync_intf.h"`。

阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束，用于解决Scalar的内存依赖。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| arg0 | 模板参数，表示需要等待的内存位置，类型为MemDsbT，可取值为：<br>&bull; ALL，等待GM和UB的访问指令（不包括其他内存位置，例如L0C Buffer、L1 Buffer）。<br>&bull; DDR，等待GM访问指令。<br>&bull; UB，等待UB访问指令。<br>&bull; SEQ，等待SEQ访问指令。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

等待的内存位置取值为ALL时，只能等待GM和Unified Buffer（UB）的访问指令，不包括其他内存位置，例如L0C Buffer、L1 Buffer。

## 调用示例<a name="section837496171220"></a>

```cpp
    if (blockIdx == 0) {
        // 先写入被依赖数据，再通过DataSyncBarrier等待GM访问完成。
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

完整样例请参考[DataSyncBarrier接口样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/05_sync_control/data_sync_barrier)。

- [DepthwiseConv2D 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/conv/depthwise_conv_2d_950.md)
- [GatherNd 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/gather_nd_950.md)
- [StridedSlice 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/strided_slice_950.md)
- [InplaceScatterUpdate 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/inplace_scatter_update_950.md)
- [ScatterNd 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/scatter_nd_950.md)
- [ScatterNdUpdate 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/scatter_nd_update_950.md)
- [AvgPool2d 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/pooling/avg_pool2d_950.md)
- [AvgPool3d 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/pooling/avg_pool3d_950.md)
- [Gather 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/index/gather_950.md)
- [Scatter 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/index/scatter_950.md)
- [AdaptiveAvgPool3D 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/pooling/adaptive_avg_pool_3d_950.md)
