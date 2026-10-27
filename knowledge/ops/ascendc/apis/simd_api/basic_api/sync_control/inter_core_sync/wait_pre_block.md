---
type: API
title: WaitPreBlock
description: 通过读取全局内存判断前序核是否完成，阻塞当前核直至前序核通过NotifyNextBlock通知，与InitDetermineComputeWorkspace配合实现确定性计算。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/WaitPreBlock.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_determine_compute_sync_intf.h
---

# WaitPreBlock<a name="ZH-CN_TOPIC_0000001771590261"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_determine_compute_sync_intf.h"`。

等待当前AIV核的所有前序核完成本轮操作。接口持续读取全局内存（`gmWorkspace`）中的同步状态；当`blockIdx`小于当前核的所有AIV核均已完成本轮操作时，当前核结束等待并继续执行，从而确保多个AIV核严格按照`blockIdx`升序执行。

本接口须与[InitDetermineComputeWorkspace](init_determine_compute_workspace.md)和[NotifyNextBlock](notify_next_block.md)配合使用：在核函数（Kernel）开始时调用InitDetermineComputeWorkspace初始化同步状态，在需要顺序执行的操作前调用本接口，操作完成后调用NotifyNextBlock通知后序核。三个接口组合使用适用于要求[确定性计算](../../atomic_operations/key_features.md)的场景。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void WaitPreBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| gmWorkspace | 输入 | 多个AIV共享的全局内存，用于存储核间顺序执行的同步状态，位于Global Memory，类型为[GlobalTensor](../../data_structures/global_tensor/index.md)。本接口通过读取该内存判断当前核是否可以继续执行。 |
| ubWorkspace | 输入 | 当前AIV的核内缓存，用于将`gmWorkspace`中的核间同步状态搬运到Unified Buffer后读取，类型为[LocalTensor](../../data_structures/local_tensor/index.md)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 要实现确定性计算，需要保证每个核调用NotifyNextBlock/WaitPreBlock接口的次数相同。如果调用了NotifyNextBlock没有与之配对的WaitPreBlock，会导致非确定性计算、输出结果可能不符合预期；如果调用了WaitPreBlock没有与之配对的NotifyNextBlock，会导致程序卡死。
- 使用接口前，请确保已经调用[InitDetermineComputeWorkspace](init_determine_compute_workspace.md)接口，初始化共享内存（`gmWorkspace`）。
- [InitDetermineComputeWorkspace](init_determine_compute_workspace.md)、[WaitPreBlock](wait_pre_block.md)和[NotifyNextBlock](notify_next_block.md)组合使用时，三个接口的`gmWorkspace`必须指向同一块Global Memory，`ubWorkspace`必须指向当前AIV上的同一块Unified Buffer，通常应向三个接口传入同一组变量。否则，WaitPreBlock可能无法从`gmWorkspace`读取InitDetermineComputeWorkspace初始化或NotifyNextBlock写入的核间同步状态，NotifyNextBlock也可能无法从`ubWorkspace`读取InitDetermineComputeWorkspace初始化或前一次NotifyNextBlock更新的同步轮次状态，导致执行顺序错误或程序卡死。

- `gmWorkspace`和`ubWorkspace`申请的空间须满足以下公式：

$$
\begin{aligned}
\text{gmWorkspace空间大小} &\geq \mathrm{GetBlockNum}() \times 32\text{字节} \\
\text{ubWorkspace空间大小} &\geq (\mathrm{GetBlockNum}() + 1) \times 32\text{字节}
\end{aligned}
$$

- [分离模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下，WaitPreBlock和NotifyNextBlock仅对AIV核生效。在AIC核中调用任一接口时，接口直接返回，不执行等待、通知或核间同步。
- WaitPreBlock和NotifyNextBlock之间仅支持插入矢量计算相关指令，对矩阵计算相关指令不生效。
- 使用该接口进行多核控制时，算子调用时指定的逻辑AI Core核数numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致核函数（Kernel）“卡死”现象。

## 调用示例<a name="section177231425115410"></a>

完整样例请参考[InitDetermineComputeWorkspace调用示例](init_determine_compute_workspace.md#调用示例)。
