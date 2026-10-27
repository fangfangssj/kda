---
type: API
title: InitDetermineComputeWorkspace
description: 基于核间顺序执行的确定性计算初始化接口，初始化GM共享内存后才能调用WaitPreBlock和NotifyNextBlock完成核间顺序执行。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/InitDetermineComputeWorkspace.md
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

# InitDetermineComputeWorkspace<a name="ZH-CN_TOPIC_0000001723825540"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_determine_compute_sync_intf.h"`。

InitDetermineComputeWorkspace是基于核间顺序执行的确定性计算的初始化配置接口，能够初始化共享内存（`gmWorkspace`）的值，完成初始化后才可以调用[WaitPreBlock](wait_pre_block.md)和[NotifyNextBlock](notify_next_block.md)。以上三个接口共同完成基于核间顺序执行的确定性计算，确定性计算的具体含义请参考[确定性计算](../../atomic_operations/key_features.md)。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void InitDetermineComputeWorkspace(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| :--- | :--- | :--- |
| gmWorkspace | 输入 | 多个AIV共享的全局内存，用于存储核间顺序执行的同步状态，位于Global Memory，类型为[GlobalTensor](../../data_structures/global_tensor/index.md)。本接口将其中的同步状态初始化为0。 |
| ubWorkspace | 输入 | 当前AIV的核内缓存，用于在Unified Buffer中暂存待写入`gmWorkspace`的核间同步状态，并保存后续接口使用的同步轮次状态，类型为[LocalTensor](../../data_structures/local_tensor/index.md)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 调用者无需预先初始化`gmWorkspace`和`ubWorkspace`。本接口会将`gmWorkspace`中的核间同步状态初始化为0，并将`ubWorkspace`中的同步轮次状态初始化为1。
- [InitDetermineComputeWorkspace](init_determine_compute_workspace.md)、[WaitPreBlock](wait_pre_block.md)和[NotifyNextBlock](notify_next_block.md)组合使用时，三个接口的`gmWorkspace`必须指向同一块Global Memory，`ubWorkspace`必须指向当前AIV上的同一块Unified Buffer，通常应向三个接口传入同一组变量。否则，WaitPreBlock可能无法从`gmWorkspace`读取InitDetermineComputeWorkspace初始化或NotifyNextBlock写入的核间同步状态，NotifyNextBlock也可能无法从`ubWorkspace`读取InitDetermineComputeWorkspace初始化或前一次NotifyNextBlock更新的同步轮次状态，导致执行顺序错误或程序卡死。
- `gmWorkspace`和`ubWorkspace`申请的空间须满足以下公式：

$$
\begin{aligned}
\text{gmWorkspace空间大小} &\geq \mathrm{GetBlockNum}() \times 32\text{字节} \\
\text{ubWorkspace空间大小} &\geq (\mathrm{GetBlockNum}() + 1) \times 32\text{字节}
\end{aligned}
$$

- [分离模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下，该接口仅对AIV核生效。在AIC核中调用时，接口直接返回，不初始化`gmWorkspace`和`ubWorkspace`。
- 使用该接口进行多核控制时，算子调用时指定的逻辑AI Core核数numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致核函数（Kernel）“卡死”现象。

## 调用示例<a name="section177231425115410"></a>

```cpp
    // 初始化共享内存（gmWorkspace）的同步状态，必须在核函数（Kernel）开始时首先调用。
    AscendC::InitDetermineComputeWorkspace(gmWorkspace, ubWorkspace);

    for(int64_t i = 0; i < tileNum; i++) {
        AscendC::LocalTensor<T> srcLocal = que.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal[i * tileCount], tileCount);

        // 等待前序核（blockIdx-1）完成操作。
        AscendC::WaitPreBlock(gmWorkspace, ubWorkspace);
        
        // 开启原子累加。
        AscendC::SetAtomicAdd<T>();
        AscendC::DataCopy(dstGlobal[i * tileCount], srcLocal, tileCount);
        AscendC::DisableDmaAtomic();
        
        // 通知后序核（blockIdx+1）当前核已完成。
        AscendC::NotifyNextBlock(gmWorkspace, ubWorkspace);
        que.FreeTensor(srcLocal);
    }
```

完整样例请参考[sequential\_block\_sync示例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/05_sync_control/sequential_block_sync)。
