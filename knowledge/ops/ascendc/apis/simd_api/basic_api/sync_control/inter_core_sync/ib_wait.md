---
type: API
title: IBWait
description: 核间存在全局内存数据依赖时插入同步语句，等待核持续读取全局内存直至被等待核通过IBSet写入通知值，实现核间同步。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/IBWait.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_operator_block_sync_intf.h
---

# IBWait<a name="ZH-CN_TOPIC_0000001487056954"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_block_sync_intf.h"`。

等待核间同步标志。不同核操作同一块全局内存且存在读后写、写后读或写后写等数据依赖时，等待核（消费者）在访问依赖数据前调用本接口。本接口持续读取同步标志，直到其值为1时将其清零并返回，以保证被等待核（生产者）已经完成相关操作，避免因核间执行顺序不确定而产生数据读写错误。

本接口须与[IBSet](ib_set.md)成对使用。被等待核（生产者）在完成相关操作后调用IBSet，IBSet和IBWait用于同一个同步事件时，相关参数配置须保持一致。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void IBWait(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx, int32_t eventID)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| isAIVOnly | 控制同步标志在`gmWorkspace`中的存储位置计算方式。取值为true（默认值）时，按AIVOnly模式计算同步标志的存储位置；取值为false时，按MIX模式计算同步标志的存储位置。MIX模式下须显式配置为false。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gmWorkspace | 输入 | 多个AIV共享的全局内存，用于存储核间同步标志，位于Global Memory，类型为[GlobalTensor](../../data_structures/global_tensor/index.md)。首次进行核间同步前，必须将其中的核间同步标志初始化为0。 |
| ubWorkspace | 输入 | 当前AIV的核内缓存，用于将`gmWorkspace`中的核间同步标志搬运到Unified Buffer后读取和修改，并将修改后的同步标志写回`gmWorkspace`。<br>类型为[LocalTensor](../../data_structures/local_tensor/index.md)，支持的[TPosition](../../aux_data_structures/t_position.md)为VECIN/VECCALC/VECOUT。 |
| blockIdx | 输入 | 同步标志对应的逻辑AIV索引，即被等待核（生产者，调用IBSet的核）的逻辑AIV索引。取值范围由算子调用时设置的逻辑AI Core核数`numBlocks`和AIV配比决定，例如`__mix__(1, 2)`修饰核函数且`numBlocks`为4时，`blockIdx`取值范围为`[0, 7]`。 |
| eventID | 输入 | 同步事件的逻辑编号，与`blockIdx`共同确定`gmWorkspace`中的同步标志，用于区分同一对生产者和消费者之间的不同软件同步事件。<br>该参数与[SetFlag/WaitFlag](../intra_core_sync/set_flag_wait_flag_isasi.md)的入参名称相同，但含义不同：本接口的`eventID`用于定位软件同步标志，SetFlag/WaitFlag的`eventID`用于硬件指令中的核内流水同步事件。<br>因此，本接口的`eventID`不适用SetFlag/WaitFlag中0-7的取值限制。取值为非负整数，建议从0开始连续编号。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 首次调用IBSet和IBWait进行核间同步前，必须将共享的`gmWorkspace`中的核间同步标志初始化为0：
    - 若同步标志的初始值为1，IBWait可能将其误判为生产者已完成操作并提前结束等待，导致同步结果错误；
    - 若初始值为其他非预期值，IBSet和IBWait可能持续等待，导致程序卡死。

- `gmWorkspace`申请的空间须满足以下公式：

    $$
    \text{gmWorkspace空间大小} \geq (\mathrm{eventStride} \times \mathrm{eventID\_max} + \mathrm{blockIdx\_max} + 1) \times 32\text{字节}
    $$

    - `eventID_max`和`blockIdx_max`分别表示共享同一`gmWorkspace`的所有IBSet和IBWait调用中，`eventID`和`blockIdx`可能取到的最大值。`blockIdx_max`不一定等于`blockIdx`合法范围的上限，例如`__mix__(1, 2)`修饰核函数且`numBlocks`为4时，`blockIdx`合法范围为`[0, 7]`；若共享同一`gmWorkspace`的所有调用中只使用`blockIdx = 0`和`blockIdx = 1`，则`blockIdx_max = 1`。
    - `eventStride`的取值与模板参数isAIVOnly有关：
        - 当`isAIVOnly`为true时，`eventStride = GetBlockNum()`;
        - 当`isAIVOnly`为false时，`eventStride = GetBlockNum() × 2`，与AIC:AIV的配比无关。

    以下以AIVOnly模式下的多事件同步场景为例，说明上述参数取值和`gmWorkspace`空间大小的计算方法。

    AIVOnly模式下`GetBlockNum() = 4`，核0作为生产者分别通过`IBSet(sync_gm, sync_buf, 0, 0)`和`IBSet(sync_gm, sync_buf, 0, 1)`通知两个不同同步事件，核2作为生产者通过`IBSet(sync_gm, sync_buf, 2, 0)`通知另一个同步事件。对应的IBWait必须分别传入相同的`blockIdx`和`eventID`，例如`IBWait(sync_gm, sync_buf, 0, 1)`只能等待`IBSet(sync_gm, sync_buf, 0, 1)`设置的同步标志；如果误传为`blockIdx = 1`或`eventID = 0`，会访问另一处同步标志并可能卡死。该场景中`blockIdx_max = 2`、`eventID_max = 1`、`eventStride = 4`，因此`gmWorkspace`至少需要`(4 × 1 + 2 + 1) × 32 = 224`字节。
- ubWorkspace申请的空间最少要求为：32Bytes。
- 使用该接口进行多核控制时，算子调用时指定的逻辑AI Core核数numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致核函数（Kernel）“卡死”现象。
- IBSet和IBWait配对使用时，除了ubWorkspace其余所有参数都必须相同，否则程序会在IBWait处卡死。
- [分离模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下，该接口仅对AIV核生效。在AIC核中调用时，接口直接返回，不等待同步标志，也不执行核间同步。

## 调用示例<a name="section177231425115410"></a>

完整样例请参考[IBSet调用示例](ib_set.md#调用示例)。
