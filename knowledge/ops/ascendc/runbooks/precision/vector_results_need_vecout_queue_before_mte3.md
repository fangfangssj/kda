---
type: "Runbook"
title: "Vector 结果搬出前须经 VECOUT 队列"
description: "Ascend C kernel 编译成功但输出出现首元素垃圾值和逐位置滞后时，检查 Vector 结果是否未经 VECOUT EnQue/DeQue 就被 DataCopyPad 搬到 GM。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.data_copy", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-17T00:58:06Z"
platforms: ["950"]
---

# Vector 结果搬出前须经 VECOUT 队列

## 现象

在编写一个累积极值 Ascend C kernel 的 `op_kernel/<op>_kernel.cpp` 并完成远程构建后，设备侧代表对拍没有报运行时错误，但 values 和 indices 在多种 dtype、多个扫描轴上同时错位：values 常保留前一轮前缀状态，indices 的首元素是未初始化样式的大整数，后续索引也呈一拍滞后。有限平局、Inf 和常规有限输入都能触发该模式，因此它不是某个特殊数值分支的公式误差。

当时 `CopyOutRow` 直接以 `tmpX_` 作为 `DataCopyPad` 源；indices 则先在 VECCALC buffer 上执行 `Cast`，随后立即由 `DataCopyPad` 搬出。代码没有用 VECOUT 队列交接 Vector 结果，也没有表达 Vector 与 MTE3 之间的事件依赖。

## 适用场景

- 工程形态：Ascend C kernel 直调，Vector 在 UB 中生成结果后由 MTE3 搬到 GM。
- API 路径：`Cast` / UB `DataCopy` → `TQue<VECOUT>` → `EnQue` / `DeQue` → `DataCopyPad`。
- 验证过的平台与 CANN 版本：Ascend 950，CANN 9.0.1。
- 适用粒度：`operator_family`，适用于把 VECCALC 结果或 Cast 结果跨流水交给 GM 搬出的 Vector kernel；具体 buffer 容量仍须按算子重新计算。
- 路线优先级：`recommended_fastpath`。在本开发会话的候选中，这是唯一既通过正确性对拍又使用标准队列事件语义的路线；这里不宣称相对其它正确同步实现有通用性能收益。
- 不适用 / 未验证：结果本来就由一个完整 VECOUT 队列生命周期生产时不要重复套队列；A2、A3、不同 CANN 版本及显式 event/barrier 的替代实现未验证。

## 根因

`tmpX_` 和 `tmpArg_` 的更新由 Vector 流水完成，`DataCopyPad` 搬出走 MTE3。直接把 VECCALC LocalTensor 或刚完成的 `Cast` 结果交给 MTE3，代码中没有一个队列事件让 MTE3 等待 Vector 写完成。于是 DMA 可能读取旧值或未初始化值，表现为首元素垃圾值和逐轮滞后，而不是稳定的算术误差。 修复的关键不只是把 buffer 的 position 改成 VECOUT，而是形成完整所有权与事件链：为本轮输出分配 VECOUT tensor，在 Vector 流水写入后 `EnQue`，由搬出侧 `DeQue` 后再 `DataCopyPad`，最后 `FreeTensor`。

## 解决配方

以下 before/after 是 trace 中 `CopyOutRow` 的函数级原样代码。

### before

代码证据：revision = 修复前 `old_string` 快照（无 Git commit，固定事件锚点仅存 manifest）；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbol = `KernelCummin<X>::CopyOutRow`；line_range = 函数完整体。

```cpp
    __aicore__ inline void CopyOutRow(int64_t m, int64_t n0, int64_t k,
                                      int64_t chunkLen)
    {
        // values 输出: 同 dtype, offset = m*R*N + k*N + n0
        int64_t outOffset = m * R_ * N_ + k * N_ + n0;
        AscendC::DataCopyExtParams copyParams{1,
            static_cast<uint32_t>(chunkLen * sizeof(X)), 0, 0, 0};
        AscendC::DataCopyPad(valuesGm_[outOffset], tmpX_, copyParams);

        // indices 输出: int32 → int64 Cast 后搬出
        AscendC::Cast(argOut_, tmpArg_, AscendC::RoundMode::CAST_NONE,
                      static_cast<uint32_t>(alignedCount_));
        AscendC::DataCopyExtParams idxParams{1,
            static_cast<uint32_t>(chunkLen * sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPad(indicesGm_[outOffset], argOut_, idxParams);
    }
```

### after

代码证据：revision = 同次修复的 `new_string` 快照（无 Git commit，固定事件锚点仅存 manifest）；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbol = `KernelCummin<X>::CopyOutRow`；line_range = 函数完整体。

```cpp
    __aicore__ inline void CopyOutRow(int64_t m, int64_t n0, int64_t k,
                                      int64_t chunkLen)
    {
        // values 输出: 同 dtype, offset = m*R*N + k*N + n0
        int64_t outOffset = m * R_ * N_ + k * N_ + n0;
        AscendC::DataCopyExtParams copyParams{1,
            static_cast<uint32_t>(chunkLen * sizeof(X)), 0, 0, 0};
        // 经 VECOUT TQue 的 EnQue/DeQue 插入 V→MTE3 与 MTE3→V 同步
        // (sqrt 样板: 输出经 outQueueZ EnQue/DeQue 后再 DataCopyPad 搬出).
        auto vOut = valuesOutQue_.template AllocTensor<X>();
        AscendC::DataCopy(vOut, tmpX_, alignedCount_);
        valuesOutQue_.EnQue(vOut);
        auto vD = valuesOutQue_.template DeQue<X>();
        AscendC::DataCopyPad(valuesGm_[outOffset], vD, copyParams);
        valuesOutQue_.FreeTensor(vD);

        // indices 输出: int32 → int64 Cast 后搬出 (Cast 是 V, 经 VECOUT 同步)
        AscendC::DataCopyExtParams idxParams{1,
            static_cast<uint32_t>(chunkLen * sizeof(int64_t)), 0, 0, 0};
        auto iOut = indicesOutQue_.template AllocTensor<int64_t>();
        AscendC::Cast(iOut, tmpArg_, AscendC::RoundMode::CAST_NONE,
                      static_cast<uint32_t>(alignedCount_));
        indicesOutQue_.EnQue(iOut);
        auto iD = indicesOutQue_.template DeQue<int64_t>();
        AscendC::DataCopyPad(indicesGm_[outOffset], iD, idxParams);
        indicesOutQue_.FreeTensor(iD);
    }
```

配套的 Init 片段也必须一起改变；以下同样是事件流原样代码，不能只复制函数而遗漏队列初始化。

代码证据：revision = 修复后的队列初始化 `new_string` 快照；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbol = `KernelCummin<X>::Init`；line_range = mask 与输出队列初始化至函数结尾。

```cpp
        // mask: 1 bit per element, 256B 对齐
        int64_t maskBytes = ALIGN_UP_BYTES((alignedCount_ + 7) / 8, ALIGN_BYTES);
        pipe_.InitBuffer(maskQue_,      1, maskBytes);
        // VECOUT 输出队列 (经 EnQue/DeQue 提供 V→MTE3 同步, 见 sqrt 样板)
        pipe_.InitBuffer(valuesOutQue_,  1, alignedCount_ * sizeof(X));
        pipe_.InitBuffer(indicesOutQue_, 1, alignedCount_ * sizeof(int64_t));

        // 一次性分配持久 LocalTensor (compute-only scratch, 跨行复用, 不 FreeTensor)
        tmpX_    = tmpXQue_.template AllocTensor<X>();
        tmpArg_  = tmpArgQue_.template AllocTensor<int32_t>();
        minDst_  = minDstQue_.template AllocTensor<X>();
        argDst_  = argDstQue_.template AllocTensor<int32_t>();
        kRow_    = kRowQue_.template AllocTensor<int32_t>();
        mask_    = maskQue_.template AllocTensor<uint8_t>();
    }
```

代码证据：revision = 修复后的类成员 `new_string` 快照；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbol = `KernelCummin<X>`；line_range = mask/output queue 与持久 LocalTensor 成员声明段。

```cpp
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> maskQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> valuesOutQue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> indicesOutQue_;
    // 持久 LocalTensor (Init 一次性 AllocTensor, 跨行复用)
    AscendC::LocalTensor<X> tmpX_;
    AscendC::LocalTensor<int32_t> tmpArg_;
    AscendC::LocalTensor<X> minDst_;
    AscendC::LocalTensor<int32_t> argDst_;
    AscendC::LocalTensor<int32_t> kRow_;
    AscendC::LocalTensor<uint8_t> mask_;
```

## 设计决策

1. 选择 VECOUT `TQue`，因为它同时表达 buffer 所有权与 Vector→MTE3 的流水依赖，修复后代码仍可由正常 pipeline 调度。
2. 放弃“直接从 VECCALC LocalTensor 搬到 GM”的路线：该路线已产生跨 dtype 的错位和未初始化索引。轨迹没有验证全流水 `PipeBarrier` 或手工 event 的替代方案，因此不把它们写成可选解法。
3. 当前路线标为 `recommended_fastpath`：它是本开发会话中唯一验证正确的标准队列路线；由于没有两条正确路线的性能对比，不外推绝对收益或跨平台最优性。

## 系统信息

- 目标：Ascend 950（设备名探测为 Ascend950PR_957c）。
- 工具链：CANN 9.0.1。
- 工程：kernel 直调；修改文件角色为 `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`。
- 涉及数据路径：VECCALC running value / running index → VECOUT queue → GM values / indices。

## 预防

- 对每个 `DataCopyPad(GM, LocalTensor, ...)` 反查 LocalTensor 的最近生产者属于哪条流水。
- 若最近生产者是 `Cast`、`Select`、`Compare` 后的 Vector 写，确认输出通过 VECOUT `EnQue` / `DeQue` 交接。
- 检查 `InitBuffer`、成员 `TQue<VECOUT>`、`AllocTensor`、`FreeTensor` 是否形成完整生命周期。
- 首元素为垃圾值、后续像前一轮结果时，优先做流水依赖审计，再改公式或 tiling。

## 验证信号

来源是一次 Ascend C 累积极值 kernel 直调开发中的设备对拍；修复前后的函数写入、构建输出和代表用例输出均在同一轨迹中可见。

- 修复前，远程运行在多种 dtype 与轴向上均显示 values/indices 错位，indices 首元素出现未初始化样式的大整数（来源平台：Ascend 950；证据等级：tool_observed）。
- 加入 VECOUT 队列后远程构建成功（来源平台：Ascend 950；证据等级：tool_observed）。
- 同轮八个非 NaN 代表检查中，values 与 indices 均逐项匹配 golden，覆盖浮点、整数、平局和 Inf 语义（来源平台：Ascend 950；证据等级：tool_observed）。

- 独立复现 R1：把 vector 计算结果经 VECOUT EnQue/DeQue 后再交给 MTE3，七组非 NaN values/indices 对拍均为 True，Inf 用例也为 True；NaN 用例数值列表与 golden 一致且 indices_eq=True，仅普通相等比较因 NaN 返回 False（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：避开把 VECIN tensor 直接作为 MTE3 输出源，改用 dataE.clone() 初始化并在 raw kernel 前同步后，构建成功，1D update/add/int32 与 2D dim0 定向输出均与 golden 一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本结论针对 Vector 生产、MTE3 消费的跨流水输出；若生产者和消费者在同一流水，依赖机制需重新判断。
- 队列深度和 buffer 大小必须纳入 UB 预算；不能机械复制容量。
- A2、A3 和其它 CANN 版本未验证，推断可迁移，但不得据此扩大 `platforms`。
- 轨迹没有验证显式 event 或 barrier 的正确性和性能，不能据此否定这些替代机制。

## 坏实践

before 中“Vector 写 LocalTensor 后立即 `DataCopyPad` 到 GM”的路径就是本轮被设备结果否定的坏实践。若 LocalTensor 的写入属于 Vector 流水而搬出属于 MTE3，这种写法没有建立跨流水依赖；“编译通过”不能证明搬出读到的是本轮结果。

## 误诊

- 把首元素垃圾值和一拍滞后当成累计公式或平局规则错误，会在 compute 上反复试错，却不改变异步搬出时序。
- 只把 buffer position 改成 VECOUT、但不形成 `AllocTensor → EnQue → DeQue → FreeTensor` 生命周期，也不能完整表达队列事件。