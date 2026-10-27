---
type: "Runbook"
title: TBuf Get 提前到搬运前致 kernel 间歇挂死
description: 重构时把 TBuf 的 LocalTensor Get<T>() 提前到 DataCopyPad/EnQue/DeQue 之前会破坏 TPipe 流程时序，连续路径 kernel 间歇性挂死；恢复 DeQue 之后获取并补 MTE2_S/V_S 同步后消除。
tags: [task.debugging, topic.memory, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-10-10T00:00:00Z'
platforms: ["950"]
aliases:
  - "改动 buffer 之后 kernel 间歇挂死"
  - "TBuf Get 顺序提前 挂死"
---

# TBuf Get 提前到搬运前致 kernel 间歇挂死

## 现象

在 Ascend950PR（CANN 9.1.0，arch35）上为 BenchNoHarnessForeachNorm 算子重构 kernel 的 chunk 处理循环（为支持非连续 Tensor 增加 stride 分支）后，ST Real 全量回归连续两轮超时：

- 第一轮 600 秒超时被杀，无任何用例输出；
- 第二轮 180 秒超时：L0_001–L0_007（连续 FP32 小张量用例）全部 PASS 后，L0_008（`{8,8,8,8}` 连续 FP32、4096 元素、p=2）打印用例头后**再无结果**——该用例由 fork 出的子进程执行，子进程卡在流同步等 kernel 完成，父进程 `waitpid` 永久阻塞。

L0_008 是纯连续用例（stride=1），与本次新增的非连续分支无关，说明挂死出现在公共路径上。该挂死具有**间歇性**：同版本二进制此前曾完整跑完一轮 74 用例。

## 适用场景

- 工程形态：AscendC kernel 开发 + ST Real NPU 回归（每用例 fork 子进程执行）。
- API 路径：`TQue::AllocTensor/EnQue/DeQue/FreeTensor`、`TBuf::Get<T>()`、`DataCopyPad`、`SetFlag/WaitFlag`（HardEvent::MTE2_S / V_S / S_V）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有使用 TQue+TBuf 混合缓冲的 AscendC kernel 重构。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：纯 TBuf（无 TQue 队列）的 kernel 未验证；挂死与 300 秒级套件耗时无关（见误诊）。

## 根因

重构时把 `workBuf0/workBuf1/resultBuf` 三个 TBuf 的 LocalTensor 获取（`Get<ComputeT>()`）从 `DeQue` 之后**提前到 `AllocTensor` 与 `DataCopyPad/EnQue/DeQue` 之间**，破坏了原有 TPipe 流程时序，连续路径出现间歇性 kernel 挂死（TPipe 内部机制未完全验证，属修复后未复现支撑的归因）。

## 解决配方

把 `Get<T>()` 调用移回各分支内、`DeQue` 之后，恢复原始顺序；同时为非连续分支补上标量读写前缺失的 `MTE2_S`/`V_S` 同步。

修复前（Get<T>() 在 DataCopyPad/EnQue/DeQue 之前，两分支共用）：

```cpp
                // CopyIn: GM -> UB
                LocalTensor<D_T_X> inputLocal = inQueue_.template AllocTensor<D_T_X>();
                LocalTensor<ComputeT> workBuf0 = workBuf0_.Get<ComputeT>();
                LocalTensor<ComputeT> workBuf1 = workBuf1_.Get<ComputeT>();
                LocalTensor<ComputeT> resultBuf = resultBuf_.Get<ComputeT>();

                if (tensorStride > 1) {
                    // 非连续 Tensor: 拷贝物理跨度 -> Cast 升 FP32 -> 标量提取有效元素
                    int64_t physicalOffset = chunkStart * tensorStride;
                    int64_t physicalCount = chunkElem * tensorStride;
                    DataCopyExtParams copyParams{1,
                        static_cast<uint32_t>(physicalCount * sizeof(D_T_X)), 0, 0, 0};
                    DataCopyPadExtParams<D_T_X> padParams{true, 0, 0, 0};
                    DataCopyPad(inputLocal, inTensorGM[physicalOffset], copyParams, padParams);
                    inQueue_.EnQue(inputLocal);
                    LocalTensor<D_T_X> inputData = inQueue_.template DeQue<D_T_X>();
                    // ... 标量提取与归约 ...
                } else {
                    // 连续 Tensor: 现有路径
                    DataCopyExtParams copyParams{1,
                        static_cast<uint32_t>(chunkElem * sizeof(D_T_X)), 0, 0, 0};
                    DataCopyPadExtParams<D_T_X> padParams{true, 0, 0, 0};
                    DataCopyPad(inputLocal, inTensorGM[chunkStart], copyParams, padParams);
                    inQueue_.EnQue(inputLocal);
                    LocalTensor<D_T_X> inputData = inQueue_.template DeQue<D_T_X>();
                    // ... Cast 与归约 ...
                }
```

修复后（Get<T>() 移入各分支 DeQue 之后；非连续分支补 MTE2_S/V_S 同步；连续路径恢复原始代码顺序）：

```cpp
                // CopyIn: GM -> UB
                LocalTensor<D_T_X> inputLocal = inQueue_.template AllocTensor<D_T_X>();

                if (tensorStride > 1) {
                    // 非连续 Tensor: 拷贝物理跨度 -> Cast 升 FP32 -> 标量提取有效元素
                    int64_t physicalOffset = chunkStart * tensorStride;
                    int64_t physicalCount = chunkElem * tensorStride;
                    DataCopyExtParams copyParams{1,
                        static_cast<uint32_t>(physicalCount * sizeof(D_T_X)), 0, 0, 0};
                    DataCopyPadExtParams<D_T_X> padParams{true, 0, 0, 0};
                    DataCopyPad(inputLocal, inTensorGM[physicalOffset], copyParams, padParams);
                    inQueue_.EnQue(inputLocal);
                    LocalTensor<D_T_X> inputData = inQueue_.template DeQue<D_T_X>();
                    LocalTensor<ComputeT> workBuf0 = workBuf0_.Get<ComputeT>();
                    LocalTensor<ComputeT> workBuf1 = workBuf1_.Get<ComputeT>();
                    LocalTensor<ComputeT> resultBuf = resultBuf_.Get<ComputeT>();

                    if constexpr (std::is_same_v<D_T_X, ComputeT>) {
                        // FP32: MTE2->S 同步后标量提取
                        event_t eventIDMTE2ToS = static_cast<event_t>(
                            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                        for (int64_t j = 0; j < chunkElem; j++) {
                            workBuf1.SetValue(j, inputData.GetValue(j * tensorStride));
                        }
                    } else {
                        // FP16/BF16: Cast 整个物理跨度到 FP32, 再标量提取
                        Cast(workBuf0, inputData, RoundMode::CAST_NONE, physicalCount);
                        PipeBarrier<PIPE_V>();
                        event_t eventIDVToS = static_cast<event_t>(
                            GetTPipePtr()->FetchEventID(HardEvent::V_S));
                        SetFlag<HardEvent::V_S>(eventIDVToS);
                        WaitFlag<HardEvent::V_S>(eventIDVToS);
                        for (int64_t j = 0; j < chunkElem; j++) {
                            workBuf1.SetValue(j, workBuf0.GetValue(j * tensorStride));
                        }
                    }
                    // S -> V 同步: 标量写完成后才能向量计算
                    event_t eventIDSToV = static_cast<event_t>(
                        GetTPipePtr()->FetchEventID(HardEvent::S_V));
                    SetFlag<HardEvent::S_V>(eventIDSToV);
                    WaitFlag<HardEvent::S_V>(eventIDSToV);

                    ProcessChunk(workBuf1, workBuf0, resultBuf, chunkElem);

                    event_t eventIDVToS2 = static_cast<event_t>(
                        GetTPipePtr()->FetchEventID(HardEvent::V_S));
                    SetFlag<HardEvent::V_S>(eventIDVToS2);
                    WaitFlag<HardEvent::V_S>(eventIDVToS2);
                    ComputeT chunkResult = resultBuf.GetValue(0);

                    if constexpr (NORM_TYPE == 3) {
                        if (chunkResult > partialResult) partialResult = chunkResult;
                    } else if constexpr (NORM_TYPE == 4) {
                        if (chunkResult < partialResult) partialResult = chunkResult;
                    } else {
                        partialResult += chunkResult;
                    }

                    event_t eventIDSToV2 = static_cast<event_t>(
                        GetTPipePtr()->FetchEventID(HardEvent::S_V));
                    SetFlag<HardEvent::S_V>(eventIDSToV2);
                    WaitFlag<HardEvent::S_V>(eventIDSToV2);
                    inQueue_.FreeTensor(inputData);
                } else {
                    // 连续 Tensor: 现有路径 (保持原始代码顺序)
                    DataCopyExtParams copyParams{1,
                        static_cast<uint32_t>(chunkElem * sizeof(D_T_X)), 0, 0, 0};
                    DataCopyPadExtParams<D_T_X> padParams{true, 0, 0, 0};
                    DataCopyPad(inputLocal, inTensorGM[chunkStart], copyParams, padParams);
                    inQueue_.EnQue(inputLocal);
                    LocalTensor<D_T_X> inputData = inQueue_.template DeQue<D_T_X>();
                    LocalTensor<ComputeT> workBuf0 = workBuf0_.Get<ComputeT>();
                    LocalTensor<ComputeT> workBuf1 = workBuf1_.Get<ComputeT>();
                    LocalTensor<ComputeT> resultBuf = resultBuf_.Get<ComputeT>();

                    if constexpr (std::is_same_v<D_T_X, ComputeT>) {
                        ProcessChunk(inputData, workBuf1, resultBuf, chunkElem);
                    } else {
                        Cast(workBuf0, inputData, RoundMode::CAST_NONE, chunkElem);
                        PipeBarrier<PIPE_V>();
                        ProcessChunk(workBuf0, workBuf1, resultBuf, chunkElem);
                    }

                    event_t eventIDVToS = static_cast<event_t>(
                        GetTPipePtr()->FetchEventID(HardEvent::V_S));
                    SetFlag<HardEvent::V_S>(eventIDVToS);
                    WaitFlag<HardEvent::V_S>(eventIDVToS);
                    ComputeT chunkResult = resultBuf.GetValue(0);

                    if constexpr (NORM_TYPE == 3) {
                        if (chunkResult > partialResult) partialResult = chunkResult;
                    } else if constexpr (NORM_TYPE == 4) {
                        if (chunkResult < partialResult) partialResult = chunkResult;
                    } else {
                        partialResult += chunkResult;
                    }

                    event_t eventIDSToV = static_cast<event_t>(
                        GetTPipePtr()->FetchEventID(HardEvent::S_V));
                    SetFlag<HardEvent::S_V>(eventIDSToV);
                    WaitFlag<HardEvent::S_V>(eventIDSToV);
                    inQueue_.FreeTensor(inputData);
                }
```

判别方法：挂死的形态特征是"某用例打印用例头后无 PASS/FAIL、子进程不退出、父进程 waitpid 阻塞、shell 超时杀进程"；对比同版本二进制历史运行可判断间歇性。

## 设计决策

- 修复选择"恢复原始调用顺序"而不是继续在提前 Get 的版本上调同步：原始顺序是此前已验证的稳定形态，回归风险最小。
- 顺带补齐非连续分支标量读写前的 `MTE2_S`（DataCopy 后标量读）与 `V_S`（Cast 后标量读）同步：`EnQue/DeQue` 只保证 MTE2→V，标量（S）读写需要显式事件同步（S 管线与 MTE 的依赖另见[标量 S pipe 误用 TQue 导致输出全常数卡](../precision/scalar_s_pipe_requires_explicit_mte_dependencies.md)）。

## 验证信号

来源说明：本卡结论来自该算子迭代三联调的多轮全量回归对比。

- 修复前：连续两轮全量运行挂死（600s 超时无输出；180s 超时卡在 L0_008，L0_001–L0_007 均 PASS）（来源平台：950；证据等级：tool_observed）。
- 修复后：三轮运行零挂死；其中 `timeout 300` 两轮超时属套件本身耗时（每用例独立 fork + ACL 初始化开销，全量需 >300 秒），`timeout 600` 一轮完整跑完 74 用例（通过 73，失败 1 为非连续用例，与本卡无关），确定性验证 3/3 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 归因机制（TPipe 时序）为推断，由"修复后多轮未复现"支撑，未追到硬件/驱动层；不能排除其他触发同症状的挂死源。
- 挂死具有间歇性（同版本二进制一次全量通过、一次挂死），"未复现"不等于"永不复现"。
- 其他 TPipe 配置（不同 BUFFER_NUM、不同 TBuf 组合）未验证。

**未决项**：

- 优先项：复现并追根因到机制层——TPipe 在 `Get<T>()` 早调用时挂死的内部行为（队列/事件状态机）未验证。
- 间歇性触发的条件（负载/时序窗口）未定位。
## 坏实践

- 重构 kernel 时"顺手"把 LocalTensor 声明提升到分支之外以减少重复代码——在 TQue+TBuf 混合场景下改变了缓冲获取与搬运指令的相对顺序。
- 把 300 秒级超时直接判为挂死复发：应先核对套件在正常情况下的单用例耗时（fork + aclInit/finalize 本身可达数秒每用例）。

## 误诊

- 曾把修复后的 300 秒超时当作挂死复发；核对输出进度与单用例耗时后确认是套件总耗时超过超时阈值，非挂死（输出进度为 tool_observed）。

## 相关卡片

- [PB-45 `TPipe::Reset()` 释放全局 `g_tpipeImpl` 事件池](pb_45_tpipe_reset_frees_the_global_g_tpipeimpl_event_poo.md)：同样表现为改 buffer 后挂死，但根因是 Init 期事件 id 预算耗尽而非 Get 顺序；若按本卡调整 Get/DeQue 顺序无效，应转去数事件 id，而不是继续试同步点。
- [arch3510 事件池 QUE_MAX_EVENT=8 的事实锚点](../performance/tile_merged_dma_eliminate_mte2_issue_bound.md)：判别「是不是超了预算」所需的数字（每核每事件类型 8，每方向独立位图）。
- [Reduce work buf 额外占 UB 致 507035](reduce_work_buf_ub_507035.md)：同族的**字节**预算形态——超限时崩在启动后而非挂死，可用来反向确认本次是挂死而非越界。
