---
type: "Runbook"
title: "GroupTailA 分核：合并分发+tail-A 助手复用"
description: 非末轴归约 Group 分支 kernel 实现配方：APT 把新 tilingKey 与同 pattern 已实现 key 合并分发到同一 Kernel 模板，ProcessGroup 按 isTailR 编译期分发，Phase1TailA/Phase2TailA 复用 Normal 分支的转置装入/归约助手与 Group 分支的 rescale 合并结构。
tags: [task.implementation, topic.activation, topic.compute.reduction, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# GroupTailA 分核：合并分发+tail-A 助手复用

## 现象

为 softmax 类算子实现最后一个计算分支（tilingKey=4 GroupTailA：合轴后 R 外 A 内的 RA 布局、A 维不满核、R 维有并行度）时，已有的两个"半边"参考实现分别是：key=3 GroupTailR（group 2D 分核骨架：Phase1 部分归约写 GM + SyncAll + Phase2 合并重算，但用 AR 布局助手）和 key=1 NormalTailA（tail-A 的转置装入/归约/CopyOut 助手，但无多核分核）。任务是把两者拼成 key=4：不新增第 5 个 kernel 类，而是让新分支搭在现有模板上落地，并保持全部既有分支零回归。落地后 UT 102/102、ST Branch-4 40/40、其余分支 46/37/24/34 全过。

## 适用场景

- 工程形态：算子仓自定义 AscendC 算子，多 tilingKey 分支增量开发（新分支 = 已有 pattern 的组合）。
- API 路径：APT 入口 `TILING_KEY_IS(k)` 分发、`AscendC::SyncAll()`（需 host 侧 `SetScheduleMode(1)`）、`DataCopyPad` GM↔UB、`asc_vf_call` 自定义 VF 合并函数、`if constexpr` 模板编译期分发。
- 验证过的平台与 CANN 版本：Ascend950PR（group 用例 blockDim=56）+ CANN 9.2.0。
- 适用粒度：`operator_family`（多分支归约类算子的 group 分支扩展）。
- 路线优先级：`recommended_fastpath`（相对"新建独立 kernel 类"的方案：少一份模板实例与注册面，助手复用已被既有分支 ST 验证）。
- 明确不适用/未验证：非归约类、单核即可吃满的分支不需要该结构；A 分块数 >1 的场景在本切片 ST 集内未覆盖（白盒 bf16 大 A 失败待修，见失效边界）。

## 设计决策

- **为何 key=4 与 key=1 合并分发到同一 `Kernel<DTYPE_X, false>` 模板**：两者同为 isTailR=false（RA 布局、转置装入、CopyOut path 2/3），差异只在运行时 `td_->rGroupCnt > 0` 是否走 group 路径。合并后 APT 只有 3 个分发臂（0/3、1/4、2），模板实例与二进制面不膨胀。
- **为何 `if constexpr (isTailR)` 编译期分发而非运行时 if**：group 的 Phase1/Phase2 内部调用链整个不同（AR 助手 vs tail-A 助手），编译期分发让每个模板实例只编入自己路径的代码，避免运行时分支与未用代码。
- **为何 Phase1/2 结构 mirror key=3 而助手全部换 tail-A 版**：group 的正确性结构（部分 dual-reduction → SyncAll → rescale 合并 → 重读输入三遍法）与布局无关，可整体照搬；装入/归约/CopyOut 的布局相关助手必须用 key=1 已验证的 tail-A 版本（`CopyInTailA/CastToFp32TailA/DoReduceMaxTailA/DoReduceSumTailA/DoSubExpTailA/DoSubExpDivCastTailA/CopyOutTailA/InitMaxCacheTailA/InitSumCacheTailA/AccumulateMaxTailA/AccumulateSumTailA`）。
- **softmax rescale 合并公式**（与 key=3 共用同一个 VF 函数 `BHSoftmaxRescaleSumVf`，它是 per-A-entry 的一维向量操作，与布局无关）：`final_m = max_r(m_part_r)`，`final_s = Σ_r s_part_r × exp(m_part_r − final_m)`，保证 `e = exp(x − final_m)/final_s` 数学正确。
- **被放弃的方案**：独立 workspace 存 m/s（v1）——因 aclnn 直调路径 workspace 入参不可用触发 507035 而回退 y 别名（完整闭环见本片 build 卡）；新建第 5 个 kernel 类——膨胀注册面且助手全部重复。
- 当前路线：`recommended_fastpath`（在 y 别名约束成立的前提下）。

## 系统信息

- toolchain：CANN 9.2.0，AscendC 自定义算子包（`operators/build.sh <pkg>` + CPack custom_opp 安装到 `$ASCEND_OPP_PATH/vendors/<OpType>`）。
- 目标平台：Ascend950PR（白盒 `Device Platform: Ascend950PR_9579`）。
- 目录布局与相关文件（与事件流实际读写一一对应）：
  - `op_kernel/bench_harness_softmax_apt.cpp` — APT 入口，tilingKey 分发（本卡改）；
  - `op_kernel/arch35/bench_harness_softmax_kernel.h` — kernel 模板（本卡改：Init/Process/ProcessGroup/Group tail-A 方法组）；
  - `op_host/bench_harness_softmax_tiling_host.cpp`、`op_host/arch35/bench_harness_softmax_tiling_arch35.cpp` — host tiling（前一任务已实现 key=4 编码，本卡只消费）；
  - `tests/ut/test_branch4_tiling.cpp` — 分支 TilingUT（独立 oracle）；
  - `tests/aclnn/test_aclnn.cpp` + `run.sh` — C++ aclnn ST。

## 完整代码

APT 入口分发（终版；key=4 与 key=1 合并，stub else 移除）：

```cpp
    if (TILING_KEY_IS(0) || TILING_KEY_IS(3)) {
        BenchHarnessSoftmaxKernel<DTYPE_X, true> kernel;
        kernel.Init(x, y, workspace, &tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(1) || TILING_KEY_IS(4)) {
        // NormalTailA (key=1, isTailR=false): RA pattern, transpose load, path 2/3.
        // GroupTailA (key=4, isTailR=false): 2D split Phase1+Phase2 + tail-A
        //   transpose load + CopyOut path 2/3 (DESIGN-BRANCH-4.md §3/§5).
        //   Both share BenchHarnessSoftmaxKernel<_, false>; rGroupCnt>0 selects group.
        BenchHarnessSoftmaxKernel<DTYPE_X, false> kernel;
        kernel.Init(x, y, workspace, &tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(2)) {
        BenchHarnessSoftmaxEmptyKernel<DTYPE_X> kernel;
        kernel.Init(x, y, workspace, &tilingData);
        kernel.Process();
    }
```

`Process()`：去掉 isTailR 守卫，key=3/4 都进 group 路径：

```cpp
        // Group mode (tilingKey=3/4): 2D split Phase1+Phase2 (DESIGN-BRANCH-3/4 §3/§5).
        if (td_->rGroupCnt > 0) {
            ProcessGroup();
            return;
        }
```

`ProcessGroup()`：核心间分工 + SyncAll 屏障 + 编译期分发：

```cpp
    __aicore__ inline void ProcessGroup()
    {
        int64_t coreId = AscendC::GetBlockIdx();
        int64_t aNeedCore = td_->aLoopCntTotal;

        if (coreId >= aNeedCore) {
            if constexpr (isTailR) {
                ProcessGroupPhase1(coreId - aNeedCore);
            } else {
                ProcessGroupPhase1TailA(coreId - aNeedCore);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();

        if (coreId < aNeedCore) {
            if constexpr (isTailR) {
                ProcessGroupPhase2(coreId);
            } else {
                ProcessGroupPhase2TailA(coreId);
            }
        }
    }
```

`Init()` 内 group workspace 绑定（y 别名；竞态机制与闭环见姊妹卡《Group Phase2 读 y 别名 workspace 与 CopyOut 竞态》）：

```cpp
        // Group mode workspace: alias output GM (y) for m/s storage, same as
        // key=3 (the framework does not reliably pass the separate workspace
        // buffer to the kernel's workspace GM_ADDR).  Phase 1 writes m/s to
        // the start of y; Phase 2 reads before CopyOut overwrites.  SyncAll
        // + fast tiny workspace reads (Pass A/B) before large Pass 3 CopyOut
        // was believed to guarantee ordering (key=3 ST cases were all
        // aSplitChunkCnt=1); the aSplitChunkCnt>1 race was later confirmed and
        // Phase 2 no longer reads workspace (final ProcessGroupPhase2TailA below).
        if (td_->rGroupCnt > 0) {
            gmWs_.SetGlobalBuffer((__gm__ float*)y);
        } else {
            gmWs_.SetGlobalBuffer((__gm__ float*)workspace);
        }
```

Group tail-A 方法组（类末尾新增，verbatim；helper 语义见设计决策）：

```cpp
    // aTotal for ARA = ∏(all A axes) = axisShape[0] × axisShape[2].
    __aicore__ inline int64_t GroupATotalTailA()
    {
        return td_->axisShape[0] * td_->axisShape[2];
    }

    // Linear A-space offset (into aTotal) for (aLoopIdx, aSub).
    // Mirrors GmBaseATailA's A-space component (dropping R/stride terms).
    __aicore__ inline int64_t WsAOffTailA(int64_t aLoopIdx, int64_t aSub)
    {
        if (td_->aSplitAxisIdx == 2) {
            int64_t a0 = (td_->aSplitChunkCnt > 0)
                ? (aLoopIdx / td_->aSplitChunkCnt) : 0;
            int64_t a2ChunkIdx = (td_->aSplitChunkCnt > 0)
                ? (aLoopIdx % td_->aSplitChunkCnt) : 0;
            return a0 * td_->axisShape[2] + a2ChunkIdx * td_->aUbFactor;
        }
        int64_t a0Base = aLoopIdx * td_->aUbFactor;
        return (a0Base + aSub) * td_->axisShape[2];
    }

    // Write m_part (cacheBuf[0..nColsAlign-1]) and s_part (cacheBuf[nColsAlign..])
    // to workspace GM fp32 (DESIGN-BRANCH-4 §8 layout: [rGroupCnt, aTotal] ×2).
    __aicore__ inline void WriteWorkspaceMsTailA(int64_t rGroupIdx, int64_t aOff,
                                                   int64_t validACols, int64_t aTotal)
    {
        int64_t wsOffM = rGroupIdx * aTotal + aOff;
        int64_t wsOffS = td_->rGroupCnt * aTotal + rGroupIdx * aTotal + aOff;

        auto cache = cacheBuf_.Get<float>();
        int64_t nColsAlign = TileAColsAlign();
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = static_cast<uint32_t>(validACols * sizeof(float));
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmWs_[wsOffM], cache[0], extParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(gmWs_[wsOffS], cache[nColsAlign], extParams);
    }

    __aicore__ inline void ProcessGroupPhase1TailA(int64_t rCoreIdx)
    {
        int64_t rNeedCore = td_->usedCoreNum - td_->aLoopCntTotal;
        int64_t rPerCore = CeilDivI64(td_->rGroupCnt, rNeedCore);
        int64_t myRGroupStart = rCoreIdx * rPerCore;
        int64_t myRGroupEnd = myRGroupStart + rPerCore;
        if (myRGroupEnd > td_->rGroupCnt) myRGroupEnd = td_->rGroupCnt;

        int64_t rLoopsPerGroup = CeilDivI64(td_->rLoopCntTotal, td_->rGroupCnt);
        int64_t aTotal = GroupATotalTailA();

        for (int64_t rGroupIdx = myRGroupStart; rGroupIdx < myRGroupEnd; ++rGroupIdx) {
            int64_t rChunkStart = rGroupIdx * rLoopsPerGroup;
            int64_t rChunkEnd = rChunkStart + rLoopsPerGroup;
            if (rChunkEnd > td_->rLoopCntTotal) rChunkEnd = td_->rLoopCntTotal;

            for (int64_t aLoopIdx = 0; aLoopIdx < td_->aLoopCntTotal; ++aLoopIdx) {
                int64_t aSubCnt = ValidASubTailA(aLoopIdx);
                int64_t validACols = ValidAColsTailA(aLoopIdx);
                for (int64_t aSub = 0; aSub < aSubCnt; ++aSub) {
                    // Pass 1: ReduceMax<RA>, linear accumulate → m_part.
                    InitMaxCacheTailA();
                    AscendC::PipeBarrier<PIPE_ALL>();
                    for (int64_t rIdx = rChunkStart; rIdx < rChunkEnd; ++rIdx) {
                        CopyInTailA(aLoopIdx, aSub, rIdx);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        CastToFp32TailA();
                        AscendC::PipeBarrier<PIPE_ALL>();
                        DoReduceMaxTailA();
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AccumulateMaxTailA(validACols);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }

                    // Pass 2: Sub+Exp → ReduceSum<RA>, linear accumulate → s_part.
                    InitSumCacheTailA();
                    AscendC::PipeBarrier<PIPE_ALL>();
                    for (int64_t rIdx = rChunkStart; rIdx < rChunkEnd; ++rIdx) {
                        CopyInTailA(aLoopIdx, aSub, rIdx);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        CastToFp32TailA();
                        AscendC::PipeBarrier<PIPE_ALL>();
                        DoSubExpTailA();
                        AscendC::PipeBarrier<PIPE_ALL>();
                        DoReduceSumTailA();
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AccumulateSumTailA(validACols);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }

                    // Write m_part + s_part to workspace (skip Div+Cast, §5 Phase 1).
                    int64_t aOff = WsAOffTailA(aLoopIdx, aSub);
                    WriteWorkspaceMsTailA(rGroupIdx, aOff, validACols, aTotal);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessGroupPhase2TailA(int64_t aLoopIdx)
    {
        // Phase 2 computes final_m / final_s / y directly from GM x via the same
        // 3-pass algorithm as key=1 NormalTailA (ProcessThreePassTailA), rather
        // than merging per-rGroup m_part/s_part from workspace.
        //
        // Rationale (DESIGN-BRANCH-4 §7 精度处理 + §8 workspace):
        //  1. The framework does not reliably pass the dedicated workspace GM_ADDR
        //     to the kernel (verified: using workspace param causes 507035 crash).
        //     Workspace is aliased to y (same as key=3).  Phase 2 reading workspace
        //     (aliased to y) races with Phase 2 CopyOut when aSplitChunkCnt > 1:
        //     early A-chunks' CopyOut overwrites workspace needed by later A-chunks.
        //     This caused case00005/06 catastrophic failure (bf16, aSplitChunkCnt=19).
        //  2. The rescaling trick (final_s = Σ s_part × exp(m_part − final_m))
        //     introduces multi-level accumulation error growing with
        //     rGroupCnt × rLoopsPerGroup, causing case00010/11 precision failure.
        //
        // By computing final_m = max(x) and final_s = Σ exp(x − final_m) directly
        // from GM x (3-pass: max → sum → div+cast), Phase 2 never reads workspace,
        // eliminating both the race and the rescaling error.  The precision matches
        // key=1 (NormalTailA) exactly.  Cost: one extra GM read pass over R in
        // Phase 2 (max pass), but correctness > performance.
        //
        // Phase 1 (R-cores) still writes m_part/s_part to workspace (aliased to y);
        // this is harmless — Phase 2 Pass 3 CopyOut overwrites all of y with the
        // correct output, and SyncAll guarantees Phase 1 completes before Phase 2.
        ProcessThreePassTailA(aLoopIdx, 1);
    }
```

## 验证信号

问题与验证来源：GroupTailA 分支 kernel 落地后的构建 + UT + 全量 aclnn ST 回归（Ascend950PR，CANN 9.2.0）。

- 构建：`./build.sh BenchHarnessSoftmax_package` 退出码 0（来源平台：950；证据等级：tool_observed）。
- 分支 TilingUT 全量：102/102 通过（15 条 branch-4 + 87 条既有）（来源平台：950；证据等级：tool_observed）。
- 全量 aclnn ST：Branch-0 46/46、Branch-1 37/37、Branch-3 34/34、Branch-4 40/40 precision PASSED，0 aclnn-FAIL（含 key=1 边界回归用例，零回归）（来源平台：950；证据等级：tool_observed）。
- kernel 头文件 `__simd_vf__|asc_vf_call` 计数 24（含新增 key=4 的 VF 调用）（来源平台：950；证据等级：tool_observed）。
- Phase2 改三遍直算后的全量回归：C++ aclnn ST 181/181 精度 PASS（Branch-0 46/46 + Branch-1 37/37 + Branch-2 24/24 + Branch-3 34/34 + Branch-4 40/40），无任何分支回归（来源平台：950；证据等级：tool_observed）。

## 性能影响

本切片未做 profiling 优化循环，无两条路线的成对相对对比数据；白盒单跑 dyn_perf_us 绝对值仅记录在 manifest provenance 中，不写入正文（不作通用预期）。

## 失效边界

- ST 验证集的 group 用例 A 总元素均不超过单 A 块（aLoopCntTotal=1）；A 分块数 >1（aSplitChunkCnt>1）场景已被白盒 bf16 用例证实存在 y 别名 workspace 读-写竞态，Phase2 的 workspace 合并形态已改为三遍直算（`ProcessThreePassTailA`，完整 before/after 见姊妹卡）；采用本配方时 Phase2 必须用直算形态。
- group 模式 m/s 必须走 y 别名绑定（workspace 入参不可用，见本片 build 卡）；换到可传 workspace 的调用路径时布局公式（`[rGroupCnt, aTotal]×2`）不变、仅换绑定。
- rescale 合并的数值行为：`final_s` 按 rGroup 顺序累加，长 R 下与 golden 的求和模型存在可增长偏差（见本片 fp32 大 R 卡）。
- 未验证：非 Ascend950 平台、`axisNum>3`（更深 ARA 合轴）场景。

**未决项**：

- 白盒 4 条失败已分流：bf16 2D 灾难性 ×2 由 y 别名 workspace 读-写竞态修复闭合（Phase2 改三遍直算，白盒复跑 case00005/06 mere≈1.5e-5 PASS，见姊妹卡）；fp32 大 R 漂移 ×2 为求和顺序问题另卡归因，未闭环。
- key=3（GroupTailR）路径的同构竞态隐患是否同样改直算未验证。
## 坏实践

- 为新 tilingKey 新建整套独立 kernel 类再复制粘贴助手——注册面与代码双膨胀，且助手副本失去与既有分支的同源修复。
- group 分支照搬 tail-R 的 AR 布局助手（`CopyIn/DoReduceMax/...`）——布局与 RA 不匹配，装入/归约/CopyOut 全链错位。
