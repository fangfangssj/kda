---
type: "Runbook"
title: "Softmax 大 R 分核：Phase1+SyncAll+Phase2 合并"
description: A 维不满核而 R 维巨大时，R 核按 rGroup 算部分 max/sum 写 GM，SyncAll 后 A 核做 max+rescale-sum 合并再重读输入输出；含跨核同步 VF 合并函数与输出 GM 复用存 m/s 的完整实现。
tags: [task.implementation, topic.activation, topic.compute.reduction, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# Softmax 大 R 分核：Phase1+SyncAll+Phase2 合并

## 现象（任务背景）

在 Ascend950PR（CANN 9.2.0，arch3510）上实现自定义算子 BenchHarnessSoftmax 的 Group 分支（tilingKey=3 GroupTailR）：输入的归约轴 R 很大（131072~2097152）、而 A 维（归约轴外的合轴元素数）很小（1/2/4），`aLoopCntTotal < coreNum` 且 `rLoopCntTotal > 1`。若沿用 key=0 的按 A 切核逻辑，只有 aLoopCntTotal 个核有活、每核串行扫全 R——正确但完全不吃满 56 核。需要一个"R 维分组 + 跨核合并"的 2D 分核实现，并保证与参考实现（`torch.nn.functional.softmax`，max-subtraction + fp32 累加语义）在 atol/rtol=2^-10（fp16）容差内一致。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel（算子仓 package，op_kernel/arch35 单头文件实现）+ aclnn ST/TTK 白盒双验证。
- API 路径：`AscendC::SyncAll()`（跨核屏障；host TilingFunc 需 `SetScheduleMode(1)`，头文件在 `cann/x86_64-linux/asc/include/interface/kernel_operator_block_sync_intf.h`，模板默认 `isAIVOnly=true`）；`DataCopyPad`（UB↔GM，`DataCopyExtParams` 按 byte 计 blockLen）；regbase VF（`__simd_vf__` 函数 + `asc_vf_call` 调用、`AscendC::Reg::*` 寄存器算子、`UpdateMask` 自动消耗 remaining）。
- 验证过的平台与 CANN 版本：Ascend950PR（TTK SOC 串 Ascend950PR_9579），CANN 9.2.0。
- 适用粒度：operator_family——"部分归约 + 全局合并"结构的归约类算子（softmax/logsumexp/分段 max+sum 统计）。
- 路线优先级：recommended_fastpath（A 不满核 + 大 R 场景吃满核的默认解）。
- 明确不适用 / 未验证场景：A 已填满核（走 key=0 按核三遍即可）；Cube 路径；需要在线流式（无法两遍扫 R）的算法。

## 解决配方（算法结构）

设 R 被切成 `rGroupCnt` 组，每组 `rLoopsPerGroup` 个 rLoop：

- **Phase 1（R 核，`coreId >= aNeedCore`）**：对每个 rGroup、每个 aLoop，两遍扫该组的 R 区间——Pass1 线性累积 `m_part`（组内最大值），Pass2 `Sub+Exp` 后线性累积 `s_part`（组内指数和）；把 `m_part`/`s_part` 以 fp32 写 GM。
- **SyncAll**：全核屏障，保证 Phase 1 写完成后再进 Phase 2。
- **Phase 2（A 核，`coreId < aNeedCore`）**：Pass A 合并 `final_m = max_r m_part_r`；Pass B 用 rescale 技巧合并 `final_s = Σ_r s_part_r × exp(m_part_r − final_m)`（softmax group rescaling：把各组按局部 max 算出的部分和折算到全局 max 下，数学上与整体 softmax 等价）；Pass 3 重读 GM x，按 `exp(x−final_m)/final_s` 计算并 `CopyOut`。

m/s 的 GM 布局：`m_part` 段 `[0, rGroupCnt×aTotal)`、`s_part` 段 `[rGroupCnt×aTotal, 2×rGroupCnt×aTotal)`，元素级索引 `[rGroupIdx×aTotal + aOff]`。本实现把该区域放在**输出 GM 开头**（`gmWs_` 绑定 y；Phase 2 读回发生在 Pass 3 `CopyOut` 覆盖之前，SyncAll 保证跨核写读序）——这是对"框架 workspace 未传递"的规避，若 workspace 可用则改绑 workspace 即可，其余代码不变。

  **已知边界（同 trace 后续证实）**：当单个 A 核需处理多个 A 分块（aSplitChunkCnt>1）时，早分块的 Pass 3 CopyOut 会覆盖晚分块尚未读取的 m/s（y 别名 workspace 读-写竞态）——key=4（GroupTailA）分支已实测灾难性失败并改为 Phase 2 三遍直算；key=3 现有 ST 用例全部 aSplitChunkCnt=1 未触发该竞态，同构隐患存在，是否同样改直算未验证。

## 完整代码

以下为终版实现（verbatim 取自会话事件流；`CopyIn/CastToFp32/DoReduceMax/AccumulateMax/DoSubExp/DoReduceSum/AccumulateSum/InitMaxCache/InitSumCache/DoSubExpDivCast/CopyOut/ValidARows/CeilDivI64` 及 `cacheBuf_/tmpBuf0_/tmpBuf1_` 等为该算子 key=0 路径已有的公共搬运/归约助手与 TBuf）。

Process 路由（Group 模式进入 `ProcessGroup`）：

```cpp
        if (td_->usedCoreNum == 0) return;
        if (AscendC::GetBlockIdx() >= static_cast<int64_t>(td_->usedCoreNum)) return;

        // Group mode (tilingKey=3): 2D split Phase1+Phase2 (DESIGN-BRANCH-3 §3/§5).
        if (td_->rGroupCnt > 0) {
            if constexpr (isTailR) {
                ProcessGroup();
            }
            return;
        }

        int64_t myStart = 0, myLoops = 0;
```

Init 中的 workspace 绑定（Group 模式复用输出 GM）：

```cpp
        gmX_.SetGlobalBuffer((__gm__ T*)x);
        gmY_.SetGlobalBuffer((__gm__ T*)y);
        // Group mode (key=3): reuse output GM for m/s workspace storage
        // (framework may not allocate the separate workspace buffer).
        // Phase 1 writes m/s to output start; Phase 2 reads before CopyOut
        // overwrites.  SyncAll guarantees write-before-read ordering.
        if (td_->rGroupCnt > 0) {
            gmWs_.SetGlobalBuffer((__gm__ float*)y);
        } else {
            gmWs_.SetGlobalBuffer((__gm__ float*)workspace);
        }
```

调度主体与 Phase 1：

```cpp
    __aicore__ inline void ProcessGroup()
    {
        int64_t coreId = AscendC::GetBlockIdx();
        int64_t aNeedCore = td_->aLoopCntTotal;

        if (coreId >= aNeedCore) {
            ProcessGroupPhase1(coreId - aNeedCore);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();

        if (coreId < aNeedCore) {
            ProcessGroupPhase2(coreId);
        }
    }

    __aicore__ inline void ProcessGroupPhase1(int64_t rCoreIdx)
    {
        int64_t rNeedCore = td_->usedCoreNum - td_->aLoopCntTotal;
        int64_t rPerCore = CeilDivI64(td_->rGroupCnt, rNeedCore);
        int64_t myRGroupStart = rCoreIdx * rPerCore;
        int64_t myRGroupEnd = myRGroupStart + rPerCore;
        if (myRGroupEnd > td_->rGroupCnt) myRGroupEnd = td_->rGroupCnt;

        int64_t rLoopsPerGroup = CeilDivI64(td_->rLoopCntTotal, td_->rGroupCnt);
        int64_t aTotal = td_->axisShape[0];

        for (int64_t rGroupIdx = myRGroupStart; rGroupIdx < myRGroupEnd; ++rGroupIdx) {
            int64_t rChunkStart = rGroupIdx * rLoopsPerGroup;
            int64_t rChunkEnd = rChunkStart + rLoopsPerGroup;
            if (rChunkEnd > td_->rLoopCntTotal) rChunkEnd = td_->rLoopCntTotal;

            for (int64_t aLoopIdx = 0; aLoopIdx < td_->aLoopCntTotal; ++aLoopIdx) {
                int64_t validA = ValidARows(aLoopIdx);

                // Pass 1: ReduceMax, linear accumulate → m_part.
                InitMaxCache();
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int64_t rIdx = rChunkStart; rIdx < rChunkEnd; ++rIdx) {
                    CopyIn(aLoopIdx, rIdx, validA);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    CastToFp32(validA);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DoReduceMax(true);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AccumulateMax(validA);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }

                // Pass 2: Sub+Exp → ReduceSum, linear accumulate → s_part.
                InitSumCache();
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int64_t rIdx = rChunkStart; rIdx < rChunkEnd; ++rIdx) {
                    CopyIn(aLoopIdx, rIdx, validA);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    CastToFp32(validA);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DoSubExp(validA);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DoReduceSum(true);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AccumulateSum(validA);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }

                // Write m_part + s_part to workspace (skip Div+Cast, §5 Phase 1).
                WriteWorkspaceMs(rGroupIdx, aLoopIdx, validA, aTotal);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
```

m/s 写出（终版）：

```cpp
    __aicore__ inline void WriteWorkspaceMs(int64_t rGroupIdx, int64_t aLoopIdx,
                                             int64_t validA, int64_t aTotal)
    {
        int64_t aOff = aLoopIdx * td_->aUbFactor;
        int64_t wsOffM = rGroupIdx * aTotal + aOff;
        int64_t wsOffS = td_->rGroupCnt * aTotal + rGroupIdx * aTotal + aOff;

        auto cache = cacheBuf_.Get<float>();
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = static_cast<uint32_t>(validA * sizeof(float));
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmWs_[wsOffM], cache[0], extParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(gmWs_[wsOffS], cache[td_->aUbFactorAlign], extParams);
    }
```

Phase 2（合并 + 重读输出）：

```cpp
    __aicore__ inline void ProcessGroupPhase2(int64_t aLoopIdx)
    {
        int64_t validA = ValidARows(aLoopIdx);
        int64_t aTotal = td_->axisShape[0];
        int64_t aOff = aLoopIdx * td_->aUbFactor;

        // ── Pass A: final_m = max over rGroups of m_part ──
        InitMaxCache();
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t rGroupIdx = 0; rGroupIdx < td_->rGroupCnt; ++rGroupIdx) {
            int64_t wsOffM = rGroupIdx * aTotal + aOff;
            auto dst = tmpBuf0_.Get<float>();
            AscendC::DataCopyExtParams extParams;
            extParams.blockCount = 1;
            extParams.blockLen = static_cast<uint32_t>(validA * sizeof(float));
            extParams.srcStride = 0;
            extParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<float> padParams;
            padParams.isPad = true;
            padParams.paddingValue = BitsToFp32(0xFF800000u);  // -inf
            AscendC::DataCopyPad(dst[0], gmWs_[wsOffM], extParams, padParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            AccumulateMax(validA);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // ── Pass B: final_s = Σ s_part × exp(m_part − final_m) ──
        InitSumCache();
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ float* finalMPtr = (__ubuf__ float*)cacheBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* finalSPtr = finalMPtr + td_->aUbFactorAlign;
        for (int64_t rGroupIdx = 0; rGroupIdx < td_->rGroupCnt; ++rGroupIdx) {
            int64_t wsOffM = rGroupIdx * aTotal + aOff;
            int64_t wsOffS = td_->rGroupCnt * aTotal + rGroupIdx * aTotal + aOff;
            auto dstM = tmpBuf0_.Get<float>();
            auto dstS = tmpBuf1_.Get<float>();
            AscendC::DataCopyExtParams extParams;
            extParams.blockCount = 1;
            extParams.blockLen = static_cast<uint32_t>(validA * sizeof(float));
            extParams.srcStride = 0;
            extParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<float> padM;
            padM.isPad = true;
            padM.paddingValue = BitsToFp32(0xFF800000u);  // -inf (max identity)
            AscendC::DataCopyPadExtParams<float> padS;
            padS.isPad = true;
            padS.paddingValue = 0.0f;  // 0 (sum identity)
            AscendC::DataCopyPad(dstM[0], gmWs_[wsOffM], extParams, padM);
            AscendC::DataCopyPad(dstS[0], gmWs_[wsOffS], extParams, padS);
            AscendC::PipeBarrier<PIPE_ALL>();
            __ubuf__ float* mPartPtr = (__ubuf__ float*)dstM.GetPhyAddr();
            __ubuf__ float* sPartPtr = (__ubuf__ float*)dstS.GetPhyAddr();
            asc_vf_call<BHSoftmaxRescaleSumVf<VL_F32_U16>>(
                finalSPtr, mPartPtr, sPartPtr, finalMPtr,
                static_cast<uint16_t>(validA));
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // ── Pass 3: Sub+Exp+Div+Cast → CopyOut (re-read x, same as key=0 Pass3) ──
        for (int64_t rIdx = 0; rIdx < td_->rLoopCntTotal; ++rIdx) {
            CopyIn(aLoopIdx, rIdx, validA);
            AscendC::PipeBarrier<PIPE_ALL>();
            CastToFp32(validA);
            AscendC::PipeBarrier<PIPE_ALL>();
            DoSubExpDivCast(validA);
            AscendC::PipeBarrier<PIPE_ALL>();
            CopyOut(aLoopIdx, rIdx, validA);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
```

rescale 合并的 VF 函数（连续向量链 Sub→Exp→Mul→Add，`UpdateMask` 每轮消耗 remaining）：

```cpp
template <uint16_t VL_F32>
__simd_vf__ inline void BHSoftmaxRescaleSumVf(
    __ubuf__ float* sumUb, __ubuf__ float* mPartUb,
    __ubuf__ float* sPartUb, __ubuf__ float* finalMUb,
    uint16_t srcK)
{
    uint32_t remaining = srcK;
    uint16_t repTimes = (srcK + VL_F32 - 1) / VL_F32;
    for (uint16_t j = 0; j < repTimes; ++j) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> mReg;
        AscendC::Reg::RegTensor<float> sReg;
        AscendC::Reg::RegTensor<float> fmReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::RegTensor<float> tmpReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::LoadAlign(mReg, mPartUb + j * VL_F32);
        AscendC::Reg::LoadAlign(sReg, sPartUb + j * VL_F32);
        AscendC::Reg::LoadAlign(fmReg, finalMUb + j * VL_F32);
        AscendC::Reg::LoadAlign(sumReg, sumUb + j * VL_F32);
        AscendC::Reg::Sub(tmpReg, mReg, fmReg, mask);
        AscendC::Reg::Exp(expReg, tmpReg, mask);
        AscendC::Reg::Mul(tmpReg, sReg, expReg, mask);
        AscendC::Reg::Add(sumReg, sumReg, tmpReg, mask);
        AscendC::Reg::StoreAlign(sumUb + j * VL_F32, sumReg, mask);
    }
}
```

## 设计决策

- **为何选 R 分组 + rescale 合并而不是按 R 切核直接算**：softmax 的除法需要全局 max 与全局和，任何 R 切分都必须跨核交换信息；rescale 技巧（`s_r × exp(m_r − final_m)`）让每组可以用自己的局部 max 先算部分和，合并时一次性折算，避免全局两遍同步。数学等价性：`exp(x−m_r)/Σ_r' s_r'` 与 `exp(x−m_global)/Σ exp(x−m_global)` 在 fp32 累加下等价（相乘的 `exp(m_r−final_m)` 因子约掉）。
- **m/s 放 GM 而不是 UB**：跨核交换必须出核；GM 布局按 `[rGroupCnt, aTotal]` 两段连续，`DataCopyPad` 的 blockLen 按 byte（`validA × sizeof(float)`），非对齐尾巴用 pad 模式补 `-inf`（max 单位元）/`0`（sum 单位元）。
- **SyncAll 的启用前提**：host TilingFunc 在 Group 判定成立时调用 `SetScheduleMode(1)`，否则 `AscendC::SyncAll()` 不可用；这是 host/kernel 的一处隐式契约。
- **合并路径用 VF（regbase）实现**：仓库工作流硬性要求实现代码走 `__simd_vf__` + `asc_vf_call`；`UpdateMask(remaining)` 按引用消耗剩余元素数，循环写法与既有 VF 助手一致。
- **被放弃的候选**：key=3 复用 key=0 逻辑（初始状态，正确但不吃满核，被本实现替换）；`WriteWorkspaceMs` 的 `Duplicate`(V)+`SetValue`(Scalar) 中转写法（调试期引入、指针修复后证明多余，终版删除）；workspace 存放 m/s（因框架未传递 workspace 改为输出 GM 复用——若传递链修好可直换绑定）。
- **当前方案属哪档路线**：recommended_fastpath（见适用场景）。

## 系统信息

- 工具链：CANN 9.2.0；构建 `cd operators && ./build.sh BenchHarnessSoftmax_package`（退出 0）；测试 `tests/aclnn/run.sh`（ST，自建 harness + 独立 ref_softmax oracle）与 TTK kernel 模式白盒。
- SOC：Ascend950PR（arch3510；56 AIV 核；TTK SOC 串 Ascend950PR_9579）。
- 目录/文件：`op_kernel/arch35/bench_harness_softmax_kernel.h`（本卡全部代码所在）；`op_kernel/arch35/bench_harness_softmax_tiling_struct.h`（`td_->rGroupCnt/aLoopCntTotal/rLoopCntTotal/aUbFactor/aUbFactorAlign/usedCoreNum/axisShape` 等 tiling 字段）；`op_host/arch35/bench_harness_softmax_tiling_arch35.cpp`（Group 判定、rGroupCnt 切分、`SetScheduleMode(1)`）；`op_kernel/bench_harness_softmax_apt.cpp`（tilingKey 分发入口）。
- 关键 API 位置：`SyncAll` 在 `cann-9.2.0/x86_64-linux/asc/include/interface/kernel_operator_block_sync_intf.h`（顶层 `include/` 搜不到）；`DataCopyExtParams` 在 `include/basic_api/kernel_struct_data_copy.h`（`srcStride/dstStride` 为 int64_t）；reg 计算在 `include/basic_api/reg_compute/`。

## 验证信号

以下信号来自该分支的实现与验收会话，均为上板实测：

- 构建 `./build.sh` 退出 0；ST `run.sh` 退出 0（来源平台：950；证据等级：tool_observed）。
- `Branch-3 (GroupTailR, tilingKey=3) summary: 34/34 precision PASSED`（覆盖 A=1/2/4 大 R、非对齐 R [1,131071]、nan/inf/-inf/zero/60000 特殊值全表、key=0 边界 [1,256] 不误路由）；Branch-0 46/46、Branch-1 37/37、Branch-2 24/24 无回归（来源平台：950；证据等级：tool_observed）。
- TTK kernel 模式白盒 82 条：79 pass；3 条 FAIL 经归因均非本分支缺陷（1 条 key=0 路径 fp16 1-ULP 硬件差异、2 条路由到未实现的 key=4）（来源平台：950；证据等级：tool_observed）。
- 硬性判据 `grep -cE '__simd_vf__|asc_vf_call'` 于 kernel 头命中 23 处（来源平台：950；证据等级：tool_observed）。

## 性能影响

本会话未做分核前后的成对性能对比（无 profiling 数据）；"R 核 + A 核吃满 56 核 vs 仅 aLoopCntTotal 核有活"的收益是结构性的，未量化。

## 预防

- 新增 Group 类分支时，host 侧同步开 `SetScheduleMode(1)`、kernel 侧只依赖 `SyncAll` 之后的 GM 可见性。
- m/s 的 GM 布局（两段连续）写进 tiling 注释，读写两侧共用同一索引公式，避免漂移。

## 失效边界

- 输出 GM 复用要求输出元素数 ≥ `rGroupCnt×aTotal×2`（m/s 两段）；workspace 传递链修好后应改回绑 workspace，代码其余部分不变。
- `DataCopyPad` 的 `blockLen` 按 byte 且非 32B 对齐尾巴依赖 pad 模式（`isPad=true` + 单位元 paddingValue）；对齐假设变化时需复核。
- 本实现假定 `usedCoreNum = aLoopCntTotal + rNeedCore`（R 核从 `coreId - aNeedCore` 起编号）；tiling 侧改核分配策略时调度代码要同步。
- 仅在 950PR（arch3510）上板验证；推断可迁移到同为 arch35 的 950DT（未验证，推断可迁移）。
- Phase 2 从 y 别名 workspace 读 m/s 的形态在 aSplitChunkCnt>1 时与自身 CopyOut 构成读-写竞态（key=4 分支已证实；本分支 key=3 用例未触发）——跨 A 分块场景须改三遍直算或绑定真 workspace，见姊妹卡《Group Phase2 读 y 别名 workspace 与 CopyOut 竞态》。

**未决项**：

- Phase 2 Pass B 每轮 `DataCopyPad` 拷 `m_part` 仅为算 `exp(m_r−final_m)`，`final_m` 在 Pass A 已得——m_part 可在合并 max 时缓存复用，少一半 GM 读（未实现）。
- workspace 传递链修复后回迁绑定（见失效边界）。
## 坏实践

- Phase 1/Phase 2 的 GM 读写之间只靠代码顺序、漏掉 `SyncAll`——跨核可见性无保证，本实现的屏障位置（`ProcessGroup` 中 `PipeBarrier + SyncAll`）不可省。
- 用 `ProcessThreePass` 之类的旧路径顶替 Group 路径交差——正确但不吃满核，等于没做本分支。

## 误诊

- 实现初期把 507035 归因于 DataCopyPad 写法（对齐/流水线/Scalar→MTE3 同步），引入了 `Duplicate+SetValue` 中转等冗余代码；根因是 workspace 指针无效（见该根因的独立卡片），指针有效后这些中转全部删除。
