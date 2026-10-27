---
type: "Runbook"
title: GetBlockNum 对齐后 TBE 多核执行仍 EZ9999
description: TBE 动态编译把 blockDim 烤为平台核数时，kernel 三处已用 AscendC::GetBlockNum() 对齐切分，GE 图路径执行仍失败：浮点例 EZ9999（error 95 MTE DDR out of range, core 0），整数例不崩但 bitwise mismatch——网格对齐修复必要而不充分，跨核一致性层未闭合。
tags: [task.debugging, topic.memory, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# GetBlockNum 对齐后 TBE 多核执行仍 EZ9999

## 现象

在 Ascend950PR（CANN 9.2.0）上验收自定义算子 BenchHarnessScatter（scatter，data→y 拷贝 + 按 indices 散布 updates）的 GEIR 图模式。GE 图路径下该算子被 TBE 动态编译（kernel_name `te_benchharnessscatter_911e6935...`），`ParseTvmBlockDim: 6` / `StreamLaunchKernelV2 ... coreDim=6`——blockDim 被烤死为平台 6 核，而 TilingFunc 的 `SetBlockDim`（单核语义）被无视（判别见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡）。kernel 源码里**已经存在**为这种"实际网格数 ≠ tiling numCores"场景做的对齐修复：三处切分函数统一用 `AscendC::GetBlockNum()` 覆盖 numCores。但即便如此，GE 图路径执行仍然失败：

- route 探针（data [4,3] fp32、indices [2,3] int32、updates [2,3] fp32，reduce=update）：`rungraph_rc=4294967295`，设备侧报 aivec 异常：

```
[ERROR] RUNTIME(...,geir_route): ... [device_error_proc_c.cc:716] ... ProcessDavidStarsCoreErrorInfo:The error from device(chipId:0, dieId:0), serial number is 289, there is an aivec error exception, core id is 0, error code = 95, dump info: pc start: 0x120041000000, current: 0x12004100058c, sc error info: 0xffffffffffff, su error info: 0xaaa4be403eba0278,0x7fff7fff18000000, mte error info: 0xdf73d5a800020265, vec error info: 0x4c601748003101ee, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x12004cc00000, mte error: 0x80000000, aic cond: 0.
```

上报为 `error_code:EZ9999`，`RunGraph: ErrorNo: 4294967295(failed)`。异常发生在 core 0、PC 偏移 0x58c（很早），`mte error: 0x80000000` 即 MTE DDR out of range。

- TTK GEIR static 8/8 执行、0 SKIP：`static_f16_i32` / `static_f32_i32` / `static_bf16_i32` 等 6 例 `EXEC_FAILURE`（`ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error!` 同一设备错误）；`static_i32_i32` 与 `static_i64_i32` 两例 kernel 跑完不崩但精度失败——i32 例 59.375%（binary_equal，13 处差异，如 `RealIndex: 000001 Expected: -1.0 Actual: 0.0`、`RealIndex: 000008 Expected: -3.0 Actual: 5.0`），i64 例 53.125%（15 处差异；判读规则见同批整数静默失配卡）。
- 同一 kernel 在 aclnn 直调路径回归 SUCCESS（`Test overwrite: PASSED` / `Test add: PASSED`）——差异只在 GE 图路径的 6 核下发。

当时代码状态：kernel 三处切分（CopyDataToY / CopyDataToAccFp32 / CopyAccToYNative）均已按 GetBlockNum() 对齐，core 0 单核 scatter 前有 `SyncAll()` + `MTE3_MTE2` 事件 + `PipeBarrier<PIPE_ALL>()` 的跨阶段同步。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：AscendC 自定义算子 kernel（多核切分 + 单核串行 scatter）+ GE 图路径（ge::Session/RunGraph）下发，辅以 TTK GEIR static/dynamic 评测。
- API 路径：`AscendC::GetBlockNum()/GetBlockIdx()`、`SyncAll()`、`SetFlag/WaitFlag<HardEvent::MTE3_MTE2>`、`DataCopyPad`、GE `StreamLaunchKernelV2`（coreDim）。
- 验证过的平台与 CANN 版本：Ascend950PR（6 AIV 核），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——"多核切分 + 单核归约/散布"形态的 kernel 在 blockDim 被外层烤死时的行为边界。
- 路线优先级：`safety_fallback`——在 TBE 回退无法消除时，kernel 网格无关化（GetBlockNum 对齐）是必要的防御，但**不足以**保证正确性；成对相对代价：相对"命中预编译二进制（blockDim=-1，SetBlockDim 生效，单核执行）"的正确路线，本路线在本会话中 0/8 通过（参照路线 aclnn 直调全 PASS，来源平台 950）。fallback 声明：仅当无法让 GE 图路径命中预编译二进制时才接受多核执行，且须先按本卡失效边界补齐跨核一致性验证。
- 明确不适用 / 未验证场景：aclnn 直调路径（实测 PASS，不涉及）；blockDim=1 的图路径执行（未观测到）；本会话任务约束"不修改 Kernel/Tiling"，故未对 kernel 做进一步修复尝试。

## 尝试与观测（负结果）

网格对齐修复（kernel 既有实现，本会话 grep + 文件读取 verbatim 证实存在）——CopyDataToY（另两处 CopyDataToAccFp32/CopyAccToYNative 同构）：

```cpp
template <typename T, typename I, int64_t RANK>
__aicore__ inline void BenchHarnessScatterKernel<T, I, RANK>::CopyDataToY()
{
    int64_t dataSize=1; for(int64_t k=0;k<td_->rank;k++) dataSize*=td_->dataShape[k];
    int64_t numCores=td_->multicore.numCores; if(numCores<1) numCores=1;
    // Align split with the actually-launched grid (GEIR graph path may compile
    // blockDim != tiling numCores; mismatch caused cid>=numCores blocks to
    // compute out-of-range GM offsets -> EZ9999 MTE DDR out of range).
    int64_t launchedBlocks=AscendC::GetBlockNum(); if(launchedBlocks>0) numCores=launchedBlocks;
    int64_t cpc=dataSize/numCores, ct=dataSize%numCores;
    int64_t cid=AscendC::GetBlockIdx();
    int64_t ms=cid*cpc+(cid<ct?cid:ct), mc=cpc+(cid<ct?1:0);
    if(mc<=0){ return; }
    int64_t maxChunk=td_->perBufBytes/sizeof(T); if(maxChunk<=0) maxChunk=1;
    for(int64_t off=0;off<mc;off+=maxChunk){
        int64_t cnt=(mc-off<maxChunk)?(mc-off):maxChunk;
        AscendC::DataCopyExtParams p; p.blockCount=1; p.blockLen=cnt*sizeof(T); p.srcStride=0; p.dstStride=0;
        AscendC::DataCopyPadExtParams<T> pad{};
        AscendC::DataCopyPad(buf_[B0].Get<T>(), gmData_[ms+off], p, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2_MTE3_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2_MTE3_);
        AscendC::DataCopyPad(gmY_[ms+off], buf_[B0].Get<T>(), p);
        // Ensure MTE3 (B0 read→GM write) completes before next iteration's MTE2
        // overwrites B0.  Without this, the next MTE2 corrupts B0 mid-read.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
    }
    AscendC::PipeBarrier<PIPE_MTE3>();
}
```

Process() 的非 fp32acc 主路径（多核拷贝 → 跨核同步 → core 0 scatter）：

```cpp
    CopyDataToY();
    AscendC::SyncAll();
    // MTE3 (copy writes) → MTE2/MTE3 (scatter reads/writes) GM coherency.
    // PipeBarrier<PIPE_ALL> may not guarantee GM write visibility across pipes.
    // Explicit MTE3_MTE2 event ensures all copy MTE3 writes are committed to GM
    // before the scatter's MTE2 reads or MTE3 writes.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
    ...
    if(AscendC::GetBlockIdx()!=0){ AscendC::PipeBarrier<PIPE_ALL>(); return; }
```

（fp32acc 路径同构：CopyDataToAccFp32 → SyncAll + MTE3_MTE2 + PipeBarrier → core 0 ScatterReduceFp32Acc → SyncAll → CopyAccToYNative。）

观测（对齐修复在场的情况下）：

- 6 核执行 f16/f32/bf16 × i32 例：EZ9999 / error 95 / mte error 0x80000000，core 0、PC 0x58c。
- i32/i64 例：不崩，但 `cst: {0: {'standard': 'binary_equal', 'pass': False, 'reason': 'bitwise mismatch'}}`，精度 59.375%（13 diff）/ 53.125%（15 diff）。
- TTK dynamic 例：kernel 执行挂起，整批 600s 超时被杀（评测侧无 --proc-timeout）。

## 根因（未追到机制层，如实记录）

**本会话未追到多核失败的确定性根因**。已排除/已确认的部分：

- 已确认：失败不是"切分越界"——GetBlockNum 对齐后每核区间 `ms..ms+mc` 均在 dataSize 内（12 元素 ÷ 6 核 = 每核 2 个）。
- 已确认：同 kernel 单核路径（aclnn 直调，SetBlockDim 生效）全部 PASS——问题与多核执行强相关。
- 未证实假设（agent 推测，进误诊段）：浮点的 crash 与整数的位失配可能同源于"多核 CopyDataToY 的 MTE3 写与 core 0 scatter 的 MTE2 读之间跨核 GM 可见性/时序"，即 SyncAll + 核内 MTE3_MTE2 事件不足以保证其他核的 GM 写对 core 0 可见；但整数失配值呈现"覆盖错位/读到旧值"模式（Expected -1 Actual 0、Expected -3 Actual 5），也可能是 updates 写入顺序问题。error 95 的具体越界访存指令未定位（PC 0x58c 未反汇编）。

## 设计决策

- 为什么当初做 GetBlockNum 对齐：注释写明"GEIR graph path may compile blockDim != tiling numCores; mismatch caused cid>=numCores blocks to compute out-of-range GM offsets -> EZ9999"——即第一层根因（切分越界）已由该修复闭合（本会话 grep 证实三处齐全）。
- 为什么 core 0 单核 scatter：散布语义需要确定性的重复索引处理顺序（与 golden 的 flat 顺序一致），多核并发 RMW 同一 target 会引入竞争——所以多核只做不相交区间的 data→y 拷贝。
- 当前方案路线档位：safety_fallback（网格无关化防御）——正确路线是让 GE 图路径命中预编译二进制（blockDim=-1 + SetBlockDim 生效），见同批判别卡。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（6 AIV 核），CANN 9.2.0，GEIR 验收套件 + TTK。
- 相关文件：`op_kernel/arch35/bench_harness_scatter_kernel.h`（CopyDataToY/CopyDataToAccFp32/CopyAccToYNative/Process，只读核对）；route 日志与 `tests/geir/build/ttk/static-result.csv`（失败观测源）。

## 预防

- kernel 切分一律以 `GetBlockNum()` 实际网格对齐（必要防御），但**不要宣称这足以保证 GE 图路径正确性**——本会话是反例。
- 图路径验收时把"实际 coreDim"（StreamLaunchKernelV2 日志）与 kernel 假设的网格数对照；coreDim≠预期时先解决编译路径问题（见同批判别卡），不要在错误网格下调试 kernel 语义。

## 验证信号

来源说明：以下信号来自该算子一次 GEIR 图模式验收会话（kernel 网格对齐修复在场、未再修改 kernel）。

- GE 图路径 6 核执行设备异常：`core id is 0, error code = 95 ... mte error: 0x80000000`（MTE DDR out of range），`RunGraph: ErrorNo: 4294967295`，EZ9999 上报（来源平台：950；证据等级：tool_observed）。
- TTK GEIR static：8/8 执行、0 SKIP——6 例 EXEC_FAILURE（EZ9999，`static_f16_i32`/`static_f32_i32`/`static_bf16_i32` 等）+ 2 例 FAIL（`static_i32_i32` 59.375% bitwise mismatch 13 diff、`static_i64_i32` 53.125% 15 diff）（来源平台：950；证据等级：tool_observed）。
- TTK GEIR dynamic：4 例挂起，整批 600s 超时被杀（来源平台：950；证据等级：tool_observed）。
- 同 kernel aclnn 直调回归：`Test overwrite: PASSED` / `Test add: PASSED`（`ACLNN BenchHarnessScatter summary: SUCCESS (overwrite=1 add=1)`）——单核路径正确（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定"TBE 动态编译多核产物 + 本 kernel 的多核拷贝/单核散布结构"；blockDim=1 或命中预编译二进制的场景不受此边界约束（后者本会话未复现）。
- 失败与 dtype 相关（浮点崩、整数错）：整数位失配的具体模式（覆盖错位 vs 旧值）未逐一归因（判读规则见同批整数静默失配卡）。
- `SyncAll` + 核内事件 + PipeBarrier 的组合是否足以保证跨核 GM 可见性——未验证（这正是未闭合的机制层）。
- 其他 SoC 核数（非 6）下的行为未验证，推断可迁移（机制与核数无关）。

**未决项**：

- **优先项：复现并追根因到机制层**——在受控 6 核下定位 error 95 的具体越界访存（反汇编 PC 0x58c 附近；对照 tiling para base 0x12004cc00000 与 GM 实际分配范围），并对整数位失配做逐元素 target 归属分析（失配索引是否集中在某核写区段/某 update 覆盖窗口）。
- "SyncAll + 核内 MTE3_MTE2 事件 + PipeBarrier"组合的跨核 GM 可见性语义需独立实验验证（可用最小双核 kernel 复现）。
- 让 GE 图路径命中预编译二进制（blockDim=-1）后，本卡边界自动缩小为"TBE 回退时的 fallback 行为"。
## 坏实践

- 认为"切分对齐了就万事大吉"——网格对齐只消除越界层失败，不解决跨核一致性/时序层失败。
- 在 GE 图路径 kernel 失败时直接深挖 kernel 语义，而不先确认 coreDim 与编译路径（本会话前序已证明该失败由 TBE 回退引发，kernel 侧修复被任务边界约束为 out-of-scope）。

## 误诊

- "GetBlockNum 对齐修复已存在，EZ9999 应已解决"——前一阶段会话的结论（LOG 记载"三处统一以 GetBlockNum() 对齐"后重建成功）；本会话重跑证明修复在场仍失败，该结论作为完整因果不成立（前会话结论属转述，agent_reported；本会话的失败观测为 tool_observed）。
- agent 对整数位失配成因的推测（CopyDataToY 多核写与 core 0 scatter 读的跨核可见性）未经任何实验证实，仅作线索保留。
