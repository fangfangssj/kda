---
type: "Runbook"
title: split-load 多 tile UB 循环 kernel 实现骨架
description: 每核任务量超单 UB tile（blockFormer > ubFormer）时，kernel 入口按 tilingKey 分派 split-load 路径：UB 内循环主块走 DataCopy、尾块走 DataCopyPad，逐 tile 复用同一 VF 计算链，TQue depth=2 自然重叠相邻 tile 搬运。
tags: [task.implementation, topic.tiling, topic.data_movement.data_copy, topic.optimizer]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# split-load 多 tile UB 循环 kernel 实现骨架

## 现象

在 Ascend950PR（CANN 9.2.0）上为 BenchHarnessApplyAdamW 实现 Branch-1（split-load，tilingKey=1，进入条件 `blockFormer > ubFormer`——每核任务量超过单个 UB tile，需要 UB 内多 tile 循环）。按测试先行流程，ST 先行写好 41 例（Branch-0 21 例 + Branch-1 20 例）；实现前跑基线：37/41 通过，4 例大 shape 全 FAIL——`b1_fp32_256M`、`b1_bf16_256M`、`b1_fp32_1048576` 输出 `actual=nan`，`b1_fp16_256M` 输出 `actual=126976`，golden 均 ≈0.0998。当时代码状态：kernel 入口 `bench_harness_apply_adam_w_apt.cpp` 只有 `if (tilingData.tilingKey == 0)` 分支并注释「tilingKey == 1 (split-load) — not implemented」——tilingKey=1 时 kernel 什么都不做，输出张量保持未初始化的垃圾值。注意同批按 32 核设计的「进入边界」shape（blockFormer=5120/9728）在实机更多核数下落回 tilingKey=0 经 full-load 路径通过，所以失败集中在这 4 个无论核数多少都稳定进入 split-load 的超大 shape。

实现 ProcessTile/ProcessSplitLoad 并在入口加分派后，41/41 全通过，Branch-0 与 TilingUT 回归不退化。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发（tilingKey 多分支 kernel，aclnn ST 直调验证）+ TTK kernel 白盒。
- API 路径：AscendC `TQue`（depth=2 双缓冲）、`DataCopy`（对齐快路径）/`DataCopyPad`（尾块安全路径）、`asc_vf_call<...>` 调 `__simd_vf__` 计算链、kernel 入口按 TilingData 的 tilingKey 分派。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0，AIV 32 核（实机核数以 `plat.GetCoreNumAiv()` 运行时查询为准）（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——逐元素类（无跨 tile 累积状态）算子的「每核多 tile」切分实现；含跨 tile 归约的算子需另加同步，不能直接套用。
- 路线优先级：recommended_fastpath（本轨迹实测的 split-load 标准结构）。
- 明确不适用 / 未验证场景：跨核/跨 tile 有数据依赖（scatter-reduce、邻域卷积等）——本骨架明确无 SyncAll，依赖「逐 tile 计算链无跨 tile 状态」这一前提；Cube/Matmul 类未验证。

## 解决配方

结构：把 per-tile 的 CopyIn→VF 计算→CopyOut 抽成共享 `ProcessTile(offset, tileLen, useDataCopy)`；Branch-0 的 `Process()` 退化为单 tile 调用；Branch-1 新增 `ProcessSplitLoad()` 做 UB 内循环——主循环 `loopNum-1` 次满 `ubFormer` tile 走 `DataCopy`（256B 对齐保证 32B 对齐），最后 1 次 `tailNum` 尾 tile 走 `DataCopyPad`；kernel 入口按 tilingKey 分派。

修复前（`op_kernel/bench_harness_apply_adam_w_apt.cpp`，split-load 无实现）：

```cpp
    if (tilingData.tilingKey == 0) {
        // Branch-0: full-load, single tile per core (DESIGN-BRANCH-0)
        KernelBenchHarnessApplyAdamW<T> kernel;
        kernel.Init((__gm__ uint8_t*)var, (__gm__ uint8_t*)grad,
                    (__gm__ uint8_t*)m, (__gm__ uint8_t*)v,
                    (__gm__ uint8_t*)y, tilingData);
        kernel.Process();
    }
    // tilingKey == 1 (split-load, Branch-1) — not implemented in this task.
```

修复后（入口分派）：

```cpp
    if (tilingData.tilingKey == 0) {
        // Branch-0: full-load, single tile per core (DESIGN-BRANCH-0)
        KernelBenchHarnessApplyAdamW<T> kernel;
        kernel.Init((__gm__ uint8_t*)var, (__gm__ uint8_t*)grad,
                    (__gm__ uint8_t*)m, (__gm__ uint8_t*)v,
                    (__gm__ uint8_t*)y, tilingData);
        kernel.Process();
    } else if (tilingData.tilingKey == 1) {
        // Branch-1: split-load, UB inner-loop multi-tile (DESIGN-BRANCH-1)
        KernelBenchHarnessApplyAdamW<T> kernel;
        kernel.Init((__gm__ uint8_t*)var, (__gm__ uint8_t*)grad,
                    (__gm__ uint8_t*)m, (__gm__ uint8_t*)v,
                    (__gm__ uint8_t*)y, tilingData);
        kernel.ProcessSplitLoad();
    }
```

完整实现见下节。

## 设计决策

- **DataCopy vs DataCopyPad 双路径**：`ubFormer` 由 tiling 按 256B 对齐推出（fp32=4672、fp16/bf16=9344，本机 UB 192KiB−8KiB 预留、10 buffer 推得），满 tile 字节数必然 32B 对齐，可走 `DataCopy` 快路径；尾 tile `tailNum` 任意，必须 `DataCopyPad`。Branch-0 单 tile 路径保留 DataCopyPad（尾块可能非对齐）。
- **共享 ProcessTile 而非复制两份流水**：逐元素算子每个 tile 的 VF 计算链完全相同、无跨 tile 状态，抽公共函数使两分支只差循环结构，回归面最小（Branch-0 21 例 + TilingUT 37 例一次通过）。
- **TQue depth=2 天然重叠相邻 tile**：UB 内循环里相邻 tile 的搬运与计算靠队列双缓冲交叠，无需手工 SetFlag/WaitFlag；逐元素无跨核归约，故不加 SyncAll。
- **标量一次性预计算进寄存器**：β、β^t、lr、sign 等在 `Init` 缓存/预计算（`oneMinusBeta1_`、`signLr_` 等），每 tile 复用，不占额外 UB。
- **分支触发的运行时依赖（重要测试设计结论）**：`blockFormer = AlignUp(dim0/coreNum, 512)` 随实机核数变化——按 32 核设计的「进入边界」shape 在核数 ≥40 时 blockFormer 缩小落回 tilingKey=0。要让 ST/白盒用例稳定命中 split-load，shape 必须大到「任意预期核数下 blockFormer 仍 > ubFormer」（本轨迹用 256M 元素与 1048576 单维，模拟 32/40/48/64/80 核均 tilingKey=1）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.2.0，`$ASCEND_HOME_PATH/x86_64-linux/ascendc/include` 头文件族。
- 目录布局与相关文件（与本卡证据一一对应）：
  - `operators/BenchHarnessApplyAdamW_package/op_kernel/arch35/bench_harness_apply_adam_w_kernel.h`——kernel 主体（ProcessTile/Process/ProcessSplitLoad/Init 缓存）。
  - `operators/BenchHarnessApplyAdamW_package/op_kernel/bench_harness_apply_adam_w_apt.cpp`——kernel 入口 tilingKey 分派。
  - `operators/BenchHarnessApplyAdamW_package/op_host/bench_harness_apply_adam_w_tiling_host.cpp`——切分公式（`ubLoop=CeilDiv(blockFormer,ubFormer)`、`ubTail=blockFormer−(loop−1)·ubFormer`、末核 blockTail 同形特化）。
  - `operators/BenchHarnessApplyAdamW_package/op_kernel/arch35/bench_harness_apply_adam_w_tiling_struct.h`——TilingData 字段（ubLoopOfFormerBlock/ubTailOfFormerBlock/ubLoopOfTailBlock/ubTailOfTailBlock 等）。
  - `operators/BenchHarnessApplyAdamW_package/tests/aclnn/test_aclnn.cpp`——ST（独立 C++ goldenAdamW oracle，不调 kernel）。
- 构建与验证命令：

```bash
source docs/BenchHarnessApplyAdamW/develop/env.sh && \
  ( cd operators && ./build.sh BenchHarnessApplyAdamW_package ) && \
  ( cd operators/BenchHarnessApplyAdamW_package/tests/aclnn && ./run.sh )
# TilingUT 回归：
( cd operators/BenchHarnessApplyAdamW_package/tests/tiling && ./run.sh )
```

## 完整代码

`bench_harness_apply_adam_w_kernel.h` 的三个核心函数（verbatim 取自终版写入；`Init` 中缓存 `ubFormer_`、`ubLoopOfFormerBlock_`、`ubTailOfFormerBlock_`、`ubLoopOfTailBlock_`、`ubTailOfTailBlock_` 等 TilingData 字段并预计算 `oneMinusBeta1_`、`oneMinusBeta2_`、`signLr_` 标量，此处从略）：

```cpp
    // =======================================================================
    // ProcessTile — per-tile CopyIn→Compute→CopyOut (DESIGN-BRANCH-1 §3/§5,
    //   shared by both branches; DESIGN-BRANCH-1 §5: VF chain identical per
    //   tile, no cross-tile accumulation).
    //
    //   useDataCopy=true  → DataCopy fast path (tile 256B-aligned, main loop)
    //   useDataCopy=false → DataCopyPad (tail-safe, last tile may be unaligned)
    // =======================================================================
    __aicore__ inline void ProcessTile(uint64_t offset, uint64_t tileLen,
                                       bool useDataCopy)
    {
        if (tileLen == 0) return;

        // ---- CopyIn params (used only for DataCopyPad path) ----
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = static_cast<uint32_t>(tileLen * sizeof(T));
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;
        AscendC::DataCopyPadExtParams<T> padParams(false, 0, 0, T(0));

        // ---- CopyIn: 4-route GM→UB (DESIGN-BRANCH-1 §3 step 1) ----
        AscendC::LocalTensor<T> varLocal  = inQueueVar.AllocTensor<T>();
        AscendC::LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
        AscendC::LocalTensor<T> mLocal    = inQueueM.AllocTensor<T>();
        AscendC::LocalTensor<T> vLocal    = inQueueV.AllocTensor<T>();

        if (useDataCopy) {
            // Main loop: ubFormer is 256B-aligned → DataCopy fast path (§2)
            uint32_t cnt = static_cast<uint32_t>(tileLen);
            AscendC::DataCopy(varLocal,  gmVar[offset],  cnt);
            AscendC::DataCopy(gradLocal, gmGrad[offset], cnt);
            AscendC::DataCopy(mLocal,    gmM[offset],    cnt);
            AscendC::DataCopy(vLocal,    gmV[offset],    cnt);
        } else {
            // Last tile: tailNum may be non-256B-aligned → DataCopyPad (§2)
            AscendC::DataCopyPad(varLocal,  gmVar[offset],  copyParams, padParams);
            AscendC::DataCopyPad(gradLocal, gmGrad[offset], copyParams, padParams);
            AscendC::DataCopyPad(mLocal,    gmM[offset],    copyParams, padParams);
            AscendC::DataCopyPad(vLocal,    gmV[offset],    copyParams, padParams);
        }

        inQueueVar.EnQue(varLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueM.EnQue(mLocal);
        inQueueV.EnQue(vLocal);

        // ---- Compute: DeQue → VF chain → EnQue (DESIGN §3 step 2) ----
        AscendC::LocalTensor<T> varIn  = inQueueVar.DeQue<T>();
        AscendC::LocalTensor<T> gradIn = inQueueGrad.DeQue<T>();
        AscendC::LocalTensor<T> mIn    = inQueueM.DeQue<T>();
        AscendC::LocalTensor<T> vIn    = inQueueV.DeQue<T>();

        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        uint32_t count = static_cast<uint32_t>(tileLen);
        uint16_t repeatTimes = static_cast<uint16_t>((count + VL - 1) / VL);

        // Invoke the fused VF compute chain (DESIGN §5, all in fp32 registers).
        // asc_vf_call passes UB physical addresses + scalar operands.
        // Per-tile VF chain is identical to full-load (no cross-tile state).
        asc_vf_call<AdamWVF<T>>(
            (__ubuf__ T*)yLocal.GetPhyAddr(),
            (__ubuf__ T*)varIn.GetPhyAddr(),
            (__ubuf__ T*)gradIn.GetPhyAddr(),
            (__ubuf__ T*)mIn.GetPhyAddr(),
            (__ubuf__ T*)vIn.GetPhyAddr(),
            beta1_, oneMinusBeta1_, beta2_, oneMinusBeta2_,
            beta1Pow_, beta2Pow_, weightDecay_, epsilon_, signLr_,
            count, VL, repeatTimes);

        // PipeBarrier between VF chain and TQue EnQue (DESIGN §3 sync)
        AscendC::PipeBarrier<PIPE_V>();

        outQueueY.EnQue(yLocal);
        inQueueVar.FreeTensor(varIn);
        inQueueGrad.FreeTensor(gradIn);
        inQueueM.FreeTensor(mIn);
        inQueueV.FreeTensor(vIn);

        // ---- CopyOut: UB→GM (DESIGN-BRANCH-1 §3 step 3) ----
        AscendC::LocalTensor<T> yOut = outQueueY.DeQue<T>();
        if (useDataCopy) {
            AscendC::DataCopy(gmY[offset], yOut, static_cast<uint32_t>(tileLen));
        } else {
            AscendC::DataCopyPad(gmY[offset], yOut, copyParams);
        }
        outQueueY.FreeTensor(yOut);
    }

    // =======================================================================
    // Process — Branch-0 full-load (tilingKey=0, DESIGN-BRANCH-0 §3).
    //   Each core processes exactly one tile (≤ ubFormer), DataCopyPad path.
    // =======================================================================
    __aicore__ inline void Process()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockNum_) return;

        bool isLastBlock = (blockIdx == blockNum_ - 1);
        uint64_t tileLen = isLastBlock ? blockTail_ : blockFormer_;
        if (tileLen == 0) return;
        uint64_t offset = blockFormer_ * blockIdx;  // element offset into GM

        // Branch-0: single tile, DataCopyPad (tail may be non-aligned, §2)
        ProcessTile(offset, tileLen, /*useDataCopy=*/false);
    }

    // =======================================================================
    // ProcessSplitLoad — Branch-1 split-load (tilingKey=1, DESIGN-BRANCH-1 §3).
    //   UB inner loop: main loop (loopNum-1) × ubFormer via DataCopy (aligned),
    //   then 1 × tailNum via DataCopyPad (tail-safe). Reuses ProcessTile VF chain.
    //   TQue depth=2 overlaps adjacent tile搬运; no SyncAll.
    // =======================================================================
    __aicore__ inline void ProcessSplitLoad()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockNum_) return;

        bool isLastBlock = (blockIdx == blockNum_ - 1);
        // §3: loopNum/tailNum selected by isLastBlock
        uint64_t loopNum = isLastBlock ? ubLoopOfTailBlock_   : ubLoopOfFormerBlock_;
        uint64_t tailNum = isLastBlock ? ubTailOfTailBlock_   : ubTailOfFormerBlock_;
        if (loopNum == 0) return;

        uint64_t baseOffset = blockFormer_ * blockIdx;  // element offset into GM

        // Main loop: loopNum-1 iterations of ubFormer (256B-aligned → DataCopy)
        for (uint64_t i = 0; i + 1 < loopNum; ++i) {
            uint64_t offset = baseOffset + i * ubFormer_;
            ProcessTile(offset, ubFormer_, /*useDataCopy=*/true);
        }
        // Last tile: tailNum (may be non-aligned → DataCopyPad)
        uint64_t tailOffset = baseOffset + (loopNum - 1) * ubFormer_;
        ProcessTile(tailOffset, tailNum, /*useDataCopy=*/false);
    }
```

## 性能影响

本轨迹未做 split-load 与 full-load 的成对性能对比（无 profiling 数据），不构成可引用的性能结论；256M 元素用例的执行时间在 ST 中可接受（41 例一轮数分钟内完成），仅作定性参考。

## 预防

- 实现 tilingKey 多分支 kernel 时，入口分派必须覆盖所有 tilingKey：未分派的 key 静默不执行，输出保持未初始化垃圾（nan / 随机大数），比报错更难定位——ST 基线先跑一遍即可暴露（无分派的 key 对应用例 actual=nan/垃圾值）。
- 设计「进入边界」测试 shape 时按公式 `blockFormer = AlignUp(dim0/coreNum, 512)` 代入**最大预期核数**验证 `blockFormer > ubFormer` 仍成立；否则用例在核数更多的设备上会静默落入另一分支，覆盖失效。
- 满块走 DataCopy 的前提是 tile 字节数 32B 对齐——由 tiling 的 256B 对齐保证；若改 tiling 对齐系数须同步复核此约束。

## 验证信号

以下信号来自该算子 Branch-1 实现会话：实现前先跑 ST 红基线，实现后同命令复跑，并回归 TilingUT。

- 实现前（tilingKey=1 无分派）：ST 37/41——`b1_fp32_256M`/`b1_bf16_256M`/`b1_fp32_1048576` 输出 actual=nan、`b1_fp16_256M` 输出 actual=126976（golden≈0.0998），Branch-0 21/21 与 Branch-1 其余 16/20（实机核数下落回 full-load）通过（来源平台：950；证据等级：tool_observed）。
- 实现后（ProcessTile/ProcessSplitLoad + 入口分派）：ST 41/41 通过（Branch-0 21/21 回归不退化 + Branch-1 20/20），run.sh 退出 0（来源平台：950；证据等级：tool_observed）。
- 切分模拟核对：256M/1048576 级 shape 在 32/40/48/64/80 核模拟下均 tilingKey=1（fp32 256M：blockFormer=8388608、ubLoop=1796；fp16 256M：ubLoop=898），按 32 核设计的边界 shape 在 40+ 核时落回 key=0（来源平台：950；证据等级：tool_observed）。
- TilingUT 回归 37/37 通过（公共 14 + 分派边界 3 + Branch-0 11 + Branch-1 9）；构建无 tiling 未注册告警；`grep __simd_vf__`/`grep asc_vf_call` 在 op_kernel 下均有匹配（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `ubFormer` 值绑定本机 UB 布局（192KiB−8KiB 预留、10 buffer、256B 对齐；fp32=4672、fp16/bf16=9344）——其它 SoC/UB 配置须按 tiling 公式重推，数值不可照抄（机制相同，未验证，推断可迁移）。
- 尾核 `blockTail` 与主核 blockFormer 的 UB 循环量分别由 `ubLoopOfTailBlock`/`ubTailOfTailBlock` 与 `ubLoopOfFormerBlock`/`ubTailOfFormerBlock` 描述，双尾块非对齐（fp32 5000、fp16 9601 一类）已覆盖；`tailNum == 0`（整除退化）由 ProcessTile 开头 `if (tileLen == 0) return` 兜底。
- 本骨架依赖「逐 tile VF 链无跨 tile 状态」——凡跨 tile 累积（滑动窗口、scatter 归约）不能直接套用，需另设跨 tile 同步与累积缓冲。
- 256M 元素 fp32 单 case 的 host/device 内存占用约 4×256M×4B + 1×256M×4B ≈ 5GB，受 HBM 上限约束（本机 131GB 富余）；更大 shape 未验证。

**未决项**：

- 实现会话收尾后，验收阶段重新构建安装并复跑 ST 时 41 例全部报 ACLNN 561002（`aclnnBenchHarnessApplyAdamWGetWorkspaceSize` 处），验收调查定位到 step 属性类型线索后被记录截断——该 561002 与本骨架实现的因果关系未闭环（见同批 E80003 para_check 卡遗留问题）。
- split-load 相对 full-load 的相对性能代价未测量（无 profiling）。
- 超过 256M 的 shape 与 fp8 等 dtype 未验证。
## 坏实践

- kernel 入口对已知 tilingKey 只写注释不写分支——静默输出垃圾值，且「进入边界」用例在多核设备上落回 full-load 后连失败都暴露不出来，形成假覆盖。
- 复制一份只改循环的 Branch-1 流水而不抽共享 ProcessTile——两份 CopyIn/VF/CopyOut 代码在后续精度修复（如 β^t double 化）时极易只改一支漏一支。

## 误诊

- 实现前误判「16/20 Branch-1 用例已通过 = split-load 已部分可用」——实际是该设备核数下多数「Branch-1 用例」落回了 full-load 分支，通过的 16 例根本没有走 split-load 路径；只有核数无关的 4 个超大 shape 暴露了真缺口。
