---
type: "Runbook"
title: "SyncAll 须配 tiling SetScheduleMode 批调度"
description: "kernel 用 AscendC::SyncAll 做全核同步而 TilingFunc 未调用 SetScheduleMode(BATCH_MODE_SCHEDULE=1) 时，多流场景存在核间同步卡死风险；在 TilingFunc 的 SetBlockDim 前设置批调度模式即修复（CANN 9.2.0 头文件不导出该常量，须自定义 constexpr）。"
tags: [task.debugging, topic.implementation.synchronization, topic.implementation.tiling]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# SyncAll 须配 tiling SetScheduleMode 批调度

## 现象

对自定义算子 BenchHarnessScatter（aclnn 交付形态，AscendC kernel + host TilingFunc，多核切分）做成品代码检视整改时，读取检视报告 `docs/develop/checks/CODE_REVIEW.md`，整体状态为 ❌失败：1 个 FAIL（F1）+ 2 个 SUSPICIOUS。唯一 FAIL 命中检视条例 TOPK-13（ascendc-topk.md #13，适用 Host/Tiling 侧）：kernel 侧 `op_kernel/arch35/bench_harness_scatter_kernel.h` 的 `Process()` 函数在 3 处使用 `AscendC::SyncAll()` 做全核同步（fp32 workspace 累加路径的 Phase A→B 交接、Phase B→C 交接，以及原生路径 `CopyDataToY()` 之后），而 tiling 侧 `op_host/arch35/bench_harness_scatter_tiling_arch35.cpp` 的 `TilingFunc` 只调用了 `context->SetBlockDim(static_cast<uint32_t>(mc.numCores))`，全文件没有任何 `SetScheduleMode` 调用——调度模式保持默认值 0。检视条例明确指出：多流场景下 kernel 侧使用 SyncAll 而 tiling 侧未用 `SetScheduleMode` 设置 `BATCH_MODE_SCHEDULE`，会导致卡死问题。当时代码处于可编译、单流测试全部通过的状态（该缺陷是静态检视发现的潜在多流风险，不是已发生的运行时故障）。

## 适用场景

- 工程形态：aclnn 自定义算子（AscendC kernel + host TilingFunc），kernel 采用多核切分且阶段间需要跨核全局同步（如"全核写 → 单核 RMW → 全核读"的分段流水），因此使用 `AscendC::SyncAll()`。
- API 路径：kernel 侧 `AscendC::SyncAll()`；host 侧 `gert::TilingContext::SetScheduleMode(const uint32_t schedule_mode)`（写 tiling 输出槽 `kOutputScheduleMode = 6`，见 `exe_graph/runtime/tiling_context.h`）；取值常量 `BATCH_MODE_SCHEDULE = 1` 在 CANN 9.2.0 中未导出，须在 tiling 侧自定义 `constexpr uint32_t`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 严格一致）。
- 适用粒度：platform_wide——同一 CANN 版本下任何使用 SyncAll 的多核 AscendC kernel，与具体算子无关。
- 路线优先级：recommended_fastpath（检视条例级正确性要求，无等价替代路线）。
- 明确不适用 / 未验证场景：不使用 SyncAll 的 kernel（无此要求）；多流卡死现象本身未在本次轨迹复现（全部回归为单设备单流 aclnn 路径，修复前后均通过——本卡针对的是检视条例指出的多流风险态）；其他 CANN 版本与 NPU 架构（常量取值与调度语义未验证，推断可迁移）；GEIR 图模式执行路径（未验证）。

## 根因

kernel 通过 `AscendC::SyncAll()` 做跨核全同步，但 tiling 侧从未调用 `SetScheduleMode` 把调度模式设为 `BATCH_MODE_SCHEDULE`（保持默认 0），多流场景下同一次算子调用的各核不保证被同批调度，SyncAll 等待的全核集合无法凑齐而卡死。

机制说明（据检视条例 TOPK-13 的规则描述，本次轨迹未复现卡死）：`SetScheduleMode` 向 tiling 输出槽 `kOutputScheduleMode`（outputs[6]）写入一个 uint32 调度模式；设为 1（批模式）保证单次算子调用的全部核在同一批内调度，使 SyncAll 的全核栅障可满足；默认 0 下不同流的核可能分开调度，全核栅障永久等待。

判别方法（可执行检查）：

1. `grep -n "SyncAll" op_kernel/*.h op_kernel/*.cpp`——凡 kernel 出现 `AscendC::SyncAll`，即为触发条件。
2. `grep -n "SetScheduleMode\|BATCH_MODE" op_host/*.cpp op_host/**/*.cpp`——tiling 侧无命中即风险态（本次检视即以该 grep 为空确认缺失）。
3. 两者不成对出现即判违规；修复后复核 tiling 侧出现 `SetScheduleMode(BATCH_MODE_SCHEDULE)`。

## 解决配方

修复位置在 tiling 侧（kernel 的 SyncAll 本身是合法的跨核同步手段，不需要改）。

修复前——tiling 的 `TilingFunc` 尾段（`op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`，只设 BlockDim，无调度模式）：

```cpp
    FillAndLogTilingData(td, split, mc, perBufBytes, rank, dim, reduceMode,
                         totalIndices, dataShapeArr, dataStridesArr, indicesShapeArr);
    td->useFp32Acc = useFp32Acc;
    td->dataNumel = dataNumel;

    context->SetBlockDim(static_cast<uint32_t>(mc.numCores));

    // tilingKey: rank≤4 → 0 (RANK_4), rank≥5 → 1 (RANK_8)
    // Note: without ASCENDC_TPL, all kernels use default tilingKey=0.
    // The rank dispatch happens at runtime via td->rank.
    // context->SetTilingKey(tilingKey);  // commented out — no TPL, default key=0
```

触发该要求的 kernel 侧上下文（`op_kernel/arch35/bench_harness_scatter_kernel.h` `Process()`，检视时状态，3 处 SyncAll 全列出；行内两处 `CopyAccToYNative()` 是同次检视的另一独立发现——copy-paste 冗余，已在同次整改中删除，与本卡无关）：

```cpp
template <typename T, typename I, int64_t RANK>
__aicore__ inline void BenchHarnessScatterKernel<T, I, RANK>::Process()
{
    if (td_->useFp32Acc) {
        // fp16/bf16 add/multiply: fp32 workspace accumulation path.
        // Phase A (all cores): data → acc (fp32), bulk vector Cast.
        CopyDataToAccFp32();
        AscendC::SyncAll();
        // MTE3 (acc writes) → MTE2/MTE3 (scatter reads/writes) coherency.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
        AscendC::PipeBarrier<PIPE_ALL>();
        // Phase B (core 0): per-element fp32 RMW on acc (4-byte atomic = safe,
        // fp32 accumulation = golden's exact precision).
        if (td_->totalIndices > 0 && AscendC::GetBlockIdx() == 0) {
            ScatterReduceFp32Acc();
        }
        // Cross-core sync: core 0's acc writes must be visible to all cores
        // before Phase C reads acc.  PipeBarrier<PIPE_ALL> is per-core only.
        AscendC::SyncAll();
        CopyAccToYNative();
        // Phase C (all cores): acc (fp32) → y (native), bulk vector Cast with
        // single round-to-nearest-even (matches golden's final astype).
        CopyAccToYNative();
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    CopyDataToY();
    AscendC::SyncAll();
    // MTE3 (copy writes) → MTE2/MTE3 (scatter reads/writes) GM coherency.
    // PipeBarrier<PIPE_ALL> may not guarantee GM write visibility across pipes.
    // Explicit MTE3_MTE2 event ensures all copy MTE3 writes are committed to GM
    // before the scatter's MTE2 reads or MTE3 writes.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
    AscendC::PipeBarrier<PIPE_ALL>();
    if(td_->totalIndices==0){ AscendC::PipeBarrier<PIPE_ALL>(); return; }
```

修复后——在 `SetBlockDim` 之前添加调度模式设置（`BATCH_MODE_SCHEDULE` 常量须自定义，取值 1）：

```cpp
    // TOPK-13: Kernel uses SyncAll() for cross-core synchronization (3 sites in
    // Process()).  In multi-stream scenarios the default schedule mode (0) can
    // cause deadlock because cores from different streams are not dispatched
    // together.  Setting BATCH_MODE_SCHEDULE (=1) guarantees all cores of a
    // single operator invocation are scheduled in the same batch, making
    // SyncAll safe.
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    context->SetBlockDim(static_cast<uint32_t>(mc.numCores));
```

## 设计决策

- **为什么自定义 `constexpr uint32_t BATCH_MODE_SCHEDULE = 1`**：对 CANN 9.2.0 的 `include/`（含 `x86_64-linux/include`）与 `opp/built-in` 全树执行 `grep -rn "BATCH_MODE_SCHEDULE\|batch_mode_schedule\|BATCH_MODE"` 均无该常量定义（仅命中 gemm_v3/mat_mul_v3 的 `BATCH_MODEL` 模板参数，无关）；取值 1 来自检视条例 TOPK-13 的正确示例（`constexpr uint32_t BATCH_MODE_SCHEDULE = 1;`），错误示例明确 `BATCH_MODE_DEFAULT = 0` 即风险态。
- **为什么放在 `SetBlockDim` 之前**：`tiling_context.h` 的实现显示 `SetScheduleMode` 只是向 tiling 输出槽 `kOutputScheduleMode` 写一个 uint32，与 `SetBlockDim` 无顺序依赖；先设调度模式再设核数使"调度配置"意图集中清晰。
- **为什么不改 kernel 去掉 SyncAll**：该算子的 fp32 累加路径要求"core 0 的 acc 写入对全部核可见后再进入 Phase C"，代码注释明确 `PipeBarrier<PIPE_ALL>` 仅 per-core 生效，跨核可见性必须依赖 SyncAll——同步原语是必要的，缺陷在 tiling 侧未配对调度模式。
- **当前方案路线档位**：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1.6），CANN 9.2.0（独立安装目录 `Ascend/cann-9.2.0`，经 `docs/develop/env.sh` source `set_env.sh` 生效）。
- 关键头文件：`$ASCEND_HOME_PATH/include/exe_graph/runtime/tiling_context.h`（`SetScheduleMode` 定义、`TilingOutputIndex::kOutputScheduleMode = 6`）。
- 相关文件：`operators/BenchHarnessScatter_package/op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`（修复点，TilingFunc）、`operators/BenchHarnessScatter_package/op_kernel/arch35/bench_harness_scatter_kernel.h`（SyncAll 触发点，Process()）、`docs/develop/checks/CODE_REVIEW.md`（检视报告，FAIL 判定）、`docs/develop/checks/CODE_REVIEW_FIX.md`（整改报告）。
- 检视规则来源：ascendc-code-review skill 的 `references/ascendc-topk.md` #13（TOPK-13）。
- 回归命令（修复后全部执行）：

```bash
source docs/develop/env.sh && cd operators && bash build.sh BenchHarnessScatter_package
# ACLNN: operators/BenchHarnessScatter_package/tests/aclnn/run.sh
# GEIR:  tests/geir/run.sh {prepare, route, infer V2, reject}
# 黑盒正向: PYTHONPATH=$TTK python3 -m ttk aclnn -i blackbox_all_exec.csv --pc=4 \
#   --task-prof false --run 1 --warmup false --device-whitelist 0 --proc-timeout 300 \
#   --plugin docs/develop/golden.py -o result_review_fix.csv
# 黑盒负向: 同上，-i blackbox_l2_neg.csv --proc-timeout 120 -o result_l2_review_fix.csv
```

同平台 2D 分核场景的 `SetScheduleMode(1)` 配套用法（含失败即返回的检查写法）另见 [group 2D 分核卡](group_2d_core_split_numblocks_align_schedule_mode.md)。

## 预防

- 任何 kernel 计划使用 `AscendC::SyncAll()` 时，TilingFunc 必须同步添加 `constexpr uint32_t BATCH_MODE_SCHEDULE = 1; context->SetScheduleMode(BATCH_MODE_SCHEDULE);`，把"kernel SyncAll ↔ tiling SetScheduleMode(BATCH_MODE_SCHEDULE)"作为代码检视固定检查项。
- 该常量在 CANN 9.2.0 未导出——不要假设 `#include` 某个头文件就能直接使用，须在 tiling 侧自定义并注释取值依据。
- superKernel / 多核同步场景同时核对 TOPK-14：调用全核同步 API（SyncAll、CrossCoreWaitFlag）时禁止做 BlockDim 限制（只在部分核调用 SyncAll 而全部核在 wait，同样卡死）。

## 验证信号

以下信号来自该算子的一次代码检视整改会话（Ascend950PR，CANN 9.2.0）：检视把"kernel SyncAll 与 tiling 调度模式不成对"判为 FAIL，实施修复后跑全量回归确认无退化。

- 修复前：检视报告 `CODE_REVIEW.md` 整体状态 ❌失败，F1 判定 FAIL（置信度 85%）：kernel `Process()` 共 3 处 `AscendC::SyncAll()`，tiling 全文件 `grep "SetScheduleMode\|BATCH_MODE"` 为空（来源平台：950；证据等级：tool_observed）。
- API 事实锚点：CANN 9.2.0 `exe_graph/runtime/tiling_context.h` 中 `ge::graphStatus SetScheduleMode(const uint32_t schedule_mode)` 向 tiling 输出槽 `kOutputScheduleMode = 6` 写 uint32；`grep -rn "BATCH_MODE_SCHEDULE" <CANN 安装>/include/ <CANN 安装>/opp/` 无任何定义命中（来源平台：950；证据等级：tool_observed）。
- 修复后构建：`bash build.sh BenchHarnessScatter_package` 编译通过，CPack 生成 `custom_opp_ubuntu_x86_64.run` 并安装 SUCCESS（来源平台：950；证据等级：tool_observed）。
- 修复后 ACLNN 回归：`Test overwrite: PASSED`、`Test add: PASSED`，summary `SUCCESS (overwrite=1 add=1)`（来源平台：950；证据等级：tool_observed）。
- 修复后 GEIR 合同检查：infer V2 10/10 PASS（3 Shape + 7 Dtype，`INFER-SUMMARY route=V2 total=10 pass=10 fail=0`）、reject 1/1 PASS（`GEIR-SUMMARY total=1 pass=1 fail=0 skip=0`）（来源平台：950；证据等级：tool_observed）。
- 修复后全量黑盒：正向 399/399 PASS（PassRate 100.00%，`--pc=4 --proc-timeout 300`）；负向 25 条中 24 FAIL（被算子正确拒绝，负向门禁预期）+ 1 PASS（已知误归类用例，dim=0 为合法输入），均与修复前基线一致、无退化（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **多流卡死现象未复现**：本轨迹全部测试为单设备单流 aclnn 路径，修复前后均无卡死——本卡消除的是检视条例 TOPK-13 指出的多流风险态，"SyncAll + 默认调度模式 → 卡死"的机制链来自规则文档而非本轨迹实测。
- 常量取值（=1）与输出槽号（kOutputScheduleMode=6）均以 CANN 9.2.0 为据；其他 CANN 版本与 NPU 架构未验证（推断可迁移），升级版本时应复核常量是否已由头文件导出。
- 不使用 SyncAll 的 kernel 不适用本卡；若业务上不需要跨核全局可见性，应优先避免引入 SyncAll（而非依赖调度模式兜底）。
- superKernel 场景"部分核调用 SyncAll 导致卡死"是另一条例（TOPK-14，BlockDim 限制问题），本卡不覆盖。

**未决项**：

- **优先项**：构造多流场景（如双 stream 并发下发同一算子）复现"SyncAll + 默认调度模式"的卡死，把机制链从规则描述升级为实测信号，并验证设置 BATCH_MODE_SCHEDULE 后卡死消失——本轨迹未复现该现象。
- `BATCH_MODE_SCHEDULE = 1` 的官方权威出处（CANN 9.2.0 安装内无定义，本卡取值来自检视规则文档示例）；CANN 升级时复核。
- `SetScheduleMode(1)` 对单流 aclnn 路径的性能影响未测量（批调度可能改变下发时序）。
- 该设置与 GEIR 图模式执行路径的协同行为未验证（该路径在本次轨迹中因另一已知限制超时跳过）。
## 坏实践

- kernel 用 SyncAll 而 tiling 只调 `SetBlockDim` 就交付——检视 FAIL 的原始状态，多流场景有卡死风险。
- 以为 `BATCH_MODE_SCHEDULE` 是 CANN 头文件导出常量、直接 `#include` 使用——CANN 9.2.0 未导出，编译不过或误造常量。
- 把"不调用 SetScheduleMode"当成与"设默认 0"等价的无害行为——检视条例明确默认 0 即风险态（错误示例即 `BATCH_MODE_DEFAULT = 0`）。

## 误诊

- 同次整改对相邻发现 S1（用 `GetInputTensor` 取 Dtype、检视建议换 `GetInputDesc`）作出"不修改"决策，依据是"当前 CANN 9.2.0 的 TilingContext 未定义 GetInputDesc"——该说法只有整改 agent 的陈述，未对头文件做工具验证（证据等级：agent_reported），读者采用前应自行 grep 确认。
- 检视曾把同函数内重复的 `CopyAccToYNative()` 调用怀疑为内存安全风险；其假设检验结论为两次调用间 acc 缓冲未被修改、结果幂等、无精度影响，属 copy-paste 冗余而非同步缺陷（证据等级：agent_reported，检视报告分析）。
