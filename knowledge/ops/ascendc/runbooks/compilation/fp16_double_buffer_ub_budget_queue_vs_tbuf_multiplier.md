---
type: "Runbook"
title: "FP16 double-buffer UB 预算漏乘 2 致 VEC_ERROR"
description: "FusedBiasLeakyRelu arch35 tiling 在 FP16+double buffer 路径计算 ubFactor 时，bytesPerElem 未对 Queue 部分乘以 DOUBLE_BUFFER_MULTIPLIER，导致 ubFactor 高估、UB 溢出 ~29%，大 shape fp16 用例触发 VEC_ERROR。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T01:47:15Z"
platforms: ["950"]
---

# FP16 double-buffer UB 预算漏乘 2 致 VEC_ERROR

## 现象

对 ops-nn 仓库的 `FusedBiasLeakyRelu`（路径 `ops-nn/activation/fused_bias_leaky_relu`）跑 TTK v2 全量 271 用例（`ops-test-kit/fused_bias_leaky_relu/fused_bias_leaky_relu_kernel_cases_v2.csv`），在 Ascend950PR + CANN 9.0.0 下，3 个 FP16 大 shape 用例全部失败：

- `fblr_perf_fp16_18M`，shape (6000, 3000)=18M，tiling_key=1（double buffer）
- `fblr_perf_fp16_33M`，shape (2048, 2048, 8)=33M，tiling_key=1
- `fblr_edge_large_1d_fp16_float16`，shape (1000000,)=1M，tiling_key=1

三者的 `dyn_precision` 都是 `NO_OUTPUT`，`metrics: {'dyn': {}, 'cst': {}, 'bin': {}}`，TTK 的 `ttk-error.log` 显式记录 `VEC_ERROR encountered`（来源 `rts_interface.py:832`）。这 3 个用例在 2026-08-01 的旧安装下全部 `PASS`，重新编译安装 ops-nn 算子后开始失败。FP32 路径与小 shape（tiling_key=0，single buffer）FP16 用例不受影响。

HBM 充足（131072 MB 仅用几 GB，`rtMalloc` 均成功），排除 OOM；多卡健康状态排查后，在健康 device 上仍复现，确认是算子 tiling 代码问题。

## 适用场景

- **工程形态**：ops-nn 算子仓开发（ops-registry-invoke 体系），TTK（ops-test-kit）kernel 模式精度/执行验证。
- **API 路径**：`op_host/arch35/fused_bias_leaky_relu_tiling_arch35.cpp` 的 `FusedBiasLeakyReluTilingFunc`；kernel 侧 FP16 走 Cast 通路（x/bias cast 到 fp32 计算、再 cast 回 fp16），Queue 用 `BUFFER_NUM=2` double buffer。
- **验证过的平台与 CANN 版本**：Ascend950PR（ascend950），CANN 9.0.0（`/usr/local/Ascend/cann-9.0.0`）。仅 950 实测。
- **适用粒度**：`operator_family`——同一 tiling 范式（Queue+Cast 混合 UB 预算、Queue 受 double buffer 翻倍而 TBuf 单份）的 FP16 element-wise 算子。
- **路线优先级**：`recommended_fastpath`——对任何「Queue 走 double buffer、同时存在非 Queue 的 TBuf/cast 单份 buffer」的 tiling，UB 预算必须按 Queue 与 TBuf 分别计倍数。
- **不适用或未验证场景**：FP32 路径（无 cast buffer，受影响不同，本卡未覆盖其正确性证明）；a2/a3 平台未验证，推断可迁移（见 `## 失效边界`）；single buffer（tiling_key=0）路径不受本 bug 影响。

## 根因

`FusedBiasLeakyReluTilingFunc` 在 FP16 分支按下式估每元素 UB 占用：

```
bytesPerElem = TENSOR_NUM * TYPE_SIZE_FP16 + FP16_CAST_BUF_NUM * TYPE_SIZE_FP32
             = 3 * 2 + 3 * 4 = 18 bytes/elem
ubFactor     = FloorAlign(FloorDiv(ubSize, bytesPerElem), ubBlockSize)
             = FloorAlign(262144 / 18, ...) ≈ 14096
```

但 kernel 在 double buffer 模式下，3 个 Queue（x/bias/y 的 GM↔UB 搬运 buffer）各占 **2 份**（`BUFFER_NUM=2`），而 3 个 Cast buffer（`xFp32/biasFp32/yFp32`，TBuf）是单份。真实每元素占用：

```
bytesPerElem_real = 3 queues * 2 (double buffer) * TYPE_SIZE_FP16
                  + 3 cast bufs * 1 * TYPE_SIZE_FP32
                  = 3*2*2 + 3*4 = 24 bytes/elem
ubFactor_real     = FloorAlign(262144 / 24, ...) ≈ 10912
```

按错误的 18 bytes/elem 算出 `ubFactor≈14096`，远大于正确的 10912；UB 实际需求 `14096 * 24 = 338304 bytes` 超过 UB 容量 `262144 bytes`，溢出 `76160 bytes (29.1%)`，运行时 Vector 核执行越界触发 `VEC_ERROR`。

机制链点破：FP16 路径的 UB 预算里，**Queue 部分受 double buffer 翻倍、Cast(TBuf) 部分不翻倍**——这两类 buffer 的倍数不同，必须分别按各自倍数相加，不能把 Queue 与 TBuf 混在一起按单一倍数计。

## 解决配方

修复 `op_host/arch35/fused_bias_leaky_relu_tiling_arch35.cpp` 的 UB 预算计算。before/after 均为 `FusedBiasLeakyReluTilingFunc` 内 typeSize/ubBlockSize/blockFactor/useDoubleBuffer 之后的 UB 预算段（函数级 verbatim，取自 trace 的 file_read 事件）。

before（修复前，trace event 713/714 的 file_read 原文）：

```cpp
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? TYPE_SIZE_FP16 : TYPE_SIZE_FP32;
    int64_t ubBlockSize = BLOCK_ALIGN_BYTES / typeSize;

    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // FP16 path needs extra Cast buffers (xFp32, biasFp32, yFp32) at FP32 size
    bool isFp16 = (dataType == ge::DT_FLOAT16);
    int64_t extraBuf = isFp16 ? FP16_CAST_BUF_NUM : 0;
    int64_t totalBufNum = TENSOR_NUM + extraBuf;

    int64_t loopCountIfDouble = CeilDiv(blockFactor,
        static_cast<int64_t>(ubSize) / typeSize / (totalBufNum * DOUBLE_BUFFER_MULTIPLIER));
    uint64_t useDoubleBuffer = (loopCountIfDouble >= DOUBLE_BUFFER_THRESHOLD) ? 1 : 0;

    // UB budget: TENSOR_NUM buffers at typeSize + extraBuf buffers at FP32 size
    int64_t bufferCount = totalBufNum * (useDoubleBuffer ? DOUBLE_BUFFER_MULTIPLIER : 1);
    int64_t ubFactor;
    if (isFp16) {
        // FP16: 3 queues × sizeof(half) + 3 cast bufs × sizeof(float) = 3×2 + 3×4 = 18 bytes per element
        int64_t bytesPerElem = TENSOR_NUM * TYPE_SIZE_FP16 + FP16_CAST_BUF_NUM * TYPE_SIZE_FP32;
        ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), bytesPerElem), ubBlockSize);
    } else {
        ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, bufferCount), ubBlockSize);
    }
```

after（修复后，trace event 959 的 file_read 原文）：

```cpp
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? TYPE_SIZE_FP16 : TYPE_SIZE_FP32;
    int64_t ubBlockSize = BLOCK_ALIGN_BYTES / typeSize;

    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // FP16 path needs extra Cast buffers (xFp32, biasFp32, yFp32) at FP32 size
    bool isFp16 = (dataType == ge::DT_FLOAT16);
    int64_t extraBuf = isFp16 ? FP16_CAST_BUF_NUM : 0;
    int64_t totalBufNum = TENSOR_NUM + extraBuf;

    int64_t loopCountIfDouble = CeilDiv(blockFactor,
        static_cast<int64_t>(ubSize) / typeSize / (totalBufNum * DOUBLE_BUFFER_MULTIPLIER));
    uint64_t useDoubleBuffer = (loopCountIfDouble >= DOUBLE_BUFFER_THRESHOLD) ? 1 : 0;

    // UB budget: TENSOR_NUM queue buffers (doubled if double buffer) + extraBuf cast bufs (single)
    int64_t queueBufNum = TENSOR_NUM * (useDoubleBuffer ? DOUBLE_BUFFER_MULTIPLIER : 1);
    int64_t ubFactor;
    if (isFp16) {
        // FP16: queue buffers × sizeof(half) + cast bufs × sizeof(float)
        // Queue buffers are doubled in double buffer mode; cast bufs (TBuf) are single
        int64_t bytesPerElem = queueBufNum * TYPE_SIZE_FP16 + FP16_CAST_BUF_NUM * TYPE_SIZE_FP32;
        ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), bytesPerElem), ubBlockSize);
    } else {
        ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, queueBufNum), ubBlockSize);
    }
```

关键差异：

1. 把 `bufferCount`（按 `totalBufNum` 整体翻倍，会把不该翻倍的 Cast buffer 也翻倍）改为 `queueBufNum = TENSOR_NUM * (useDoubleBuffer ? 2 : 1)`，只对 Queue 部分翻倍。
2. FP16 分支 `bytesPerElem` 的 Queue 项用 `queueBufNum`（已含 double buffer 倍数），Cast 项 `FP16_CAST_BUF_NUM * TYPE_SIZE_FP32` 保持单份。
3. FP32 分支同步用 `queueBufNum`（FP32 无 Cast buffer，整体即 Queue 数）。

重编安装：`source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash && bash build.sh --ops=fused_bias_leaky_relu --soc=ascend950 --pkg -j16`（workdir `/workspace/cann/ops-nn`），随后安装到 `opp/vendors/custom_nn` 并清 TTK/atc kernel cache。

## 系统信息

- **工具链**：CANN 9.0.0（`/usr/local/Ascend/cann-9.0.0`），TTK `python3 -m ttk`。
- **目标平台**：Ascend950PR（ascend950），8 卡 131072 MB HBM。
- **相关文件**（与事件流实际读写一一对应）：
  - `ops-nn/activation/fused_bias_leaky_relu/op_host/arch35/fused_bias_leaky_relu_tiling_arch35.cpp`（修复点，事件 713/714 read before、717 edit、959 read after）
  - `ops-nn/activation/fused_bias_leaky_relu/op_kernel/arch35/fused_bias_leaky_relu.h`（kernel 侧 BUFFER_NUM/TENSOR_NUM 定义，事件 707 read）
  - `ops-nn/activation/fused_bias_leaky_relu/op_kernel/arch35/fused_bias_leaky_relu_tiling_data.h`、`..._tiling_key.h`
  - `ops-test-kit/fused_bias_leaky_relu/fused_bias_leaky_relu_golden_plugin.py`（golden，本卡不涉及）
- **常量**：`TENSOR_NUM=3`、`FP16_CAST_BUF_NUM=3`、`TYPE_SIZE_FP16=2`、`TYPE_SIZE_FP32=4`、`DOUBLE_BUFFER_MULTIPLIER=2`、`DOUBLE_BUFFER_THRESHOLD=2`、`BLOCK_ALIGN_BYTES=32`。

## 设计决策

- **为何 Queue 与 Cast buffer 倍数不同**：Queue 是 GM↔UB 的搬运 buffer，kernel 用 `BUFFER_NUM=2` 做双缓冲以重叠搬运与计算，运行时各占 2 份 UB；Cast buffer（`xFp32/biasFp32/yFp32`）是 kernel 内 `TBuf`，单份即可，不随 double buffer 翻倍。UB 预算必须分别按各自倍数累加。
- **被放弃的写法**：原 `bufferCount = totalBufNum * (useDoubleBuffer ? 2 : 1)` 把 Queue 与 Cast 一起翻倍——在 FP32 路径（无 Cast buffer）下碰巧等价，但 FP16 路径下既错误（漏乘 Queue）又自相矛盾（Cast 也被翻倍），是 bug 来源。
- **为何不改 kernel**：kernel 的 `BUFFER_NUM=2` 是性能需要（搬运/计算重叠），不应削弱；正确做法是在 tiling 侧如实反映 kernel 的 buffer 占用。
- **路线归属**：`recommended_fastpath`——任何 Queue+TBuf 混合的 UB 预算都应按此分别计倍数，不是 fallback。

## 验证信号

修复后重编安装并重跑原 VEC_ERROR 用例（TTK `ttk op` 子命令，fp16 + tiling_key=1 大 shape）：

- 修复前：`fblr_perf_fp16_18M`、`fblr_perf_fp16_33M`、`fblr_edge_large_1d_fp16_float16` 均 `dyn_precision=NO_OUTPUT`、`metrics: {'dyn': {}, 'cst': {}, 'bin': {}}`，`ttk-error.log` 记录 `VEC_ERROR encountered`（来源平台：950；证据等级: tool_observed）。
- 修复后：`fblr_perf_fp16_18M` 不再 NO_OUTPUT，产生精度指标 `MERE=0.00019371121015865356`（threshold 0.0009765625，通过）、`MARE=0.1033301055431366`（该 case 残留的 golden dtype 不匹配是另一独立问题）（来源平台：950；证据等级: tool_observed）。
- 小 shape `fblr_edge_smallval_float16`（fp16 + tiling_key=0，single buffer）修复前后均不触发 VEC_ERROR，与本根因无关（来源平台：950；证据等级: tool_observed）。

## 性能影响

本卡为 runtime 正确性修复，无性能故事（修复前直接 VEC_ERROR 无输出，修复后才可比较）。未验证精度影响（精度影响由后续 golden dtype 修复独立处理）。

## 预防

- 写 tiling UB 预算时，先列清 kernel 侧每类 buffer 的份数（Queue 是否 double buffer、TBuf 是否单份），再分别按倍数累加 `bytesPerElem`，禁止把 Queue 与 TBuf 混按单一倍数。
- 对 FP16 + Cast 通路，单独走 `bytesPerElem = queueBufNum * sizeof(half) + castBufNum * sizeof(float)` 分支，而非套用 FP32 的 `bufferCount` 公式。
- TTK 出现 `VEC_ERROR` + `NO_OUTPUT` + 大 shape + 特定 tiling_key 时，优先核对 tiling 的 UB/GM 预算算式，而不是先怀疑设备健康或 OOM。

## 失效边界

- **未验证，推断可迁移**：a2/a3 平台未实测；同 tiling 范式（Queue double buffer + Cast TBuf 单份）的 FP16 element-wise 算子推断可迁移，但 UB 容量、`DOUBLE_BUFFER_THRESHOLD`、`BLOCK_ALIGN_BYTES` 在不同 arch 可能不同，迁移时需用目标 arch 的 `GetCoreMemSize(UB)` 与 kernel buffer 定义复核算式。
- **不覆盖**：FP32 路径、single buffer 路径的正确性证明；Cast buffer 是否真为 TBuf 单份的 kernel 侧 verbatim 证据（event 707 read 被 redact）。

**未决项**：

- 【优先】完整 DoubleBuffer 配置（tiling `bufNum=2` + kernel 队列 depth=2）上板复验——该配置在另一独立 trace 的增量内未上板验证，启用时须重跑全量 ST。
- `fblr_perf_fp16_18M` 残留的 golden dtype 不匹配（MARE 超阈）为另一独立问题，未在本卡闭环。
## 坏实践

- 把 `bufferCount = totalBufNum * (useDoubleBuffer ? 2 : 1)` 当万能公式套到 FP16 Cast 通路——对 FP32 碰巧等价，对 FP16 既有「Queue 漏乘」又有「Cast 误乘」，掩盖性强。
- TTK 见 `VEC_ERROR`/`NO_OUTPUT` 就先怀疑设备 Alarm 或 OOM，未先核对 tiling UB 算式。

## 误诊

本开发会话 在定位本根因前曾长时间误诊为 `compile info not contain [_pattern]` / `INVALID_TILING`：因删除 `opp/vendors/customize` 目录破坏环境、以及 ops-nn `build.sh --ops=fused_bias_leaky_relu` 在 build 目录残留了 advance_step 的旧 `libcust_opmaster_rt2.0.so`（`FusedBiasLeakyRelu` count=0）所致。期间 agent 一度尝试修改公共 `asc_op_compile_base/asc_op_compiler/op_tiling.py` 注入 `_pattern`，被用户驳回「别的算子可以编译，不要修改公共代码」。真正解法是清理 build 目录残留 .so 后重编安装，与本次 UB 根因无关，记录于此供下个 agent 跳过该弯路。

## 复现增量：argmax 实例——bufNum 作为 tiling 字段的多路径量纲公式 + 回退隔离

另一独立轨迹并入 argmax 算子（arch35/CANN 9.1.0）的同根因实例：kernel 输入队列从单缓冲升级 DoubleBuffer（`BUFFER_NUM` 1→2、`TQue<VECIN, 1>` → `TQue<VECIN, 2>`）而 tiling 的 perElementBytes 不含 bufNum 因子时，`InitBuffer(inQueueX, BUFFER_NUM, ubFactorR_*dtypeSize)` 实际分配 = **BUFFER_NUM × ubFactorR × dtypeSize**，是预算的 2 倍——ST 失败数从 2 条暴涨到 16 条（14 条统一 `流同步失败: 507035`）。回退 `BUFFER_NUM=1/TQue depth=1` 后 14 条 507035 全部恢复（回退隔离证明失败随缓冲深度配置确定性地出现/消失）。

修复模式（两侧量纲对齐，bufNum 从 `tiling->bufNum` 读入、与 kernel BUFFER_NUM 同源约定，避免两处魔法数）：

```cpp
uint64_t bufNum = static_cast<uint64_t>(tiling->bufNum);
...
// Float-AR / Int32-AR（Cast 路径）：
uint64_t perElementBytes = needsCastToFloat ?
    (bufNum * dtypeSize + computeDtypeSize) :
    (bufNum * computeDtypeSize);
// Int32-AR 树归约路径（含 cmp 掩码字节的保守式）：
uint64_t perElementBytes = bufNum * computeDtypeSize + computeDtypeSize + 1;  // bufNum*4 + 4 + 1
// Int64-AR：
uint64_t perElementBytes = bufNum * dtypeSize;  // bufNum * 8
```

启用/修改队列深度前先写两侧量纲等式核对：`BUFFER_NUM × ubFactorR × dtypeSize + 固定缓冲 ≤ UB 预算`，其中 tiling 的 perElementBytes 必须含同一 bufNum；DoubleBuffer 变更单独成轮提交并跑全量 ST。止血可先回退单缓冲（safety_fallback，牺牲搬运/计算重叠的吞吐收益，未量化）。

**独立 trace 新增验证信号**（均为 tool_observed）：

- BUFFER_NUM=2 + 预算无 bufNum 时：ST 143/159，其中 14 条 `[FAIL] 流同步失败: 507035`（来源平台：950；证据等级：tool_observed）。
- 回退 BUFFER_NUM/TQue depth=1 后：同批失败消失（156/159，仅剩已知 int64 框架路由 2 条 + tiling 桩大 R 1 条）（来源平台：950；证据等级：tool_observed）。
- 补 bufNum 因子 + 修复 tiling 桩后（bufNum=1 下公式为恒等）：157/159 恢复（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界**：`bufNum=2` + kernel depth=2 的完整 DoubleBuffer 配置在该轨迹内未上板验证（后续会话完成 UT 期望对齐后启用），启用时须重跑全量 ST；非 VECIN 队列（TBuf 固定缓冲）不涉及本口径。
