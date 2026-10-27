---
type: "Runbook"
title: "Max/Min 逐元素非标量广播 须 Duplicate 至同长 buffer"
description: "AscendC count-based Max/Min(dst,src0,src1,count) 是逐元素比较，src1 必须是与 src0 同长度的完整 tensor，不做标量广播；要用标量做 clamp 须先 Duplicate(scalar) 填充同长度 tensor，且该 buffer 容量须 ≥ N×sizeof(float)（32B clampBuf 远不够），应复用阶段内已空闲的 N×float buffer。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.comparison_logic", "topic.compute.quantization"]
status: draft
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# Max/Min 逐元素非标量广播 须 Duplicate 至同长 buffer

## 现象

`add_rms_norm_dynamic_quant` 算子修复“Abs 丢符号”后重测，大 N mismatch 降至约 2%，但小 N 仍高（M=1,N=16,fp16：0.9375；M=4,N=128,fp16：0.4785；M=64,N=100,fp16：0.9384）。继续追踪：NPU `y` 值仍是非常小的值（0, 1, -1, 2），而 `scale` 正确、`q_xout`（从 xOut 用 FP32 重算的参考）与 `y_ref` 完全一致。定位到 Step 5e 的 `ClampToInt8Range`——原实现用高维 `Max/Min` 配 `src1RepStride=0` 做标量广播。把 clamp 改成简单 `Max<float>(data, data, clampLocal, count)`（`clampLocal` 只 `SetValue(0, ±val)`）后，mismatch 反而全面升至约 0.93。最终改用 `Duplicate<float>(clampLocal, val, count)` 先填充同长度 tensor、再 `Max/Min(data, data, clampLocal, count)`，并把 `clampLocal` 从 32B 的 `clampBuf_` 换到 N×float 的 `workBuf2_`（xOutFp32，clamp 阶段已空闲）后，mismatch 降至 0.4-0.8%、再查差异分布全部 ≤±1。

## 适用场景

- **工程形态**：AscendC count-based Max/Min(dst,src0,src1,count) 是逐元素比较。
- **API 路径**：AscendC `Max/Min`（count-based Level 2 接口）、`Duplicate`、UB `TBuf` 复用。
- **验证平台与版本**：ascend910b（dav-2201），CANN 9.0.0。不在注册平台集合 {950,a3,a2}，`platforms` 为空表。
- **适用粒度**：`operator_family`（任何在 UB 上用标量做逐元素 clamp/比较广播的算子）。
- **路线优先级**：`recommended_fastpath`（Duplicate 标量到同长度 buffer 再做逐元素 Max/Min）。
- **不适用/未验证场景**：高维 `Max/Min` 配正确的 `src1RepStride=0` 广播若 mask/repStride 计算正确亦可，但本 trace 证明其 mask/repStride 易错（小 N tail 分支尤其），不推荐。

## 根因

两个叠加根因：

1. **count-based `Max/Min` 是逐元素、非标量广播**。`Max<float>(dst, src0, src1, count)` 执行 `dst[i]=max(src0[i], src1[i])`，`src1` 必须是与 `src0` 同长度的完整 tensor。只对 `src1` 做 `SetValue(0, val)`（仅第 0 元素赋值）后直接 `Max(count)`，会比较 `src1` 中**未初始化的其余元素**，把 `data` 损坏成垃圾值（实测 mismatch 升至 0.93）。标量广播必须先把标量 `Duplicate` 到同长度 tensor，再逐元素 `Max/Min`。
2. **广播 buffer 容量须 ≥ N×sizeof(float)**。原 `clampBuf_` 仅 `pipe_.InitBuffer(clampBuf_, 32)`（32 字节，为放单个标量设计），对 `Duplicate<float>(clampBuf_, val, count)`（N=4096 需 16KB）严重不足，`Duplicate` 会越界写相邻 UB 区域。须改用一个在 clamp 阶段已空闲、容量为 N×float 的 buffer（本 trace 用 `workBuf2_`=xOutFp32，Step 4 已写回 GM）。

高维 `Max/Min`（带 `mask/repeatTime/{src0RepStride,src1RepStride,...}`）的 `src1RepStride=0` 广播在原理上可行，但 mask/fullRepeats/tailCount/repStride 的计算对小 N 的 tail 分支易错，本 trace 放弃该路径。

## 解决配方

修复前（原 `ClampToInt8Range`，高维 `Max/Min` 配 `src1RepStride=0`，mask/repStride 计算易错）：

```cpp
    // Clamp FP32 values to [-128, 127] using Max/Min with src1RepStride=0 broadcast
    __aicore__ inline void ClampToInt8Range(
        AscendC::LocalTensor<float>& data,
        int32_t count)
    {
        auto clampLocal = clampBuf_.Get<float>();

        // Set -128.0 for Max (clamp lower bound)
        clampLocal.SetValue(0, -128.0f);
        // Max: data = max(data, -128)
        // Using high-dimension API with src1RepStride=0 for broadcast
        // mask = count (number of elements per repeat)
        // For count elements, we need to compute repeat and mask
        int32_t elementsPerRepeat = 64;  // 256 bytes / 4 bytes = 64 FP32 elements
        int32_t fullRepeats = count / elementsPerRepeat;
        int32_t tailCount = count % elementsPerRepeat;
        int32_t repStride = static_cast<int32_t>(nColsAlign_) / 8;  // in blocks of 8 floats (32 bytes)

        // Process full repeats
        if (fullRepeats > 0) {
            uint64_t mask = 64;
            AscendC::Max<float>(data, data, clampLocal,
                mask, static_cast<uint8_t>(fullRepeats),
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0});
        }
        // Process tail
        if (tailCount > 0) {
            uint64_t mask = static_cast<uint64_t>(tailCount);
            auto dataTail = data[fullRepeats * elementsPerRepeat];
            AscendC::Max<float>(dataTail, dataTail, clampLocal,
                mask, 1,
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0});
        }

        // Set 127.0 for Min (clamp upper bound)
        clampLocal.SetValue(0, 127.0f);
        // Min: data = min(data, 127)
        if (fullRepeats > 0) {
            uint64_t mask = 64;
            AscendC::Min<float>(data, data, clampLocal,
                mask, static_cast<uint8_t>(fullRepeats),
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0});
        }
        if (tailCount > 0) {
            uint64_t mask = static_cast<uint64_t>(tailCount);
            auto dataTail = data[fullRepeats * elementsPerRepeat];
            AscendC::Min<float>(dataTail, dataTail, clampLocal,
                mask, 1,
                {1, 1, 1, static_cast<uint8_t>(repStride), static_cast<uint8_t>(repStride), 0});
        }
    }
```

修复后（用 `Duplicate` 把标量填充到同长度 `workBuf2_`，再逐元素 `Max/Min`）：

```cpp
    // Clamp FP32 values to [-128, 127] using Duplicate + Max/Min
    // Uses workBuf2_ as temp buffer for the broadcast values (N × sizeof(float))
    __aicore__ inline void ClampToInt8Range(
        AscendC::LocalTensor<float>& data,
        int32_t count)
    {
        // Use workBuf2_ as temp for broadcast (xOutFp32 no longer needed at this point)
        auto clampLocal = workBuf2_.Get<float>();

        // Clamp lower bound: max(data, -128)
        AscendC::Duplicate<float>(clampLocal, -128.0f, count);
        AscendC::Max<float>(data, data, clampLocal, count);

        // Clamp upper bound: min(data, 127)
        AscendC::Duplicate<float>(clampLocal, 127.0f, count);
        AscendC::Min<float>(data, data, clampLocal, count);
    }
```

## 设计决策

- **放弃高维 Max/Min**：高维接口的 `mask`/`fullRepeats`/`tailCount`/`repStride` 计算对小 N 的 tail 分支易错（小 N mismatch 仍 0.94），改用更简单的 count-based 接口 + `Duplicate` 广播。
- **广播 buffer 选 `workBuf2_` 而非 `clampBuf_`**：`clampBuf_` 仅 32B（为单标量设计），`Duplicate` N 个 float 需 N×4 字节。`workBuf2_`（xOutFp32）在 Step 4 已写回 GM、Step 5e 时空空闲，且容量为 N×float，复用其视图作广播 buffer；`Get<float>()` 只返回视图、非新分配，顺序使用下与 Step 5a 的 `absBuf`（也复用 `workBuf2_`）不冲突。
- **路线属 `recommended_fastpath`**：count-based + Duplicate 比高维接口更可读、更不易错。

## 预防

在 UB 上用标量做逐元素 clamp/比较时，模式固定为：`Duplicate(buf, scalar, count)` 填充同长度 tensor → `Max/Min(data, data, buf, count)`；不要只 `SetValue(0, scalar)` 后直接 `Max(count)`（会用未初始化值逐元素比较）。广播 buffer 容量须 ≥ count×sizeof(T)，不要用为单标量设计的小 buffer（如 32B）。

## 验证信号

来源：本轮算子精度联调的 clamp 修复迭代。修复后（Abs 修复 + 本 clamp 修复一并生效）M=128,N=4096,fp16 实测差异分布：`diff=-1` 共 956 元素、`diff=0` 共 522340 元素、`diff=+1` 共 992 元素，`|diff|<=1` 占 100.00%、`|diff|==0` 占 99.63%、max abs diff=1.0。剩余 ±1 为 FP32 vs FP64 量化边界，非本卡根因。（来源平台：未定位；证据等级: tool_observed）
## 失效边界

本卡覆盖“标量广播 clamp 的正确实现与广播 buffer 容量约束”。剩余 ±1 拼差属 FP32 vs FP64 边界，不在此卡。本 trace 验证平台为 ascend910b（dav-2201）；count-based Max/Min 与 Duplicate 的语义在其它 AICore 微架构推断一致——未验证，推断可迁移。

## 误诊

- **误诊 1**：先尝试把高维 `Max/Min` 换成 `SetValue(0, val)` + `Max<float>(data, data, clampLocal, count)`，假设 count-based 接口会广播 `clampLocal[0]`。实测 mismatch 反升至约 0.93——count-based 接口是逐元素比较，`clampLocal` 未初始化元素参与比较，损坏数据。此误诊揭示了根因 1（逐元素非广播）。
- **误诊 2**：改用 `Duplicate<float>(clampLocal, val, count)` + `Max/Min(count)`，但 `clampLocal` 仍取自 32B 的 `clampBuf_`。仍不正确——`clampBuf_` 容量不足 N×float，`Duplicate` 越界。此误诊揭示了根因 2（广播 buffer 容量）。