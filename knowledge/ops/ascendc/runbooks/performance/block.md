---
type: "Runbook"
title: "逐 block 串行启动退化多核并行"
description: "Ascend950 上按行切分多核的算子，若 launch 函数用 host 侧 for 循环逐 block 调 <<<1, nullptr, stream>>>，多核并行会退化为'启动开销 × block 数'的串行链；改单次 <<<numBlocks>>> + kernel 内 GetBlockIdx/GetBlockNum 自算行区间后，case1 耗时 42866µs→705µs（约 60× 相对）、20/20 精度不变。"
tags: ["task.optimization", "topic.implementation.pipeline", "topic.tiling", "topic.performance.optimization.optimization_overview"]
status: stable
sources: []
created_at: "2026-08-18T07:57:43Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 逐 block 串行启动退化多核并行

## 现象

Ascend950（dav-3510，16 AIV 核）上优化 `AddRmsNormDynamicQuant`（Add+RmsNorm+AbsMax+DynamicQuant 融合）算子。精度已全过 20/20，但性能评测显示 kernel 远低于计算能力：case1（M=8192 N=8192）耗时 42866µs、相对基线加速比 0.0135，多 case 加速比量级 0.002~0.025；case6（M=524288 N=128，行数最多）耗时 1484430µs。代码审计发现 launch 函数把多核切分写成 host 侧 `for (b = 0; b < numBlocks; ++b)` 循环，每轮用 `add_rms_norm_dynamic_quant_kernel<T><<<1, nullptr, stream>>>(..., startRow, endRow, ...)` 启动一个单 block 核函数。

## 适用场景

- **工程形态**：ops-direct-invoke-flash Kernel 直调核函数构建（`op_kernel/*_kernel.cpp` 的 `extern "C" launch_*` 函数 + `__global__ __aicore__` kernel 模板）。
- **API 路径**：`kernel<<<numBlocks, nullptr, stream>>>(...)` 单次多块启动；kernel 内 `AscendC::GetBlockIdx()` / `AscendC::GetBlockNum()` 自算行区间；`platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv()` 取核数。
- **验证过的平台**：Ascend950（dav-3510），16 AIV 核；CANN 版本未在 trace 中显式记录。
- **适用粒度**：`platform_wide`——Ascend950 上任何按行（token）切分多核、launch 函数当前用 `for` + `<<<1>>>` 的算子。
- **路线优先级**：`recommended_fastpath`——单次 `<<<numBlocks, nullptr, stream>>>` 启动 + kernel 内自算行区间。
- **不适用或未验证场景**：a2/a3 平台未验证（`GetBlockIdx/GetBlockNum` 与 `<<<numBlocks>>>` 是 AscendC 跨平台语义，但核数与 `GetCoreNumAiv` 返回值不同，tiling 上限需复核）；非按行切分的算子（cube/matmul 切分模型）不适用本判别。

## 判别方法

- **触发（trigger）**：按行切分多核的算子精度全过、但性能评测加速比量级在 0.002~0.025、单 case 耗时远高于基线（case1 42866µs vs 基线 577µs），且随行数增长劣化（case6 1484430µs）。tool_observed（eval stdout）。
- **探针（probe）**：静态检查 `launch_*` host 函数体，确认是否存在 host 侧 `for (...; b < numBlocks; ...)` 循环包住 `kernel<<<1, nullptr, stream>>>(..., startRow, endRow, ...)`——即每次 launch 只启动 1 个 block、行区间由 host 算好传入。可执行检查：在同一 `launch_` 函数体内同时 grep 到 `<<<1,` 与 `for` 循环头即命中。
- **观测（observation）**：触发态 eval stdout 显示所有 case 加速比量级 0.002~0.025、case1 42866µs、case6 1484430µs（tool_observed，raw transcript Turn 67 Tool Results）。改单次 `<<<numBlocks, nullptr, stream>>>` + kernel 内 `GetBlockIdx/GetBlockNum` 自算 `[startRow, endRow)` 后，同 case1 耗时降到 705µs（约 60× 相对）、20/20 精度不变（tool_observed，raw transcript Turn 76 Tool Results）。
- **判读（interpretation）**：`for + <<<1>>>` 每次 launch 只让 1 个 block 跑，16 个 AIV 核没有在同一次 launch 里同时跑而是排队启动，多核并行退化为"启动开销 × block 数"的串行链；每次 launch 还付一次 host→device 同步开销，乘以 numBlocks 倍。kernel 内每行又有 `PIPE_ALL` barrier、`Duplicate` 清零、gamma 重复搬运等固定开销，行数越多放大越严重。单次 `<<<numBlocks>>>` 让 16 核在同一次 launch 里真并行、启动开销只付一次，故 case1 约 60× 相对提升。行区间从 host 算移到 kernel 内 `GetBlockIdx/GetBlockNum` 自算，tiling 结构体只剩 `numBlocks`，与同仓 `dequant_swiglu_quant` 的 `<<<numBlocks>>>` + `GetBlockIdx` 模式一致。

## 验证信号

- 触发态 eval（raw transcript Turn 67 Tool Results）：case1 elapsed=42866µs baseline=577.3µs speedup=0.0135、case6 elapsed=1484430µs、多 case 加速比量级 0.002~0.025、20/20 精度全过（来源平台:Ascend950；证据等级:tool_observed）。
- 改单次 `<<<numBlocks>>>` 启动后 eval（raw transcript Turn 76 Tool Results）：case1 elapsed=705µs（约 60× 相对提升）、20/20 精度不变（来源平台:Ascend950；证据等级:tool_observed）。