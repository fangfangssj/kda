---
type: "Runbook"
title: "kernel 源码改动未生效因 binary 缓存"
description: "改 kernel 源码并重编后 NPU 运行时仍跑旧 binary、输出与改前逐字节一致；主机侧 volatile 变量不进 device binary hash，加 PRINTF 才能改 hash 强制加载新 binary。"
tags: ["task.debugging", "topic.implementation.debugging"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# kernel 源码改动未生效因 binary 缓存

## 现象

在 Ascend950 上调试 ApplyCamePart2 Branch-0 fp32 Kernel 时，多次出现「改了 kernel 源码、`build.sh` 重编成功、`run.sh` 重跑，但输出与改动前**逐字节一致**」的现象。典型场景：把 `ComputeSumR` 由 `DataCopy` 读 r 改为 `GetValue` 直读，并加 `static volatile int force_recompile_vN = N;` 试图改 binary hash，重编重跑后 `sumR_` 与 `u` 的值纹丝不动。随后在 kernel 内加一行 `PRINTF("sumR=%f\n", sumR_);`，重跑发现 `sumR_` 立即变为正确值（4.6/1.0）、B0-S0/S1/S2 精度修复生效并 PASSED；把 PRINTF 注释掉重跑，输出又回到错误值——即「加 PRINTF → 修复生效，去 PRINTF → 退回错误」可复现切换。

出问题时的「无效强制重编」写法（kernel 源码内，verbatim，取自 apply_came_part2_apt.cpp 多轮 edit 的公共片段）：

```cpp
// Force binary recompilation (dummy attr to change kernel hash)
static volatile int force_recompile_v4 = 44;

extern "C" __global__ __aicore__ void apply_came_part2(
    GM_ADDR grad, ...)
{
    // ... kernel body unchanged across force_recompile_vN bumps ...
}
```

该 `force_recompile_vN` 在多轮修复中被从 `_v2=42` 一路 bump 到 `_v8=48`，均未使改动生效。

## 适用场景

- **工程形态**：Kernel 直调（ops-registry-invoke，Ascend C device kernel）+ `build.sh` 重编 + `run.sh` 上板 ST 运行。
- **API 路径**：`build.sh <op>_package`（CANN `npu_op_kernel_sources`/`npu_op_kernel_library` CMake 宏编译 kernel）；kernel binary 缓存目录（`kernel_meta` 等）；`PRINTF`（device 侧打印，受 `ASCENDC_DUMP` 开关控制）。
- **验证平台与版本**：Ascend950（CANN 1.0.0, TimeStamp 202602）。trace 内仅在此平台实测。
- **适用粒度**：`platform_wide`（凡 Ascend950 kernel 增量重编场景均可能命中）。
- **路线优先级**：`recommended_fastpath`——判定 binary 是否生效的标准做法是加一行 device 侧 `PRINTF`（改变 device binary hash）做对照；若加 PRINTF 后行为变化即确诊缓存未刷新。生产清理应清 `kernel_meta`/`build` 目录全量重编，而非依赖主机侧 volatile 变量。
- **不适用或未验证场景**：a2/a3 平台未验证（推断同 binary 缓存模型，但未实测）；首次全量编译（无缓存）不命中；纯 host 侧（op_host/tiling）改动不涉及 device binary，不在本卡范围。

## 判别方法

**触发条件**：kernel 源码改动 + `build.sh` 重编成功 + `run.sh` 重跑后，输出（`u`/`sumR_`/`r_out` 等任何值）与改动前逐字节一致，且改动应产生可观测差异（如改了 `sumR_` 计算路径，期望 `sumR_` 变化）。

**可执行探针**：在 kernel 内改动点附近加一行 device 侧 `PRINTF`（即便 `ASCENDC_DUMP=0` 不打印，PRINTF 调用本身改变 device binary hash），重编重跑，对比输出：
1. 若加 PRINTF 后输出变化（改动生效）→ 确诊 binary 缓存未刷新。
2. 若加 PRINTF 后输出仍不变 → 排除缓存问题，改去查源码改动是否真的进入编译单元（路径/宏/CMake 源文件列表）。

```cpp
// 在改动点附近加（device 侧，改变 binary hash）：
PRINTF("probe sumR=%f n=%lld\n", sumR_, (long long)n_);
```

对照实验：加 PRINTF 跑一次 → 注释 PRINTF 重跑一次 → 对比两次输出。

**原始观测**（Ascend950，ApplyCamePart2 Branch-0，多轮修复迭代）：
- 改 `ComputeSumR`（GetValue 替代 DataCopy）+ bump `force_recompile_v4` 后，`sumR_` 与 `u` 输出与改前完全一致。
- 加 `PRINTF("sumR=%f", sumR_);` 后重跑：`sumR_=4.600000`（B0-S0，正确）、`1.000000`（B0-S1）、`4.600000`（B0-S2），B0-S0/S1/S2 精度达 1e-5 阈值并 PASSED。
- 移除该 PRINTF 重跑：输出回到改动前的错误值（`u[0] got -3.00000042e-01` 等），B0-S0/S1/S2 再次 FAILED。
- LOG 记录：`NPU运行时缓存旧二进制致修复代码未生效（PRINTF验证新binary加载但输出不变）`——即此前曾用 PRINTF 确认新 binary 已加载，但当时改动未触及真正根因，故输出仍不变；本卡区分「binary 未生效」与「binary 已生效但改动无效」两层。

**机制解释**：CANN 编译流水线把 device kernel 编译为二进制并按内容 hash 缓存（`kernel_meta` 等目录）。增量重编时，若编译器判定 kernel 源码的 device 侧可生成内容未变（hash 不变），则复用缓存的旧 binary，运行时加载的仍是旧 binary。主机侧的 `static volatile int force_recompile_vN` 是 host/global 侧符号，**不进入 device kernel binary 的内容 hash**，故 bump 它不会使缓存失效。`PRINTF` 是 device 侧代码，调用会改变 device binary 的可生成内容 → hash 变化 → 缓存失效 → 运行时加载新 binary。这就是「加 PRINTF 生效、去 PRINTF 退回」的机制：PRINTF 充当了无意中的 binary hash 扰动项。注意 `ASCENDC_DUMP=0` 只关闭打印输出，不撤销 PRINTF 调用对 binary hash 的影响。

## 解决配方

经 trace 验证可行的判定与清理方法：

1. **判定 binary 是否生效**：用上述 PRINTF 对照实验——加 PRINTF 后输出变化即确诊缓存未刷新。这是最小成本、可复现的判定法。
2. **强制加载新 binary**：清 `kernel_meta` 与 `operators/<op>_package/build` 目录后全量重编（`rm -rf build && bash build.sh <op>_package`），不依赖 host 侧 volatile 变量。
3. **不要**用 `static volatile int force_recompile_vN` 试图改 device hash——它不进 device binary，无效；本 trace 中从 `_v2` bump 到 `_v8` 均未生效。

注意：本卡无完整可执行 after 代码块需求（diagnostic_rule 形态，核心是判别方法与机制解释，非函数级代码修复）；上述清理命令为 trace 内实际使用的 bash 命令（`rm -rf operators/ApplyCamePart2_package/build && bash operators/build.sh apply_came_part2_package`）。

## 验证信号

本卡验证信号来自 ApplyCamePart2 Branch-0 fp32 Kernel 在 Ascend950 上的多轮修复迭代（同一 trace）。

- 改 `ComputeSumR` 源码 + bump `force_recompile_v4` 后，`sumR_`/`u` 输出与改前逐字节一致（来源平台：Ascend950；证据等级: tool_observed）。
- 加 `PRINTF("sumR=%f", sumR_);` 后重跑，`sumR_=4.600000/1.000000/4.600000`（正确），B0-S0/S1/S2 精度达 1e-5 阈值并 PASSED（来源平台：Ascend950；证据等级: tool_observed）。
- 移除该 PRINTF 重跑，输出回到改动前错误值，B0-S0/S1/S2 再次 FAILED（来源平台：Ascend950；证据等级: tool_observed）。
- LOG 记录「NPU运行时缓存旧二进制致修复代码未生效」（来源平台：Ascend950；证据等级: artifact_verified）。

## 失效边界

- 本结论仅在 Ascend950（CANN 1.0.0, 202602）实测。
- a2/a3 平台：未验证，推断可迁移（CANN 编译缓存模型跨平台同源），但未实测，采用前需先在目标平台用 PRINTF 对照实验确认。
- 缓存键的精确构成（源文件 mtime vs. 内容 hash vs. 编译选项 hash）未在 trace 内探明——本卡只确认「device 侧内容变化（PRINTF）能失效缓存、host 侧 volatile 不能」，不对更细的缓存键边界作出结论。
- 清 `kernel_meta` 是否等价于 PRINTF 改 hash、是否需同时清 `build` 目录——trace 内观察到 `rm -rf build && bash build.sh` 的全量重编用法，但未做「只清 kernel_meta 不清 build」的对照实验。

## 坏实践

- 用 `static volatile int force_recompile_vN = N;` 并不断 bump N 试图改 device hash——它不进 device binary，从 `_v2` bump 到 `_v8` 均无效，是最常见的徒劳操作。
- 见「改了源码、build 成功、输出没变」就怀疑自己改错了代码逻辑，而不先排除 binary 缓存——会把真正生效的修复误判为无效，反而回退正确改动。
- 把「加 PRINTF 就过、去 PRINTF 就不过」误判为「PRINTF 改变了 kernel 时序/语义修复了 bug」——PRINTF 在本场景只是 binary hash 扰动项，真正起作用的是被缓存遮蔽的源码改动；时序敏感的 Heisenbug 才是另一回事，不可混淆。

## 误诊

- 初判曾把「输出不变」归因为「自己的 `ComputeSumR` 改动逻辑不对」而回退——实际改动逻辑正确，只是 binary 未生效。PRINTF 对照实验可区分这两层（证据等级: agent_reported；记录调试路径）。
- LOG 早期条目「PRINTF验证新binary加载但输出不变」曾导致误判「binary 已生效、改动无效」——实际是当时改动未触及真正根因（GM 回读可见性，见相关卡 *GM SetValue 写回后 GetValue 读到旧值*）；binary 生效与改动触及根因是两个独立条件，需分别验证。

## 预防

- 凡 kernel 源码改动后跑 ST，先做 PRINTF 对照实验确认 binary 生效，再判断改动逻辑对错。
- 增量重编不可靠时，默认 `rm -rf <op>_package/build && bash build.sh <op>_package` 全量重编，不依赖 host 侧 volatile 变量。
- 评审 kernel 修复时，若 PR 含 `static volatile int force_recompile_vN` 这类「强制重编」占位变量，应要求改用全量清理重编并删除占位符——它是 binary 缓存问题的症状而非解法。