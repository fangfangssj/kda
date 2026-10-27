---
type: "Runbook"
title: "arch35 kernel 上板报 507035 实为 arch 不匹配"
description: "编译通过的 ascend950（arch35）SIMT kernel 在 Ascend910_93（arch22）板上运行 L0 用例 20/22 失败、报 ACL_ERROR_RT_VECTOR_CORE_EXCEPTION（507035）；判别方法为先核对 build --soc 与 npu-smi 硬件 arch，再怀疑 kernel 逻辑。该路线在 arch22 板上 banned，只能改路线。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.execution_model.simt"]
status: draft
sources: []
created_at: "2026-08-18T04:29:45Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: []
---

# arch35 kernel 上板报 507035 实为 arch 不匹配

## 现象

开发 aclblasTbsv（BLAS Level-2 三角带状矩阵求解）算子，设计采用标量路径（n<128）+ SIMT 路径（n≥128）双路径、单核执行，目标芯片 `ascend950`（arch35 / dav-3510），FP32。算子代码与 ST 测试均已用 `bash build.sh --ops=stbsv --soc=ascend950` 编译通过。

进入汇合联调运行 `./build/test/tbsv/stbsv/stbsv_test` 后，22 条 L0 用例里只有 2 条通过（`n=0`、`k=0`+unit 的快速返回用例，kernel 根本不执行），其余 20 条全部失败，gtest 输出：

```
    Which is: 0
Unexpected NPU error code: 6
[  FAILED  ] Stbsv/StbsvArch35Test.CsvDriven/TC_L0_22, where GetParam() = TC_L0_22 (14 ms)
[==========] 22 tests from 1 test suite ran. (1772 ms total)
[  PASSED  ] 2 tests.
[  FAILED  ] 20 tests, listed below:
[  FAILED  ] Stbsv/StbsvArch35Test.CsvDriven/TC_L0_01, where GetParam() = TC_L0_01
... (TC_L0_02 .. TC_L0_22 中除快速返回外全失败)
20 FAILED TESTS
```

错误码即 `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（507035）——kernel 在 AI Core 上崩溃。代码刚编译通过、没有任何报错，容易被当成 kernel 逻辑 bug 去查。

## 适用场景

- **工程形态**：算子 kernel 直调 / 上板 ST 测试（ops-blas `build.sh --ops=<op> --soc=<soc>` + `./build/test/.../<op>_test`）。
- **API 路径**：AscendC SIMT 编程模型（arch35 / ascend950 专属），`__global__` kernel + `GlobalTensor` / `TQue` / `LocalTensor` / `KERNEL_TASK_TYPE_DEFAULT` 任务类型派生。
- **验证过的平台与 CANN 版本**：失败信号实测于 Ascend910_93（arch22 / dav-2201）板卡，驱动 `Version=25.5.1`、`ascendhal_version=7.35.23`、CANN `cann-9.0.0`；目标 SOC `ascend950`（arch35）的 kernel 在本 trace 中未在真实 950 板上验证通过。arch22 不在注册平台词表 `{950, a3, a2}` 内，按未定位处理 → `platforms: []`。
- **适用粒度**：`platform_wide`（凡是为 arch35/ascend950 编译的 SIMT kernel 落到 arch22 板上都会复现，与具体算子无关；本 trace 以 tbsv/stbsv 为观测样本）。
- **路线优先级**：`banned`——在 arch22 硬件上运行 arch35 SIMT kernel 这条路线禁止采用；既无法通过、也无代码侧修复能使其通过，必须改路线。
- **明确不适用或未验证场景**：① arch35 kernel 在真实 Ascend950 板上的精度/性能本 trace 未验证；② 若已有 arch22（SIMD/membase）实现目录，`--soc=ascend910_93` 不会跳过、可在 arch22 板正常上板——那条路径本 trace 未实现、未验证；③ 非 SIMT（纯 SIMD/tensor）arch35 kernel 是否在 arch22 上同样报 507035 未在本 trace 验证（判别方法仍适用，但归因需另行确认）。

## 判别方法

形态：`diagnostic_rule`（触发 → 可执行探针 → 原始观测 → 解释）。本卡不展示 before/after 代码修复（无真代码修复，只有判别与改路线）。

**触发信号（任一即应怀疑 arch 不匹配，先于 kernel 逻辑排查）**：

1. kernel 用 `--soc=ascend950`（或任何 arch35/ascend950 目标）编译通过、无报错，但上板 ST 立刻大量失败、报 `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（507035，"Unexpected NPU error code: 6"）。
2. 失败 pattern 呈"全崩 + 极少数通过"，且通过的多是 `n=0` / `k=0` / unit 这类不执行 kernel 主体的快速返回用例（它们 PASSED 不是因为算得对，而是因为 kernel 没跑）。

**可执行探针（逐条在受影响的机器上跑，verbatim 取自本 trace 的 bash 工具调用）**：

```bash
# 1) 看板卡真实型号 / arch（不是看 build target）
npu-smi info 2>/dev/null | head -20
npu-smi info -t board -i 0 2>/dev/null | grep -E "Chip|Name|Type" | head -5
# 2) 看驱动 / hal 版本
cat /usr/local/Ascend/driver/version.info 2>/dev/null | head -5
# 3) 看 build 实际选定的 SOC / NPU_ARCH / arch 目录（与上面 npu-smi 比对）
bash build.sh --ops=<op> --soc=<soc> 2>&1 | grep -E "SOC_VERSION=|NPU_ARCH=|SOC_ARCH_DIRS=|Skipping test"
```

**原始观测（本 trace 实测）**：

`npu-smi info` 输出（seq 695，来源平台：未定位/arch22 Ascend910_93；证据等级：tool_observed）：

```
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip  Phy-ID              | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 1     Ascend910           | OK            | 164.8       45                0    / 0             |
| 0     2                   | 0000:0B:00.0  | 0           0    / 0          3418 / 65536         |
...
```

`cat /usr/local/Ascend/driver/version.info`（seq 319，来源平台：未定位/arch22；证据等级：tool_observed）：

```
Version=25.5.1
ascendhal_version=7.35.23
aicpu_version=1.0
...
Innerversion=V100R001C23SPC006B220
```

`bash build.sh --ops=stbsv --soc=ascend950` 的 CMake 配置阶段（seq 326，来源平台：未定位/arch22；证据等级：tool_observed）：

```
-- SOC_VERSION=ascend950, NPU_ARCH=dav-3510
-- SOC_ARCH_DIRS=arch35
...
```

改用 `--soc=ascend910_93` 重建（seq 326，来源平台：未定位/arch22；证据等级：tool_observed）：

```
-- SOC_VERSION=ascend910_93, NPU_ARCH=dav-2201
-- SOC_ARCH_DIRS=arch22
...
-- Skipping test 'stbsv': operator implementation not found for 'ascend910_93' ; (checked blas/stbsv/ and blasLt/*/); test directory 'test/tbsv/stbsv/' contains no test sources for 'ascend910_93'. Please verify that 'aclblasStbsv' supports 'ascend910_93'.
```

**解释（interpretation，agent 的根因分析）**：build target 是 `ascend950`（arch35 / dav-3510，SIMT 编程模型），板卡真实是 `Ascend910_93`（arch22 / dav-2201）。arch35 SIMT kernel 二进制指令在 arch22 的 vector core 上无法执行，runtime 直接抛 `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（507035）。这不是 kernel 逻辑错误：编译能过、`n=0`/`k=0` 快速返回用例（不进入 kernel 主体）能"通过"，恰恰说明 launch 通道本身没问题、问题出在 arch35 指令在 arch22 上无法运行。本算子在 arch22 上没有实现目录（`blas/tbsv/stbsv/` 下只有 `arch35/`，无 `arch22/`），所以 `--soc=ascend910_93` 被 CMake 跳过——没有可运行的 arch22 版本。

## 误诊

本 trace 的 developer subagent 在最终定位 arch 不匹配之前，花了约 250 个事件怀疑 kernel 逻辑并做了多轮无果改动，是典型的"507035 当 kernel bug 调"的坑（以下为 agent_reported，非实测信号，不进 `## 验证信号`）：

- 怀疑 `GlobalTensor::GetValue` / `SetValue` 逐元素读写、`TQue` / `LocalTensor` / `TPipe::InitBuffer` 用量过大，尝试逐段剥离子进程隔离。
- 怀疑 `KERNEL_TYPE_AIV_ONLY` 宏未标记导致 kernel type auto-derivation 失败，全仓 grep 该宏定义。
- 怀疑 `TbsvBandIdx` 返回 `int64_t` 做数组索引越界，打算强转 `uint32_t`。
- 对 kernel 做了若干次试探性 edit，重建后 507035 不变——这些 edit 的 verbatim 内容因 markdown 转写被截为"_see below_"，未能从事件流逐字恢复，故本卡不内联其代码；它们无一是 507035 的真因。

真正的判别点是 `## 判别方法` 的探针：先 `npu-smi info` + `cat version.info` 比对 `--soc`，再动 kernel。

## 预防

- 在跑任何上板 ST 之前，先 `npu-smi info` 确认板卡真实型号/chip，再选 `--soc`；`--soc` 必须与板卡 arch 同族，而非与"算子设计目标芯片"盲目对齐。
- 算子按 arch35（SIMT）设计但开发机只有 arch22 板时，汇合联调必然被这一步卡住：要么换 950 板，要么为该算子补 `arch22/`（SIMD/membase）实现目录——不要试图改 kernel 逻辑绕过 507035。
- 把"`n=0`/`k=0`+unit 这类快速返回用例独通过、其余全 507035"这一 pattern 作为 arch 不匹配的特征指纹，写进 ST 报告的失败判读规则，避免后续 agent 反复在 kernel 侧排障。

## 验证信号

本卡验证信号仅含本 trace 内 `tool_observed` 实测；误诊段的"noop kernel works"等只有 agent 叙述、未恢复干净 tool_result，按 hearsay 处理，不进本段。

- L0 22 用例：2 PASSED / 20 FAILED，失败用例报 `Unexpected NPU error code: 6` / `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（507035）；通过的两条为 `n=0`、`k=0`+unit 快速返回用例，不执行 kernel 主体（来源平台：未定位/arch22 Ascend910_93；证据等级：tool_observed）。
- `npu-smi info` 显示板卡 `NPU Name = Ascend910`、Phy-ID 2/3（来源平台：未定位/arch22；证据等级：tool_observed）。
- `cat /usr/local/Ascend/driver/version.info`：`Version=25.5.1`、`ascendhal_version=7.35.23`、`Innerversion=V100R001C23SPC006B220`（来源平台：未定位/arch22；证据等级：tool_observed）。
- `--soc=ascend950` build 配置：`SOC_VERSION=ascend950, NPU_ARCH=dav-3510, SOC_ARCH_DIRS=arch35`（来源平台：未定位/arch22；证据等级：tool_observed）。
- `--soc=ascend910_93` 改建：`SOC_VERSION=ascend910_93, NPU_ARCH=dav-2201, SOC_ARCH_DIRS=arch22`，CMake 跳过 `stbsv`（`operator implementation not found for ascend910_93`）（来源平台：未定位/arch22；证据等级：tool_observed）。
- arch35 kernel 与 ST 框架编译通过（`build.sh --ops=stbsv --soc=ascend950` 无 error、产出 `stbsv_test`）（来源平台：未定位/arch22；证据等级：tool_observed）。

未在本 trace 验证的项（显式标注）：① stbsv 在真实 Ascend950 板上的精度（本 trace 无 950 板，2/22 通过用例不构成精度证据）；② 改路线（补 arch22 实现或换 950 板）后的通过情况——本 trace 未实施。

## 失效边界

- arch35 SIMT kernel 在 arch22（Ascend910_93）硬件上运行 = banned，本 trace 实测必报 507035，无代码侧修复路径；只能改路线。
- 推断可迁移：本判别方法（507035 + "全崩 + 仅快速返回用例独通过" pattern → 先查 arch 不匹配）对任何 arch35 SIMT kernel 落到 arch22 板的算子都应适用，属 `platform_wide`；但未验证于 tbsv/stbsv 以外的算子，标注"未验证，推断可迁移"。
- arch22（Ascend910_93）不在注册平台词表 `{950, a3, a2}` 内，`platforms: []`；Ascend950 因本 trace 未在 950 板上实测通过，亦不写入 `platforms`。
