---
type: "Runbook"
title: "kernel 输出=输入：binary 不随源码变化即守卫吞体"
description: "950（Ascend950, arch 3510）上 kernel 被 launch 但输出与输入逐字节一致、用例全 FAIL，且 device binary 体积异常小。判别=改 kernel 体源码后重编比较 .o 体积/sha256：若多次改动 hash 始终不变，说明 #if __CCE_AICORE__==<arch> 守卫不匹配新架构把整个 kernel 体条件编译排除，编出空壳 binary。移除守卫后 binary 变大、V1 全 PASS。"
tags: ["task.debugging", "topic.compilation", "topic.quality.compatibility.version_migration"]
status: stable
sources: []
created_at: "2026-08-18T07:11:01Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# kernel 输出=输入：binary 不随源码变化即守卫吞体

## 现象

AdvanceStep 适配 950（Ascend950, arch 3510）后，TTK 跑 250 用例全部 FAIL，且所有输出 tensor 与输入逐字节相同——kernel 被 launch 了但没做任何事。plog 显示 `tilingKey is 0, functionEntry=0, Kernel launch successfully`，确认 kernel 已加载执行。binary 文件 `AdvanceStep_..._high_performance.o` 只有 2136 字节，对包含 DataCopyPad/循环/算术的 kernel 异常小。

## 适用场景

- **工程形态**：kernel 入口按 `#if __CCE_AICORE__ == <arch>` 预处理守卫分流路径，迁移到新架构。
- **API 路径**：`#if __CCE_AICORE__ == <arch>` 预处理守卫；`TILING_KEY_IS(k)` 运行时宏；`asc_opc --soc_version=Ascend950` 编译。
- **验证平台与版本**：Ascend950（arch 3510）/ CANN 9.0.0。trace 内仅在此平台实测。
- **适用粒度**：`platform_wide`（任何用 `#if __CCE_AICORE__ == 220` 守卫包裹整个 kernel 体、迁移到 950 的算子都会踩）。
- **路线优先级**：`recommended_fastpath`——先按 binary 是否随源码变化判定守卫是否吞体，再决定是否移除守卫、改用运行时 `tilingKey`/`specNum` 选路径。
- **不适用或未验证场景**：A2/A3 继续走原 `TILING_KEY_IS(1/2)` 分支（未验证，推断可迁移）。

## 判别方法

**触发条件**（tool_observed）：kernel 被 launch（plog 有 `Kernel launch successfully`、`functionEntry=0`）但输出 tensor 与输入逐字节一致、用例全 FAIL；且编译产出的 `.o` 体积异常小（本例 2136B）。

**可执行探针**：在 kernel 体内做一处明确改动（如把 `GlobalTensor.SetValue` 改写为 `DataCopyPad`，或加一行不影响语义的 device 侧语句），重编后比较 `.o` 的体积与 sha256：

```bash
# 编译后取 device binary 体积与 hash
stat -c '%s' <op>_*.o
sha256sum <op>_*.o
```

判定：

1. 若 binary 体积/hash 随源码改动而变化 → 源码进入编译单元，守卫未吞体，转去查数值根因。
2. 若 binary 体积/hash 在多次 kernel 体改动后**始终不变** → 源码未进入 device binary，守卫把整个 kernel 体条件编译排除了。

**原始观测**（Ascend950，AdvanceStep 多轮迭代）：

- 250 用例全 FAIL，输出==输入；binary 2136B，hash `a4fd7d5dc04f6b94f3ff2dc70a040832`。
- 把 `GlobalTensor.SetValue` 重写为 `DataCopyPad` 后重编，binary 仍为 2136B、hash 仍 `a4fd7d5d...`——多次 kernel 体改动 hash 一律不变。
- 移除 `#if __CCE_AICORE__ == 220` 守卫后重编，binary 变为 16656B、hash `ba206ef86af1e74608924f83ad5e34d5`；V1 用例 130/130 PASS，前 10 用例全 PASS。

**机制解释**（interpretation，agent 根因分析）：kernel 入口用预处理守卫按架构条件编译（示意，从 agent 在 trace 中 seq 810 的叙述恢复，非逐字 file_read）：

```cpp
#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1)) { ... }       // V1 Legacy
    else if (TILING_KEY_IS(2)) { ... }  // V2 Spec
#endif
```

950（Ascend950, arch 3510）的 `__CCE_AICORE__` 宏值既不等于 220 也不等于 350 → `#if` 不匹配 → 整个 kernel 函数体被条件编译完全跳过 → `asc_opc` 编出的 `.o` 只有函数序言/尾声的空壳 → runtime launch 后什么都不做 → 输出=输入。正因函数体被守卫排除，函数体内任何源码改动（`SetValue`↔`DataCopyPad`）都不进二进制——这就是“binary hash 在多次改动后恒不变”的根因。守卫移除后，原 `GlobalTensor.SetValue` 写法 V1 直接 130/130 PASS，证明 `SetValue` 在 950 本可用，先前“SetValue 在 950 不生效”为误诊。

## 验证信号

- 250 用例全 FAIL，输出 tensor==输入（来源平台：950；证据等级: tool_observed）
- binary 2136B 空壳，hash `a4fd7d5dc04f6b94f3ff2dc70a040832`（来源平台：950；证据等级: tool_observed）
- 改写 kernel 体（GlobalTensor.SetValue→DataCopyPad）后重编，binary 仍 2136B、hash 仍 `a4fd7d5d...`（来源平台：950；证据等级: tool_observed）
- 移除 `#if __CCE_AICORE__` 守卫后重编，binary 16656B、hash `ba206ef86af1e74608924f83ad5e34d5`（来源平台：950；证据等级: tool_observed）
- 守卫移除后 V1 用例 130/130 PASS、前 10 用例全 PASS（来源平台：950；证据等级: tool_observed）

## 失效边界

- 仅在 Ascend950（arch 3510）/ CANN 9.0.0 实测。
- A2/A3：保留 `TILING_KEY_IS(1/2)` 分支原行为不变，未验证，推断可迁移。
- 守卫移除后的最终重构版（删除 `TILING_KEY_IS`、仅按 `tilingData.specNum` 运行时选 V1/V2、`SetTilingKey(0)` 单入口）属工程重构，其 before/after 逐字代码未从 trace 恢复，本卡只固化“守卫吞体”判别规则。
- `__CCE_AICORE__` 在 950 的确切宏值未在 trace 中直接打印，仅由 binary 体积变化间接确认守卫是根因。

## 坏实践

- 见“kernel 输出=输入”就先怀疑 `GlobalTensor.SetValue` 在新平台不生效，没先看 binary 体积/hash 是否随源码变化——本 trace 在这条误诊上花了大量时间。
- 反复清 `build/`、`$HOME/atc_data/kernel_cache/`、删 built-in 头文件试图强制重编——binary 不变的根因是守卫让函数体为空，清缓存无效。

## 误诊

- **误诊：`GlobalTensor.SetValue` 在 950 不生效**（agent_reported）：agent 按 ascendc-precision-debug 速查表把 `SetValue` 列为嫌疑并改用 `DataCopyPad` 重写，但 binary hash 始终不变（`a4fd7d5d...`）这本身就是“源码改动没进二进制”的信号。守卫移除后原 `SetValue` 写法 V1 直接 130/130 PASS，证伪。
- **误诊：编译器缓存导致 binary 不更新**（agent_reported）：实际 binary 不变是因守卫让函数体为空，被守卫排除的源码改动不影响空壳二进制。