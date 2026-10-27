---
type: "Runbook"
title: "507035 — standalone ccec launch 失败"
description: "Ascend950 上 standalone ccec benchmark 构建成功，但所有 kernel warmup 在 aclrtSynchronizeStream 返回 507035；用于区分已排除的 UB 假设与尚待继续定位的 launch/binary 路径问题。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 507035 — standalone ccec launch 失败

## 现象

在 Ascend950 / CANN 9.0.0-beta.2 上，GM→UB standalone benchmark 由 `ccec --cce-aicore-lang` 编译成功并打印 `BUILD OK`。运行时设备信息与 bench 表头能够正常打印，但 56 AIV 和 28 AIC 两种配置的每个用例都在 warmup 后由 `aclrtSynchronizeStream` 返回 `507035`，没有有效带宽结果。

## 适用场景

- **工程形态**：host 与 kernel 同 TU 的 standalone kernel benchmark，经 `<<<>>>` 启动；不适用于 op_host/op_kernel 或 aclnn 注册算子。
- **验证平台与版本**：Ascend950，CANN 9.0.0-beta.2。
- **适用粒度**：`exact_task`。
- **路线优先级**：`diagnostic_only`。本卡只记录已核实的失败边界和排障分流，不提供修复路线。

## 判别方法

1. 确认 `ccec` 已返回 `BUILD OK`，错误只出现在 kernel launch 后的 stream sync。
2. 同时用 vector core 数和 ai core 数启动；若两者均在 warmup 返回 507035，可排除单纯的核数选择问题。
3. 把 `InitBuffer` 控制在探测到的 UB 容量以内后重跑。该 trace 中从约 2 MiB 缩至 162 KiB（小于探测到的 216 KiB）后错误不变，因此原始 UB 超配是真实缺陷，但不是剩余 507035 的充分解释。
4. 分 TU 尝试若在 host 链接阶段出现 `device .o is incompatible with elf64-x86-64`，只能说明 device object 不能由普通 host linker 直接链接；它不是运行期 507035 的根因证明。

上述观测把问题范围缩小到 kernel body 以外的编译、封装或 launch 路径，但 trace 没有继续隔离到具体注册符号、binary 格式或 runtime 行为，不能据此指定唯一机制。

## 验证信号

- `ccec --cce-aicore-lang` 返回 `BUILD OK`（来源平台：950；证据等级：tool_observed）。
- 56 AIV 与 28 AIC 两种配置的所有用例均为 `warmup sync failed: 507035`，有效带宽为 0（来源平台：950；证据等级：tool_observed）。
- UB buffer 降到 162 KiB 后错误不变（来源平台：950；证据等级：tool_observed）。
- 分 TU host 链接报 device object 与 `elf64-x86-64` 不兼容（来源平台：950；证据等级：tool_observed）。

## 失效边界

- return-only kernel 仍失败、CANN 9.2.0 构建可运行等信息仅存在于 agent 叙述，没有可核工具输出，不作为结论。
- 非 Ascend950、CANN 9.2.0 及以后、框架注册算子均未验证。
- `507035` 的精确错误类别和 launch 失败机制未由该 trace 证实。

## 坏实践

- 把已排除的 UB 超配继续当作剩余 507035 的唯一根因。
- 把 device `.o` 的 host linker 不兼容错误直接等同于运行期 launch 失败机制。
- 把未落盘的 agent 成功叙述当作修复信号。
