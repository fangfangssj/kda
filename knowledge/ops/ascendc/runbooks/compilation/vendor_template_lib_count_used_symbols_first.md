---
type: "Runbook"
title: "vendor 模板库移植前先数实际使用符号"
description: "port 携带 catlass/tl_templates 等 vendor include 树前，先 grep kernel 实际使用的符号；典型情况只用 2-3 个宏/包装（如 CATLASS_DEVICE、tl::ascend::Fill），一行级 prelude 即可替代，删掉整个 vendor include 面后编译更干净。判别方法：在 kernel 源文件范围内枚举使用符号并人工核 use（grep 只缩圈，文本去重会把未用声明扫入并截断混合大小写名），不是行数统计。"
tags: ["task.implementation", "topic.compilation"]
status: draft
sources: []
created_at: "2026-09-24T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["AscendC::Duplicate", "tl::ascend::Fill"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。

# vendor 模板库移植前先数实际使用符号

## 适用场景

port_a3_to_a5（或任何跨环境移植）携带 catlass、tl_templates 等 vendor 模板库 include 树的 kernel。

## 步骤

1. 枚举**实际使用符号**（不是 include 了什么，也不是行数）：`grep -oE 'tl::ascend::[A-Za-z_]+|Catlass[A-Za-z_:]*|CATLASS_[A-Z_]+' <kernel 源文件> | sort -u` 只作缩圈——文本去重 ≠ 已用符号（会把未使用声明扫进来、且 `[A-Z_]+` 会截断 LayoutL 这类大小写混合名），**必须限定在 kernel 源文件范围（不要扫 vendor/archived 公共头），并对结果人工核 use**。本例核完恰好两个被使用：`CATLASS_DEVICE`（`__forceinline__ __aicore__`）、`tl::ascend::Fill`（`AscendC::Duplicate`）；2 个未使用的 namespace using 声明不计入。
2. 对每个符号判断可否一行级替代：实测 `CATLASS_DEVICE` ≡ `__forceinline__ __aicore__`、`tl::ascend::Fill` ≡ `AscendC::Duplicate`——10 行 prelude 全覆盖。
3. 删除 vendor include 面，用 prelude 重编译。

## 验证方式

删除整个 vendor 树后在目标代首编即过（arch35 / CANN 9.2.0 first-build clean）；静态检查与运行时行为不变（符号语义逐字等价）。

## 失效边界

仅当实际使用符号确实可枚举替代时成立；使用 vendor 模板真实逻辑（非宏/薄包装）的 kernel 不适用，须按真实依赖评估。
