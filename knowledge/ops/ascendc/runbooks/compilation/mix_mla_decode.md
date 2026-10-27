---
type: "Runbook"
title: "Mix合图编译超时致 MLA decode 不可用"
description: "Ascend950 (DAV_3510) CANN 9.1.0 上对 mla_prolog_quant_d decode 算子用 sg_set_scope 做 Mix合图 (CV 融合)，编译被 shell 超时终止；缩小到最小 scope、单 unroll 路径仅 1 个 mix kernel 仍超时，判定该路线对本算子族 banned。"
tags: ["task.debugging", "topic.implementation.pipeline", "topic.programming_model.execution_unit.cube"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["mix_kernel"]
---

# Mix合图编译超时致 MLA decode 不可用

## 现象

在 Ascend950 (950/DAV_3510) CANN 9.1.0 上对 `mla_prolog_quant_d` (DeepSeek V3.2 MLA Prolog decode 非量化) 算子按 Mix合图 (CV 融合) 流程优化：用 `pypto.set_pass_options(sg_set_scope=5001)` 与 `sg_set_scope=-1` 包裹 `mla_prolog_quant_compute` 循环体（Matmul_qa(C) → RmsNorm_qa(V) → Matmul_qb(C) → RoPE(V) 等多次 CV 交替），运行编译。默认 timeout 编译被 shell 终止；放宽门槛重试仍未完成。随后做 4 次系统性降复杂度实验，全部超时。

## 适用场景

- **工程形态**：kernel 直调（PyPTO `@pypto.frontend.jit` 算子，通过 `pypto.set_pass_options(sg_set_scope=...)` 触发 Mix合图 pass）。
- **API 路径**：`pypto.set_pass_options(sg_set_scope=<正整数>)` 开启 CV 融合 scope，`sg_set_scope=-1` 关闭；配合 `cube_nbuffer_setting`/`vec_nbuffer_setting` 调 nbuffer=1 降低并发缓冲，配合 `unroll_list` 减少代码路径。
- **验证过的平台与 CANN 版本**：Ascend950 (950/DAV_3510)，CANN 9.1.0（来源平台：950）。
- **适用粒度**：`operator_family`（含 rms_norm 归约 + NZ 格式 matmul 的 CV 交替段算子，如 MLA/attention 类）。
- **路线优先级**：`banned`（对本算子族 Mix合图不可应用，单 mix kernel 即超编译门槛）。
- **不适用/未验证场景**：不含 rms_norm 归约或 NZ 格式 matmul 的简单 CV 交替算子未在本 trace 验证；其他平台 (a3/a2) 未验证，本限制与 950 mix 编译器 pass 相关，跨平台不可直接外推。

## 失败路线

banned：Mix合图 (sg_set_scope) 对 `mla_prolog_quant_d` decode 算子不可应用，不应再尝试。

4 次降复杂度实验的路径与结果（均超编译门槛被终止）：

| 实验 | scope | unroll_list | nbuffer | mix kernel 数 | 编译时间 |
|---|---|---|---|---|---|
| 全循环体 (首次) | CV 全合整个循环体 | [8,4,2,1] | 默认 | 多 | 超门槛被终止 |
| 缩小 scope | Matmul_qa→RmsNorm_qa→Matmul_qb | [8,4,2,1] | 默认 | 多 | 超门槛被终止 |
| 实验1 | CV 全合循环体 | [4,2,1] | =1 | 3 (mix0/mix3/mix7) | 超门槛被终止 |
| 实验2 | CV 全合循环体 | [2,1] | =1 | 2 | 超门槛被终止 |
| 实验3 | 最小 scope (RmsNorm_qa→Matmul_qb 单 V→C 段) | [2,1] | =1 | 2 (mix0/mix1) | 超门槛被终止 |
| 实验4 | 最小 scope | [1] | =1 | 1 (mix0) | 超门槛被终止 |

关键判据：实验4 是可降的极限——最小 scope（仅 RmsNorm_qa→Matmul_qb 一条 V→C 通路）+ unroll_list=[1]（仅 1 个代码路径）+ nbuffer=1，仅生成 1 个 mix0 kernel，编译仍超门槛被终止。说明问题不在配置规模，而在单个 mix kernel 内部的 rms_norm+matmul 复杂度。缩小 scope/unroll 反复重试无效，首次全循环体超时后不应继续耗费编译时间。

## 验证信号

- 全循环体 sg_set_scope=5001/-1 + 默认 unroll 编译被 shell 以 timeout 终止（来源平台：950；证据等级：tool_observed）
- 缩小 scope 至 pre_compute_2d 单 CV 段 + 默认 unroll 编译被超时终止（来源平台：950；证据等级：tool_observed）
- 实验1 CV全合 + unroll=[4,2,1] + nbuffer=1 生成 mix0/mix3/mix7 三个 mix kernel，编译超时被终止（来源平台：950；证据等级：tool_observed）
- 实验3 最小 scope (RmsNorm_qa→Matmul_qb) + unroll=[2,1] + nbuffer=1 生成 mix0/mix1 两个 mix kernel，编译超时被终止（来源平台：950；证据等级：tool_observed）
- 实验4 最小 scope + unroll=[1] + nbuffer=1 仅 1 个 mix0 kernel，编译仍超时被终止（来源平台：950；证据等级：tool_observed）

## 失效边界

- 已验证不可应用：Ascend950/DAV_3510 + CANN 9.1.0 + `mla_prolog_quant_d` decode (含 rms_norm + NZ matmul)。
- 未验证，推断可迁移：其他含 rms_norm 归约 + NZ 格式 matmul 的 950 decode 算子（如 MLA/attention 族）大概率命中同一编译复杂度问题，未实测。
- 未验证：非 950 平台 (a3/a2) 的 mix 编译器 pass 行为不同，本限制不可直接外推。