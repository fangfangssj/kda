---
type: Runbook
title: harness 生成的 pass_b_runner 对返回裸 Tensor 的参考实现一律报 0/N，是输出命名不匹配而非精度问题
description: 'CPU bundler 把裸 Tensor 存成 {"y": tensor}，生成的 runner 的 _same() 在真值叶为 dict 时要求候选也是 dict，于是每个用例都被标 precision mismatch。'
tags:
- task.debugging
- topic.precision
status: draft
sources: []
created_at: '2026-09-25T08:59:06Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- '950'
aliases: ["CAND-EC-GENERATED-PASSB-BARE-TENSOR"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# harness 生成的 pass_b_runner 对返回裸 Tensor 的参考实现一律报 0/N，是输出命名不匹配而非精度问题

## 证据

- CPU bundler 把裸 Tensor 存成具名映射 `{"y": tensor}`：
  `phase_o25_generator.py:1189` → `return {"y": out.detach().cpu()}`；
  `o25_gen_helpers.py:330 _normalize_outputs` 采用同一命名。
- 生成的 runner 不做这一步：`verifier_artifact_contract.py:185 def _same(actual, expected)`，
  `:195` 要求 `isinstance(actual, dict)`；该文件中 `_normalize_outputs` 出现 **0 次**。

实测：生成 runner **0/234**，每个用例标 "precision mismatch"；同一 kernel、同一评分函数
（`precision_tier1._ecosystem_t1`）、补上 bundler 的命名后 **159/234**。
只改这一步就改结果，因此测的是这一步而非 kernel 数值。

**候选不能改成返回 dict**：保留的 Pass-A oracle 里 `expected` 存的就是 `Model.forward`
返回的裸 Tensor，返回 dict 会让 Pass A 失败。裸 Tensor 是正确的候选契约，
**归一化属于 verifier**。

上游缺陷单：私有沉淀仓 仓 issue #31（私有仓，不可匿名访问；内容见下文转述）

## 适用范围与边界

- 适用：任何 forward 返回**单个裸 Tensor** 且进到 Pass B 的算子。
- **与"结构化输出"缺陷相反**：O5 target-model grader 拒绝**多输出/结构化** forward
  （见 a5_ops issue #20），组件与形状都与本卡相反，不要混为一条。
- **危险性在可辨识度**：这个 0/N 在算子读到的收据里与真实精度失败**不可区分**，
  所以看到整齐的 0/N 先核输出命名，再怀疑数值。
- 实测环境：soc=Ascend950DT_9582、cann=9.1.0；代码事实在 a5_ops `main@8aada6d69998` 上复验存在。

## 适用场景

- **工程形态**：CPU bundler 把裸 Tensor 存成 {"y": tensor}，生成的 runner 的 _same() 。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
