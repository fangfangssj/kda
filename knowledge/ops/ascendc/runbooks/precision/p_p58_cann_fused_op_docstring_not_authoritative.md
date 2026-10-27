---
type: Runbook
title: "CANN 融合算子 docstring 不是权威算法——对抗探针检测"
description: "CANN 融合算子的 docstring 可能与实际 kernel 算法分叉（如动态量化的除数 clamp、swiglu mode 派发）；用对抗性小幅值探针一次检测，别拿 docstring 逐字写 kernel。原 legacy 锚 P-P58 族。"
tags: [task.debugging, topic.precision]
status: draft
sources: []
created_at: "2026-09-22T05:30:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: [P-P58]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CANN 融合算子 docstring 不是权威算法（P-P58 族）

## 机制

docstring 描述的安全护栏（如"小 ε 处 clamp 除数"）可能与融合算子实际 kernel 分叉。按 docstring 逐字写 `qs_div = max(out_scale, 1e-10)` 会让退化行（幅值小到 `amax / N_levels << ε`）输出全零，而参考融合算子直接用未 clamp 的小幅值产出非零饱和 int8。同族另一轴：swiglu_mode 的 mode 标志派发（P-P58.X）——不同轴向，同一根因（docstring 不权威）。

## 一发检测探针

写一个小幅值常量探针：所有输入取 ~5e-7 或更小（`amax / N_levels` 低于 docstring 的 clamp ε），跑参考实现，看 max 位置上有没有非零 int8。有 = 分叉证实，kernel 里去掉 clamp。预探针仪式 ~5 分钟，省掉多轮 precision-probe 循环。

## 边界与陷阱

- Pass A（benchmark）输入幅值正常时 clamp 永不触发——没有对抗性 Pass B/边缘数据集探针，这个缺陷不可见。
- 覆盖到的算子族：AddRmsNormDynamicQuant、RmsNormDynamicQuant、SwigluDynamicQuant、GroupQuantize 等动态量化类；第二个同族确认后可升级为统一的"docstring 不权威"模式族。

## 适用场景

- **工程形态**：CANN 融合算子的 docstring 可能与实际 kernel 算法分叉（如动态量化的除数 clamp、swiglu mode 派发）。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
