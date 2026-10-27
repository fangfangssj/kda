---
type: Runbook
title: "When the reference path's output depends on RNG global state, both ref and cand must agree on a seed"
description: "When the reference path's output depends on RNG global state, both ref and cand must agree on a seed"
tags: [task.debugging, topic.data_movement, topic.performance.analysis, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-95-when-the-reference-path-s-output-depends-on-rng-gl.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-95"]
---

# When the reference path's output depends on RNG global state, both ref and cand must agree on a seed

> 原 OKF v1 卡号：**OL-95**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - source forward(x, ...args) does any of:

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process / verification
- **Loaded by**: aog-kernel-worker / orchestrator when the source benchmark `Model.forward` constructs an `nn.Module` (or other RNG-state-driven object) **inline** with random initialization, AND verification is bit-exact comparison
- **Trigger**: source `forward(x, ...args)` does any of:
  - `m = nn.Module(...)` inline (random parameter init — `nn.Conv*`, `nn.Linear`, `nn.LSTM`, etc.)
  - `w = torch.randn(...)` / `torch.randint(...)` inline (RNG-driven tensor materialization)
  - Any path where the function's outputs depend on Python global RNG state at call time
- **Problem**: the verification harness calls `Model.forward(x, ...)` and `ModelNew.forward(x, ...)` back-to-back inside one test case. Between the two calls, RNG global state advances. So ref and cand see DIFFERENT random tensors → bit-exact comparison ALWAYS fails by construction, no kernel can save it.
- **Diagnostic signature**: `max_abs_diff` is unbounded (essentially comparing random output to random output) — no recognizable "kernel bug" pattern. The fact that the diff doesn't shrink with kernel iterations is the diagnostic: more iterations don't help because the wrong-by-construction inputs are different on every run-pair.
- **Fix (both halves)**:
  1. **Benchmark side** — add a `seed` kwarg to `Model.forward` and call `torch.manual_seed(seed)` at the start of the function body. Default `seed=0`. `get_input_groups()` reads `seed` from the JSON case attrs (defaulting to 0) so existing JSON cases keep working unchanged. This is upstream-PR territory.
  2. **ModelNew side** — mirror the seed handling exactly:
     ```python
     def forward(self, x, ...args, seed=0):
         torch.manual_seed(seed)
         conv = nn.Conv1d(...).to(x.device)         # same construction as ref
         w = conv.weight.contiguous()
         b = conv.bias.contiguous() if bias else None
         return _ext.run_kernel(x, w, b, ...args, has_bias=bias)
     ```
     The `nn.<Module>` constructor consumes RNG in a deterministic order, so given the same `seed`, the resulting parameters are bit-identical to ref's. The candidate kernel only computes the pure function (e.g. `conv1d(x, w, b, ...)`) with no RNG dependency, so its output matches ref's bit-for-bit (within FMA tolerance).
- **Distinction from OL-89 (prose-spec docstring decomp)**: OL-89 is for "reference is unrunnable; reconstruct semantics from docstring spec or embedded Python decomp". OL-95 is for "reference IS runnable but its outputs depend on RNG state; both halves must agree on seed". Both belong to the same family of "verification harness can't compare ref vs cand naively"; OL-89 fixes the source of the values, OL-95 fixes the order in which they're drawn.
- **Evidence**: 6_ConvStandard1d (2026-04-29 cold-start of conv family) — ref uses inline `nn.Conv1d(...).to(x.device)(x)`. With both halves applied (upstream PR adds `seed` kwarg + ModelNew mirrors), Pass A 50/50 PASS (atol=rtol=0.01) + Pass B 16/16 PASS at same tolerance, max_abs_diff = 9e-4 (FMA accumulation order drift). Without ModelNew mirror, every case would fail max_abs_diff > 1.
- **Other instances (predicted)**: any benchmark op whose `Model.forward` constructs an `nn.Module` inline with random init — explicitly the conv family `7_ConvStandard2d / 8_ConvStandard3d / 9_ConvDepthwise2d / 10_ConvTranspose2d` (same seed mechanism applies). More broadly, any RNG-touching `forward()` in level-3 / level-4 ops.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-95（category=process / verification，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
