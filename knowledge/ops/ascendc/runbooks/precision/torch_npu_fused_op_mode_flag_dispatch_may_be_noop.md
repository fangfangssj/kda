---
type: Runbook
title: "torch_npu fused-op spec gap — mode-flag dispatch may be no-op"
description: "移植带 mode 整数属性的 torch_npu 融合算子时，docstring 宣称的 mode 专属语义可能被底层 CANN 算子忽略（按 mode=0 一致派发）；须以小 shape RNG 重放探针 × 候选公式实测，勿按 docstring 实现。"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["P-P58.X"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P58.X: torch_npu fused-op spec gap — mode-flag dispatch may be no-op

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；Scenario 为移植通过整数属性宣称多种 mode 的 `torch_npu.<fused_op>`（正文 Scenario 行）；证据锚定 op#11 DequantSwigluQuant（CANN 9.0.0，2026-04-21，正文 Evidence 段）。
- 验证来源：卡内无 verified_on 行；16 候选公式扫描（K1-K16）与 408/408 行匹配证据见正文 Evidence 段；同族另一轴向（divisor-clamp 发散）见 CAND-PP85 卡 adversarial_divisor_clamp_probe_docstring_divergence.md。
- 说明：本卡内容蒸馏自 domains/precision.md 的 P-P58.X 小节（该域卡其余小节与本仓同号 v1 卡同 lesson，未单独建域卡；见 logs/2026-09-19.md 批次 10 条目）。


**Severity**: HIGH | **Mode**: any | **Scenario**: porting a `torch_npu.<fused_op>` that advertises multiple "modes" via an integer attribute

**Criterion**: When the `torch_npu` op docstring describes mode-specific semantics (e.g. "mode=1 is variant with params X, Y, Z") but trusting the docstring leads to wrong-formula kernels that pass superficial QA on one mode and fail on the other. The underlying CANN fused op may **ignore** mode-specific parameters and dispatch identically across mode values.

**Detection heuristic**:
1. Implement the kernel per docstring.
2. If ≥50% of cases at one mode value fail while the other mode passes bit-exact, **do not assume the kernel formula is off by a small factor**.
3. Probe instead: test candidate formulas including "mode=1 dispatches identically to mode=0".
4. Small-shape RNG-replayed probes (1-2 rows, default + non-default attrs) × candidate formulas; compute per-row pre-quant amax rel_diff. Unambiguous winner shows ≤1e-7 rel_diff while others ≥1e-3.

**Evidence (op#11 DequantSwigluQuant, 2026-04-21)**:
- `torch_npu.npu_dequant_swiglu_quant` docstring advertises `swiglu_mode=1` as "variant with `clamp_limit`, `glu_alpha`, `glu_bias`".
- Initial kernel implemented mode=1 per docstring: `self·sigmoid(self·α)·(other+β)` → 25/50 PASS (mode=0 all PASS, mode=1 all FAIL with 28% mismatch, max_abs_diff up to 174).
- Precision-probe iter 4 ran 16-candidate formula sweep (K1-K16) per-row on 7 mode=1 cases covering default AND non-default `(clamp_limit, glu_alpha, glu_bias)` attribute combos:
  - K7 = `silu(self)*other` (== mode=0 formula): mean rel_amax_diff = 3.75e-9 (fp32 noise), 408/408 rows match
  - K1 docstring formula: mean rel_amax_diff = 7.81e-3 (orders of magnitude worse)
  - K8 variant with β inside sigmoid: mean = 8.67e-4 (still orders worse)
- Fix: replace mode=1 branch with mode=0 body → 50/50 bit-exact immediately.
- Root cause: underlying `aclnnDequantSwigluQuantV2` on CANN 9.0.0 does NOT branch on `swiglu_mode` for the activation step; `clamp_limit/glu_alpha/glu_bias` are received but unused.

**Mitigation**:
1. Implement the empirically correct formula.
2. Keep docstring-advertised parameters in kernel signature/fields for **ABI compatibility** (callers still pass them; kernel silently ignores).
3. Document the spec gap in the `else` branch inline comment with reference to the probe report.

**Anti-patterns to avoid**:
- ❌ Trusting torch_npu docstring without per-mode probe validation
- ❌ Writing complex "best-effort variant" kernel code for mode=1 when actual implementation is mode=0
- ❌ Claiming mismatch is "1-ULP boundary OL-83" without formula-sweep probe confirmation — OL-83 residuals are max_abs_diff=1 on int8, NOT max_abs_diff 30-174

**Cross-ref**:
- `output/npukernelbench/src/kernels/11_DequantSwigluQuant/probe_report.md` — full iter 1-4 trace
- `output/npukernelbench/src/kernels/11_DequantSwigluQuant/knowledge_update.md` — structured entry
- OL-83 (torch_npu 1-ULP drift) — distinct signature (≤ ±1 int8 diff; this P-P58.X has 30-174 int8 diff)

**Why this matters for future ports**: any CANN fused op with a mode/variant parameter should be treated as suspect until probed. Docstring semantics ≠ implementation semantics. This pattern will likely recur on other CANN fused ops where internal dispatch was consolidated across versions but docs were not updated.

---