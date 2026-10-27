---
type: Runbook
title: "Fused residual-add + normalization backward — the two add-branch input grads are bit-identical; emit grad_x to both output buffers from one compute pass"
description: "Fused residual-add + normalization backward — the two add-branch input grads are bit-identical; emit grad_x to both output buffers from one compute pass"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP106"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP106 — Fused residual-add + normalization backward — the two add-branch input grads are bit-identical; emit grad_x to both output buffers from one compute pass

## 适用场景

- 适用平台与范围：soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (fused residual-add + rms/layer norm grad)（卡内 applies_to 行）。
- 验证来源：soc=Ascend910_V220; cann=9.0.0 (fused_add_rmsnorm_grad 4/4 PASS)（卡内 verified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (fused residual-add + rms/layer norm grad)`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (fused_add_rmsnorm_grad 4/4 PASS)`
`status: UNCONFIRMED — single op; needs a 2nd fused-add-norm op (e.g. add+layernorm grad) to confirm`

**Pattern**: for a "fused residual-add + `<normalization>`" backward where the forward is `x = a + b; y = norm(x) * w`, the two add-branch input grads are BIT-IDENTICAL: `d_a == d_b == grad_x`. This is the identity Jacobian of `x = a + b` (∂x/∂a = ∂x/∂b = I), so there is NO separate backward to compute for the second branch. Emit `grad_x` to BOTH output GM buffers from ONE compute pass. The marginal cost over the non-fused norm-grad is exactly **+1 input load + 1 extra grad store** — NOT a second backward.

```
forward:  x = a + b;  y = (x * rsqrt(mean(x^2)+eps)) * w   // add + rmsnorm
backward: grad_x = r*(g - (x*r*r)*sum_H(g*x)/H)            // the rms_norm_grad grad_x
          d_a = grad_x;  d_b = grad_x                       // identity branch — store twice, no recompute
```

**Evidence**: fused_add_rmsnorm_grad (2026-06-03, port_a3_to_a5 V220, KB-only / analytic backward derived from scratch, bit-exact vs fp64 autograd oracle, max_diff ~1e-16) — 4/4 PASS first verify, 0 precision-fix iters. Structurally the op is "`rms_norm_grad`'s grad_x emitted to two output buffers + the same grad_w cross-row reduction"; the leading `x = a + b` add is the only forward delta and it vanishes to an identity on the backward side.

**Promote when**: a 2nd fused-add-norm backward (e.g. add + layernorm grad) confirms the d_a==d_b==grad_x identity-store recipe transfers — i.e. it's a norm-family rule, not a one-op coincidence.

**Cross-ref**: rms_norm_grad / OL-103 (the norm-family transcendental NR-Rsqrt backward technique this fused op consumes — the add-branch dup is the only structural delta on top), OL-75 (the partial+reduce template the grad_w cross-row reduction uses), CAND-PP104 (sibling norm-family backward launch-overhead candidate), CAND-PP109 (the shared-load/different-grad multi-output sibling — PP106 here is identical-grad duplication).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP106，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
