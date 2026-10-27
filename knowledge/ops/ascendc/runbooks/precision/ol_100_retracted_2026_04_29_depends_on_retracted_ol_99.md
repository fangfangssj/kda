---
type: Runbook
title: "[RETRACTED 2026-04-29 — depends on retracted OL-99]"
description: "[RETRACTED 2026-04-29 — depends on retracted OL-99]"
tags: [task.debugging, topic.precision, topic.data_movement]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-100-retracted-2026-04-29-depends-on-retracted-ol-99.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-100"]
---

# [RETRACTED 2026-04-29 — depends on retracted OL-99]

> 原 OKF v1 卡号：**OL-100**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - precision sweep failure that probe (via OL-99 method) classifies as MARE-eps artifact

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

**Original claim**: "When a composite-op kernel matches CPU at hardware ULP but fails MARE, the fix is verifier-side, not kernel-side."

**Why retracted (same day)**: this entry's premise was that "kernel matches CPU at hardware ULP" — but for op#28 (the supporting case), CANN actually achieves bit-exact (0 ULP) match with CPU. So the premise "you've reached hardware ULP" was unverified — we never compared against CANN baseline before declaring the floor. The decision tree below is invalid because branch 1 was prematurely entered.

**Replacement** (TBD): a corrected decision tree should require an explicit "CANN bit-exact check" step before declaring hardware ULP. Defer until enough empirical evidence accumulates from the Q4 sweep classification.

**Action items**:
- Do NOT use this entry's decision tree.
- Future MARE failures: probe BOTH `(our_kernel vs CPU)` AND `(CANN vs CPU)`. Only when CANN is also non-bit-exact AND at the same ULP bound is "verifier-side" a candidate explanation.

- **Category (legacy)**: process / verification
- **Loaded by**: Lead/Reviewer when deciding "fix kernel" vs "fix verifier" vs "accept as-is"
- **Trigger**: precision sweep failure that probe (via OL-99 method) classifies as MARE-eps artifact

### Decision tree

```
Sweep failure on op X
  ↓
OL-99 probe: |ref|<1e-4 AND |abs_diff|<1 dtype-ULP?
  ├─ YES → MARE artifact
  │        ├─ Document op X as "ULP-correct, MARE artifact"
  │        ├─ File upstream issue: dtype-scaled atol floor
  │        ├─ DO NOT modify kernel
  │        └─ Update sweep status to PARTIAL_VERIFIED_AS_HARDWARE_ULP_LIMIT
  │
  └─ NO → real precision bug
           ├─ Probe further (FMA? algorithm? reduction order?)
           ├─ Apply Kind-1 (parametric) or Kind-2 (architectural) fix
           └─ Re-test
```

### Key insight: kernel cannot be more precise than the dtype

Hardware fp32/fp16/bf16 arithmetic produces results at the dtype's representable precision. A kernel that's 1-ULP off from CPU is hitting that floor. Demanding "MARE < 10×threshold" with eps=1e-7 effectively asks for sub-ULP precision on near-zero outputs — physically impossible.

### Cross-reference

- Implements the user-stated principle (Discord 2026-04-29): "如果自己实在推导不出方法（包括 precision prob），同时 cann 能符合精度要求，可以参考 cann，但必须以知识的形式（足够通用）沉淀"
- For op#28 specifically: kernel is at 1 bf16 ULP from CPU → ULP-correct. The 4/50 sweep result is MARE-eps artifact, not bug. Documented in op#28's `verification.json` under new schema field `ulp_correctness` (TODO when schema bumps).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-100（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
