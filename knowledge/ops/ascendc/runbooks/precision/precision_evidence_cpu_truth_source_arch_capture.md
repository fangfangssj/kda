---
type: Runbook
title: "Precision evidence — CPU truth and source-arch behavior capture"
description: "Precision evidence — CPU truth and source-arch behavior capture"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["REFERENCE_SOURCE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-97 — OL-97: Precision evidence — CPU truth and source-arch behavior capture

> 关联：同一 OL-97 主题已由既有卡 [ol_97_precision_target_cpu_truth_vs_cann_bit_match.md](ol_97_precision_target_cpu_truth_vs_cann_bit_match.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：paradigm=ascendc（卡内 applies_to 仅声明范式；适用范围细节见正文）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行。

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: process / precision / verification
- **Loaded by**: aog-kernel-worker / aog-precision-probe / aog-kernel-optimizer **whenever the op has a numerical-precision verifier in Phase D**
- **Trigger**: any migration workflow whose verifier compares generated output against CPU truth or an authorized source-arch behavior capture.

### Why this OL exists

Two complementary evidence channels are useful:

1. **CPU-truth target** — the reference is `torch.<op>(x.cpu().float())`, executed on CPU with fp32 internal precision, then cast back to native dtype. This is the **production-line precision-eval skill** (`Ascend/agent-skills/skills/ascendc-operator-precision-eval`) convention. The kernel is being measured against the *mathematical* answer.
2. **Source-arch behavior capture** — run the user-selected arch22 operation through its declared public entry point and preserve inputs, outputs, attributes, and environment. This records observable source behavior.

Thresholds come from the operator contract; do not inherit global constants from another verifier.

The same result can agree with one channel and disagree with the other because reduction order and
primitive precision differ. CPU fp64 anchors the math; source-arch capture anchors observable
migration behavior. A disagreement must be classified, not waived.

### Decision rule

Read `REFERENCE_SOURCE` in PROGRESS.md (V3.3.8 field) plus the verifier shape:

| If the verifier compares against … | Then the precision target is | Apply patterns from |
|------------------------------------|------------------------------|---------------------|
| `Model.forward` on **CPU** (or OL-89 docstring decomp) | **CPU-truth** | "CPU-truth" column below |
| authorized selected operation on the **source-arch NPU** | **source behavior** | source-capture column below |

### Applying the two channels

- Always evaluate the declared equation against CPU fp64, including boundary, tail, optional-input,
  and non-finite cases.
- For migration, replay the same declared cases on the selected arch22 source and archive the
  observable capture.
- For generated backward kernels, derive the oracle from the forward equation and compare with
  CPU fp64 autograd; the source capture is supplementary when a source backward contract exists.
- Record any CPU-versus-source disagreement by dtype, shape region, reduction order, and tolerance.
  Never tune a formula solely to imitate an opaque rounding signature.

The KB manifest must name the evidence channel used by each check. Target implementations,
binaries, profiles, archives, and sibling worktrees may be consulted as prior-art for generation
and research, but they are not migration or backward truth.
- **Performance comparison fairness**: when comparing perf to aclnn, the kernel must compute the *same answer*; matching CANN avoids the "but we returned a different number, that's why we're faster" trap.

The OL says **choose your target deliberately**, and document the choice in `analysis.md`.

### Evidence

- 2026-04-29 sweep of 9 archived level-3 ops against br_430's NPU-ref MERE/MARE harness: matmul family lost 4-9 fp32 cases per op to MARE outliers; conv family lost 100% of fp32 cases to FMA-reorder. Sample of op#1 BatchMatmul against CPU-ref via `precision_eval_cpu_ref.py` shows the same 6/51 fail count — the failures are MERE-dominated by near-zero-ref-value MARE amplification regardless of which reference is used. The kernels are **correct** numerically (max_abs_err ~1e-3); the metric is sensitive to outputs containing tiny values.
- Production-line skill `ascendc-operator-precision-eval` (`gitcode.com/Ascend/agent-skills`) explicitly uses CPU as ground truth via `torch.<op>(x.cpu().float()).to(dtype)`. Confirms CPU-truth as the canonical "math correctness" target.
- Tracking doc: `docs/precision_eval_cpu_truth_migration_2026_04_29.md`.

---
