---
type: Runbook
title: "A reference model's `forward()` must accept the harness case-dict kwargs shape, not only the flat positional signature — else native/cpu-truth provisioning fail-closes silently and grading degrades to a stricter fallback"
description: "<!-- applies_to_backend: all -->"
tags: [task.debugging, topic.data_movement, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-270-a-reference-model-s-forward-must-accept-the-harnes.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-270"]
---

# A reference model's `forward()` must accept the harness case-dict kwargs shape, not only the flat positional signature — else native/cpu-truth provisioning fail-closes silently and grading degrades to a stricter fallback

> 原 OKF v1 卡号：**OL-270**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - <!-- applies_to_backend: all -->

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; phase=provisioning`
`verified_on: soc=Ascend950PR; cann=9.0.0 (rms_norm port_a3_to_a5)`

**Principle**: the harness's native / cpu-truth provisioners (`provision_native_capture` and the CPU-truth path) call the reference model as `model(**case)`, where `case` is the full case_gen record `{idx, name, shape, inputs:{...}, meta}` — NOT the op's flat positional signature `(x, gamma, epsilon)`. A `forward()` written to accept ONLY the flat positional args raises `unexpected keyword 'idx'`, the provisioner **fail-closes** (native_capture absent), and grading silently falls back to a stricter baseline truth source. For a near-zero / tight-tolerance op this can quietly regress the carve-out; the failure is invisible because it looks like "native capture just wasn't produced."

**Contract to satisfy** — the model's `forward()` must accept BOTH call shapes:
```python
def forward(self, x=None, gamma=None, epsilon=None, *, inputs=None, **kwargs):
    if inputs is not None:            # harness case-dict call: model(**case)
        x = inputs["x"]; gamma = inputs["gamma"]; epsilon = inputs.get("epsilon", eps_default)
    # ... flat positional call (x, gamma, epsilon) also works
```
Absorb the record's extra top-level keys (`idx`, `name`, `shape`, `meta`) via `**kwargs`, and unpack the nested `inputs` dict.

**Detection**: `provision_native_capture` logs `unexpected keyword 'idx'` (or another record key) and produces no native_capture; downstream grading is running against the fallback truth source rather than the native capture you intended.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Evidence
- rms_norm kw-1 (2026-07-02, A5 Ascend950PR_957b, port_a3_to_a5, CANN 9.0.0): model.py initially accepted only `(x, gamma, epsilon)`; `provision_native_capture` fail-closed on `unexpected keyword 'idx'`. Fixed by absorbing `**kwargs` + unpacking the nested `inputs` key. All 33 cases still PASSed T1 strict here (so no observed regression), but for a near-zero op the silent fallback would have degraded the fp32 carve-out.

## Other instances (predicted)
Any op-gen mode that authors a fresh reference `model.py` / `model_new_ascendc.py` and relies on the native-capture or CPU-truth provisioner: port_a3_to_a5 CPU-truth-deferred, backward-gen, cross-gen-port. Also any harness that dispatches the reference via `model(**case_record)` rather than positional args.

Cross-ref: OL-97 (cpu-truth side), OL-262 (verify the reference/truth-source before blaming the kernel — same "a silently-wrong truth source masquerades as a kernel result" family).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-270（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
