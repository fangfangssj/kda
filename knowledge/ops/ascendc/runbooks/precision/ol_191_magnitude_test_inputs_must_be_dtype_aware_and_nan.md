---
type: Runbook
title: "Magnitude test-inputs must be dtype-aware, and nan/inf-reference cases are DEGENERATE (unscoreable), not FAIL"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (test-infra: case generation + scoring)"
tags: [task.debugging, topic.precision, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-191-magnitude-test-inputs-must-be-dtype-aware-and-nan.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
aliases: ["OL-191"]
---

# Magnitude test-inputs must be dtype-aware, and nan/inf-reference cases are DEGENERATE (unscoreable), not FAIL

> 原 OKF v1 卡号：**OL-191**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (test-infra: case generation + scoring)

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (test-infra: case generation + scoring)`
`verified_on: test-infra (dtype-independent — fp16/bf16 input generation + oracle scoring)`

**Principle (two coupled rules)**:
1. **dtype-aware magnitude bands** — a fixed large-magnitude input band (e.g. `[1e20, 1e30]`) overflows EVERY fp16 (`finfo.max ≈ 6.55e4`) and bf16 input to `inf` before the kernel even runs, making the case test nothing. Scale magnitude bands to the dtype: cap at `sqrt(finfo.max)/16` (leaves headroom for a Q@K-style accumulation to stay finite). Same logic for small_mag vs `finfo.tiny`.
2. **Degenerate-reference guard in the scorer** — for a deep-accumulation op (especially BACKWARD: dq/dk/dgk are sums over contraction axes), large inputs overflow the low-precision GRADIENT range even when inputs are finite. When the fp64 ORACLE reference itself is nan/inf for a case, NO kernel (ours, cv-agent, CANN) can match it — the case is UNSCOREABLE. The scorer MUST classify it DEGENERATE (excluded from pass/total), NOT FAIL. Misclassifying as FAIL penalizes the kernel for an inherent low-precision limitation.

Concrete anchor (scorer guard):
```python
if any(not torch.isfinite(ref[k].float()).all() for k in ("dq","dk","dweights")):
    ndegen += 1; continue   # oracle nan/inf -> unscoreable, NOT a kernel fault
```

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Evidence
- lightning_indexer_grad (A3, 2026-05-27): 4 large_mag cases (#6 #7 #25 #26) had fp64-autograd oracle = nan/inf (gradients overflow fp16/bf16); classified DEGENERATE → final 34/34 scoreable PASS + 4 degenerate, instead of a misleading 34/38 "fail".

## Other instances (predicted)
Any low-precision (fp16/bf16) op test harness with magnitude-sweep coverage; especially backward / reduction / softmax-denominator / cumulative ops where accumulation can overflow even when inputs are in-range.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-191（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
