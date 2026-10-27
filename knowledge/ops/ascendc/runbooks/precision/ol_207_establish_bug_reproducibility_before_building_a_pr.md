---
type: Runbook
title: "Establish bug-reproducibility BEFORE building a proxy-investigation — a non-reproducible defect's proxy-refutations are not bug-relevant"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.implementation.atomic, topic.data_movement, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-207-establish-bug-reproducibility-before-building-a-pr.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-207"]
---

# Establish bug-reproducibility BEFORE building a proxy-investigation — a non-reproducible defect's proxy-refutations are not bug-relevant

> 原 OKF v1 卡号：**OL-207**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=all; cann=all; bisheng=n/a; op_class=all

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 graybox 2026-06-04/05)`

**Principle**: when a defect is observed, FIRST establish that it is **reproducible** under controllable conditions before building any proxy-investigation around a hypothesized mechanism. A **non-reproducible** defect's proxy-refutations tell you **nothing** about the real defect — you are testing a stand-in for a ghost. A "fix" for a non-reproducible defect is **unverifiable** (you can show neither fail-before nor succeed-after); say so explicitly rather than chasing a deterministic proof the non-reproducibility makes impossible.

**How to apply**:
- On a defect report, gate the investigation: run controllable stressors to establish reproducibility FIRST. If NOT reproducible under any controllable condition → it is a **documented residual / known-risk**, not a demonstrated blocker; do NOT spawn a mechanism-hunt.
- Distinguish **proxy-refuted** ("this candidate doesn't achieve the proxy property X") from **bug-refuted** ("this candidate doesn't prevent the real defect") — they are the same thing ONLY if the proxy property is PROVEN to be the defect mechanism.

**Concrete anchor**: FA-A5 graybox — a CCU-263/`507015` cross-core cascade was observed ONCE, confounded by a concurrent now-gone external job. Instead of asking "is this reproducible?", the team built an elaborate isolation-proxy (a scoping-probe for "does a cross-group flag alias") and refuted 4 hypotheses against that proxy. A later deterministic test proved isolation is NOT even the cascade mechanism (vendor uses the same global flags and does not cascade) → the entire proxy-investigation had measured the wrong thing. Owner's catch: "你跟我说他只复现过一次？那么后面凭什么认为我们的修复有问题呢？".

**Evidence**: FA-A5 graybox 2026-06-04/05 — one external-confounded `507015` fire; isolation-proxy built + 4 hypotheses refuted against it; then isolation deterministically shown NOT the mechanism → whole proxy-investigation wasted. Owner-collapsed to a documented residual `DEBT-FA-CASCADE-UNREPRO` (ROADMAP §6, commit `3f201374` — the committed refutation-ledger evidence). Cross-ref `feedback_faithful_port_copy_host_not_just_kernel` (the host/launch differentiator that the proxy-hunt obscured), OL-208 (the deterministic-test-target discipline), OL-209 (refutation-ledger discipline).

**Other instances (predicted)**: any rare/non-deterministic runtime fault (race / cross-core / atomic / contention-triggered); any "intermittent" customer bug — establish a controllable repro before spending tokens on a mechanism-hunt.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-207（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
