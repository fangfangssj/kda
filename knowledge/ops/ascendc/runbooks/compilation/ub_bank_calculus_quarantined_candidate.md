---
type: Runbook
title: "Quarantined candidate: source-bound UB bank-layout hypotheses for an A5 conflict probe"
description: "隔离候选（不得投送）：sanitized David 源描述的 UB bank group/端口/子 bank 布局假说，仅作 A5 冲突探针设计输入；A5 的 bank 映射、仲裁、延迟与收益均未验证；OL-50 仍是 canonical 通则。"
tags:
- task.debugging
- topic.compilation
status: deprecated
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-WIKI-A5-UB-BANK-CALCULUS"]
---

# CAND-WIKI-A5-UB-BANK-CALCULUS

## 适用场景

- 适用平台与范围：none —— quarantined candidate，不声明任何生产目标适用性（卡内 applies_to 行）；validation_target 为 A5 / Ascend950PR，status=not_validated。
- 验证与未验证边界：见卡内 verified_on / unverified_on / source_scope 行（逐字节保留于正文）。
- 隔离状态：CANDIDATE / QUARANTINED FROM AUTOMATIC USE（卡内 status 行）——仅为反馈环保留，不得投送；canonical 既有卡为 runbooks/compilation/ol_50_avoid_ub_bank_conflicts_consecutive_vec_instructio.md（OL-50）。


> lifecycle: quarantined（KB_INDEX 标记；不得投送，仅为反馈环保留）

## CAND-WIKI-A5-UB-BANK-CALCULUS: source-bound bank-layout hypotheses for an A5 conflict probe

`applies_to: none — quarantined candidate; no production target applicability is claimed`

`source_scope: architecture=David; exact_soc=unsourced; backend=unsourced`

`validation_target: A5 / Ascend950PR; status=not_validated`

`verified_on: exact sanitized David-bank source bytes and 25/25 referenced image bytes; no current-A5 device run`

`unverified_on: Ascend950PR bank/subbank mapping, arbitration, latency, compiler scheduling, and performance benefit`

`status: CANDIDATE / QUARANTINED FROM AUTOMATIC USE — source supplement to OL-50, not NEW_VALIDATED_SOURCE`

- **Candidate hypothesis**: for two same-cycle UB reads, reason from the mapped bank
  group before changing arithmetic. The sanitized source describes a David layout
  with eight bank groups, two banks per group, a `2R0W` or `1R1W` group port limit,
  and four subbanks per bank for gather/scatter. Its worked diagrams classify two
  contiguous 256-byte reads as conflict-free at an odd 256-byte separation and
  conflicting at an even separation. This is a useful *probe design*, not an A5
  fact: current A5 code must derive the addresses actually issued and measure the
  same access with a conflict-free control.
- **Safe use**: when msprof/UB traces localize a load stall, build an A/B that changes
  only the candidate bank mapping (for example, adjacent-vs-even-separated read
  bases) while preserving bytes, vector width, instruction count, output, and
  determinism. A gather/scatter arm must also vary subbank placement explicitly;
  ordinary contiguous-load results do not validate the source's subbank claims.
- **Do not infer**: do not copy the David port counts, 256-byte parity rule, stated
  arbitration order, or `VGather` alignment recommendation into an A5 kernel as a
  rule. Do not pad/realign if it changes the logical index map or adds enough traffic
  to erase the measured gain. Existing OL-50 remains the canonical general rule.
- **Evidence binding**: sanitized source lines 24-132 and every referenced image are
  byte-bound in the
  batch-1 source manifest（A5Ops 绑定外部引用；本语料未收录目标）.
  Representative diagrams were visually checked, but neither authorship nor current
  A5 applicability is authenticated.
- **Promotion gate / falsifier**: promote only after at least two independent
  Ascend950PR ops reproduce the same address-to-conflict relation with exact
  source/binary/case identity, unchanged precision/determinism, and retained timeline
  evidence. One correctly mapped A5 counterexample, or equal timing after controlling
  traffic and instruction count, rules out the transferred rule for that scope.
