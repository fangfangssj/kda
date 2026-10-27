---
type: Runbook
title: "Brent-Kung / Blelloch work-efficient scan LOSES to Hillis-Steele on an ISSUE-BOUND A5 vector unit — op-issue-count dominates total element-work"
description: "Brent-Kung / Blelloch work-efficient scan LOSES to Hillis-Steele on an ISSUE-BOUND A5 vector unit — op-issue-count dominates total element-work"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-BRENT-KUNG-WORK-EFFICIENT-SCAN-LOSES-ISSUE-BOUND"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-BRENT-KUNG-WORK-EFFICIENT-SCAN-LOSES-ISSUE-BOUND — Brent-Kung / Blelloch work-efficient scan LOSES to Hillis-Steele on an ISSUE-BOUND A5 vector unit — op-issue-count dominates total element-work

## 适用场景

- 适用平台与范围：soc=Ascend950PR (V351/arch35)；cann=9.1.T500；bisheng=AIV；op_class=scan/SSM（小 N 状态空间，如 Mamba selective_scan）；dtype=fp32。
- 验证来源：selective_scan_source_a5 fwd-SIMD perf-loop iter-4（2026-06-23，A5/Ascend950PR_957b）白盒实测的反模式记录（NOT pursued）；含 contiguous 对照实验隔离 issue-count 成因。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.T500; bisheng=AIV; op_class=scan/SSM (small-N state-space, e.g. Mamba selective_scan); dtype=fp32`
**Source**: selective_scan_source_a5 fwd-SIMD perf-loop iter-4 (2026-06-23, A5/Ascend950PR_957b) | **Validation status**: anti-pattern, whitebox-measured, NOT pursued

**Concept (the instinct)**: the textbook says a work-efficient parallel scan (Brent-Kung / Blelloch, O(L) total work + O(log L) depth) beats the work-inefficient Hillis-Steele (O(L log L) work). So porting the scan from Hillis-Steele (HS) to Brent-Kung (BK) should cut total work ~3.5× and win.

**Why it LOSES (the A5 vector unit is ISSUE-BOUND, not work-bound)**: measured BK ran **11.6× SLOWER** than HS despite doing **3.5× LESS total element-work**. A control test isolated the cause:
- A **contiguous same-op-count BK (BKc)** control — BK's op structure but contiguous (non-strided) access — measured only **+1.1%** over HS. So the strided sub-granule access of the BK tree is NOT the killer.
- The real **11.5× killer is op-ISSUE-COUNT**: BK's up-sweep/down-sweep tree issues MANY small per-node ops, each only N=16 wide; HS issues FEW maximally-wide contiguous-L passes (~4096 elements each). On A5's issue-bound vector unit (a width-16 op issues in the same time as a width-128 op — OL-231's measured W=16 ≈ W=128 fact), **issue-count dominates total element-work**: 3.5× less work spread over many-times-more issues is a net loss.

**The general principle**: on an issue-bound vector unit, the scan optimum is **few maximally-wide contiguous ops**, NOT minimal total work. HS-over-contiguous-L (few wide passes) IS the issue-optimum; the work-efficient tree (many narrow ops) is exactly the wrong shape. Any restructure of the scan into more/smaller ops loses, regardless of how much element-work it saves.

**Boundary (where BK might compete)**: N(dstate) ≥ 64 changes the calculus — at full-granule N each tree node is a full-lane op, so the issue-count penalty per node is amortized over real lane-work and the work-efficiency can matter. For N=16 (single-head Mamba scan) HS wins decisively.

**Promote when**: a 2nd issue-bound vector op reproduces "work-efficient algorithm loses to wide-contiguous despite less total work, isolated to op-issue-count by a contiguous-control test", OR an N≥64 variant where BK competes. Cross-ref: [OL-231](ol_231_unoptimized_prototype_not_architecture_verdict.md) (issue-bound W=16≈W=128 anchor + the architecture-floor consolidation; this is the iter-4 lever in the 8-lever failure set), OL-245 (per-VF-call issue overhead — the same issue-count-dominates principle for regbase re-entry), P-P106 (the scan structure). Whitebox trace: `workspace/ss_perf_loop/whitebox_log.md` (iter-4). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-BRENT-KUNG-WORK-EFFICIENT-SCAN-LOSES-ISSUE-BOUND，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
