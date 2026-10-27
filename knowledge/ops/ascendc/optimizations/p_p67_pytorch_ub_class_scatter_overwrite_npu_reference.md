---
type: Optimization
title: "PyTorch-UB-class scatter-overwrite — NPU reference is non-deterministic, kernel cannot chase"
description: "Problem class: torch.index_put_(accumulate=False) and torch.scatter_(reduce=None) (\"assignment scatter\") on inputs with duplicate indices are PyTorch-undefined-behavior per docs. NPU torch_npu resolve"
tags: [task.debugging, topic.precision, topic.implementation.atomic, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p67-pytorch-ub-class-scatter-overwrite-npu-reference.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# PyTorch-UB-class scatter-overwrite — NPU reference is non-deterministic, kernel cannot chase

> 原 OKF v1 卡号：**P-P67**。

## 优化点

**Problem class**: `torch.index_put_(accumulate=False)` and `torch.scatter_(reduce=None)` ("assignment scatter") on inputs **with duplicate indices** are PyTorch-undefined-behavior per docs. NPU `torch_npu` resolves duplicate writes via parallel hardware-thread scheduling — winner per dup slot is **non-deterministic across runs** when contention is high enough.

**Concrete signature** (op#19 pp-2 measurement, fp16/bf16/fp32, K ∈ {64..16384}, fan_in=K/N≈0.5 with allow_dup_indices):
- 5-run NPU bit-eq check: 3/17 small-K cases (K ≤ 128) deterministic; 14/17 K ≥ 256 cases NON-deterministic
- Pairwise max abs diff across 5 NPU runs: 3.2 .. 4.39 (fp16, K=2560)
- ~1.2 % of slots flip identity between any two runs

**Anti-pattern (what NOT to do)**:
- ❌ Don't write a deterministic kernel rule (last-wins / first-wins / wave-chunk / AIV-block) and expect it to match. Best rule pp-2 found: WF32 (wave-firstwins W=32) at 61% mean / 31% min match. Far below the 90% acceptance bar.
- ❌ Don't use atomicAdd / SetAtomicAdd / SyncAll / multi-core scatter to "match NPU's parallelism". This re-introduces kernel non-determinism (DEBT-053 motivator); fails determinism gate; STILL doesn't bit-match the ref because the ref's randomness is HW-scheduling-dependent, not algorithm-dependent.
- ❌ Don't add case-specific predicates (`if K > threshold use rule X else rule Y`). OL-85 anti-overfitting violation.

**Correct response (what TO do)**:
1. **Precision-probe Step 1** (mandatory): replicate verifier's exact input gen, run NPU 5× on cloned buffers with explicit `torch.npu.synchronize()`, save pairwise diff matrix. Probe template: `workspace/indexput/probes/pp2_step1_dup_det.py`, `pp2_step1b_case33_isolated.py`.
2. **Precision-probe Step 2** (mandatory before classifying): exhaustive structured-rule search (R1 last/first, R3/R4 wave-chunk × W∈{8..512}, R5 sort-stable, R6 AIV-block × A∈{1,2,4,8,16,20,40}, R7 round-robin). Record per-case match rates. Templates: `pp2_step2_rule_search.py`, `pp2_step3_aiv_rule.py`.
3. **If best rule ≥ 99% on every case** → kernel-implementable, write it as deterministic single-thread algorithm.
4. **If best rule ≤ 90% mean OR 5-run pairwise diff > 0** → **REQUIREMENT** verdict per OL-90. Kernel ships with deterministic single-core single-source-order scan (matches CPU torch semantics, not NPU torch). Failures are spec-level UB, not kernel bugs.

**Verifier-side mitigations** (OL-90 lists 4): alt-ref hook (CPU-torch fallback for UB-trigger cases), case-gen `allow_dup_indices=False` default, dual-reference REPORT row, per-case tolerance loosen with documented rationale.

**Cross-reference**:
- **OL-90** (PyTorch-UB-class detection + verifier-side mitigation) — full canonical entry
- **OL-85** (logic-first, anti-overfitting) — forbids case-specific predicate hacks
- **OL-88** (ref non-det preflight) — sibling class; OL-88 = CANN op internal race, P-P67 = spec-level user-input UB
- **P-P61** (kernel runtime determinism) — kernel side stays deterministic regardless of ref behavior; non-det ref is no excuse for non-det kernel
- **DEBT-053** (op#19 sequential-reset) — the motivator that drove discovering this pattern

**Evidence summary**:
- op#19 IndexPut 29/46 PASS (22/22 acc=True + 7/7 acc=False no-dup + 0/17 acc=False with-dup); deterministic-kernel by construction (single AIV core, THREAD_NUM=1, no atomic, 5-run gate 46/46 IDENTICAL)
- pp-2 probes: `workspace/indexput/probes/pp2_npu_dup_det_5run.json` (Step 1), `pp2_step2_rule_match.json` (Step 2 R1-R5), `pp2_step3_aiv_rule.json` (Step 2 R6-R7)
- pp-2 verdict: REQUIREMENT — no deterministic kernel can match a non-deterministic ref. Published 17/46 gap is the canonical result until verifier-side alt-ref lands (DEBT recommendation).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P67，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: `index_put_(accumulate=False)` / `scatter_(reduce=None)` on duplicate indices is PyTorch-UB; NPU torch resolves duplicates via HW thread scheduling — non-deterministic across runs (op#19 pp-2: 14/17 cases jitter, max pairwise diff 4.39 fp16). No deterministic rule (last/first/wave-chunk W∈{8..512}/AIV-block A∈{1..40}/round-robin) reaches 90% match (best WF32 mean 61%, min 31%). Kernel MUST NOT use atomic / multi-core scatter to "match NPU randomness" (re-introduces kernel non-det); MUST NOT use case-specific predicates (OL-85 violation). Ship deterministic single-core CPU-semantic scan; gap is REQUIREMENT not kernel bug. Mitigation = verifier-side alt-ref / case-gen `allow_dup_indices=False`. See OL-90 for full protocol.；优先级: CRITICAL。

> 索引行原文：`| P-P67 | **PyTorch-UB-class scatter-overwrite — NPU reference non-deterministic, kernel cannot chase** | scatter_add | `index_put_(accumulate=False)` / `scatter_(reduce=None)` on duplicate indices is PyTorch-UB; NPU torch resolves duplicates via HW thread scheduling — non-deterministic across runs (op#19 pp-2: 14/17 cases jitter, max pairwise diff 4.39 fp16). No deterministic rule (last/first/wave-chunk W∈{8..512}/AIV-block A∈{1..40}/round-robin) reaches 90% match (best WF32 mean 61%, min 31%). Kernel MUST NOT use atomic / multi-core scatter to "match NPU randomness" (re-introduces kernel non-det); MUST NOT use case-specific predicates (OL-85 violation). Ship deterministic single-core CPU-semantic scan; gap is REQUIREMENT not kernel bug. Mitigation = verifier-side alt-ref / case-gen `allow_dup_indices=False`. **See OL-90 for full protocol**. | **CRITICAL** |`
