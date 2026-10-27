---
type: Runbook
title: "SIMD class kernel + SIMT `__simt_vf__` kernel coexistence in same `.so` corrupts SIMT path"
description: "SIMD class kernel + SIMT `__simt_vf__` kernel coexistence in same `.so` corrupts SIMT path"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PB-SIMD-SIMT-COEXIST"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PB-SIMD-SIMT-COEXIST — SIMD class kernel + SIMT `__simt_vf__` kernel coexistence in same `.so` corrupts SIMT path

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；观测平台 Ascend950PR_9579 / CANN 9.0.0（卡内 Source 行），适用场景为同一 `.so` 同时构建 SIMD 类 kernel（Init/Process + TPipe/TQue/TBuf）与 SIMT `__simt_vf__` kernel 的混合模块（正文 Symptom 段）。
- 验证来源：卡内无 verified_on 行；1 op 证据（26_AvgPool3d kw-2 SIMT 回归观测，2026-05-03），卡内自述 hypothesis plausible but unverified、尚无第 2 op 最小复现（正文 Validation status 段）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Severity**: HIGH (would block any Kind-2 SIMD rewrite of multi-path SIMT operators)
**Source**: 26_AvgPool3d kw-2 (2026-05-03 Ascend950PR_9579, CANN 9.0.0) — 8 SIMD UB-tile rewrite iterations + 1 revert.
**Validation status**: 1 op evidence (SIMT regression observed); hypothesis plausible but unverified; no minimal repro on a 2nd op yet.

**Symptom**: when a kernel module emits BOTH a SIMD class kernel (Init/Process pattern with `TPipe + TQue + TBuf`) AND a SIMT `__simt_vf__` kernel in the same `.so`, the SIMT path may produce subtly wrong results — including for SIMT cases that worked correctly when the SIMD class was absent.

**Specific evidence (op#26 AvgPool3d kw-2)**:

The SIMT generic path was a verbatim copy of kw-1's `avgpool3d_vf<T>` (renamed to `avgpool3d_simt_fallback_vf<T>` for kw-2's mixed build). In kw-1 (SIMT-only build) it passed 72/72 cases; in kw-2 (SIMD class + SIMT in the same `.so`) the same SIMT code FAILED on 4-7 cases per dtype:

| Dtype | kw-1 SIMT-only | kw-2 SIMD+SIMT |
|---|---|---|
| fp32 | 52/52 PASS | 42-44/52 PASS |
| fp16 | 10/10 PASS | 5-6/10 PASS |
| bf16 | 10/10 PASS | 5-6/10 PASS |

Failing cases mix BOTH fast-path-routed (SIMD class) AND generic-routed (SIMT) — the SIMT regression cannot be explained by a SIMD class bug alone.

**Plausible mechanism (not yet confirmed)**:
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` set in both SIMD class entry-points AND SIMT entry-points
- SIMD class instantiates `TPipe + TQue + TBuf` state that may persist across kernel boundaries
- SIMT path's grid-stride loop with `__gm__ T*` scalar reads may pick up stale event/queue state from prior SIMD invocation
- OR mixed `__global__ __aicore__` kernels with `Simt::VF_CALL<...>` may force compiler to a different code-gen mode for the SIMT TU

**Mitigation strategies (proposed, not validated)**:
1. **One-pattern-per-.so rule**: either all-SIMT or all-SIMD class. For multi-path kernels (fast + generic), implement BOTH paths via the same pattern (generic path also uses Init/Process class with conditional logic in `Process()`).
2. **Separate .so per path**: build SIMD class in `_op_simd.so`, SIMT in `_op_simt.so`, dispatch at Python pybind level.
3. **Debug-instrument the SIMT regression**: write canary GM values at start/end of SIMT kernel to confirm whether SIMT is being called at all or its state is corrupted by prior SIMD invocation.

**Workaround applied (op 26)**: reverted to kw-1 SIMT-only baseline (72/72 PASS, 0.14× perf). Structural-ceiling exit at 0.14× — ko-1's 3-axis ablation already confirmed HBM scalar-load latency is the SIMT bottleneck; the architectural shift to SIMD class is blocked by this coexistence issue.

**Promotion criteria** (CAND → PB-N):
1. Reproduce on a second op where mixed SIMD class + SIMT `__simt_vf__` build exhibits SIMT regression vs SIMT-only build
2. Minimal repro: 2-kernel `.so` with one trivial SIMD class kernel and one SIMT scalar kernel reading from independent GM buffers — does the SIMT kernel produce different output when SIMD class is linked?
3. Identify exact compiler/linker code-gen difference between mixed and SIMT-only builds (`bisheng -S` or equivalent)
4. Confirm with CANN team if this is intentional ABI vs unintended pipeline-state leak

**Related**:
- PB-9 (DataCopy localDst/localSrc silent corruption — same family: pipeline-state correctness gotchas)
- OL-63 (TQue VECIN depth — same family: pipeline configuration affects code-gen)
- A3 op#13_Cat finding 2026-04-25 (KERNEL_TASK_TYPE_DEFAULT macro affects code-gen path — sibling concern)
- aog-self-critic C5 (no premature platform-blame — must reproduce on 2nd op before promoting to PB-N)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PB-SIMD-SIMT-COEXIST，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
