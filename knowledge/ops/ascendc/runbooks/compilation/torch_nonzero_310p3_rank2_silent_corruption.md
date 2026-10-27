---
type: Runbook
title: "`torch.nonzero` on Ascend310P3 NPU silently returns content-corrupted output for rank≥2"
description: "`torch.nonzero` on Ascend310P3 NPU silently returns content-corrupted output for rank≥2"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-121 — PB-121: A5Ops additive extension

> 关联：同一 PB-121 主题已由既有卡 [pb_121_torch_nonzero_on_ascend310p3_npu_silently_return.md](pb_121_torch_nonzero_on_ascend310p3_npu_silently_return.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；torch_npu 2.9（aarch64）；op_class=数据依赖形状的 stream-compaction 算子（nonzero 及同类 where / masked_select / unique）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0；torch_npu 2.9（22_Nonzero 310p benchmark Pass A CANN 腿 + 专用判别探针，2026-08-15）。

**PB-121** — **torch.nonzero on Ascend310P3 NPU silently returns CONTENT-CORRUPTED output for rank≥2**: correct row count + dtype, WRONG rows; nondeterministic first-bad-row (439→1017→285 identical-seed), intermittent at benchmark scale (26/50 vs 0/50 across processes), rank-1 correct; Pass A measured `cann_ref_agrees_truth=24/50`. PB-106 EXCLUDED by a dedicated minimal vendor-op probe (`probes/npu_ref_discrimination.py`) — the corruption is in the vendor op's own output. ACTION: never grade this op family vs model.py-on-NPU on this SoC/CANN — Path-A split (`model.py` vendor-verbatim perf baseline, `model_cpu_truth.py` oracle, `precision_eval_two_tier.py` evaluator); check data-dependent-shape siblings (where/masked_select/unique) for the same class. 22_Nonzero 310p 2026-08-15: Path-A graded 50/50 + 518/518 bit-exact. Does NOT transfer to A5 (P-P80/P-P83 graded clean there). Cross-ref [OL-68](ol_68_some_torch_npu_fused_ops_are_unavailable_on_ascend.md) (Case A extended), PB-106, OL-89, OL-90, P-P80, P-P83. backend=ascendc.

- Former A5Ops ID: `PB-121`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9 (aarch64); op_class=data-dependent-shape stream-compaction ops (nonzero + siblings: where / masked_select / unique)`
`verified_on: soc=Ascend310P3; cann=9.0.0; torch_npu 2.9 (22_Nonzero 310p benchmark Pass A CANN leg + dedicated discrimination probe, 2026-08-15)`
`unverified_on: other CANN / torch_npu versions; soc=Ascend950PR / Ascend910 (the A5 nonzero lineage P-P80/P-P83 graded clean against the NPU reference — this entry does NOT transfer to those SoCs)`

**Symptom**: `torch.nonzero(x_npu)` for **rank≥2** inputs returns output with the SAME row count and integer dtype but WRONG rows. The corruption is run-to-run **nondeterministic** (identical-seed rank-2 input: first bad row 439 → 1017 → 285 across consecutive runs) and intermittent at benchmark scale (26/50 cases wrong in one process, 0/50 in another — order/load-correlated). **Rank-1 is correct** (12/12). Canonical Pass A measured `cann_ref_agrees_truth=24/50` — the vendor leg itself disagrees with CPU truth about half the time.

**Discrimination (PB-106 excluded)**: reproduced by a dedicated minimal probe (`workspace/22_Nonzero/probes/npu_ref_discrimination.py`) running the vendor op alone on identical-seed inputs — no host-side `.to()`/staging between D2D and launch (PB-106's surface). The corruption is in the VENDOR op's own output, not in our runner's staging.

**Action**: for this op family on this SoC/CANN, do NOT grade against `model.py`-on-NPU. Path-A split: `model.py` stays vendor-verbatim (perf baseline only), `model_cpu_truth.py` is the precision oracle, canonical evaluator = `src/scripts/precision_eval_two_tier.py`; document in `verification.json` precision.method (OL-68 Case A topology, OL-89/OL-90 lineage). Check nonzero-like siblings (data-dependent-shape ops) for the same class before trusting their NPU reference leg.

**Evidence**: 22_Nonzero 310p (2026-08-15, benchmark mode): probe discrimination + Pass A CANN leg 24/50 agreement; shipped kernel graded Path-A — Pass A 50/50 + Pass B 518/518 bit-exact vs CPU truth (`workspace/22_Nonzero/verification.json` method fields; device run — `hardware/target/ascend310p.md` still carries the documented-stale pre-device `device.runtime: NOT_RUN` flag, same note as PB-120).

**Cross-ref**: OL-68 (Case A taxonomy — extended by this entry's 2026-08-15 evidence row), PB-106 (host-staging race — excluded by the discrimination probe), OL-89 / OL-90 (Path-A CPU-truth topology), P-P80 / P-P83 (A5 nonzero lineage — clean there; does not transfer). backend=ascendc.
