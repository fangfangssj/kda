---
type: Runbook
title: "torch.nonzero on Ascend310P3 NPU silently returns CONTENT-CORRUPTED output for rank≥2"
description: "torch_npu 的 `torch.nonzero` 在 310P3 上 rank≥2 静默返回内容错误的结果（行数对、行内容错，非确定首坏行）——厂商 op 坑。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-121"]
---

# torch.nonzero on Ascend310P3 NPU silently returns CONTENT-CORRUPTED output for rank≥2

**PB-121** — **torch.nonzero on Ascend310P3 NPU silently returns CONTENT-CORRUPTED output for rank≥2**: correct row count + dtype, WRONG rows; nondeterministic first-bad-row (439→1017→285 identical-seed), intermittent at benchmark scale (26/50 vs 0/50 across processes), rank-1 correct; Pass A measured `cann_ref_agrees_truth=24/50`. [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) EXCLUDED by a dedicated minimal vendor-op probe (`probes/npu_ref_discrimination.py`) — the corruption is in the vendor op's own output. ACTION: never grade this op family vs model.py-on-NPU on this SoC/CANN — Path-A split (`model.py` vendor-verbatim perf baseline, `model_cpu_truth.py` oracle, `precision_eval_two_tier.py` evaluator); check data-dependent-shape siblings (where/masked_select/unique) for the same class. 22_Nonzero 310p 2026-08-15: Path-A graded 50/50 + 518/518 bit-exact. Does NOT transfer to A5 (P-P80/P-P83 graded clean there). Cross-ref [`OL-68`](ol_68_some_torch_npu_fused_ops_are_unavailable_on_ascend.md) (Case A extended), [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md), [`OL-89`](../precision/ol_89_when_the_active_reference_is_non_deterministic_loo.md), [`OL-90`](../precision/ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md), P-P80, P-P83. backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9 (aarch64); op_class=data-dependent-shape stream-compaction ops (nonzero + siblings: where / masked_select / unique)`
`verified_on: soc=Ascend310P3; cann=9.0.0; torch_npu 2.9 (22_Nonzero 310p benchmark Pass A CANN leg + dedicated discrimination probe, 2026-08-15)`
`unverified_on: other CANN / torch_npu versions; soc=Ascend950PR / Ascend910 (the A5 nonzero lineage P-P80/P-P83 graded clean against the NPU reference — this entry does NOT transfer to those SoCs)`

**Symptom**: `torch.nonzero(x_npu)` for **rank≥2** inputs returns output with the SAME row count and integer dtype but WRONG rows. The corruption is run-to-run **nondeterministic** (identical-seed rank-2 input: first bad row 439 → 1017 → 285 across consecutive runs) and intermittent at benchmark scale (26/50 cases wrong in one process, 0/50 in another — order/load-correlated). **Rank-1 is correct** (12/12). Canonical Pass A measured `cann_ref_agrees_truth=24/50` — the vendor leg itself disagrees with CPU truth about half the time.

**Discrimination (PB-106 excluded)**: reproduced by a dedicated minimal probe (`代码片段 22_Nonzero/npu_ref_discrimination.core.snippet.py（见文末）`) running the vendor op alone on identical-seed inputs — no host-side `.to()`/staging between D2D and launch (PB-106's surface). The corruption is in the VENDOR op's own output, not in our runner's staging.

**Action**: for this op family on this SoC/CANN, do NOT grade against `model.py`-on-NPU. Path-A split: `model.py` stays vendor-verbatim (perf baseline only), `model_cpu_truth.py` is the precision oracle, canonical evaluator = `precision_eval_two_tier.py`（a5_ops 路径，未迁移）; document in `verification.json` precision.method (OL-68 Case A topology, OL-89/OL-90 lineage). Check nonzero-like siblings (data-dependent-shape ops) for the same class before trusting their NPU reference leg.

**Evidence**: 22_Nonzero 310p (2026-08-15, benchmark mode): probe discrimination + Pass A CANN leg 24/50 agreement; shipped kernel graded Path-A — Pass A 50/50 + Pass B 518/518 bit-exact vs CPU truth (`22_Nonzero/verification.json` method fields; device run — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) still carries the documented-stale pre-device `device.runtime: NOT_RUN` flag, same note as PB-120).

**Cross-ref**: [`OL-68`](ol_68_some_torch_npu_fused_ops_are_unavailable_on_ascend.md) (Case A taxonomy — extended by this entry's 2026-08-15 evidence row), [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (host-staging race — excluded by the discrimination probe), [`OL-89`](../precision/ol_89_when_the_active_reference_is_non_deterministic_loo.md) / [`OL-90`](../precision/ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md) (Path-A CPU-truth topology), P-P80 / P-P83 (A5 nonzero lineage — clean there; does not transfer). backend=ascendc.

## 内嵌代码片段

### 22_Nonzero/npu_ref_discrimination.core.snippet.py

```python

def main():
    torch.manual_seed(42)

    # ---- leg 1: vendor torch.nonzero, CPU truth vs NPU execution ----------
    print("== leg 1: torch.nonzero NPU vs CPU (vendor reference health) ==")
    mism = []
    trials = []
    for i in range(12):
        n = 997 + 173 * i
        x = torch.randn(n)
        x[:: 2 + i % 4] = 0
        ref = torch.nonzero(x)
        got = torch.nonzero(x.npu()).cpu()
        ok, det = cmp_nonzero(ref, got)
        trials.append({"i": i, "n": n, "ok": bool(ok), "detail": det})
        if not ok:
            mism.append(i)
    # one rank-2 + one as_tuple + one fp16, to see shape dependence
    for tag, x, at in [
        ("rank2", torch.randn(37, 61), False),
        ("rank2-tuple", torch.randn(37, 61), True),
        ("fp16", torch.randn(4096, dtype=torch.float16), False),
    ]:
        x = x.clone()
        x.view(-1)[::3] = 0
        ref = torch.nonzero(x, as_tuple=at)
        got = torch.nonzero(x.npu(), as_tuple=at)
        got = tuple(t.cpu() for t in got) if at else got.cpu()
        ok, det = cmp_nonzero(ref, got)
        trials.append({"i": tag, "ok": bool(ok), "detail": det})
        if not ok:
            mism.append(tag)
    for t in trials:
        if not t["ok"]:
            print(f"  MISMATCH {t['i']}: {t['detail']}")
    print(f"leg1 verdict: {len(trials) - len(mism)}/{len(trials)} agree "
          f"(mismatches: {mism})")
```
