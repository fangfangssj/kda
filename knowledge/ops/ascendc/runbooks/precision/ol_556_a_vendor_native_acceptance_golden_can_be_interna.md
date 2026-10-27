---
type: Runbook
title: "a vendor-native acceptance golden can be internally non-deterministic — run the golden twice and self-diff it"
description: "厂商 golden 自身可能不确定——评分前先自跑两遍 self-diff，有 diff 的 case 降级用 CPU truth 评，绝不为匹配随机结果改 kernel"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-556"]
---

# a vendor-native acceptance golden can be internally non-deterministic — run the golden twice and self-diff it

**OL-556** — **a vendor-native acceptance golden can be internally non-deterministic — run the golden twice and self-diff it before grading, fall back to an independent truth (CPU) for `vendor_self_diff > 0` cases.** Decision rule: self-diff==0 → grade vs vendor; self-diff>0 → grade those cases vs CPU truth and report the split; never patch the kernel to match a racy draw ([`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md)). add_v2 310p 2026-08-14 (vendor broadcast add self-diffed). Cross-ref [`PB-122`](../compilation/pb_122_vendor_torch_add_broadcast_is_non_deterministic.md), OL-319, [`PB-111`](../compilation/pb_111_cann_aclnnincreflashattention_torch_npu_npu_incr.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=broadcast elementwise under vendor-native-golden acceptance, extensible to any vendor op with an internal race`
`verified_on: soc=Ascend310P3; cann=9.0.0 (add_v2 310p, 2026-08-14)`
`unverified_on: soc=Ascend910, Ascend950PR — a vendor op's internal determinism is per-SoC/per-build; self-diff-probe the golden before assuming it is stable on any other environment`

- **Category**: harness / acceptance methodology
- **Trigger**: the acceptance truth regime is vendor-native (`310p_vendor_native`), and the op is a broadcast/strided elementwise (or any vendor op with a plausible internal race: reduction order, scatter, top-k tie-break).
- **Principle**: a vendor-native golden is a measurement instrument, and like any instrument it can be NON-DETERMINISTIC on its own (self-diff across identical draws), not just off-spec (OL-319). Grading a candidate against a self-diffing golden makes the verdict depend on which draw was compared — the precision test is flaky-by-construction through no fault of the ported kernel. Before trusting a vendor-native golden, PROBE its self-consistency: run the golden twice on the same inputs and compare; if `vendor_self_diff > 0`, those cases are ungradable against the vendor and must fall back to an independent truth (CPU decomposition of the documented algorithm) or be flagged/skipped.
- **Concrete anchor (add_v2 310p)**: vendor `torch.add` broadcast self-diffed 8320/8672/43583/… elements across 3 draws on broadcast cases, while the ported kernel matched CPU truth bit-exactly (10/10, 0 diff) in isolation. The acceptance result (38/40) was the golden disagreeing with itself. Harness guard: run `Model.forward` twice; a `vendor_self_diff` gate flags the cases the vendor cannot grade.
- **Mitigation paths (decision rule) — applies to ANY vendor-native-golden op**: (a) golden self-diff == 0 on all cases → grade normally against the vendor; (b) golden self-diff > 0 on some cases → those cases are vendor-ungradable: grade them against CPU truth and report the split, never against the racy vendor draw. Do NOT patch the kernel to match a specific racy draw (OL-85).
- **Evidence**: add_v2 310p (2026-08-14, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): cases 19/30/31 self-diff on broadcast add; same-shape controls deterministic. `add_v2/knowledge_update.md` F1 + PP-candidate.
- **Other instances (predicted)**: any broadcast/strided elementwise op under vendor-native truth; vendor sort/topk tie-break; vendor scatter/reduction with unspecified order; fused ops whose vendor impl has an internal race. Rule of thumb: self-diff-probe the golden once per op class before relying on it.
- **Cross-ref**: [`PB-122`](../compilation/pb_122_vendor_torch_add_broadcast_is_non_deterministic.md) (the platform fact — vendor broadcast-add non-determinism), OL-319 (the deterministic-but-off-spec sibling: two-tier truth ladder), [`PB-111`](../compilation/pb_111_cann_aclnnincreflashattention_torch_npu_npu_incr.md) (vendor-behavior-is-spec), [`PB-114`](../compilation/pb_114_ascend310p_torch_npu_f_gelu_aclnngelu_ignores_th.md)/[`PB-116`](../compilation/pb_116_ascend310p_torch_npu_has_no_fp64_double_dtype_si.md) (truth-regime and fp64 caveats), [`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md) (anti-overfit), [`OL-83`](ol_83_1_ulp_boundary_mismatch_signature_probe_must_be_ex.md) (two-valid-paths). backend=ascendc.

## 补充证据（2026-10-10 合并自 `vendor_golden_nondeterministic_self_diff_before_grading.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-556 — OL-556: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=broadcast elementwise under vendor-native-golden acceptance, extensible to any vendor op with an internal race。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (add_v2 310p, 2026-08-14)。

**OL-556** — **a vendor-native acceptance golden can be internally non-deterministic — run the golden twice and self-diff it before grading, fall back to an independent truth (CPU) for `vendor_self_diff > 0` cases.** Decision rule: self-diff==0 → grade vs vendor; self-diff>0 → grade those cases vs CPU truth and report the split; never patch the kernel to match a racy draw (OL-85). add_v2 310p 2026-08-14 (vendor broadcast add self-diffed). Cross-ref [PB-122](../compilation/vendor_torch_add_broadcast_nondeterministic_dav_m200.md), [OL-319](vendor_ref_diverging_from_spec_two_tier_ladder.md), [PB-111](../compilation/pb_111_cann_aclnnincreflashattention_torch_npu_npu_incr.md). backend=ascendc.

- Former A5Ops ID: `OL-556`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=broadcast elementwise under vendor-native-golden acceptance, extensible to any vendor op with an internal race`
`verified_on: soc=Ascend310P3; cann=9.0.0 (add_v2 310p, 2026-08-14)`
`unverified_on: soc=Ascend910, Ascend950PR — a vendor op's internal determinism is per-SoC/per-build; self-diff-probe the golden before assuming it is stable on any other environment`

- **Category**: harness / acceptance methodology
- **Trigger**: the acceptance truth regime is vendor-native (`310p_vendor_native`), and the op is a broadcast/strided elementwise (or any vendor op with a plausible internal race: reduction order, scatter, top-k tie-break).
- **Principle**: a vendor-native golden is a measurement instrument, and like any instrument it can be NON-DETERMINISTIC on its own (self-diff across identical draws), not just off-spec (OL-319). Grading a candidate against a self-diffing golden makes the verdict depend on which draw was compared — the precision test is flaky-by-construction through no fault of the ported kernel. Before trusting a vendor-native golden, PROBE its self-consistency: run the golden twice on the same inputs and compare; if `vendor_self_diff > 0`, those cases are ungradable against the vendor and must fall back to an independent truth (CPU decomposition of the documented algorithm) or be flagged/skipped.
- **Concrete anchor (add_v2 310p)**: vendor `torch.add` broadcast self-diffed 8320/8672/43583/… elements across 3 draws on broadcast cases, while the ported kernel matched CPU truth bit-exactly (10/10, 0 diff) in isolation. The acceptance result (38/40) was the golden disagreeing with itself. Harness guard: run `Model.forward` twice; a `vendor_self_diff` gate flags the cases the vendor cannot grade.
- **Mitigation paths (decision rule) — applies to ANY vendor-native-golden op**: (a) golden self-diff == 0 on all cases → grade normally against the vendor; (b) golden self-diff > 0 on some cases → those cases are vendor-ungradable: grade them against CPU truth and report the split, never against the racy vendor draw. Do NOT patch the kernel to match a specific racy draw (OL-85).
- **Evidence**: add_v2 310p (2026-08-14, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): cases 19/30/31 self-diff on broadcast add; same-shape controls deterministic. `workspace/add_v2/knowledge_update.md` F1 + PP-candidate.
- **Other instances (predicted)**: any broadcast/strided elementwise op under vendor-native truth; vendor sort/topk tie-break; vendor scatter/reduction with unspecified order; fused ops whose vendor impl has an internal race. Rule of thumb: self-diff-probe the golden once per op class before relying on it.
- **Cross-ref**: PB-122 (the platform fact — vendor broadcast-add non-determinism), OL-319 (the deterministic-but-off-spec sibling: two-tier truth ladder), PB-111 (vendor-behavior-is-spec), PB-114/PB-116 (truth-regime and fp64 caveats), OL-85 (anti-overfit), OL-83 (two-valid-paths). backend=ascendc.
