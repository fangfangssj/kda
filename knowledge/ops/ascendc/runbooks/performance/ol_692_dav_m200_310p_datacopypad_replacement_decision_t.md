---
type: Runbook
title: "dav-m200 (310P) DataCopyPad-replacement decision tree — 4 branches, GM→UB logical/aligned stride split with the padding-not-written gotcha, UB→GM tail-safe writeback (floor + last-32B window), NEVER round the write up over adjacent logical outputs"
description: "每个 DataCopyPad 调用点在 310P 上过四分支决策树：能证明 32B 对齐就 plain DataCopy；GM→UB 非对齐用 compat wrapper；UB→GM 非对齐走尾安全写法"
tags: [task.optimization, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-692"]
---

# dav-m200 (310P) DataCopyPad-replacement decision tree — 4 branches, GM→UB logical/aligned stride split with the padding-not-written gotcha, UB→GM tail-safe writeback (floor + last-32B window), NEVER round the write up over adjacent logical outputs

## 优化点

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; op_class=data-movement ports from DataCopyPad-bearing source (A5/220x/910B source kernels)`
`verified_on: soc=upstream-documented — Ascend-SACT op-ascend-migration skill (gitcode, commit bf37a53c, Apache-2.0), derived from ops-transformer PR87 (moe_finalize_routing 310p) + multi-op 310P migration samples; soc=in-project corroboration — PB-59 (DataCopyPad absent both directions on dav-m200) + P-P123 (our own in-kernel tail strategy, 2 device-verified ops)`
`unverified_on: soc=the compat-wrapper implementations themselves are NOT device-measured in-project — our P-P123 is an independent device-verified alternative; merge caution: treat wrapper behavior as upstream-documented, verify per-op on device`

**Principle**: for EVERY `DataCopyPad` call on a 310P path, run the 4-branch decision tree:
1. **Provably 32B-aligned, no padding semantics needed** → plain `DataCopy` on the 310P branch (per-element counts, `blockLen/32` datablocks).
2. **GM→UB, `DataCopyPad` tolerated non-32B logical length** → compat wrapper (see below); then CHECK whether later compute reads the padded tail.
3. **UB→GM, logical output not 32B-aligned** → tail-safe writeback (see below); **never AlignUp the write length and overwrite neighbors**.
4. **`DataCopyPad` only in the non-310P `#else` branch** → acceptable fallback, NOT an unmigrated defect.

**GM→UB wrapper semantics** (PR87 `DataCopyPadCustom`): logical element count = `blockLen / sizeof(T)`; UB row width = `AlignUp(elem, 32B/sizeof(T))`; GM pointer advances by LOGICAL elements per row, UB pointer by ALIGNED elements. **The wrapper does NOT write the `padParams` values into the UB tail** — if later compute reads padded elements, they must be explicitly zeroed/filled (`DataCopyAtomicCompat`'s tail zero-fill + `SetAndWait<MTE2_S/V_S/S_MTE3>` chain is the pattern). Ext-param call sites keep an Ext wrapper (`DataCopyPadExtCustom`), do NOT force them into plain `DataCopyParams`.

**UB→GM tail-safe**: aligned → plain `DataCopy`. Non-aligned + `blockCount==1` → write the 32B-aligned floor as main body, then handle the last 32B window: in-UB copy of the tail `(mainElem..elem-1) → (mainElem..mainElem+elemPerBlock-1)` with `MTE2_S → V_S → S_MTE3` fences, then a 1-block `DataCopy` from the window start (`dstGm[elem-elemPerBlock]`). `elem < 32B/sizeof(T)` → scalar `GetValue`/`SetValue` loop (with the same fences). **`blockCount>1`**: either per-row tail-safe recursion + `PipeBarrier<PIPE_MTE3>`, or PROVE each GM row-end has legal padding — do NOT assume aligned-stride writes are safe (that's exactly the adjacent-output overwrite class).

**Other instances (predicted)**: any 910B/A2/220x-source port whose kernel uses `DataCopyPad`/`DataCopyExtParams` — the decision tree replaces the A5-era pad-copy assumption on dav-m200; also review `WaitFlag/SetFlag` ordering is preserved when swapping copy APIs (keep original timing unless the target path needs re-review).

**Cross-ref**: PB-59 (the absence), P-P123 (our alternative in-kernel tail strategy), [`OL-167`](../compilation/ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md) (element-count facts), [`OL-407`](../compilation/ol_407_datacopy_count_is_an_element_count_on_every_arch.md) (over-copy trap), [`OL-415`](../compilation/ol_415_datacopyparams_stride_slots_are_direction_asymme.md) (DataCopyParams 32B-datablock semantics), [`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md)/[`OL-409`](ol_409_index_metadata_stable_at_kernel_launch_host_writ.md) (scalar sync needs), [`OL-25`](../compilation/ol_25_tbuf_has_no_automatic_synchronization_must_use_tqu.md) (TQue/TBuf sync). backend=ascendc.
