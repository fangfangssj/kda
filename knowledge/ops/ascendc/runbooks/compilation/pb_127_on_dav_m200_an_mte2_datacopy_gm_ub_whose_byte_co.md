---
type: Runbook
title: "on dav-m200, an MTE2 `DataCopy` (GM→UB) whose BYTE count is not a 32B multiple leaves a stale tail — bytes past the down-rounded quantum are never written; bulk-copy the aligned prefix and scalar-widen the tail"
description: "非 32B 对齐字节数的行加载跑得动但精度坏，误差阶梯随超出末个量子的字节数放大，fp16 可出 NaN；32B 对齐的全 1e-8 干净。"
tags: [task.debugging, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-127", "DataCopy"]
---

# on dav-m200, an MTE2 `DataCopy` (GM→UB) whose BYTE count is not a 32B multiple leaves a stale tail — bytes past the down-rounded quantum are never written; bulk-copy the aligned prefix and scalar-widen the tail

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all (any GM→UB row/window copy with arbitrary row length)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (23_HyenaFftSizePaddingRfft, 2026-09-05 — binary discrimination over 9 S-values × 3 dtypes, DEVICE-RUN)`
`unverified_on: soc=Ascend950PR, Ascend910_V220 (PB-22 documents the 220x per-TBuf 32B LIMIT — a different symptom; V351 DataCopyPad evidence inside PB-22 suggests unaligned handling differs there; exact unaligned-count behavior unmeasured)`

- **Severity**: CRITICAL (silent wrong data — the consumer reads stale window bytes; no error, no assert).
- **Symptom**: row loads with unaligned byte counts run clean but fail precision in a magnitude ladder that tracks how far past the last 32B quantum the row ends: fp32 S=100 (400B) max|dR|=1.75e+02; S=98 (392B) 4.5e-1; fp16 S=200 (400B) NaN; EVERY 32B-aligned S passed at 1e-8. The bf16 scalar-load path was immune at all S (it never uses bulk MTE2).
- **Mechanism**: the device-level manifestation of OL-561's clause-3 header contract — unaligned counts are silently rounded DOWN, so the sub-32B tail of the destination window is never written and keeps stale bytes. Distinct from PB-22 (220x: counts >32B truncate to the first 32B regardless of alignment): here an aligned count of ANY size is fine; only the count's 32B alignment matters.
- **Fix**: bulk-copy only the 32B-aligned prefix — `fullElems = S/8*8` (fp32) / `S/16*16` (fp16) — and scalar-widen the tail elements from `__gm__` (`23_HyenaFftSizePaddingRfft/kernel/hyenarfft_kernel.h:583-596 [snippet-unavailable: source workspace not on this box]`). This also avoids the GM overread a padded full-row copy would do at row ends. PB-105: no `DataCopyPad` on dav-m200 to absorb the tail natively.
- **Evidence**: 23_HyenaFftSizePaddingRfft 310p (2026-09-05, failures_ledger precision iter 2 — the two `v220_prebuild_check` warnings at LoadRow flagged exactly these sites): binary discrimination matrix 9 S-values × 3 dtypes as above; post-fix 49/49 PASS_T1 strict + 78/78 edge vs recomputed fp64 (`23_HyenaFftSizePaddingRfft/verification.json`; `23_HyenaFftSizePaddingRfft/failures_ledger.md:8`).
- **Other instances (predicted)**: any variable-row-length GM→UB staging on dav-m200 (varlen packing, ragged tails, attention row windows); same construction discipline as CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL's 256B-quantum rule, here derived from the violating side.
- **Cross-ref**: [`OL-561`](ol_561_the_per_soc_datacopy_impl_header_is_the_alignmen.md) (the impl-header contract this confirms on device), [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md) (220x 32B limit — the sibling rule), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (no DataCopyPad on dav-m200), CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL. backend=ascendc.
