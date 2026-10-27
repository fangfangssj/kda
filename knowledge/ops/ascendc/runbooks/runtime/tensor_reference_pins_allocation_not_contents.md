---
type: Runbook
title: "An at::Tensor reference pins the allocation, not the contents: caching anything derived from a tensor needs a content guard, and PyTorch's version counter is it"
description: "An at::Tensor reference pins the allocation, not the contents: caching anything derived from a tensor needs a content guard, and PyTorch's version counter is it"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-511"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-511 — OL-511: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=any; cann=any; bisheng=n/a; op_class=any op caching weight-derived or input-derived tensors across calls; build=pybind/torch extension。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention, main 2026-09-05 (defect reproduced on 3 tensors, fix verified on each)。

**OL-511** — **a `at::Tensor` reference pins the ALLOCATION, not the CONTENTS: caching anything derived from a tensor needs a CONTENT guard, and PyTorch's version counter is it.** Caching batch-invariant weight preprocessing keyed on `data_ptr + sizes + dtype` looks safe — holding a reference does stop the address being recycled by a different tensor — but it does NOT stop the caller mutating that tensor in place, and `wk_in.to(dtype).contiguous()` returns the CALLER'S OWN OBJECT when the input is already that dtype and contiguous, so the alias is handed straight back. Result: every key field still matches and a stale derivative is served silently. Reproduced on 4-42: `max_abs` 4.5 (repacked conv weights), 5.27 (folded weights), 0.017 (folded bias). Fix within a bounded scope: compare `t.unsafeGetTensorImpl()->version_counter().current_version()` on the CALLER's tensors. ⚠️ **This covers only in-place modification that PyTorch actually TRACKS through a shared version counter.** It does NOT cover `.data` writes, raw/external writes to the storage, aliases constructed with an independent version counter, or inference tensors — so it is **not** a general guarantee of cache safety, only a guard against the tracked case that was measured here. **Every cached input needs its own guard** — guarding one leaves the others stale while the control stays green. backend=all.

- Former A5Ops ID: `OL-511`
- Fusion disposition: `additive_new`

`applies_to: soc=any; cann=any; bisheng=n/a; op_class=any op caching weight-derived or input-derived tensors across calls; build=pybind/torch extension`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention, main 2026-09-05 (defect reproduced on 3 tensors, fix verified on each)`
`unverified_on: non-PyTorch frontends — the version-counter mechanism is PyTorch-specific; the underlying hazard (reference pins allocation, not contents) is general`

- **Category**: correctness / caching
- **Trigger**: caching a repack, fold, transpose, or any derived tensor across forward calls to avoid recomputing batch-invariant work.
- **Why the caching is worth doing at all**: a reference implementation typically holds prepared weights persistently in its module, OUTSIDE the timed region, so recomputing them per call is an artifact of the ported decomposition rather than required work. Removing it makes the comparison like-for-like.
- **The trap, precisely**: `data_ptr` equality answers "is this the same storage", not "are the bytes the same". A held reference prevents *reuse of the address by a different tensor*; it does not prevent *mutation of that tensor*.
- **Control that actually tests it**: populate the cache, MUTATE the source in place (through an alias, to cover both routes), call again, compare against the uncached path. Crucially, first run a POSITIVE CONTROL ON THE MUTATION — with the cache off, output before vs after must DIFFER; otherwise the mutation is not reaching the kernel and the control is void. A first attempt at this control was inert because it mutated a field the forward clears after the first call, and both guarded and unguarded modes "passed".
- **Evidence**: main 2026-09-05, 4-42; unguarded mode reproduced stale results on all three cached tensors, guarded mode matched the uncached path exactly (`max_abs` 0.0), unchanged-weight behaviour bit-identical on all 50 cases.
- **Cross-ref**: OL-510 (this is one instance of that family), OL-507. backend=all.
