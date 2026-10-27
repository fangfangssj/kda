---
type: Runbook
title: "CANN ascendc cmake incremental device-object merge is flaky"
description: "CANN ascendc cmake 增量构建的设备目标合并不稳定——ld.lld 报 unknown file type，必要时全量重建"
tags: [task.debugging, topic.implementation.synchronization, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-125"]
---

# CANN ascendc cmake incremental device-object merge is flaky

**PB-125** — **CANN ascendc cmake incremental device-object merge is flaky** — `ld.lld` fails "unknown file type" on an UNCHANGED .o that compiled clean in the same invocation. Build-infra flake, not a source regression; always clean-rebuild (acceptance lane's ASCENDC_CLEAN_BUILD=1 is the default anyway). sort 310p 2026-08-18. [CANN 9.1.0, ascendc_kernel_cmake, build-infra]

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=n/a; op_class=all (kernel-local CMakeLists builds on the ascendc_kernel_cmake flow)`
`unverified_on: other CANN versions / SoC targets on the same cmake flow (the flake is a tool-flow property; re-observe before scoping wider)`

- **Severity**: LOW-MEDIUM — build-infra flake (no silent data risk), but it burns a debug iteration if misread as a source-level regression: the failing .o compiled fine and is unchanged since the last good build.
- **Symptom**: an incremental rebuild fails at the device-object merge/link step with `ld.lld` reporting "unknown file type" against an existing, unchanged object file.
- **Workaround**: always clean-rebuild (`rm -rf` the cmake build dir, or the acceptance scripts' `ASCENDC_CLEAN_BUILD=1`, which is the default on the benchmark-acceptance lane anyway). Do NOT attempt to repair the incremental path per-op.
- **Detection**: link-step "unknown file type" naming a .o that compiled clean in the SAME build invocation → tool-flow flake, not a code bug.
- **Evidence**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 3, build-infra class): incremental device-.o merge failed "unknown file type"; clean rebuild succeeded; kernel shipped 26/26 bit-exact.
- **Cross-ref**: (the standalone cmake flow this flake lives in). backend=ascendc.

**孪生卡 pb-178 合并的独有内容**（pb-178-cann-ascendc-cmake-incremental-device-object-mer.md 已删除）：

> - **Evidence (3rd op — 64B-sector granularity refinement, does NOT soften the DataCopy-only default)**: swi_glu_quant 310p (2026-09-02, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN, CPU-truth 35/35 PASS, det 5/5) — per-row fp32 scale values written as 4-byte scalar `SetValue` stores by CONCURRENT cores were silently LOST when the packed `[M]` rows shared a cache line. Shipped fixes: (a) give every row's scale its own 64B slot (stride-16 fp32) = one scalar writer per 64B sector → clean + deterministic; (b) route row widths < 64B to a single core. Granularity reconciliation with the group_norm_silu probe: that probe's per-core "exclusive 32B datablocks" sit two-per-64B-sector, i.e. two writers per cache line — consistent with the drop unit being the 64B sector, and with swi_glu's one-writer-per-64B-sector stores surviving. Treat any scalar-store-vs-DataCopy choice at 64B granularity as needing a 2nd op's repro before relying on it; the DataCopy-only default stands.

> - **Evidence (2nd op)**: instance_norm_grad 310p (2026-09-02, benchmark-acceptance lane, failures_ledger build iter 3 `instance_norm_grad/failures_ledger.md`): cwd-drift nested `build/build` artifact + stale .o from an interrupted parallel build → link-step "unknown file type"; absolute-path clean rebuild fixed (rc=0, `PyInit_instance_norm_grad_ext` exported). Same box / CANN family as sort.

> - **Evidence (3rd op — stale-device-object flavor, make-dependency gap)**: swi_glu_quant 310p (2026-09-02, benchmark-acceptance lane, failures_ledger build iter 6 `swi_glu_quant/failures_ledger.md`): a HOST-side pybind11.cpp-only edit did NOT invalidate the prebuilt `kernels_preprocess` device object → subsequent merge/link produced a stale build; `rm -rf` clean rebuild restored it. Distinct trigger from the ld.lld "unknown file type" merge flake but the same prescription: clean-rebuild, do NOT debug the incremental path per-op.
