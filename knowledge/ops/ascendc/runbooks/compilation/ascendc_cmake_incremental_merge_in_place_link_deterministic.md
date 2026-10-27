---
type: Runbook
title: "CANN ascendc cmake incremental device-object merge is deterministic, not flaky — the merge links each kernel object in place"
description: "CANN ascendc cmake incremental device-object merge is deterministic, not flaky — the merge links each kernel object in place"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p", "950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-125 — PB-125: A5Ops additive extension

> 关联：同一 PB-125 主题已由既有卡 [pb_125_cann_ascendc_cmake_incremental_device_object_mer.md](pb_125_cann_ascendc_cmake_incremental_device_object_mer.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）与 Ascend950DT_9582；CANN 9.1.0；op_class=all（ascendc_kernel_cmake 流程上的 kernel 本地 CMakeLists 构建）。
- 验证来源：soc=Ascend950DT_9582（10_ConvTranspose2d，2026-09-11）— 症状与 REL->EXEC 机制均已实测。

**PB-125** — **CANN ascendc cmake incremental device-object merge is DETERMINISTIC, not flaky** — the preprocess merge links each kernel object **IN PLACE**, so after one successful build the `.o` is a linked `EXEC` ELF rather than a relocatable `REL`, and the next incremental build fails `ld.lld: unknown file type` against it **every time**. Always clean-rebuild (acceptance lane's ASCENDC_CLEAN_BUILD=1 is the default anyway). [CANN 9.1.0, ascendc_kernel_cmake, build-infra]

- Former A5Ops ID: `PB-125`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200) AND Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=all (kernel-local CMakeLists builds on the ascendc_kernel_cmake flow)`
`verified_on: soc=Ascend950DT_9582 (10_ConvTranspose2d, 2026-09-11) — symptom AND the REL->EXEC mechanism`
`symptom_correlated_only: soc=Ascend310P3 (sort, 2026-08-18) — the same link-step symptom was observed, but the in-place-link mechanism was NOT measured there; do not claim it for 310P3 without 310P3 mechanism evidence`
`unverified_on: other CANN versions on the same cmake flow`

- **CORRECTION (2026-09-15, 10_ConvTranspose2d on Ascend950DT_9582)**: this entry previously called the failure a "flake" and a "tool-flow property", on single-SoC symptom evidence, and asked for re-observation before scoping wider. That re-observation happened on a second SoC and the causal claim did not survive: the failure is **deterministic and mechanistic**. The reading matters operationally — "flake" invites a retry, and a retry here **always** fails; only a clean build works.
- **Mechanism (MEASURED, not inferred)**: the merge step runs `ld.lld -m aicorelinux -Ttext=0 <path>/X.cpp.o -o <the same X.cpp.o path>` for both `aiv_obj` and `aic_obj` — output path identical to input. After one build, `file` / `readelf -h` on that `.o` report `ELF 64-bit LSB EXECUTABLE` / `Type: EXEC`, where a freshly compiled object reports `REL`. The second `cmake --build` then dies at `merge_aiv_obj_text` / `merge_aic_obj_text` because the input is no longer relocatable.
- **Severity**: LOW-MEDIUM — no silent data risk, but it burns a debug iteration whenever it is misread as a source-level regression: the failing `.o` compiled fine and is unchanged since the last good build.
- **Symptom**: an incremental rebuild fails at the device-object merge/link step with `ld.lld` reporting "unknown file type" against an existing, unchanged object file.
- **Workaround**: always clean-rebuild (`rm -rf` the cmake build dir, or the acceptance scripts' `ASCENDC_CLEAN_BUILD=1`, which is the default on the benchmark-acceptance lane anyway). Do NOT attempt to repair the incremental path per-op, and do NOT retry the incremental build expecting a different outcome.
- **Detection**: link-step "unknown file type" naming a `.o` that compiled clean in the SAME build invocation. Confirm with `readelf -h` on the named object: `Type: EXEC` means it was already linked in place, so the build directory is spent.
- **Operational upside (same measurement)**: `--clean` builds of the same source at the same in-container path are byte-reproducible — a clean → incremental-fail → clean chain produced the identical DSO sha256 `c48d5899d2301cf8…` on both clean legs. DSO-identity receipts depend on exactly this property.
- **Evidence**: Ascend310P3 **symptom only** — sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 3): incremental device-`.o` merge failed "unknown file type"; clean rebuild succeeded; kernel shipped 26/26 bit-exact. Ascend950DT_9582 **symptom AND mechanism**, retained in this exact from `main@20da0bff3baf0152f22ea47b469c64580ecf9bfb` / !1115: `docs/evidence/kb_op310_20260911/objfile_type_evidence.txt` (`file`/`readelf -h`, REL vs EXEC), `docs/evidence/kb_op310_20260911/build_B_incremental.log:20-23` (the in-place link command and both failing merge targets) and `:31-32` (the driver commands), `docs/evidence/kb_op310_20260911/build_A_clean.log` + `docs/evidence/kb_op310_20260911/build_C_clean_again.log` + `docs/evidence/kb_op310_20260911/dso_shas.txt` (both clean legs rc=0, identical DSO sha).
- **Cross-ref**: CAND-BENCH-STANDALONE-KERNEL-PACKAGING (the standalone cmake flow this lives in). backend=ascendc.
