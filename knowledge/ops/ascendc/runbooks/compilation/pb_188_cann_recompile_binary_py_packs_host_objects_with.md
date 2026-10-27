---
type: Runbook
title: "CANN `recompile_binary.py` packs host objects with `glob(\"**/*.o\", recursive=True)` which does NOT traverse dot-directories — a build task dir whose path contains a dot-component makes the installed host object glob-invisible → `TooFewObj` at `lib/libkernels.a`"
description: "构建目录路径含点开头的隐藏段（如 `.lanes/lane3`）时 Python glob 不下钻，`TooFewObj` 误报——对象文件其实在。"
tags: [task.debugging, topic.compilation, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-188", recompile_binary, "TooFewObj"]
---

# CANN `recompile_binary.py` packs host objects with `glob("**/*.o", recursive=True)` which does NOT traverse dot-directories — a build task dir whose path contains a dot-component makes the installed host object glob-invisible → `TooFewObj` at `lib/libkernels.a`

<!-- applies_to_backend: all -->

`applies_to: soc=any (CANN host tooling, SoC-independent); cann=9.0.0; bisheng=n/a; op_class=all (any build task dir layout handed to CANN's packer)`
`verified_on: 5_MatmulTransB 310p kw-1 iter-2 (2026-09-07): task dir under .lanes/lane3/build_task → TooFewObj; same sources under dot-free lane_builds/lane3_build_task → clean pack`
`unverified_on: other CANN versions' packer implementation (glob semantics are Python-level and likely long-standing; not version-swept)`

- **Severity**: MEDIUM — build-time failure whose message names neither the path shape nor the glob semantics, so it reads as a missing-object/link error and invites debugging the wrong layer.
- **Symptom**: `TooFewObj` at `lib/libkernels.a` when the build task directory path contains a dot-component (e.g. `.lanes/lane3/build_task`), while the object file demonstrably exists on disk.
- **Mechanism**: Python `glob("**/*.o", recursive=True)` does not descend into hidden (dot) directories — a dot path COMPONENT hides the entire subtree beneath it from CANN's host-object packer (内部路径（不公开）recompile_binary.py:69 — CANN-install script, external to this repo; line number as observed in the 2026-09-07 session, unauditable from this repo).
- **Mitigation**: keep build task dirs dot-component-free. Landed: `deploy_to_npu_lane.sh:223-224`（a5_ops 路径，未迁移） uses `lane_builds/lane${LANE}_build_task` (comment in the script records the constraint).
- **Evidence**: 5_MatmulTransB 310p (2026-09-07) `5_MatmulTransB/failures_ledger.md` build-failures iter-2 row; the move from `.lanes/` to `lane_builds/` resolved the pack with no other change.
- **Other instances (predicted)**: any tooling that stages builds under dot-dirs (`.cache/`, `.tmp/`, `.lanes/`) and hands the path to CANN's packer; any glob-based artifact collection over trees that may contain dot-dirs.
- **Cross-ref**: [`OL-408`](../precision/ol_408_the_cpu_truth_reference_must_replicate_the_kerne.md) (the lane-isolation fix whose directory choice this constrains), PB-90 (the vendored build-driver family — this one is CANN-side, not vendored). backend=all.

## 适用场景

- **工程形态**：构建目录路径含点开头的隐藏段（如 `.lanes/lane3`）时 Python glob 不下钻。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
