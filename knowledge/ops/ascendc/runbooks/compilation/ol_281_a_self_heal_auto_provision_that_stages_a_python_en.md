---
type: Runbook
title: "a self-heal / auto-provision that stages a Python entrypoint MUST provision its FULL dependency closure (the imported package + relative-path deps), not just the entrypoint file — else fresh-container `ModuleNotFoundError` + empty-stderr phantom compile failure"
description: "Provenance: caught by an independent run's clean-CCD user-view cannbot v3.17.0 e2e (2026-07-24) — a truly-fresh A5 container (no pre-staged build baseline). Root was in a5_ops (not port-specific); fix"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-281-a-self-heal-auto-provision-that-stages-a-python-en.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-281", "ModuleNotFoundError"]
---

# a self-heal / auto-provision that stages a Python entrypoint MUST provision its FULL dependency closure (the imported package + relative-path deps), not just the entrypoint file — else fresh-container `ModuleNotFoundError` + empty-stderr phantom compile failure

> 原 OKF v1 卡号：**OL-281**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - on a genuinely fresh container (no baseline pre-staged by setup_a3_isolated_container.sh), the build dies with ModuleNotFoundError: No module named 'build_capab

> Provenance: caught by an independent run's clean-CCD user-view cannbot v3.17.0 e2e (2026-07-24) — a truly-fresh A5 container (no pre-staged build baseline). Root was in a5_ops (not port-specific); fixed in `deploy_to_npu.sh` #241 (`ba2603e31`).
>
> **Two-agent cross-validation (2026-07-24):** independently reproduced by scan (selective_scan SIMD work) within the hour — a DIFFERENT path + DIFFERENT container, SAME root cause. Two independent hits on distinct workloads confirm this is a **general harness defect** (every fresh container hits it), not an e2e-specific artifact — which is exactly why the fix belongs at the a5_ops source (single SoT, #241) so every agent + customer benefits, rather than being patched per-consumer. (an independent run's cannbot port KB mirrors this as a pointer to OL-281, not a duplicate — canonical entry authoritative, downstream = pointer.)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR (a5) + Ascend910 (a3); cann=9.x; any deploy/self-heal that stages a build entrypoint into a fresh container`
`verified_on: deploy_to_npu.sh DEBT-197 self-heal — build_ascendc.py-alone → ModuleNotFoundError: build_capabilities on fresh container; fixed to provision the full trio + package-dir guard (#241, 5 self-heal tests + import proof)`

**Symptom**: on a genuinely fresh container (no baseline pre-staged by `setup_a3_isolated_container.sh`), the build dies with `ModuleNotFoundError: No module named 'build_capabilities'` surfaced as an **empty-stderr phantom compile failure** (no diagnostics) — because the DEBT-197 self-heal copied only `build_ascendc.py` into `$BUILD_ROOT/utils/`, but that entrypoint does `from build_capabilities import ...` (inserts its own dir into `sys.path`) and needs `cann_stubs/` (DEBT-110 include). Both deps ARE shipped in `patches/` — the self-heal just never laid them alongside.

**Principle**: a self-heal / auto-provision that claims "turnkey, zero manual provisioning" must provision the entrypoint's **entire dependency closure** — the imported package(s) + any relative-path sibling deps it resolves — not just the entrypoint file. A partial provision that satisfies "the file exists" but not "the file can import its deps" is *worse* than no self-heal: it yields a diagnostic-free phantom failure. Two guards: (a) provision the full closure, idempotent + per-piece; (b) after provisioning, test the imported **package DIRECTORY** exists (not just the entrypoint file) and emit a CLEAR diagnostic naming the missing module — never an empty-stderr failure.

**Discovery method (reusable)**: only a **true-fresh-container** user-view e2e exercises a self-heal's completeness — a pre-staged / warm container hides the gap because the baseline is already present. Anti-cheat corollary: the finalize gate correctly REJECTED the kernel worker's on-site "complete the deploy script" edit as `infra_baseline_paper_over` (P96/P9) — a harness-baseline gap is fixed at the harness source, never papered over by a worker editing tracked infra.

**Cross-ref**: the "Fix Harness for Next Customer, Not Patch Single Archive" rule (CLAUDE.md) — root was in a5_ops `deploy_to_npu.sh` (single SoT), fixed there so every fresh-container customer benefits + the cannbot port re-syncs it rather than patching only the port. Anchors: `deploy_to_npu.sh` DEBT-197 self-heal (local + remote modes), `patches/{build_ascendc.py, build_capabilities/, cann_stubs/}`. backend=ascendc.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-281（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
