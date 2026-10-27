---
type: Runbook
title: "Backward-mode op-gen uses the pybind `.so` paradigm end-to-end — the `op_host/` build contract (PB-33) and the Phase-O5 workspace resync BOTH have backward-specific carve-outs the worker must handle itself"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; kernel_type=backward"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-272-backward-mode-op-gen-uses-the-pybind-so-paradigm-e.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-272"]
---

# Backward-mode op-gen uses the pybind `.so` paradigm end-to-end — the `op_host/` build contract (PB-33) and the Phase-O5 workspace resync BOTH have backward-specific carve-outs the worker must handle itself

> 原 OKF v1 卡号：**OL-272**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; kernel_type=backward

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; kernel_type=backward`
`verified_on: soc=Ascend950PR; cann=9.0.0`

### Principle

A backward (gradient) op generated via the backward plugin does **not** follow the generic `op_host/` mirror paradigm that the standard kw brief and PB-33 assume. Two independent divergences bite the worker if unhandled:

1. **Build paradigm — standalone pybind `.so`, not `op_host`.** The backward plugin overrides `check_op_host_completeness → None`, so the PB-33 `op_host/` completeness gate does NOT apply. Build `kernel/pybind11.cpp` + `<op>_kernels.cpp` + `<op>_kernel.h` → `_<op>_ext`. Do not author the PR4778-style `op_host/*_def.cpp` / `*_tiling.{cpp,h}` mirror — it is dead weight in backward mode. (Confirmed via `plugins/backward/__init__.py:85-109`.)

2. **Precision-closure self-deploy.** Phase O5's `_resync_workspace_to_container` `push_files` list does **not** include `verify_<op>.py` NOR the precision helpers (`backward_cdv_collect` / `backward_cdv_grade` / `precision_cannbot_adapter` / `cannbench_grader`). The resync rsync only syncs `kernel/**` plus top-level `*.py` / `*.json` / `*.pt`. So the worker's OWN deploy must ship the precision closure. Robust pattern: copy the precision closure into `kernel/precision/` (which IS synced under `kernel/**`), and have `verify_<op>.py` insert BOTH `HERE` and `HERE/kernel/precision` on `sys.path`.

3. **Minor, same context — parse profiler CSV with stdlib `csv`, not pandas.** The container py311 env lacks pandas; parse `operator_details.csv` with `csv.DictReader`, summing the `Device Self Duration(us)` column.

### Evidence

- gelu_spec_grad kw-1 (2026-07-02, A5 Ascend950PR_957b, CANN 9.0.0, backward elementwise): built via `kernel/pybind11.cpp` + `<op>_kernels.cpp` + `<op>_kernel.h` → `_<op>_ext` (no op_host); precision closure copied into `kernel/precision/` + `verify_<op>.py` sys.path'd both dirs → 6/6 PASS graded on-NPU end-to-end. stdlib-csv parse used (no pandas in container).

### Other instances (predicted)

Every backward-plugin op (`ascendc-backward-gen`): elementwise grads, norm-family backward, reduction backward. Any op-gen mode whose Phase-O5 resync excludes verify-side helpers from `push_files` — the "ship your own verify closure under a synced subtree" pattern generalizes to any deploy path that rsyncs only `kernel/**`. Cross-ref PB-33 (the op_host contract this carves out of), OL-113 (a different deploy-path stale-source hazard).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-272（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
