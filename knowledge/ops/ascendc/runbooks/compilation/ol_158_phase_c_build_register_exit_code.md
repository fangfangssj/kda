---
type: Runbook
title: "arch22->arch35 Phase C: build+register activation criteria and build.sh exit-code interpretation"
description: "Ops with no torch CPU/AICPU fallback verify 0/N FAIL (EZ1001/Get regInfo failed) until the A5 kernel is built+registered; a non-zero build.sh --pkg exit can be plugin-only, so check the .o/.json."
tags: [task.debugging, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-158-phase-c-build-register-exit-code.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-158"]
---

# arch22->arch35 Phase C: build+register activation criteria and build.sh exit-code interpretation

> 原 OKF v1 卡号：**OL-158**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - port_a3_to_a5 op with no torch CPU/AICPU fallback verifies 0/N FAIL with `EZ1001 / Get regInfo failed / does not support opType` because no ascend950 kernel is registered
> - `build.sh --pkg --ops=<op> --soc=ascend950` exits non-zero after the ascendc kernel targets already produced their .o/.json

## 现象 / 触发

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=port_a3_to_a5`. Source:
`workspace/top_k_top_p_sample/knowledge_update.md` (kw-1, 2026-05-14). Refines the ctc_loss_v3 PoC
framing ("Phase C out of scope" in ROADMAP §1.5 / REPORT.md §6.5).

- An op with **no torch CPU/AICPU fallback** verifies **0/N FAIL** with
  `EZ1001 / Get regInfo failed / does not support opType` — because verify dispatches strictly into
  aclnn and no ascend950 kernel has been built+registered.
- A **non-zero exit from `build.sh --pkg --ops=<op> --soc=ascend950`** does NOT necessarily mean the
  ascendc kernel failed.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

**(1) When to attempt Phase C (build+register)**: the legacy "Phase C build+register is out of PoC
scope" framing was driven by ops whose torch reference has an **AICPU fallback** — verify reports
PASS without the A5 kernel ever being loaded, so build+register adds no signal. For ops with **no
torch CPU/AICPU fallback**, build+register is NOT optional: without a registered ascend950 kernel,
verify ALWAYS reports 0/N FAIL. Attempt Phase C when BOTH:
1. an on-host `ops-nn-port/` (or equivalent staged build tree) exists at a discoverable path
   (typical `/home/<user>/ops-nn-port/`), AND
2. the op has no torch fallback — i.e. `torch_npu.<op>` / `torch.ops.npu.<op>` dispatches strictly
   into aclnn with no CPU implementation.
Add this signal-check to the kw_brief Phase C decision. Counter-condition (skip Phase C, accept PoC
scope): the op's reference HAS an AICPU fallback (`F.ctc_loss`, some `torch.nn.functional`
reductions) — verify PASSes even without the new kernel installed.

**(2) Interpreting `build.sh --pkg` exit code**: the plugin-generation targets (proto/tf, proto/onnx,
`optf_plugin_nn_obj`, `op_nn_onnx_plugin_obj`) compile generated cpp that calls `git rev-parse` on
cann-cmake metadata; in a flat checkout (`ops-nn-port/` without the parent cann-cmake repo) those
targets fail with `fatal: not a git repository` AFTER the ascendc kernel targets have already
produced their `.o`/`.json`. Verify which side failed by inspecting the artifacts before re-running.

## 解决配方

```bash
# After build.sh exits (whether 0 or non-zero):
find build -name "*.o" -path "*ascend950*/<op_snake>/*"
find build -name "binary_info_config.json" -path "*ascend950*"
```
If both the per-dtype `.o`+`.json` and the per-soc `binary_info_config.json` exist, the kernel build
succeeded — proceed to registry install (see CAND-A3A5-15) regardless of the `build.sh` exit code.
If they are missing, the kernel side failed and the actual error is upstream of the plugin failures
in the log. Faster triage: use the `--opkernel` target when `--pkg` fails on the framework side.
