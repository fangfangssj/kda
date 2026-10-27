---
type: Runbook
title: "A deployed .so failing 'RuntimeError: Unsupported soc version: Ascend950PR 957b' while npu-smi shows Health=OK is a torch_npu / cpython BUILD mismatch — not a host/CANN/device fault"
description: "A .so built against cpython-3.12 (or a py312 torch_npu) REJECTS the 957b silicon bin; a *.cpython-312-*.so is the smoking gun. Rebuild against the known-good py311 + torch_npu 2.7.1.post5 cp311 pairing — ssh-verify npu-smi before any 'host-down' escalation."
tags: [task.debugging, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-244-unsupported-soc-version-build-mismatch.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-244"]
---

# A deployed .so failing 'RuntimeError: Unsupported soc version: Ascend950PR 957b' while npu-smi shows Health=OK is a torch_npu / cpython BUILD mismatch — not a host/CANN/device fault

> 原 OKF v1 卡号：**OL-244**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - A deployed kernel .so (or pybind extension) fails at import/run with 'RuntimeError: Unsupported soc version: Ascend950PR 957b' while npu-smi info independently shows the NPU Health=OK

## 现象 / 触发
`applies_to: soc=Ascend950PR_957b; cann=9.0.0 (set_env) / 9.1.T500 (build); op_class=all (build/env)`. `verified_on: host=npu_dev3 (203.0.113.171)`.

A deployed kernel `.so` (or pybind extension) fails at IMPORT/RUN with `RuntimeError: Unsupported soc version: Ascend950PR 957b` while `npu-smi info` independently shows the NPU Health=OK. It masquerades as a host / CANN / device regression. (`unverified_on: soc=Ascend910_V220 (A3) — the cp311-vs-cp312 mechanism may apply but the exact reject string is A5-specific`.)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
This is a **torch_npu / cpython build mismatch**, NOT a host / CANN / device regression. A `.so` built against cpython-3.12 (or a py312 torch_npu) REJECTS the 957b silicon bin. The fix is to rebuild against the known-good interpreter+torch_npu pairing, not to hunt the environment or escalate "host-down". P9 reflex still applies — but the discriminating probe here is trivially cheap (ssh + `npu-smi`), so RUN IT before concluding host-regression. A whitebox once ran an exhaustive env hunt and wrongly concluded host-regression; the host was healthy and the `.so` was simply py312.

Concrete anchor (npu_dev3, host 203.0.113.171): known-good env = `/root/miniconda3/envs/py311/bin/python` (python 3.11) + torch_npu 2.7.1.post5 cp311 (installed from `/data`) + `source cann-9.0.0/set_env.sh` + the lib64 path additions. Sanity: `torch.tensor([1.0]).npu()` -> NPU OK, soc reported `Ascend950PR_957b`, device_count=3. A correctly rebuilt extension is named `*.cpython-311-*.so`; a `*.cpython-312-*.so` is the smoking gun for this failure.

## 证据
- selective_scan_source_a5 (3) verify (2026-06-22, A5/Ascend950PR_957b): a py312-built `.so` produced a phantom "host-down" escalation; rebuilding under py311 (torch_npu 2.7.1.post5 cp311) loaded cleanly and ran. Whitebox-derived (the env hunt was the wrong path; npu-smi was healthy throughout).
- Predicted: any A5 deployment where the build interpreter drifts from the runtime interpreter; any "Unsupported soc version" at import on a card that npu-smi reports healthy — cross-check the `.so`'s `cpython-3XX` tag against the active venv first.
- Cross-ref: OL-189 (rotate-to-fresh-NPU — the OTHER thing that masquerades as a kernel/host fault; OL-244 is the build-tag variant); the A5 py311 + cann-9.0.0 set_env known-good recipe; P9 of ANTI_PRESSURE_PROTOCOLS.
