---
type: Runbook
title: "msprof / torch_npu.profiler ANALYZE stage fails `Operation not permitted` (EPERM) in the `.171` npu_dev3 container despite Privileged=true → use NPU `aclrtEvent` elapsed_time device-time as the fallback"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.precision, topic.data_movement, topic.performance.analysis, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-51-msprof-torch-npu-profiler-analyze-stage-fails-oper.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["PB-51", "aclrtEvent"]
---

# msprof / torch_npu.profiler ANALYZE stage fails `Operation not permitted` (EPERM) in the `.171` npu_dev3 container despite Privileged=true → use NPU `aclrtEvent` elapsed_time device-time as the fallback

> 原 OKF v1 卡号：**PB-51**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`

In the `.171` (203.0.113.171) `npu_dev3` container, both `torch_npu.profiler` and the `msprof` CLI collect raw profiling data but the **analyze/export stage fails with `Operation not permitted` (EPERM)** — so per-launch isolated `kernel_details.csv` device-time is UNAVAILABLE there, even though the container reports `Privileged=true` (the EPERM is from a profiling-syscall/securityfs restriction the privileged flag doesn't cover).

- **Workaround**: measure device-time via NPU events — `aclrtEvent` / `torch.npu.Event(enable_timing=True)` `start/end` + `elapsed_time` around the kernel launch (this is device-side timing, NOT host wall-clock). Validate it once against any host where msprof analyze DOES work (e.g. a committed msprof number) — observed agreement ~7%. The Event device-time is a valid same-host A/B basis (the EPERM is an export-tooling limit, not a timing-accuracy limit).
- **Distinct from the .141 policy**: `.141` FORBIDS msprof by team policy (different reason); `.171` ALLOWS msprof but its analyze stage EPERMs in npu_dev3. So for `.171` perf A/B, use NPU-Event device-time.
- **Status**: OPEN (container/securityfs profiling-syscall restriction).
- **Evidence**: selective_scan bwd perf (2026-06-30, .171 npu_dev3, CANN 9.0.0/9.1.0, PR #71): msprof analyze EPERM on both tools; NPU-Event device-time used + cross-validated ~7% vs the committed msprof figure.
- **Cross-reference**: [[feedback_report_perf_only_from_probe_device_time]] (the device-time-not-wall discipline — NPU-Event IS device-time, satisfies it), OL-245 (the perf A/B this surfaced under).

<!-- 迁移自 porter kb/target/ascendc/（PB-51，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
