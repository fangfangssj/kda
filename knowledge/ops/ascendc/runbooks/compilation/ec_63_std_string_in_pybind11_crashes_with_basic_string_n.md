---
type: Runbook
title: "`std::string` in pybind11 crashes with `basic_string null` SIGSEGV on V220 ARM64 (bisheng/GCC ABI mismatch)"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all (pybind11 host code)"
tags: [task.debugging, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-63-std-string-in-pybind11-crashes-with-basic-string-n.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3]
aliases: ["EC-63", "std::string"]
---

# `std::string` in pybind11 crashes with `basic_string null` SIGSEGV on V220 ARM64 (bisheng/GCC ABI mismatch)

> 原 OKF v1 卡号：**EC-63**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - pybind11 module loads successfully, but any call to the wrapped function crashes with basic_string null SIGSEGV (exit code 139). Same binary works on x86 host,

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all (pybind11 host code)`

- **Severity**: CRITICAL (kernel compiles, pybind11 loads, crashes at first call with SIGSEGV)
- **Status**: CONFIRMED 2026-05-22 28_Interpolate a3-ds
- **Symptom**: pybind11 module loads successfully, but any call to the wrapped function crashes with `basic_string null` SIGSEGV (exit code 139). Same binary works on x86 host, crashes on V220 ARM64.
- **Root cause**: GCC (host, x86) and bisheng (device, ARM64) have different std::string ABIs. The `std::string` parameter in pybind11 function signature is constructed by pybind11 from Python str, but the underlying memory layout differs between host and device.
- **Fix**: Replace `const std::string&` parameters with `const char*` or `py::str` in pybind11 wrappers. Convert to C string before passing to kernel launch.
- **Detection**: grep for `std::string` in kernel/pybind11.cpp. Any match → replace with C-string alternative.
- **Evidence**: 28_Interpolate a3-ds (2026-05-22, V220): `interpolate_forward(..., const std::string& mode_str, ...)` crashed with SIGSEGV 139 on A3 NPU0. Kernel compiled and built successfully.
- **Cross-ref**: OL-180 (CANN env init for .so loading) — both are pybind11-level V220-specific host issues.

<!-- 迁移自 porter kb/target/ascendc/（EC-63，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
