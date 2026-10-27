---
type: Runbook
title: "PadTiling name conflict with CANN built-in"
description: "自定义 tiling 结构体与 AscendC::tiling 内置 <Name>Tiling 族撞名导致编译 ambiguous——用 <OpName>KernelTiling 式命名规避；310P3 rms_norm 二次实证"
tags: [task.debugging, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-19-padtiling-name-conflict-with-cann-built-in.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-23T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-19"]
---

# PadTiling name conflict with CANN built-in

> 原 OKF v1 卡号：**EC-19**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - error: reference to 'PadTiling' is ambiguous

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 310P 系列（310P3 编译宿主实测）；机制为编译期名称冲突，与设备无关。

- **Error pattern**: `error: reference to 'PadTiling' is ambiguous`
- **Root cause**: CANN `kernel_tiling.h` defines `PadTiling` in `AscendC::tiling` namespace and imports it via `using`. Custom struct with same name conflicts.
- **Scope**: not Pad-specific — `kernel_tiling.h` defines a family in `AscendC::tiling` and imports it into scope; any outer struct reusing a built-in `<Name>Tiling` can become ambiguous.
- **Fix**: use an op-qualified name such as `<OpName>KernelTiling` or `<OpName>OpTiling` rather than a bare built-in-family name.
- **Evidence**: Pad V2 first build (2026-04-09); rms_norm on Ascend310P3 (2026-08-09) independently hit the same error for `RmsNormTiling` (built-in definition anchored at 内部路径（不公开）kernel_tiling.h:565), compiled after renaming the custom type to `RmsNormKernelTiling`. This is a build-host fact, not a device-runtime claim.

<!-- 迁移自 porter kb/target/ascendc/（EC-19，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

### rms_norm 片段（重命名后的 tiling 结构体）

```cpp
// Section A: rms_norm_kernel.h
constexpr uint32_t BUFFER_NUM = 1;
// Tiling struct — packed by pybind11.cpp, read by kernel via byte-by-byte GM read (OL-77).
// NOTE: named RmsNormKernelTiling to avoid conflict with AscendC::tiling::RmsNormTiling
// built into the AscendC headers (kernel_tiling.h:565).
struct RmsNormKernelTiling {
    uint32_t numRow;
    uint32_t numCol;
    float epsilon;
};
```
