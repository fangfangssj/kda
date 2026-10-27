---
type: Runbook
title: "PadTiling name conflict with CANN built-in"
description: "PadTiling name conflict with CANN built-in"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["error: reference to 'PadTiling' is ambiguous", "PadTiling"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-19 — EC-19: PadTiling name conflict with CANN built-in

> 关联：同一 EC-19 主题已由既有卡 [ec_19_padtiling_name_conflict_with_cann_built_in.md](ec_19_padtiling_name_conflict_with_cann_built_in.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：paradigm=ascendc（卡内 applies_to 仅声明范式；非 Pad 特有，任何复用内置 `<Name>Tiling` 名的外层 struct 均可能触发，见正文 Scope 行）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（Pad V2 首次构建 2026-04-09；rms_norm Ascend310P3 2026-08-09 独立复现——构建宿主事实，非 device-runtime 声明）。

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `error: reference to 'PadTiling' is ambiguous`
- **Root cause**: CANN `kernel_tiling.h` defines `PadTiling` in `AscendC::tiling` namespace and imports it via `using`. Custom struct with same name conflicts.
- **Scope**: this is not Pad-specific. `kernel_tiling.h` defines a family in
  `AscendC::tiling` and imports it into scope; any outer struct that reuses a
  built-in `<Name>Tiling` can become ambiguous.
- **Fix**: use an op-qualified name such as `<OpName>KernelTiling` or
  `<OpName>OpTiling` rather than a bare built-in-family name.
- **Evidence**: Pad V2 first build (2026-04-09); rms_norm on Ascend310P3
  (2026-08-09) independently hit the same error for `RmsNormTiling`, whose
  built-in definition is anchored at 内部路径（不公开）kernel_tiling.h:565, and compiled
  after renaming the custom type to `RmsNormKernelTiling`
  (内部路径（不公开）workspace/rms_norm/kernel/rms_norm_kernel.h:15-19). This is a
  build-host fact, not a device-runtime claim.
