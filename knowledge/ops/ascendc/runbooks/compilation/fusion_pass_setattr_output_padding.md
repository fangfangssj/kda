---
type: "Runbook"
title: "fusion pass SetAttr output_padding 非根因"
description: "为 Conv3DBackpropFilterV2 新增 output_padding proto 属性后图通路报 EZ0009 groups 不匹配；曾在 fusion pass 补 SetAttr(output_padding) 试图修复，但该修复不解 groups 报错——真实根因是 fusion pass transposed=true 时未交换 input/gradOutput，与 output_padding 传递无关，banned。"
tags: ["task.debugging", "topic.compilation", "topic.implementation"]
status: stable
sources: []
created_at: "2026-08-18T05:58:44Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# fusion pass SetAttr output_padding 非根因

## 现象

在 Ascend950（arch35、CANN 9.1.0）上为 `Conv3DBackpropFilterV2` 新增 `output_padding` proto 属性后，图通路（fusion pass → V2 算子 → tiling 层）运行时报 `EZ0009: x_channel / filter_channel != groups`。曾假设根因是 fusion pass（`conv3d_backprop_filter_to_v2_fusion_pass.cpp`）创建 V2 节点时未显式 `SetAttr("output_padding", ...)`，导致 `output_padding` 不进入 runtime attrs，`GetAttrNum()-1` 误读 `useHf32` 破坏 attrs 缓存、连带 `GROUPS_INDEX` 读到错误值。据此在 fusion pass 中补 `SetAttr("output_padding", ...)`，但该修复未解决 groups 报错——groups 不匹配的真实根因是 fusion pass 在 transposed=true 时未交换 input/gradOutput（姊妹诊断已 tool_observed 确认），与本卡的 output_padding 传递无关。

## 适用场景

- **工程形态**：图通路（fusion pass → V2 算子 → tiling 层），Ascend950 上编译 + UT。
- **API 路径**：`Conv3DBackpropFilter` → fusion pass → `Conv3DBackpropFilterV2` → `SetOutputPaddingAttr` / `SetGroupsAttrs`。
- **验证过的平台与 CANN 版本**：Ascend950（arch35），CANN 9.1.0。
- **适用粒度**：`operator_family`——所有在 `Conv3DBackpropFilterV2` proto 中新增属性且走 fusion pass 的场景。
- **路线优先级**：**banned**——不得把"在 fusion pass 中补 `SetAttr(output_padding)`"当作 EZ0009 groups 报错的修复；它只修正 output_padding 进入 runtime attrs 的传递，与 groups 不匹配无关。
- **明确不适用/未验证场景**：aclnn API 通路不受此问题影响（output_padding 经 API 层正确传递）；a2/a3 未验证，不得据本卡推断可迁移。

## 失败路线

**banned 路线：在 fusion pass `Replacement` 中为 V2 节点补 `SetAttr("output_padding", convBpAttr.outputPadding)` 以图消除 EZ0009 groups 报错。**

- **尝试**：在 `conv3d_backprop_filter_to_v2_fusion_pass.cpp` 创建 V2 节点后增加 `SetAttr("output_padding", ...)`；配套在 `conv_backprop_fusion_base_pass.{h,cpp}` 的 `ConvBackpropAttrs` / `GetNodeAttrs` 中读取 `output_padding`（容错，不存在则默认全零）。fusion pass 主文件的 edit before/after 从 trace 内 tool_call（oldString/newString）逐字恢复：

  before（fusion pass 未设置 output_padding）：

  ```cpp
      conv3dBackpropFilterV2Node->SetAttr("_op_impl_mode_enum", convBpAttr.opImplModeEnum);
  ```

  after（fusion pass 显式设置 output_padding）：

  ```cpp
      conv3dBackpropFilterV2Node->SetAttr("_op_impl_mode_enum", convBpAttr.opImplModeEnum);
      conv3dBackpropFilterV2Node->SetAttr("output_padding", convBpAttr.outputPadding);
  ```

- **客观失败**：该修复的闭合证据（编译/UT 通过、图通路 runtime 复跑）未在 trace 中作为 tool_result 捕获（被 markdown_transcript adapter 截断为 `_see below_`），无 tool_observed 信号确认 EZ0009 被消除；且 EZ0009 groups 报错的真实根因（fusion pass transposed=true 时未交换 input/gradOutput）不受此修复触及（来源平台：950；证据等级：tool_observed）。
- **边界**：`SetAttr("output_padding", ...)` 只让框架把 `output_padding` 加入 runtime attrs、使 `GetAttrNum()` 与 PR 代码 `OUTPUT_PADDING_INDEX=8` 假设一致；它不解 groups 不匹配。把此修复当作 groups 报错修复是误诊，banned。
- **收敛**：groups 报错的真正修复见 transposed 交换诊断（aclnn API 层 `CheckConvShapePlus` / `CalculateConv3DTransposeBackward` 存在该交换、fusion pass 缺失），不在本卡范围。

## 验证信号

本段只收本 trace 内 tool_observed 实测，agent 推断的机制假设不进此段。

- 图通路运行时捕获 `EZ0009: x_channel / filter_channel != groups` 报错（来源平台：950；证据等级：tool_observed）
- fusion pass 主文件 `conv3d_backprop_filter_to_v2_fusion_pass.cpp` 中 `SetAttr("output_padding", ...)` 的 edit before/after（oldString/newString）从 trace tool_call 逐字恢复（来源平台：950；证据等级：tool_observed）

## 根因

**本卡不主张机制层根因。** 曾假设的 attrs 缓存破坏链（`SetAttr 缺失 → output_padding 不入 runtime attrs → GetAttrNum()-1 偏移 1 位 → 误读 useHf32(bool) 为 ContinuousVector* → 非法指针 → attrs 缓存破坏 → GROUPS_INDEX 读错 → EZ0009`）属 agent 推断，未在 trace 内以 tool_result 验证（grep `PADDING_INDEX` / `GROUPS_INDEX` / `OUTPUT_PADDING_INDEX` 的输出未作为 tool_result 捕获，runtime attrs 布局机制未实测）。该假设被姊妹诊断（transposed 交换缺失）证伪为 groups 报错的主因，故本卡以失败路线收敛、不发布为修复。
