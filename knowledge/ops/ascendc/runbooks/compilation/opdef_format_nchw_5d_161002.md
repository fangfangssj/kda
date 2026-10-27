---
type: "Runbook"
title: "OpDef FORMAT_NCHW 致 5D 报 161002"
description: "同时支持 4D/5D 的算子在 OpDef 注册 FORMAT_NCHW 会让 5D 张量在 GetWorkspaceSize 阶段报 ACLNN error 161002；NCHW 只对 4D 有效，该路线 banned，应保持 FORMAT_ND 并由 tiling 据 shape 推断布局。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.data_movement.layout_transform"]
status: stable
sources: []
created_at: "2026-08-18T07:25:08Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# OpDef FORMAT_NCHW 致 5D 报 161002

## 现象

INTrainingReduceGrad 算子同时支持 4D（`[N,C,H,W]`）与 5D（`[N,D,C,H,W]`）输入。为了"在 tiling 层校验并路由 NCHW/NHWC 格式"，一度在 `op_host/in_training_reduce_grad_def.cpp` 把 OpDef 的输入格式从 `FORMAT_ND` 改为 `FORMAT_NCHW`，并在 tiling 写了 `ValidateFormat()` 校验格式字段、给 `TilingData` 加了 `format` 字段，同时把 tiling UT 与 aclnn ST 的测试张量也改成 `FORMAT_NCHW`。

build 通过后跑 aclnn ST：rank4（4D）用例全过，但 rank8（5D）用例全部失败，逐行报：

```
Test FAILED/R8-D1 (GetWorkspaceSize error 161002)
Test FAILED/R8-V0 (GetWorkspaceSize error 161002)
Test FAILED/R8-V1 (GetWorkspaceSize error 161002)
...
==== ACLNN ST rank8-broadcast: 15/15 cases PASSED ====
```

末行 "15/15 cases PASSED" 是 TTK 对"用例被正确拒绝也算 PASS"的计数，**不代表 5D 用例真正跑通**——前面逐行 `Test FAILED/R8-* (GetWorkspaceSize error 161002)` 才是真实运行结果。error 161002 是 `GetWorkspaceSize` 阶段格式不支持。

## 适用场景

- **工程形态**：算子仓开发，`op_host/*_def.cpp` 的 OpDef 注册 + `op_host/arch35/*_tiling_arch35.cpp` 的 tiling。
- **API 路径**：`REG_OP(...).Input(...).Format({ge::FORMAT_..., ge::FORMAT_...})`；`gert::StorageFormat`；`aclnnXxxGetWorkspaceSize`。
- **验证过的平台与 CANN 版本**：Ascend950 + CANN 9.0.0。
- **适用粒度**：`platform_wide`——所有"同时支持 4D 与 5D（或更高维）且物理格式走 ND"的算子。
- **路线优先级**：**banned**——在 OpDef 注册 `FORMAT_NCHW` 给 5D 算子是错的；正确路线是 `FORMAT_ND` + tiling 据 shape 推断布局。
- **不适用/未验证场景**：仅支持 4D 的算子可用 NCHW（未在本 trace 验证）；NHWC/NCDHW 在 OpDef 的注册行为未在本 trace 验证（仅推理）。

## 失败路线

### 尝试路线（banned）：OpDef 注册 `FORMAT_NCHW` + tiling `ValidateFormat()`

在 OpDef 把每路输入注册为 `Format({ge::FORMAT_NCHW, ge::FORMAT_NCHW})`，并在 tiling 加 `ValidateFormat()` 校验格式字段、给 `TilingData` 加 `format` 成员。测试侧也把 tiling UT 三个测试文件的 `gert::StorageFormat` 从 `FORMAT_ND` 改成 `FORMAT_NCHW`（这一处 edit 可恢复：`gert::StorageFormat fmt(ge::FORMAT_NCHW, ge::FORMAT_NCHW, gert::ExpandDimsType());`）。

**运行结果**：rank4（4D）用例全过；rank8（5D）用例全部 `GetWorkspaceSize error 161002`。把测试也改成 NCHW 后 5D 仍报 161002——根因不在测试侧格式，而在 NCHW 物理格式本身对 5D 不存在。

**根因（解释）**：CANN 的 `ACL_FORMAT_NCHW` 仅对 4D 张量有效；5D 张量（`[N,D,C,H,W]`）没有 NCHW 这一物理格式，OpDef 强制 NCHW 会让 5D 输入在 `GetWorkspaceSize` 阶段被拒。NCHW/NHWC 是 ND 的**逻辑语义特例**——物理存储都是 ND，只是维度语义解释不同；format 字段在 tiling 层不可靠，布局应由 shape 模式推断。用户点破："def 中应该还是接收 ND 格式的输入，NCHW 和 NHWC 应该只是 ND 的一种特殊形式"，遂整体回退。

### 边界（正确路线）：OpDef 保持 `FORMAT_ND` + tiling 据 shape 推断

回退后 OpDef 每路输入注册 `FORMAT_ND`（终版从修复后 file_read 逐字恢复，`op_host/in_training_reduce_grad_def.cpp`）：

```cpp
            .Input("dy",     {"variance", "mean", "res_gamma", "res_beta", "gamma", "x"})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
```

tiling 不再持有 `format_` 成员、`TilingData` 不再含 `format` 字段、`ValidateFormat()` 整体删除；改由 `ValidateShape` 校验 variance 最后两维为 1、`ComputeNumElements` 取最后两维乘积作为 `num`，对 4D/5D 统一成立（终版从 file_read 逐字恢复，`op_host/arch35/in_training_reduce_grad_tiling_arch35.cpp`）：

```cpp
    int64_t ComputeNumElements(int64_t dy_rank)
    {
        auto& dy = raw_input_shapes_[0];
        // num = H * W (last two spatial dims), same for 4D and 5D
        return dy[dy_rank - 2] * dy[dy_rank - 1];
    }
```

关键点：`num` 用 `dy[dy_rank-2] * dy[dy_rank-1]`（最后两维 H*W），不依赖 format 字段；4D 与 5D 同一公式。机制推导——variance 的空间维必为 1（batchnorm/groupnorm 的统计量按空间归约后只剩通道维）：本算子 `ValidateShape` 要求 variance **最后两维为 1**，即默认 NCHW 排布；NHWC 排布（中间两维为 1）会被该检查拒绝，若需支持 NHWC 需扩展 shape 模式判别（本 trace 未实现）。

## 验证信号

- 修复前 rank8(5D) 用例逐行 `Test FAILED/R8-* (GetWorkspaceSize error 161002)`（来源平台：950；证据等级：tool_observed）。
- 把测试侧也改成 FORMAT_NCHW 后 rank8 仍报 161002，证明根因不在测试侧格式（来源平台：950；证据等级：tool_observed）。
- 改回 FORMAT_ND + 删除 `ValidateFormat`/`format_` 后，`./build.sh INTrainingReduceGrad_package` 输出 `SUCCESS`（来源平台：950；证据等级：tool_observed）。
- tiling UT `15/15 cases PASSED`（含 5d_fp32 / 5d_fp16 / 4d_large / 4d_hw1 / empty / 各 reject case）（来源平台：950；证据等级：tool_observed）。
- OpDef 终版 `Format({ge::FORMAT_ND, ge::FORMAT_ND})` 与 `ValidateShape`/`ComputeNumElements` 终版逐字从 file_read 恢复（来源平台：950；证据等级：artifact_verified）。