---
type: "Runbook"
title: "InferShape 返回 identity 致空 kernel 假通过"
description: "op_host InferShape 用 *output=*input 占位时，aclnn 框架对 identity 用例走旁路 copy，空 kernel 也有用例通过，掩盖 kernel 未输出的缺陷。"
tags: ["task.debugging", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-17T03:21:10Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# InferShape 返回 identity 致空 kernel 假通过

## 现象

在 ascend950 上开发 BatchToSpace 算子（ACLNN / ops-registry-invoke）。调试 kernel"无可见输出"问题时，把 kernel 的 `Process` 改成完全空函数（不执行任何 DataCopyPad），ST 仍有用例通过：

- identity InferShape（stub `*context->GetOutputShape(0) = *context->GetInputShape(0)`）+ 空 kernel：非 Branch-1 identity 用例通过。
- 正确 InferShape（按 BatchToSpace 语义算输出 shape）+ 空 kernel：非 Branch-1 用例仍通过、Branch-1 用例 0 通过。
- 正确 InferShape + 真 kernel（但 tiling 数据被截断）：仅 Branch-1 int64 identity 用例通过。

这些"通过"并非 kernel 写对，而是框架在 `InferShape` 判定输入输出同 shape 时对 identity 用例走了旁路 copy（或输出缓冲初值恰好等于 golden），产生假通过，掩盖了 kernel 实质未输出的缺陷。

## 适用场景

- 工程形态：ACLNN 算子仓开发，op_host 侧实现 `InferShape`（`gert::InferShapeContext`）。
- API 路径：`static graphStatus InferShape(gert::InferShapeContext* context)` → `*context->GetOutputShape(0) = *context->GetInputShape(0)`（identity stub，常见占位写法）。
- 验证过的平台与 CANN 版本：Ascend950 / CANN 9.1.0。
- 适用粒度：`operator_family`——任何输出 shape ≠ 输入 shape 但开发期用 identity InferShape stub 占位的 ACLNN 算子（BatchToSpace/SpaceToBatch/Reshape/Transpose 等形状变换类）。
- 路线优先级：`recommended_fastpath`——`InferShape` 必须按算子真实语义计算输出 shape，不可长期保留 `*output=*input` stub；调试"空 kernel 却有通过"时第一探针即查 InferShape 是否返回 identity。
- 不适用/未验证场景：输出 shape 恰好等于输入 shape 的算子（真 identity 算子）不适用；单测 TilingUT 不经 aclnn 框架、不触发此旁路；其它平台未验证（推断可迁移，见 `## 失效边界`）。

## 判别方法

### trigger

kernel 实质未输出（空 Process 或 tiling 截断致 DMA 不执行）却仍有非零 ST 用例通过，且通过的多为 identity 用例（block_shape=[1,1] crops=0 等输入即输出的场景）。

### probe

1. 把 kernel `Process` 改成空函数（不调用任何 DataCopyPad），重 build 跑全量 ST：

```bash
cd /workspace/<工程>/operators && ./build.sh BatchToSpace_package 2>&1 | tail -3
cd BatchToSpace_package/tests/aclnn && ./run.sh 2>&1 | tail -5
```

2. 同时检查 op_host InferShape：

```bash
grep -n "InferShape\|GetOutputShape\|GetInputShape" \
  /workspace/<工程>/operators/BatchToSpace_package/op_host/batch_to_space_host.cpp
```

若 InferShape 形如 `*context->GetOutputShape(0) = *context->GetInputShape(0);`（identity）且空 kernel 仍有用例通过，即命中本规则。

3. 把 InferShape 改成按算子语义计算正确输出 shape，重 build 跑 ST：空 kernel 通过数应下降（假通过消失），真 kernel 通过数不变（直到 tiling 截断被单独修复）。

### observation

本次 trace 内（tool_observed，由实际 `./run.sh` 全量 ST 执行产出）：

- identity InferShape + 空 kernel：非 Branch-1 identity 用例通过（来源平台：950；证据等级: tool_observed）。
- 正确 InferShape + 空 kernel：非 Branch-1 用例仍通过、Branch-1 用例 0 通过——证明非 Branch-1 通过不来自 kernel，而是框架旁路（来源平台：950；证据等级: tool_observed）。
- 正确 InferShape + 真 kernel（tiling 仍截断）：仅 Branch-1 int64 identity 用例通过（来自 kernel 的少量正确输出）（来源平台：950；证据等级: tool_observed）。

### interpretation

aclnn 框架在 `InferShape` 判定输入输出 shape 一致时，对 identity 用例（输入即输出）可能走框架层 copy 旁路，不依赖算子 kernel 的输出；或输出 GM 初值/框架拷贝恰好等于 golden。这使空 kernel 也"通过"用例，造成"kernel 已工作"的假象，掩盖真实的 tiling 截断。此解释系由"空 kernel + identity InferShape → 非零通过""修正 InferShape 后空 kernel 仍非零通过（说明非 InferShape 直接触发，而是 identity 用例本身被框架旁路）"的对照推断，属 interpretation 而非 tool_observed 的机制层确认。

## 系统信息

- 工具链：Ascend C / CANN 9.1.0，ops-registry-invoke 流程。
- 目标平台：Ascend950。
- 相关文件：`operators/BatchToSpace_package/op_host/batch_to_space_host.cpp`（InferShape 修复点）、`operators/BatchToSpace_package/tests/aclnn/test_aclnn.cpp`（ST 框架，调用 `aclnnBatchToSpace`）。

## 设计决策

1. 为何 identity stub 危险：开发期常用 `*output=*input` 作 InferShape 占位（编译能过、shape 不报错），但对形状变换算子它长期错误，且会触发框架 identity 旁路产生假通过，推迟暴露 kernel 真正缺陷。正确做法是按算子语义计算（BatchToSpace 输出 `[N*bh, outH, outW, D*bw]` 等）。
2. 本卡修复 InferShape 后通过数下降（移除假通过），并未让 Branch-1 通过数上升——它净化了诊断基线，使"少量通过来自 kernel 真实输出"可被确认，从而把注意力引向 tiling 截断真因。

## 验证信号

- identity InferShape + 空 kernel → 非零用例通过（来源平台：950；证据等级: tool_observed）
- 正确 InferShape + 空 kernel → 非零用例仍通过（非 Branch-1）、0 通过（Branch-1）（来源平台：950；证据等级: tool_observed）

## 性能影响

无性能维度信号。

## 预防

- 形状变换类算子的 `InferShape` 必须按算子语义计算输出 shape，禁止长期保留 `*output=*input` stub；提交前 grep `GetOutputShape.*GetInputShape` 命中即告警。
- TDD 红绿基线判读：kernel 未实现时 ST 预期 0 通过；若空 kernel 出现非零通过，先查 InferShape identity 旁路，再查 tiling 截断。

## 失效边界

- Ascend950 / CANN 9.1.0 已实测。其它平台/版本未验证——aclnn 框架 identity 旁路是否在所有 CANN 版本一致未确认，未验证，推断可迁移。
- 真 identity 算子（输出 shape 恒等于输入）不适用本规则。