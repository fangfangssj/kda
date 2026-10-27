---
type: "Runbook"
title: "RunOpApi 后缺 synchronize 致 .to 越序错"
description: "cummin 算子 op_plugin 在 OpCommand::RunOpApi 后、idx32.to(kInt64) 前缺 synchronize，导致 .to 在 kernel 完成前执行而产生错误输出；判别要点是 RunOpApi 后对同一输出张量的 torch 操作前须显式 sync。"
tags: ["task.debugging", "topic.implementation.synchronization"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# RunOpApi 后缺 synchronize 致 .to 越序错

## 现象

在 Ascend 950（NpuArch dav-3510）上用 ops-direct-invoke-flash 工作流实现 cummin 算子。原始 op_plugin（`csrc/ops/cummin/op_plugin/cummin_plugin.cpp`）的 `cummin_npu` 流程为：对非末维输入先 `transpose(dim, rank-1).contiguous()`，再用 `at_npu::native::OpCommand::RunOpApi("Cummin", acl_call)` 启动 kernel，随后 `c10_npu::getCurrentNPUStream().synchronize()`，再 `idx32Work.to(torch::kInt64)` 与 `transpose(rank-1, dim).contiguous()` 回写。

为降开销，agent 在一次重写 plugin 时**移除了 `RunOpApi` 与 `.to(kInt64)` 之间的 `synchronize()`**，期望让 `.to` 与 transpose-back 全异步。重建后 sanity check（小规模 NPU 调用比对）立即出错：`dim=-1` 的 float32 与 bfloat16 case、`dim=0` 的 float16 与 float32 case 结果失配——`.to(kInt64)` 与 transpose-back 在 kernel 尚未完成时就跑到了同一输出张量上，产生越序读写。

## 判别方法

### 触发

`OpCommand::RunOpApi(<name>, <acl_call>)` 启动 kernel 后、对同一输出张量做 torch 侧操作（`.to` / `.contiguous` / `transpose`）前**缺 `c10_npu::getCurrentNPUStream().synchronize()`** → sanity check 出现按 dtype/dim 分裂的错误输出（dim=-1 float32/bf16、dim=0 fp16/fp32 失配）。

### 探针

1. 检查 op_plugin 中 `RunOpApi` 与后续对同一输出张量的 torch 操作之间是否有显式 `synchronize()`。
2. 复现：移除该 `synchronize()`，重建并跑 sanity check；若出现按 dtype/dim 分裂的错误输出（`.to`/transpose 越序读写 kernel 尚在写入的张量），即确诊。
3. 反向验证：补回 `synchronize()`，sanity check 应恢复正确。

修复 edit（before/after verbatim 取自 trace edit 事件）：

before（buggy：RunOpApi 后直接 .to，无 sync）：
```cpp
    at_npu::native::OpCommand::RunOpApi("Cummin", acl_call);

    auto indicesWork = idx32Work.to(torch::kInt64);
```

after（补回 synchronize）：
```cpp
    at_npu::native::OpCommand::RunOpApi("Cummin", acl_call);
    c10_npu::getCurrentNPUStream().synchronize();

    auto indicesWork = idx32Work.to(torch::kInt64);
```

### 观测

- 移除 `RunOpApi` 与 `.to(kInt64)` 之间的 `synchronize()` 后，sanity check 出现错误输出：`dim=-1` 的 float32 与 bfloat16 case、`dim=0` 的 float16 与 float32 case 结果失配。

### 判别

`OpCommand::RunOpApi("Cummin", acl_call)` 通过当前 NPU stream 启动 kernel，但后续 `idx32Work.to(torch::kInt64)`、`transpose(rank-1, dim).contiguous()` 等 torch 操作可能不在同一 stream 上排队，或 RunOpApi 内部使用独立 stream 管理。没有显式 `synchronize()` 时，这些 torch 操作不会等待 kernel 完成，直接读写 kernel 尚在写入的 `idx32Work`/`valuesWork`，产生越序读写与错误输出。这不是精度逻辑错误，而是 stream 排序错误——同一输出张量被 kernel 写入端与 host 侧 cast/transpose 端并发访问。

## 适用场景

- 工程形态：op_plugin 侧 kernel 启动 + 后续 torch 张量操作（`.to` / `.contiguous` / `transpose`）混用。
- API 路径：`at_npu::native::OpCommand::RunOpApi(<name>, <acl_call>)` + `c10_npu::getCurrentNPUStream()`。
- 验证过的平台与 CANN 版本：Ascend 950 / NpuArch dav-3510（本 trace 实测）。
- 适用粒度：`operator_family`（凡 op_plugin 用 RunOpApi 启动 kernel、随后对同一输出张量做 torch 侧 dtype/layout 操作的算子均适用）。
- 路线优先级：`recommended_fastpath`（RunOpApi 后、对同一张量的 torch 操作前显式 synchronize 是正确性 fastpath）。
- 不适用或未验证场景：kernel 启动后**不**对输出张量做任何 host 侧 torch 操作（纯 NPU 内链路即返回）时未必需要此 sync，本 trace 未验证该场景；a2/a3 平台未验证。

## 验证信号

- 移除 `RunOpApi` 与 `.to(kInt64)` 之间的 `synchronize()` 后，sanity check 出现错误输出：`dim=-1` 的 float32 与 bfloat16 case、`dim=0` 的 float16 与 float32 case 结果失配（来源平台：950；证据等级：tool_observed）
- 在原位补回 `c10_npu::getCurrentNPUStream().synchronize()` 的 edit（before/after verbatim 取自 trace edit 事件）（来源平台：950；证据等级：artifact_verified）

## 误诊

agent 一度认为移除 sync 仅影响 profiler 对 NPU kernel 的检测（`no_npu_kernel_detected`），试图通过去 sync 让 profiler 看到更多事件来提分；实际去 sync 直接破坏正确性，与 profiler 检测无关（agent_reported，非实测结论，已在此纠正）。

## 预防

- 在 op_plugin 中，凡 `RunOpApi` 后紧跟对同一输出张量的 torch 侧操作（`.to` / `.contiguous` / `transpose` / `view`），默认在二者间显式 `c10_npu::getCurrentNPUStream().synchronize()`。
- 若要去 sync，须先证明 kernel 启动与后续 torch 操作在同一 stream 上严格排队（本 trace 未证实，贸然去 sync 即破坏正确性）。
- 去 sync 的动机应来自实测性能瓶颈定位，而非基于 profiler 检测的猜测。

## 坏实践

- 基于"profiler 检测不到 NPU kernel"的猜测就去掉 `synchronize()`，未先验证 stream 排序，直接破坏正确性。
- 把 profiler 检测问题与 stream 排序问题混为一谈。

## 失效边界

- 本 trace 仅在 Ascend 950 / dav-3510 实测；a2、a3 平台未验证，推断 RunOpApi 的 stream 语义在 torch_npu 跨平台一致，故**未验证，推断可迁移**，但 stream 管理细节可能随 CANN/torch_npu 版本变化。
- 若 kernel 启动后不对输出张量做任何 host 侧 torch 操作，本 sync 是否必需未在本 trace 验证。