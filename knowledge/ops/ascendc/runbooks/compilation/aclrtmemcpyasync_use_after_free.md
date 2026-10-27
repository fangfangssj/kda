---
type: "Runbook"
title: "aclrtMemcpyAsync 局部源生命周期缺陷"
description: "直调算子把局部 std::vector 的 data 指针交给 aclrtMemcpyAsync 后立即返回，源码上无法保证 host 源存活到异步拷贝完成；该缺陷需独立修正，但现有 trace 未证明它是后续 507035 的唯一根因。"
tags: ["task.debugging", "topic.data_movement.data_copy", "topic.implementation.synchronization"]
status: stable
sources: []
created_at: "2026-08-18T05:59:22Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# aclrtMemcpyAsync 局部变量 use-after-free

## 现象

在 Ascend950 上跑 deepseek-v4 推理（把 `MoeInitRoutingGroupQuant` 等算子替换为 `npu_test::moe_init_routing_group_quant` 直调实现后），模型能成功跑完 prefill + warmup + 约 22 个 decode step，随后崩溃。异步堆栈指向 MTP model 的 `hc_post`（不准）；开 `ASCEND_LAUNCH_BLOCKING=1` 后精确定位到最后一次 kernel 打印是 `MoeInitRoutingGroupQuant`，错误码 `507035`（vector core exception）。报错原文片段：

```
NPU function error: device error type 3, error code is 507035
[Error]: The vector core execution is abnormal.
```

崩溃发生在多次成功调用之后，是典型的"偶发、与时序相关"症状——大多数 step 异步拷贝执行很快，安全；当 stream 被前序 kernel 占用时拷贝被推迟，读到已释放内存。

## 适用场景

- **工程形态**：ops-direct-invoke 直调算子（在 `*_torch.cpp` host 代码里手动 `aclrtMemcpyAsync` host→device）。
- **API 路径**：`aclrtMemcpyAsync(devPtr, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream)`；判别对照 API 为同步 `aclrtMemcpy`（去掉 stream 参数）。
- **验证过的平台与 CANN 版本**：Ascend950（deepseek-v4 `ci_950` 配置，CANN tools 路径 `/home/d00657570/Ascend/cann`）；a2/a3 未验证。
- **适用粒度**：`operator_family`——所有在 host 代码里对**局部变量/栈变量**做 `aclrtMemcpyAsync` 的直调算子通用。
- **路线优先级**：`diagnostic_only`——本卡确认 host 源生命周期违反异步调用约束；不把它等同于整起推理崩溃的闭环根因。
- **不适用/未验证场景**：a2/a3 平台未验证，推断可迁移（aclrtMemcpyAsync 语义跨 Ascend 平台一致）；纯 device→device 拷贝不涉及 host 生命周期，不在本卡范围。

## 判别方法

**触发条件**（tool_observed 失败症状）：直调算子替换进模型后，推理在 prefill + warmup + 约 22 个 decode step 后偶发崩溃，错误码 `507035`（vector core exception）。异步堆栈指向的 `hc_post` 不准，需要进一步夹逼定位。

**可执行探针**：
1. 设 `ASCEND_LAUNCH_BLOCKING=1`（同步下发、阻塞到每条 kernel 完成），重跑推理，观察崩溃前最后一次 kernel 打印落在哪个算子。
2. 若最后一次 kernel 为 `MoeInitRoutingGroupQuant` 等在 host 代码里手动做 host→device 拷贝的直调算子，打开该算子 `*_torch.cpp`，grep `aclrtMemcpyAsync`，逐个调用检查 host 源指针是否来自**函数内局部对象**（`std::vector`、栈 struct、局部 `at::Tensor` 副本）。
3. 对每个命中调用，确认该局部对象的生命周期是否在 `aclrtMemcpyAsync` 排入 stream 后、stream 排空前结束（典型信号：调用出现在函数末尾、对象在函数返回时析构）。
4. 可把命中的 async 拷贝临时改为同步 `aclrtMemcpy` 做对照；只有崩溃消失或有内存检查证据时，才能进一步把运行期故障归因于该生命周期缺陷。仅错误码变化不足以闭环。

**原始观测**（Ascend950，deepseek-v4 推理）：
- `ASCEND_LAUNCH_BLOCKING=1` 后崩溃前最后一次 kernel 打印为 `MoeInitRoutingGroupQuant`，错误码 `507035`（vector core exception）。
- 源码审查命中 `moe_init_routing_group_quant_torch.cpp` 函数末尾（约第 156-158 行）对局部 `Phase1Result p1` 的 `expandedRowIdx`（`std::vector<int32_t>`）与 `expertCount`（`std::vector<int64_t>`）调用 `aclrtMemcpyAsync(... p1.expandedRowIdx.data() ... npuStream)` 与 `aclrtMemcpyAsync(... p1.expertCount.data() ... npuStream)`，`p1` 在函数返回时析构。
- 把这两处 async→sync 重跑后，错误码从 `507035` 变为 `507015`，仍约 22 个 decode step 后崩溃。该结果只能说明执行路径发生变化，不能证明原 507035 已由这两处拷贝解释。

**机制边界**：`aclrtMemcpyAsync` 返回时并不保证拷贝完成，而 `p1` 在函数返回时析构，因此源码缺少“host 源存活到拷贝完成”的保证，这是独立且可确认的生命周期缺陷。该 trace 没有直接观测失败调用中 DMA 读取已释放内存，也没有在修正后消除模型崩溃；“device 数据污染后导致 507035”只能作为待验证假设。

## 预防

- 直调算子 host 代码里，凡 `aclrtMemcpyAsync` 的 host 源指针来自局部对象，都必须显式保证其生命周期覆盖异步拷贝完成；可采用同步拷贝或延长所有权并在完成后释放。
- Review checklist：搜 `aclrtMemcpyAsync` 的每个调用，确认 host 指针在 stream 排空前不会被析构/改写。

## 坏实践

- 误以为 `stream(true)`（函数开头同步 stream）能保护函数末尾的 async 拷贝——`stream(true)` 只保证开头时刻 stream 空闲，不保证后续排入的 async 拷贝在函数返回前执行。
- 误以为 `std::vector::data()` 返回的指针在 vector 析构后仍有效。

## 误诊

- 先怀疑 `KvCompressEpilog` 的 `getCurrentNPUStream` vs `getDefaultNPUStream` 流选择（本 trace 之前已修，但未解决崩溃）；再怀疑 `static at::Tensor tilingTensor` 跨流竞态（确实存在，是另一独立坑）。这两处修复后错误位置移动，才暴露 `MoeInitRoutingGroupQuant` 的 use-after-free。

## 验证信号

- `ASCEND_LAUNCH_BLOCKING=1` 后崩溃前最后一次 kernel 打印为 `MoeInitRoutingGroupQuant`，错误码 `507035`（vector core exception），prefill+warmup+约 22 decode step 后偶发（来源平台：950；证据等级：tool_observed）
- 两处 async→sync 重跑后错误码从 `507035` 变为 `507015`，但崩溃未消除（来源平台：950；证据等级：tool_observed；不构成根因闭环）
- 缺陷存在于源码：`moe_init_routing_group_quant_torch.cpp` 函数末尾约第 156-158 行确为 `aclrtMemcpyAsync(... p1.expandedRowIdx.data() ... npuStream)`（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 本卡结论（aclrtMemcpyAsync 不应用于生命周期短于 stream 排空时间的 host 数据）跨 Ascend 平台语义一致，a2/a3 推断可迁移但**未验证，推断可迁移**。
- 仅适用于 host→device 拷贝；device→device 拷贝的源数据生命周期由 stream 顺序保证，不在本卡范围。
- 若 host 数据生命周期被显式延长（堆分配 + 显式 stream sync 后释放），async 可用，本卡的"一律同步"建议不适用。