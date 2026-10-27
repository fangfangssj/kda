---
type: "Runbook"
title: "TopK BF16 tie 索引顺序与 CPU 不一致"
description: "AscendC TopK 经 Host Cast float32 后，BF16 tie 值的索引顺序与 torch.topk CPU 不同；逐元素 torch.equal 比较索引会误报，须按值/索引集合判别为 tie 顺序现象而非精度 bug。"
tags: ["task.testing", "topic.compute.ordering", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
aliases: ["tie_ordering"]
---

# TopK BF16 tie 索引顺序与 CPU 不一致

## 现象

在 cann-bench `top_k` 算子评测中，`pytest tests/top_k/test_top_k.py` 对 `shape=(3,20), k=5, dim=-1, largest=True, dtype=torch.bfloat16` 用例报 `AssertionError: TopK indices mismatch`。NPU 实测输出索引 `tensor([[ 3, 11, 12,  0,  4], [18, 14, 11, 0, 15], [ 0, 13, 16,  7, 14]])`，CPU `torch.topk` 期望 `tensor([[ 3, 12, 11,  0,  4], [18, 14, 11, 0, 15], [ 0, 13, 16,  7, 14]])`。差异仅出现在第 0 行的第 1、2 个索引（`11` 与 `12` 互换），其余完全一致；对应的 top-k 值集合相同。本算子在 plugin 层把 BF16 先 Cast 为 float32 再调 `AscendC::TopK<float>`、结果 Cast 回 BF16。

## 适用场景

- **工程形态**：评测/测试（`pytest tests/top_k/test_top_k.py`，golden 用 CPU `torch.topk`）。
- **API 路径**：`AscendC::TopK<float, false, true, false, TOPK_NORMAL>`（BF16 经 Host Cast float32）；CPU 侧 `torch.topk(x, k, dim, largest)` 作 golden。
- **验证过的平台与 CANN 版本**：Ascend 910B3（DAV_2201），CANN 9.0.0。仅 a2 实测。
- **适用粒度**：`operator_family`（所有经 Host Cast float32 后调 TopK 高阶 API 的排序类算子，在 tie 值存在时均可能命中）。
- **不适用或未验证场景**：FP32/FP16 原生路径的 tie 行为未在本 trace 触发（用例未覆盖 tie），不主张"FP32 无此问题"；DAV_3510 等支持 RADIX_SELECT 的架构未验证。

## 判别方法

判定一条 TopK 索引 mismatch 是否属于本 tie 顺序现象（而非真精度 bug）：

1. **触发条件（trigger）**：dtype 为 BF16（或经 Host Cast float32 的整型），且输入存在 tie 值（多个元素数值相等）；失败信号为 pytest `AssertionError` 且仅索引逐元素不等。
2. **可执行探针（probe）**：分别取 NPU 与 golden 的 top-k **值**集合和**索引**集合，按行比较：
   - `torch.equal(npu_values.sort().values, golden_values.sort().values)` 是否为 True（值集合一致）；
   - `torch.equal(torch.sort(npu_indices).values, torch.sort(golden_indices).values)` 是否为 True（索引集合一致，仅顺序不同）。
3. **原始观测（observation）**：本 trace 失败用例满足"值集合一致 + 索引集合一致 + 仅 tie 组内顺序不同"——第 0 行 `11/12` 互换，其余两行完全一致；对应 top-k 值相同。
4. **解释（interpretation）**：BF16→float32 Cast 后，原本因 BF16 精度而相等的值在 float32 下仍比较为相等（tie 未被 Cast 打破）；TopK 高阶 API 对 tie 值的索引顺序未定义，其内部 Sort+MrgSort+Extract 的稳定顺序与 CPU `torch.topk` 实现不同，导致同一 tie 组内索引排列差异。这不是 kernel 计算错误，是 tie 顺序未定义语义的体现。若值集合已不等，则不是本现象，应另查真精度 bug。

## 验证信号

本次信号来自 cann-bench top_k 一次开发会话（CANN 9.0.0 / Ascend 910B3）。

- BF16 用例失败：NPU 索引 `[3,11,12,0,4]` vs 期望 `[3,12,11,0,4]`，仅第 0 行 `11/12` 互换，对应 top-k 值集合一致。（来源平台：a2；证据等级：tool_observed）
- 同次 pytest 其余用例通过（FP32/FP16/INT8/UINT8/INT32/INT64、largest true/false、dim=1/-2、k=1/3/5/10/2048）。（来源平台：a2；证据等级：tool_observed）

## 预防

- TopK 族算子的评测脚本，索引比较一律用集合/排序后比较，禁止裸 `torch.equal(indices)`；在 tie 值存在的随机数据下逐元素 equal 必然间歇性误报。
- 生成测试数据时刻意构造 tie 值（如重复元素）以暴露顺序差异，但断言须容忍 tie 组内索引顺序。

## 失效边界

- 仅 a2（DAV_2201）实测；BF16 tie 顺序差异是否在 DAV_3510/RADIX_SELECT 路线上同样存在未验证（推断可迁移，但 RADIX_SELECT 内部排序算法不同，顺序可能又不同）。
- FP32/FP16 原生路径（不经 Cast）的 tie 顺序未在本 trace 触发，不主张其无此问题。
- INT 类型经 Host Cast float32 同样可能命中，但本 trace 未给出 INT tie 的具体 mismatch 输出，未单独验证。