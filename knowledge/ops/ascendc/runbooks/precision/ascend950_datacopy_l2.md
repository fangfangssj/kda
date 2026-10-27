---
type: "Runbook"
title: "Ascend950 DataCopy 写回 L2 驱逐丢对角线"
description: "Ascend950 上 fill_diagonal_v2 dense path 用 DataCopy 批量写回整个 tile 到 GM 时写入经多核共享 L2 cache，64 核并行写入合计约 8MB 超出 L2 容量，后 32 核 dirty cache line 在末尾 DataCacheCleanAndInvalid 前被驱逐未刷回 GM，对角线位置丢失；判别方法为 dump 比对定位对角线 out==input 且分界线落核 31/32 与 4MB L2 临界点一致。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Ascend950 DataCopy 写回 L2 驱逐丢对角线

## 适用场景

- **工程形态**：kernel 直调（ops-math conversion 目录构建 vendor，TTK kernel 模式评测）。
- **API 路径**：AscendC kernel API `DataCopy`（UB→GM，MTE3 pipeline）、`SetValue`（GlobalTensor，Scalar pipeline）、`DataCacheCleanAndInvalid<T, CacheLine>`。
- **验证平台与版本**：Ascend950（dav-2201，8 davinci devices），CANN 9.0.0，TTK `-d -c -b=release` 双千双万 tensorflow golden。
- **适用粒度**：`operator_family`（多核并行、CopyOut 用 DataCopy 批量写回整个 tile 且末尾依赖 DataCacheCleanAndInvalid 刷回的 inplace 写算子）。
- **不适用或未验证场景**：sparse path（已用 SetValue + DataCacheCleanAndInvalid，无此问题）；非 Ascend950 平台的 L2 cache 行为未验证，推断可迁移但未实测，不写入 frontmatter platforms。

## 判别方法

**触发**（tool_observed 失败）：dense path（tilingKey=1，wrap=true + totalLength>1000000 + step 阈值）48 个用例中 46 个精度失败，golden 匹配率 99.65-99.96%；sparse path（tilingKey=0）238/238 全部 100% golden 通过。`--pc=1` 串行重跑 2 个失败 dense 用例仍失败，排除并发抖动。首批全量 240/286 PASS，46 失败全部集中在 dense-path（fdv2_0239-0286）。

**探针**（可执行检查）：

1. 对失败用例 `(2000000, 8) float32` dump 输出，逐元素与 golden 比对，统计不匹配位置分布、是否落在对角线、`out==input` 比例。
2. 移除 dense path 末尾 `DataCacheCleanAndInvalid`，重跑同一用例，观察未填充数量变化。
3. 切换缓存清理策略（末尾一次 `ENTIRE_DATA_CACHE` / per-tile `ENTIRE_DATA_CACHE` / per-tile `SINGLE_CACHE_LINE` / `DcciDst::CACHELINE_OUT` / `SetFlag<HardEvent::MTE3_S>`+`WaitFlag`+`SyncAll`），重跑 48 dense 用例统计 PASS 数。

**观测**（tool_observed）：

- dump 比对：1907 个不匹配**全部在对角线位置**（1907/1907），`out==input`（位置未被填充而非被错填）；不匹配从 idx=999993（block 31 末尾）开始，block 0-30 全部正确，block 32-62 大量未填充——"前半核正确、后半核丢失"的缓存驱逐模式。
- 移除 `DataCacheCleanAndInvalid` 后未填充从 1907 升至 666404/666667，证明 `DataCopy` UB→GM 写入停留 L2 未刷回 GM，必须显式清理。
- 缓存策略对照：末尾一次 `ENTIRE_DATA_CACHE` 4/48 PASS、per-tile `ENTIRE_DATA_CACHE` 1/48、per-tile `SINGLE_CACHE_LINE` 0/48、`CACHELINE_OUT` 99.90% 仍失败、`SetFlag`+`WaitFlag`+`SyncAll` 后再清理 99.90% 仍失败——含 invalid 语义的策略均无法根治。

**判读**（根因解读）：dense path `CopyOut` 用 `DataCopy`（MTE3 pipeline，UB→GM）批量写回整个 tile（含未被修改的非对角线位置）。写入经多核共享 L2 cache：64 核并行每核写约 125KB，合计约 8MB 超出 L2 容量（反推约 4MB）。前 31 核 ×125KB≈3.86MB<4MB，dirty line 留在 cache 内由末尾 `DataCacheCleanAndInvalid` 刷回 GM；后 32 核（block 31-62）的 dirty line 在到达清理点前被前核驱逐，未刷回 GM，对角线写入丢失。`SetFlag<HardEvent::MTE3_S>`+`WaitFlag`+`SyncAll` 只保证 MTE3 pipeline 完成，不保证数据已落 GM，与观测一致。关键判据三联：不匹配全在对角线且 `out==input`、分界线落核 31/32 之间、与 4MB L2 临界点一致；invalid 语义策略无法根治容量型驱逐。

## 验证信号

来源：本次 fill_diagonal_v2 TTK 重跑会话（`-d -c -b=release`，tensorflow golden，双千双万精度）。

- dump 比对：dense path 失败用例 1907 个不匹配全在对角线位置、`out==input`、block 0-30 正确而 block 31-62 丢失（来源平台：Ascend950；证据等级: tool_observed）
- 缓存策略对照：移除 `DataCacheCleanAndInvalid` 后 666404/666667 未填充；`ENTIRE_DATA_CACHE`/`SINGLE_CACHE_LINE`/`CACHELINE_OUT`/per-tile flush 均 ≤4/48 PASS（来源平台：Ascend950；证据等级: tool_observed）
- 修复路线确认：CopyOut 改 `SetValue` 只写对角线 + 末尾一次 `DataCacheCleanAndInvalid` 后重跑全量 286/286 PASS，dense path 48/48、sparse path 238/238 三模式均 100% golden，`PRECISION_STATUS: PASS`（来源平台：Ascend950；证据等级: tool_observed）