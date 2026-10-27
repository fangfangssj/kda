---
type: "Runbook"
title: "TilingData 追加数据超额致 INVALID_TILING"
description: "在 TilingData struct 之后追加标量向量数据并 SetDataSize(sizeof(struct)+vecBytes) 时，若总大小超过框架默认 capacity（arch35/CANN 9.0.0 实测 16384 字节），框架在 host→device 传输前校验 DataSize<=capacity 失败，报 INVALID_TILING（capacity 16384 < required N）；判别依据为错误消息显式给 capacity 与 required 两侧数值，且 SetDataSize 不改 capacity。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-18T07:36:56Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# TilingData 追加数据超额致 INVALID_TILING

## 适用场景

- **工程形态**：ops-registry-invoke 自定义算子仓，host Tiling 用 `GetRawTilingData()` / `GetTilingData<T>()` 写 TilingData，并在 struct 之后追加额外数据（如全精度标量向量）；kernel 用 `GET_TILING_DATA_WITH_STRUCT` + 手动偏移读取尾区。
- **API 路径**：`GetTilingData<T>()`（内部 `SetDataSize(sizeof(T))`）/ `GetRawTilingData()` → `SetDataSize(sizeof(struct)+vecBytes)` → 框架校验 `DataSize <= capacity` → `INVALID_TILING`。
- **验证过的平台与 CANN 版本**：Ascend950PR_9579（arch35/dav_c310），CANN 9.0.0。
- **适用粒度**：`platform_wide`——所有走 TilingData struct + 尾区附加数据的算子，均受同一 capacity 校验约束。
- **不适用或未验证场景**：只写 struct 本身（不超默认 capacity）不受影响；a2/a3 同框架同机制，推断同一 capacity 约束但未实测，不同 SoC 版本 capacity 可能不同，须以目标平台 `GetCapacity()` 实测为准。

## 判别方法

**触发**（tool_observed 失败）：在 fused_bias_leaky_relu arch35 kernel（Ascend950PR，CANN 9.0.0）调试标量传输精度时，为绕开 `Duplicate`/`Muls` 立即数截断，把全精度标量向量（`[slope, ...]` 与 `[scale, ...]`，各 `ubFactor` 个 float）追加到 TilingData struct 之后的尾区，kernel 用 `DataCopy` 从尾区搬入 UB。host 调 `GetRawTilingData()` 拿原始 buffer，写入 `ubFactor × 2 × sizeof(float)` 字节并 `SetDataSize(sizeof(struct)+vecBytes)`。单元素测试用例（shape=(1,)）下 `ubFactor` 被设得很大（8192），struct + 8192×2×4 = 65576 字节，运行即报（verbatim，取自会话内运行 tool_result）：

```text
INVALID_TILING
TilingData capacity 16384 < required 65576
```

**探针**（可执行检查）：

1. 在 host tiling 写尾区前，读 `GetRawTilingData()->GetCapacity()` 取目标平台默认 capacity（arch35/CANN 9.0.0 实测返回 16384）。
2. 按 `required = sizeof(TilingDataStruct) + appendedBytes` 反推上限：`maxAppendedElems = (capacity - sizeof(struct)) / (elemSize × vecCount)`；把追加元素数 / `ubFactor` cap 到该上限以内。
3. 重跑单用例，观察 `INVALID_TILING` 与 `capacity < required` 行是否消失（capacity 与 required 两侧数值同时出现在错误消息中，是本判据的判别特征）。

**观测**（tool_observed）：

- 运行报 `INVALID_TILING`，消息显式给出 `capacity 16384 < required 65576`——capacity 侧（框架分配的 buffer 上限）与 required 侧（`SetDataSize` 设定的有效数据大小）两侧数值都在，证明校验发生在 `DataSize > capacity` 时。
- `ubFactor=8192` 时 required=65576；把追加规模降到 2048×2×4+struct=16416 时仍超（16416>16384），与“capacity 是固定 16KB 上界、与 DataSize 设置无关”一致。

**判读**（根因解读）：TilingData buffer 有一个框架在 host 侧分配的默认 `capacity`（arch35/CANN 9.0.0 实测 16384 字节 = 16KB）。`GetTilingData<T>()` 调用 `tiling_data->SetDataSize(sizeof(T))` 只设置“有效数据大小”为 struct 大小，**不改变 capacity**；`GetCapacity()` 返回的是 buffer 最大容量，默认只够放 struct。当在 struct 之后追加数据并 `SetDataSize(sizeof(struct)+vecBytes)`，框架在 host→device 传输前校验 `DataSize <= capacity`，超出即 `INVALID_TILING`。关键判据二联：错误消息同时给出 `capacity` 与 `required` 两侧数值（区别于“尾区数据未传到 device”的 DataSize 锁死问题）；`SetDataSize` 只改 DataSize 不改 capacity，故减小 required 至 capacity 以内是判别后的可行方向。

## 验证信号

来源：一次 fused_bias_leaky_relu arch35 tiling 调试（Ascend950PR_9579，arch35/dav_c310，CANN 9.0.0）。

- 追加标量向量（`ubFactor=8192`，struct+8192×2×4=65576 字节）并 `SetDataSize(sizeof(struct)+vecBytes)` 后，运行报 `INVALID_TILING`：`TilingData capacity 16384 < required 65576`；降到 2048×2×4+struct=16416 仍超 capacity 仍 `INVALID_TILING`（错误消息同时给 capacity 与 required 两侧数值，证明校验为 `DataSize>capacity`）（来源平台：950；证据等级：tool_observed）

## 预防

在 TilingData struct 之后追加数据前，先 `GetRawTilingData()->GetCapacity()` 查目标平台容量，按 `SetDataSize(sizeof(struct)+vecBytes) <= capacity` 反推最大元素数；标量向量优先考虑用小常量向量（如 8 个 float = 32 字节，满足 DataCopy 最小对齐）反复复用，而非按 `ubFactor` 等长分配——单元素用例下 `ubFactor` 会被设得很大（如 8192），瞬间超 16KB。

## 误诊

最初以为是 `GetTilingData<T>()` 的 `SetDataSize(sizeof(T))` 把 DataSize 锁死在 struct 大小导致尾区数据不传到 device——这确实也是一个相关问题（须改用 `GetRawTilingData()` 手动设 DataSize），但**不是** `INVALID_TILING` 的根因；`INVALID_TILING` 是 capacity 校验（`DataSize > capacity`），与 DataSize 设置是否锁死是两层。判据：`INVALID_TILING` 消息显式带 `capacity < required`，而“DataSize 锁死”表现为尾区读到零值、无该校验错。