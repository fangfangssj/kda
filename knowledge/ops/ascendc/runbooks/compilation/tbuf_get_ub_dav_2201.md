---
type: "Runbook"
title: "独立 TBuf.Get 返回重叠 UB 地址 dav-2201"
description: "dav-2201 上 Compute 里多个独立 TBuf.Get<float>() 会返回互相重叠的 UB 地址使张量互相覆盖，1M 元素 fp32 测试 is_pass=False matched_ratio=0.10；根因不在 UB 总量不足而在独立 TBuf 分配器该 size 段的地址分配不可靠，判别靠 correctness 测试 + 地址重叠探针。"
tags: ["task.debugging", "topic.implementation.memory_management"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# 独立 TBuf.Get 返回重叠 UB 地址 dav-2201

## 适用场景

- 工程形态：kernel-direct-invoke 工程或 aclnn op_kernel，`Compute` 里需同时持有 >6 个 float LocalTensor 且彼此生命周期重叠。
- API 路径：`TPipe::InitBuffer` + `TBuf::Get<T>` + `TBuf::GetWithOffset<T>(size, byteOffset)`。
- 验证过的平台与 CANN 版本：dav-2201（Atlas A2 族，a2），CANN 9.0.0 实测。a3 共享 dav-2201 架构但未实测。
- 适用粒度：`platform_wide`（dav-2201 上独立 TBuf 的地址分配行为）。
- 不适用或未验证场景：buffer 数量 ≤4 且每个都很小时独立 TBuf 可能不重叠（未验证边界）；非 dav-2201 架构未验证。

## 判别方法

触发（tool_observed）：在 dav-2201（CANN 9.0.0）上的 Spence 算子 kernel-direct-invoke 工程里，`Compute` 声明 8 个独立 `TBuf`（`zBuf/wBuf/pBuf/tmpBuf/lnwBuf/ln1wBuf/resBuf/sBuf`），各 `pipe_->InitBuffer(buf, TILE_LENGTH * sizeof(float))`（`TILE_LENGTH=2048`，每 buffer 8KB）后用 `TBuf::Get<float>()` 取 tensor。1M 元素 fp32 测试 `is_pass=False matched_ratio=0.10`。

触发态 verbatim 代码（取自 trace seq 837 Edit 的 OLD，独立 TBuf 取址）：

```
        auto zLocal = zBuf.Get<float>();
        auto wLocal = wBuf.Get<float>();
        auto pLocal = pBuf.Get<float>();
        auto tmpLocal = tmpBuf.Get<float>();
        auto lnwLocal = lnwBuf.Get<float>();
        auto ln1wLocal = ln1wBuf.Get<float>();
        auto resLocal = resBuf.Get<float>();
        auto sLocal = sBuf.Get<float>();
```

探针：对持有 >6 个独立 float `TBuf` 的 kernel 跑 1M 元素 fp32 correctness 测试；若 `is_pass=False` 且 `matched_ratio` 远低于 1，再写单元素单 buffer 的最小 kernel 打印/比较各 `TBuf::Get<float>()` 返回地址是否落入重叠区间。

观测（tool_observed）：1M 元素 fp32 测试 `is_pass=False matched_ratio=0.10`。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：dav-2201 上 192KB UB 总量对 8 个 8KB buffer 理论只占 ~64KB，远未超量，但独立 `TBuf::Get<float>()` 返回地址互相重叠——往 `wLocal` 写入的值被读到 `ln1wLocal` 地址，`Select` 的 `dst=resLocal` 输出 `ln1wLocal` 内容（具体数值 w=0.267 读成 -1.322 属 agent 调试陈述，kernel run 输出被 transcript adapter 截断为 `_see below_`，未作为 tool_observed 闭环）。根因不在 UB 总量不足，而在独立 TBuf 分配器在该 size 段（`TILE_LENGTH=2048`/8KB）对独立 `TBuf` 的 `bufferAddr` 分配不可靠；把 `TILE_LENGTH` 调小到 128/256 仍重叠，说明与 size 段无关、是独立 TBuf 分配路径的固有行为。绕开方式是用单 `workBuf` + `GetWithOffset<T>(size, byteOffset)` 由调用方显式字节偏移互不重叠分配（该修复的 verbatim after 与修复后通过信号未从 trace 恢复）。

## 验证信号

来源：本 trace 在 dav-2201（Atlas A2 族，a2，CANN 9.0.0）上对 Spence kernel-direct-invoke 工程做 1M 元素 fp32 correctness 测试。

- 1M 元素 fp32 测试 `is_pass=False matched_ratio=0.10`（独立 `TBuf.Get<float>()` 取址，修复前多 tile 综合表现）。（来源平台：a2；证据等级：tool_observed）

## 失效边界

- dav-2201（a2）实测；a3 共享 dav-2201 架构，未验证，推断可迁移。
- buffer 数量很少（≤4）或每个 buffer 很小时独立 TBuf 可能不重叠——具体边界未验证，遇到时先做地址重叠探针。
- 本卡只判别独立 `TBuf.Get` 地址重叠这一根因；多 tile 场景还需配合正确的 GM offset。

## 坏实践

- 看到 UB 总量（192KB）够就假设独立 TBuf 不会重叠——实际重叠与总量无关，是分配器在该 size 段的行为。该"靠 UB 总量余量判断不重叠"的路线标记为 `banned`。
- 把 `GetWithOffset` 的第二参数当元素偏移传（实际是字节偏移），导致地址错位。
- 用 `TILE_LENGTH` 调小（128/256）来"绕过"重叠——治标不治本，独立 TBuf 在小 size 下仍重叠。
