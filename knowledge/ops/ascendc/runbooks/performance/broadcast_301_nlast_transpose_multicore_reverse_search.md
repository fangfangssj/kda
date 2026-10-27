---
type: "Runbook"
title: "301 非连续多核启用：preferMultiCore 守护"
description: "brc 301 NLastTranspose 非连续 case 多核未触发因 preferMultiCore 对连续/非连续混用；须 `preferMultiCore && !inputAllContiguous` + DoTiling(0,0,true) 强制。"
tags: ["task.optimization", "topic.tiling", "topic.data_movement.layout_transform", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["schMode301"]
---

# 301 非连续多核启用：preferMultiCore 守护

## 现象

aclnnAdds 的 Add kernel brc schMode 301（NLastTranspose，非连续 NLast 转置）非连续 case 在纯净基线 opbase `2cd7568` 上多核利用率极低：cannsim 显示 `blockNum=4`，仅 4/64 核工作（~6.25%）。整体慢，`below_compute_capacity`。301 的 `DoBroadcastTilingNLastTranspose` 路径在 `preferMultiCore` 未正确传递时走单核/少核切分，非连续 case 的 fusedProduct（ubOuter×M）虽大于 coreNum 但 blockFormer 仍取 1 且未触发反向推导。

## 适用场景

- **工程形态**：注册算子工程（ops-math/math/add）op_host tiling + opbase broadcast 非连续 tiling 头。
- **API 路径**：`aclnnAdds` → Add kernel arch35 brc → `DoBroadcastOpTilingNLastTranspose`（schMode 301）。
- **验证平台与版本**：Ascend 950PR_9589 真机 + cannsim dav-3510，CANN 9.2.0，opbase `2cd7568`（baseline_no_pr692）。
- **适用粒度**：`operator_family`（broadcast NLastTranspose 非连续模板的多核启用通用模式）。
- **路线优先级**：`recommended_fastpath` — 非连续 301 case 应优先启用多核反向推导。
- **不适用或未验证场景**：连续 case（inputAllContiguous=true）不在此路径；schMode 303（last-two-axes transpose）走独立 D2 路径；a2/a3 未验证（推断可迁移，同 dav 架构族）。

## 根因

`broadcast_tiling.h` 的 `DoBroadcastOpTilingNLastTranspose` 把 `preferMultiCore` 原样透传给 `broadcastTilingParams`，没有区分连续/非连续。对 301 非连续 case，`preferMultiCore` 应仅在**非连续**时强制为 true（连续 case 已有连续模板的高效路径，强制多核反而退化）。同时 `add_tiling_arch35.cpp` 调用 `brcBaseTiling.DoTiling()` 不带第三参数，未启用强制多核 tiling 入口（DoTiling 的 `bool preferMultiCore` 形参缺省为 false）。

机制链：`preferMultiCore 对连续也 true` → `连续 case 走多核路径退化` + `非连续 case preferMultiCore 语义不明确` → `DoBroadcastTilingNLastTranspose 内反向推导未触发` → `blockNum=fusedProduct（小），少核`。

## 完整实现

before/after verbatim 函数级（两个文件，两处独立改动）：

`pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h`（DoBroadcastOpTilingNLastTranspose 内）：
```cpp
// before
        broadcastTilingParams.inputAllContiguous = inputAllContiguous;
        broadcastTilingParams.preferMultiCore = preferMultiCore;
// after
        broadcastTilingParams.inputAllContiguous = inputAllContiguous;
        broadcastTilingParams.preferMultiCore = preferMultiCore && !inputAllContiguous;
```

`math/add/op_host/arch35/add_tiling_arch35.cpp`（AddsTiling::RunTiling 内，`replaceAll=true`）：
```cpp
// before
        ret = brcBaseTiling.DoTiling();
// after
        ret = brcBaseTiling.DoTiling(0, 0, true);
```

`DoTiling(0, 0, true)` 的第三参数 `true` = 强制 preferMultiCore（与 broadcast_tiling.h 的 `preferMultiCore && !inputAllContiguous` 配合：非连续时 tiling 入口强制多核，broadcast_tiling.h 再用 `!inputAllContiguous` 把连续 case 拉回 false）。

## 设计决策

- **为何选 `preferMultiCore && !inputAllContiguous` 而非无条件 true**：连续 case（inputAllContiguous=true）已有 schMode 202/2 连续高效模板，强制多核路径会使其退化；只在非连续时启用多核反向推导，避免连续回归。
- **为何同时改 DoTiling 第三参数**：仅改 broadcast_tiling.h 的字段不够——`DoTiling()` 缺省第三参数=false，根本不进入多核 tiling 入口；必须 `DoTiling(0, 0, true)` 才能让 `BroadcastTilingNLastTranspose` 的 `preferMultiCore` 分支生效。两处是同一根因的两个必要修正点。
- **被放弃的方案**：仅在 add_tiling_arch35.cpp 加 `DoTiling(0,0,true)` 不改 broadcast_tiling.h → 连续 case 也被强制多核，回归。

## 验证信号

3D case `[M=8,32,1024]`（s301 3d_medium_fp32_nc）真机 st_runner --verify + plog 输出（绝对 aiv 时序仅作 provenance，已隐去，保留 block_dim 与 pass/fail）：

- 3D 3d_medium_fp32_nc 真机 verify `pass=1|nfail=0/262144|maxErr=0.000000`，schMode=301 fusedProduct=32（来源平台：950；证据等级: tool_observed）
- msopprof 对比（base=纯净基线，opt=加守护后）：block_dim（bd）从 4-26 提升到 32-43，非连续 FP32/INT64 medium-large case speedup 1.20-1.25x（参照路线=纯净基线 opbase 2cd7568），精度全 PASS（来源平台：950；证据等级: tool_observed）

## 性能影响

非连续 301 FP32/INT64 medium-large case 相对纯净基线 speedup 1.20-1.25x（参照路线=纯净基线 opbase 2cd7568，来源平台 950）。FP16 case 收益不明显（0.92-1.05x），因 FP16 连续 case 本就快、非连续 NDDMA 开销占比不同。精度全 PASS（nfail=0）。

## 预防

- 改动 brc tiling 的多核启用时，连续/非连续必须分路径：用 `inputAllContiguous` 做守护，避免连续模板被强制多核退化。
- `DoTiling` 的 `preferMultiCore` 形参缺省为 false，调用方须显式传 `true` 才能进入多核 tiling 入口——不要假设 broadcast_tiling.h 内的字段赋值会自动生效。

## 失效边界

- schMode 303（last-two-axes transpose）不在此路径，走独立 D2。
- FP16 非连续 case 收益不稳定，未验证为通用收益。
- 未验证，推断可迁移：a2 / a3（同 dav 架构族），未实测。
