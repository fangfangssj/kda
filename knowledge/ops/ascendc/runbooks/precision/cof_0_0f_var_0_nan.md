---
type: "Runbook"
title: "除零兜底 cof=0.0f 致 var=0 而非 NaN"
description: "ReduceStdV2Update tiling 在 N-correction<=0 时防御性写 cof=0.0f，使 var=sq_sum*0=0，与 oracle torch.var(N=1,correction=1) 输出 NaN 不一致；判别触发为除零用例 actual=0.0 vs golden=NaN。"
tags: ["task.debugging", "topic.precision", "topic.compute.reduction"]
status: stable
sources: []
created_at: "2026-08-18T05:31:17Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 除零兜底 cof=0.0f 致 var=0 而非 NaN

## 现象

ReduceStdV2Update 算子迭代一联调验证 ST 用例 18（L0_STD_014 FP32 除零保护，`x=[2,1,3]`、`dim=[1]`、`N=1`、`unbiased=true` 即 `correction=1`、`if_std=true`）失败：NPU `actual=0.0`，`golden=NaN`。这是除零场景：`denom=N-correction=1-1=0`，数学上 `var=sq_sum/0` 未定义。oracle `torch.var(correction=1)` 在 `N=1` 时输出 NaN。golden 函数 `ComputeGolden` 对除零正确输出 NaN，但 NPU kernel 输出 0.0，差异来自 tiling 函数 `FillAndLogTilingData` 的除零保护分支。

## 适用场景

- **工程形态**：aclnn L2 单算子调用（ST 黑盒测试），归约类算子带 `correction` 参数。
- **API 路径**：`aclnnReduceStdV2UpdateGetWorkspaceSize` → l0op → tiling 函数 `ReduceStdV2UpdateTilingFunc` → `FillAndLogTilingData` 计算 `cof` → `tilingdata.cof` → kernel `PostElewiseVar` 中 `Muls(f32Reg, f32Reg, cof_, mask)`。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510 / arch35），CANN 9.2.0。
- **适用粒度**：`operator_family`——所有归约类算子带 `correction` 且 tiling 计算 `cof` 的场景。
- **路线优先级**：`recommended_fastpath`——在 tiling 内把除零分支的 `cof` 改写为 NaN 使 `var=sq_sum*NaN=NaN`，1 行修复，不依赖上层 aclnn 拦截。
- **不适用或未验证场景**：未在 A3/A2 验证；上层 aclnn 接口主动拦截 `N<=correction` 返回 NaN 是另一种治本方案，本卡不覆盖其实现。

## 判别方法

- **触发**：归约算子带 `correction` 参数的除零用例（`N-correction<=0`，如 `N=1, correction=1, denom=0`）ST 失败，现象为 `actual=0.0`（有限值）而 `golden=NaN`；编译通过、无 507035、无段错误，仅该边界用例数值不符 oracle。
- **可执行探针**：在 tiling 函数 `FillAndLogTilingData` 内定位 `cof` 计算分支，查看 `denom<=0` 分支写的是 `0.0f` 还是 NaN；同时跑 ST `L0_STD_014`（`x=[2,1,3], dim=[1], N=1, unbiased=true`）确认 `actual=0.0 vs golden=NaN`。也可把 `denom` 临时改成 `N-correction>0` 的用例对照：通过即坐实除零分支为根因。
- **原始观测**：ST `L0_STD_014` 用例 18 NPU 输出 `actual=0.0`，golden 期望 `NaN`（除零保护），该用例 FAIL；同批次其余非除零用例 PASS。
- **解释**：tiling `FillAndLogTilingData` 内 `denom<=0` 的防御性分支写 `cof=0.0f`（注释声称"上层 aclnnVar/aclnnVarCorrection 拦截返回 INF/NAN，kernel 防御性写 0"），但实测 `aclnnReduceStdV2Update` L2 接口 `CheckParams` 只查 null/dtype/shape，未拦截 `N<=correction`，直接传到 kernel。机制链：`denom=0 → cof=0.0f → var=sq_sum*0.0=0.0`（IEEE 754 任意有限数乘 0 为 0，含 `sq_sum!=0`）；oracle 在 `denom==0` 时输出 `NaN`。`if_std=true` 时 host 端 `sqrt(0)=0` vs golden `sqrt(NaN)=NaN`，差异进一步放大。根因是 tiling 用 `0.0f` 表达"除零兜底"，与 oracle 的 NaN 语义不一致；正确兜底应写 NaN（`var=sq_sum*NaN=NaN`）。

定位到的根因分支（tiling `FillAndLogTilingData`，`denom<=0` 分支）：

```cpp
const int64_t denom = N - ctx.correction;
if (denom > 0) {
    ctx.cof = 1.0f / static_cast<float>(denom);
} else {
    // N<=correction 由上层 aclnnVar/aclnnVarCorrection 拦截返回 INF/NAN，kernel 防御性写 0
    ctx.cof = 0.0f;
}
```

## 验证信号

本卡验证来源：ReduceStdV2Update 迭代一联调验证 ST 会话（Ascend950 / dav-3510 / CANN 9.2.0）。

- ST `L0_STD_014` 用例 18（FP32 除零保护，`N=1, correction=1, denom=0`）NPU `actual=0.0` vs `golden=NaN`，该除零边界用例 FAIL（来源平台:Ascend950；证据等级:tool_observed）

## 预防

- tiling 内凡有"防御性写 0"的除零分支，须对照 oracle（`torch.var`/`torch.std`）的对应边界行为：oracle 输出 NaN 时 tiling 也应写 NaN，不应写 0——`0.0f` 是合法 `cof` 值，不应混用其表达"除零"语义。
- 注释"上层会拦截"须实际核对上层 aclnn `CheckParams` 是否真拦截，不得假设；本例 L2 接口未拦截 `N<=correction`，兜底分支被命中。

## 失效边界

- Ascend950 / CANN 9.2.0 实测；A3/A2 未验证，推断可迁移（IEEE 754 NaN 语义跨平台一致）——未验证，推断可迁移。
- 上层 aclnn 接口若未来增加 `N<=correction` 主动拦截返回 NaN，tiling 的 NaN 兜底分支仍作为防御性兜底无害保留。
- `Muls` 对 NaN 输入的行为依赖 RegBase `Muls` 实现；本卡未独立验证其他 AscendC `Muls` 实现下 NaN 的传播。
