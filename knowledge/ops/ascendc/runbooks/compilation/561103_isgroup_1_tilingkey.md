---
type: "Runbook"
title: "561103: isGroup=-1 哨兵致 tilingKey 路由失配"
description: "多分支归约算子 TilingFunc 重构后非 Group 分支未显式置 ctx.isGroup=0，-1 哨兵使 ChooseTilingKey 二值路由全分支失配返回 -1，TilingFunc GRAPH_FAILED 致 aclnn 561103。"
tags: ["task.debugging", "topic.tiling", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 561103: isGroup=-1 哨兵致 tilingKey 路由失配

## 现象

在为 `euclidean_norm`（归约范数，ascend950，5 个 tilingKey 0-4）实现 Branch-3（Group-TailR）Kernel 后，运行黑盒 ST `operators/euclidean_norm_package/tests/aclnn && ./run.sh`：之前 63/63 全过的 b0/b1（Normal 分支）用例集体回归为运行时错误 **561103**，b2（EMPTY）仍过；b3（Group）报 361001。本次改动仅是给 kernel 增加 Group 分支调度，未触碰 b0/b1 代码路径——回归来自 TilingFunc 侧：TilingFunc 重构时把原本无条件的 `ctx.isGroup = 0` 替换为"仅在命中 Group 时 `ctx.isGroup = 1`"，于是非 Group 用例里 `isGroup` 沿用结构体默认值未变更。

## 适用场景

- 工程形态：aclnn 算子仓开发（op_host TilingFunc + 多分支单二进制 kernel），非 Kernel 直调。
- API 路径：`OpParaFunc` TilingFunc → `ChooseTilingKey` → 内部 dispatch；ST 经 `aclnnEuclideanNormGetWorkspaceSize` → `aclnnEuclideanNorm`。
- 验证过的平台与 CANN 版本：Ascend950（spec.yaml/AddConfig + ST 上板），CANN 9.1.0。
- 适用粒度：`operator_family`——凡 tiling 上下文用"哨兵值 + 二值路由函数"模式的多分支归约/规约类算子。
- 路线优先级：`recommended_fastpath`（结构体字段哨兵设计本身合理，只需保证写入侧覆盖所有非默认分支）。
- 不适用/未验证场景：单分支算子（无 `ChooseTilingKey` 路由）；a2/a3 未验证（推断可迁移，见失效边界）。

## 判别方法

**触发**（tool_observed）：TilingFunc 重构（如新增 Group 分支调度）后，原本通过的 Normal 分支 ST 用例集体报 561103，且改动未触及 Normal 路径代码。

**探针**（可执行）：

1. `grep -n 'isGroup\|ChooseTilingKey\|return -1' op_host/<op>*_def.cpp`——检查结构体字段默认值与路由函数分支覆盖。
2. 在 TilingFunc 末尾打印 `ctx.isGroup` 与 `ChooseTilingKey(ctx)` 返回值，对照 ST 失败用例。

**原始观测**（verbatim，after-state file_read，tool_observed）：

- `EuclideanNormCtx` 字段定义：`int32_t isGroup = -1;  // -1 unset; 0=normal, 1=group`（哨兵默认值）。
- `ChooseTilingKey` 仅判 `==1` 与 `==0`，无 `== -1` 兜底分支：

```cpp
int32_t ChooseTilingKey(const EuclideanNormCtx& ctx)
{
    const int32_t emptyKind = static_cast<int32_t>(ctx.emptyKind);
    if (emptyKind == 1 || emptyKind == 2) { return 2; }  // EMPTY (EMPTY_A or EMPTY_R)
    if (ctx.isGroup == 1 && ctx.isTailR == 1) { return 3; }  // Group Tail-R
    if (ctx.isGroup == 1 && ctx.isTailR == 0) { return 4; }  // Group Tail-A
    if (ctx.isGroup == 0 && ctx.isTailR == 1) { return 0; }  // Normal Tail-R
    if (ctx.isGroup == 0 && ctx.isTailR == 0) { return 1; }  // Normal Tail-A
    return -1;  // invalid (unset flags)
}
```

**判别与解释**（interpretation，agent 根因分析，非 API 规范主张）：`isGroup` 字段以 `-1` 为"未设置"哨兵，`ChooseTilingKey` 是二值全等路由（只认 0/1），对哨兵 `-1` 无兜底分支 → 返回 -1。TilingFunc 重构把"无条件 `ctx.isGroup = 0`"改为"仅在命中 Group 时 `ctx.isGroup = 1`"，于是非 Group 分支的 `isGroup` 沿用结构体默认 `-1` → `ChooseTilingKey` 返回 -1 → `kTilingFail`/`GRAPH_FAILED` → aclnn 侧 561103。

机制链：`isGroup 默认 -1` →（非 Group 路径未覆写）→ `ChooseTilingKey: 无 -1 分支 → return -1` → `TilingFunc: tilingKey != 期望 → kTilingFail/GRAPH_FAILED` → `aclnn 561103`。该因果链中"哨兵 + 二值路由"由结构体定义与 `ChooseTilingKey` 共同构成（二者在修复前后未变），故判别闭环。

## 验证信号

- 修复后 ST `run.sh` 63/63 通过（22 b0 + 16 b1 + 7 b2 + 18 b3），b0/b1 的 561103 回归消除（来源平台：950；证据等级：tool_observed）。
- after-state `op_host/euclidean_norm_def.cpp` 中 `EuclideanNormCtx::isGroup = -1` 哨兵字段、`ChooseTilingKey` 路由函数（无 `-1` 兜底分支）与非 Group 分支显式 `ctx.isGroup = 0` 均经 file_read 逐字确认（来源平台：950；证据等级：artifact_verified）。

## 失效边界

- a2/a3：未验证，推断可迁移（tiling 上下文哨兵模式与平台无关，但 `ChooseTilingKey` 分支表与具体算子的 tilingKey 编号绑定，迁移时需对照目标算子路由表）。
- 单分支算子无 `ChooseTilingKey` 路由，不适用。
- 若结构体改为 `bool isGroup`（无 -1 哨兵）则本判别不成立——本卡仅针对 `int32_t = -1` 哨兵设计。

## 坏实践

- 误判 561103 为 kernel 实现 bug 而去改 kernel 代码——根因在 host 侧 tiling 路由，kernel 未执行到。该路线标记为 `banned`。
- 把 `isGroup` 改为默认 0 绕过——掩盖"未设置"语义，后续 Group 触发判定将无法区分"真 Normal"与"未设置"。

## 误诊

- 本 trace 早期曾把 b3 的 361001 误判为"形状不匹配"，实际 361001 来自 `aclrtBinaryGetFunctionByEntry` 找不到 tilingKey 对应的 kernel 二进制（单二进制设计中误调 `SetTilingKey` 触发，属不同根因）；561103 与 361001 不可合并。