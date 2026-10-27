---
type: "Runbook"
title: "ACLNN 507035 — stub kernel tiling 数据未初始化"
description: "接口桩 kernel（仅 InitSocState+REGISTER_TILING+GET_TILING+PipeBarrier）配合不调用 GetTilingData/不写任何 tiling 字段的 stub TilingFunc 时，aclnn 冒烟在 aclrtSynchronizeStream 报 507035=ACL_ERROR_RT_VECTOR_CORE_EXCEPTION。判别并给出 stub TilingFunc 必须初始化 tiling buffer 的修复方向。"
tags: ["task.debugging", "topic.tiling", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ACLNN 507035 — stub kernel tiling 数据未初始化

## 现象

为 TileWithAxis 算子（ascend950/arch35）在 Task 14 阶段建立基础 ACLNN 工程后，按"接口桩"要求编写最小 kernel：仅 `AscendC::InitSocState()` + `REGISTER_TILING_DEFAULT(TileWithAxisTilingData)` + `GET_TILING_DATA(tilingData, tiling)` + `AscendC::PipeBarrier<PIPE_ALL>()`，对应 stub TilingFunc 只设 `BlockDim(1)` 与 workspace，不调用 `context->GetTilingData<>()`、不写任何 tiling 字段。运行 11 dtype 冒烟（`tests/aclnn/run.sh`）：`aclnnTileWithAxisGetWorkspaceSize` 与 `aclnnTileWithAxis` 均 OK，但 `aclrtSynchronizeStream(stream)` 返回 **507035**，11/11 全部 `PRECISION_STATUS: FAIL / GOLD: NO_OUTPUT`。

## 适用场景

- **工程形态**：aclnn 算子开发（ops-registry-invoke）的接口桩（stub）阶段——kernel/TilingFunc 只搭骨架不做算法。
- **API 路径**：kernel 入口 `__global__ __aicore__ void <op>(...)` + `REGISTER_TILING_DEFAULT` / `GET_TILING_DATA`；Host `TilingFunc` 内 `context->GetTilingData<T>()` / `context->SetBlockDim()`。
- **验证过的平台与 CANN 版本**：Ascend950（arch35），CANN 9.1.0。
- **适用粒度**：`operator_family`（任何 AscendC 算子在 stub 联调阶段）。
- **路线优先级**：`recommended_fastpath`。
- **明确不适用或未验证场景**：已实现真实计算逻辑的 kernel 不在本卡范围；a2/a3 未验证（推断可迁移，见 `## 失效边界`）。

## 判别方法

1. **触发信号**：aclnn `GetWorkspaceSize` 与 launch 两步均返回 OK，但 `aclrtSynchronizeStream` 返回 507035；多 dtype 一致失败（系统性，非特定 case）。
2. **可执行探针**：`grep -rn "507035\|107035" $ASCEND_TOOLKIT_HOME/.../include/acl/` 查错误码定义；对照 `add_custom_package` 模板的 TilingFunc（它调用 `context->GetTilingData<AddCustomTilingData>()` 并写字段 `tiling->totalLength = totalLength;` 等）。
3. **原始观测**：507035 = `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`（vector core 异常，类段错误）。stub kernel 未访问任何 GM、未做 DataCopy，仅读 tiling buffer；stub TilingFunc 未调用 `GetTilingData<>()`、未写 tiling 字段，tiling buffer 内容为未初始化/全零。
4. **解释**：framework 依据 kernel 中 `REGISTER_TILING_DEFAULT(TileWithAxisTilingData)` 按 struct 大小分配 tiling buffer，但 TilingFunc 必须通过 `context->GetTilingData<T>()` 取指针并写入字段以完成初始化；stub TilingFunc 跳过该调用导致 tiling buffer 未被正确初始化，`GET_TILING_DATA` 读到未初始化数据，vector core 触发异常。即使 kernel 不使用 tilingData 字段，buffer 也必须被 TilingFunc 写过。

## 解决配方

stub TilingFunc 必须调用 `context->GetTilingData<TilingDataT>()` 取得指针并写入至少一个字段（如 `totalLength`），再 `SetBlockDim`/workspace；kernel 侧保持接口桩不变。修复后 11/11 冒烟通过。

**before**（stub kernel，verbatim 取自 trace reasoning seq3351）：

```cpp
extern "C" __global__ __aicore__ void tile_with_axis(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(TileWithAxisTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

配合的 stub TilingFunc 不调用 `GetTilingData<>()`、不写 tiling 字段 → 507035。

**after**（stub TilingFunc 按 add_custom 模板初始化 tiling buffer，参考模板 verbatim 取自 trace reasoning seq3362）：

```cpp
// TilingFunc 内（add_custom 模板模式）
TileWithAxisTilingData* tiling = context->GetTilingData<TileWithAxisTilingData>();
tiling->totalLength = totalLength;      // 至少写一个字段完成初始化
// tiling->blockLength = blockLength;
// tiling->tileLength  = tileLength;
context->SetBlockDim(1);
```

kernel 侧接口桩保持 `InitSocState + REGISTER_TILING_DEFAULT + GET_TILING_DATA + PipeBarrier` 不变即可通过冒烟。

> 注：本 trace 中 stub TilingFunc 的终版 AFTER 写入事件被 adapter redaction（content='_see below_'），无法逐字恢复完整函数体；上方 after 为按 add_custom 模板（trace reasoning seq3362 引用）与机制解释重建的最小形态，字段名沿用 TileWithAxisTilingData。完整终版 verbatim 未在 trace 中恢复—。

## 验证信号

本次开发会话（TileWithAxis Task 14 接口桩阶段）修复后实测：

- 11 个 dtype 冒烟用例全部 `PASSED`，`run.sh` 退出码 0（来源平台：950；证据等级: tool_observed）。
- 重新跑 6 个验收门禁（build 退出 0、无 tiling 注册告警、so 符号 `aclnnTileWithAxis`/`aclnnTileWithAxisGetWorkspaceSize` 存在、11 个 kernel `.o`、无 add_custom 残留）全部通过（来源平台：950；证据等级: tool_observed）。

## 失效边界

- Ascend950（arch35）实测；a2/a3 未验证，推断可迁移（`REGISTER_TILING_DEFAULT`/`GetTilingData` 为跨 arch 通用 API），未验证。
- "必须写 tiling 字段"的下限（最少写几个字段、是否可只写 BlockDim）未在本 trace 中边界探测；安全做法是写满所有 kernel 会经 `GET_TILING_DATA` 读取的字段。
- 若 kernel 已实现真实计算并仍报 507035，则属真实越界/非法访问，不属本卡范围。

## 预防

- 接口桩阶段 TilingFunc 至少调用 `GetTilingData<T>()` 取指针并写一个字段，再跑冒烟；不要只设 BlockDim/workspace。
- 冒烟报 507035 时先与 add_custom 模板对齐 TilingFunc 是否写 tiling 数据，再怀疑 kernel 逻辑。

## 坏实践

- 误以为 `REGISTER_TILING_DEFAULT` + 空 TilingFunc 即可让 stub kernel 跑过（framework 分配了 buffer 但 TilingFunc 未写入）。
- 把 507035 当作"空 kernel 硬件不接受"去加无关计算逻辑掩盖（真正缺的是 tiling buffer 初始化，不是计算）。

## 误诊

- 起初怀疑是 kernel 参数个数（4 vs 5）或缺少 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，经对照 add_custom 模板排除（agent_reported，非实测信号）。