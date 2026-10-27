---
type: "Runbook"
title: "303 D2 末轴分块破坏 no-Cast DAG x2 加载"
description: "brc 303 last-axis block distribution 在 AddWithoutCastCompute（FP32/INT32/INT64）下破坏 x2 标量 brc 加载时序；d2TotalWork/coreNum/ubFormer 是 D2 路径的能力门禁，不满足时必须禁用 D2。"
tags: ["task.debugging", "topic.precision", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["schMode303", "last_axis_block_distribution", "x2_missing"]
---

# 303 D2 末轴分块破坏 no-Cast DAG x2 加载

## 现象

在不含 PR#692 的纯净 opbase 基线（commit `2cd7568`）上对 aclnnAdds 的 Add kernel（brc arch35 模板）做 303 专项优化时，实施 D2「last-axis block distribution + 100% NDDMA 密度」方案后出现**按 dtype 分裂的精度失败**：

- **FAIL case**：FP32 / INT32 / INT64 — 走 `AddWithoutCastCompute` DAG（`CopyInBrc(x1) → CopyInBrc(x2) → Add → CopyOut`，Add 直接读 CopyInBrc 输出 buffer，无中间节点）。
- **PASS case**：FP16 / BF16 — 走 `AddWithCastCompute` DAG（`CopyInBrc → Cast → Add`，Cast 中间节点提供额外同步）。
- **失败模式**：`err=1.0`，输出 = `x1 + 0`，即 **x2 标量未参与 Add**（x2 输入缺失）。
- **边界特征**：非最后 tile PASS、最后 tile FAIL → 边界条件问题；仅在小 shape（_001 view [1,8,128]）真机复现，_002（view [1,32,512]）在加守护后 PASS。

报错原文（st_runner --verify 远端真机输出）：
```
>>>VERIFY:s1_head3d_float_001|pass=0|nfail=.../maxErr=1.000000
lin=8, out=0   # x2 缺失，Add 结果只有 x1 + 0
```

## 判别方法

### 触发

D2 last-axis block distribution tiling 路径 + no-Cast DAG（FP32/INT32/INT64 的 `AddWithoutCastCompute`）→ x2 标量 brc 加载缺失，输出 `x1 + 0`。触发条件：brc 模板 schMode 303（last-two-axes transpose）下启用 D2 每核单 tile 末轴分发，且 DAG 不含 Cast 中间节点。

### 探针

1. 检查 tiling 轴切分：D2 是否按末轴（lastAxis）切成 `ubFormerLastAxis` 块、每核单 tile 分发（`blockNum = d2TotalWork`）。
2. 检查 IR/DAG 是否含 Cast 节点：FP32/INT32/INT64 走 `AddWithoutCastCompute`（`CopyInBrc → Add` 直连无中间节点），FP16/BF16 走 `AddWithCastCompute`（含 Cast）。仅 no-Cast DAG 失败 → 确认 Cast 提供的中间同步掩盖了 x2 加载时序问题。
3. 检查 `CopyInBrc` 缓存加载条件 `lastAxisLoopIdx<=1 || ==N-1`：D2 每核单 tile 下中间 tile 的 `lastAxisLoopIdx` 既不 `<=1` 也不 `==N-1`，x2 不被加载。

配套 tiling 参数（`broadcast_tiling_noncontiguous.cpp` 的 `DoBrodcastTilingLastTranspose`，before/after verbatim 取自 trace file_read 事件）——诊断时核对以下参数是否处于 buggy/fixed 取值：

axisBeforelastAxisSplitFactor 收紧（128→64，限制末轴前一轴的切分粒度）：
```cpp
// before
    int64_t axisBeforelastAxisSplitFactor = 128 / minDtypeBlockAlignSize;
// after
    int64_t axisBeforelastAxisSplitFactor = 64 / minDtypeBlockAlignSize;
```

末轴 ubFormer 减半（防止末轴单块过大越过 UB 对齐边界）：
```cpp
// before
        // ubFormer 不可能在最后一根轴
        ubFormer = maxElemNum / curProduct;
// after
        // ubFormer 不可能在最后一根轴
        ubFormer = maxElemNum / curProduct;
        ubFormer = (ubFormer + 1) / 2;
```

minDtypeAlignLt 作用域修正（`minDtypeBits` 在插入点未定义，须用 `computeParams.minDtypeBits`）：
```cpp
// before（编译错：minDtypeBits 未在作用域）
            int64_t minDtypeAlignLt = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;
// after
            int64_t minDtypeAlignLt = BLOCK_LENGTH * BROADCAST_BITS_NUM / computeParams.minDtypeBits;
```

### 观测

- D2 _002 系列 5 dtype PASS：`>>>VERIFY:...|pass=1|nfail=0/16384|maxErr=0.000000`，Block Dim=64（D2 触发）。
- _001 系列（view [1,8,128]）真机 FAIL：`>>>VERIFY:s1_head3d_float_001|pass=0|nfail=.../maxErr=1.000000`，`lin=8, out=0`（x2 缺失，Add 结果只有 x1 + 0）；guard 后回退原路径才 PASS。
- 失败按 dtype 分裂：FP32/INT32/INT64（no-Cast DAG）FAIL，FP16/BF16（Cast DAG）PASS。

### 判别

D2 每核单 tile 切分与 no-Cast DAG 的 `CopyInBrc` 缓存加载条件（`lastAxisLoopIdx<=1 || ==N-1`）不兼容：中间 tile 的 `lastAxisLoopIdx` 既不 `<=1` 也不 `==N-1`，x2 NDDMA brc 不触发，Add 直连读到未初始化/旧 buffer → `x1 + 0`（x2 缺失，2× 因子丢失）。Cast 中间节点提供额外同步，故 FP16/BF16 幸免。问题不在 x2 gmOffset（标量恒为 0，正确），而在加载触发条件与 D2 切分的不兼容。

## 适用场景

- **工程形态**：注册算子工程（ops-math/math/add）+ opbase broadcast 非连续 tiling，aclnnAdds 接口触发 Add kernel brc 模板。
- **API 路径**：`aclnnAdds` → `Add kernel` arch35 brc schMode 303（last-two-axes transpose，`isOnlyLastTwoAxesTransposed=true`）。
- **验证平台与版本**：Ascend 950PR_9589 真机，CANN 9.2.0，opbase commit `2cd7568`（baseline_no_pr692）。
- **适用粒度**：`operator_family`（broadcast 类 brc 非连续模板的 last-axis block distribution 通用约束）。
- **能力门禁**：D2 仅在守护条件满足时启用；不满足时禁用 D2。该门禁让 _002 转正，但 _001 小 shape 仍失败，不能宣称问题已完整修复。
- **不适用/未验证场景**：连续 schMode 202/204 不涉及；schMode 301/302（NLast transpose）不使用 D2 路径；_001（shape [1,8,128]）真机仍 FAIL，未验证，推断可迁移平台（a2/a3）未验证。

## 验证信号

- D2 _002 系列 5 dtype PASS，`maxErr=0.000000`，Block Dim=64（来源平台：950；证据等级：tool_observed）
- _001 真机 FAIL，`pass=0|maxErr=1.000000`，`lin=8, out=0`（x2 缺失），guard 回退原路径后 PASS（来源平台：950；证据等级：tool_observed）
- 失败按 dtype 分裂：FP32/INT32/INT64 no-Cast DAG FAIL，FP16/BF16 Cast DAG PASS（来源平台：950；证据等级：tool_observed）

## 预防

- 任何改动 brc 模板「每核 tile 数」或「末轴分发」的策略，必须对**所有 dtype**（含 no-Cast DAG 的 FP32/INT32/INT64）跑精度回归，不能只验 FP16/BF16。
- no-Cast DAG 的 `CopyInBrc → Add` 直连缺少中间同步节点，对 tile 切分边界敏感；改动切分时检查 CopyInBrc 的缓存加载条件（`lastAxisLoopIdx<=1 || ==N-1`）是否与新切分兼容。

## 坏实践

- 用 FP16/BF16 PASS 作为 D2 全量上线的依据——Cast 中间节点掩盖了 no-Cast DAG 的 x2 加载时序问题。
- 临时加 `false &&` 禁用 D2 后见 PASS 就收工，不深究 _001 机制——根因未追到机制层。

## 误诊

- 初判「cannsim 仿真限制」——_001 在 cannsim FAIL 时被归为仿真小 stride 限制，后真机复现确认是真实 bug。
- 推测「brc 强制加载可修复」——已尝试无效，最终靠守护回退原路径。

## 失效边界

- _001（shape [1,8,128]）真机仍 FAIL：guard 使其回退原路径才 PASS，D2 本身未修复该 shape。
- 未验证，推断可迁移：a2 / a3 平台（同 dav 架构族），未在本 trace 实测。
- schMode 301/302（NLast transpose）不使用 D2 路径，本卡守护条件仅适用于 303 last-two-axes transpose。
