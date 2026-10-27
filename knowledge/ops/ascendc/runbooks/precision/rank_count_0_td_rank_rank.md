---
type: "Runbook"
title: "低 rank 零填充致 count=0 用 td_->rank 非 RANK"
description: "多 rank 共用 RANK=4 模板时 host tiling 对低 rank tensor 把 max_bro_shape[rank..3] 零填充；kernel 若用模板 RANK 作维度循环上界，inner_count *= 0 致 count=0 不处理任何元素，仅真实 4D shape 通过。判别：低 rank 用例全 FAILED 而唯 4D PASSED 时查 kernel 是否用 RANK 而非 td_->rank 作循环上界。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T05:33:58Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 低 rank 零填充致 count=0 用 td_->rank 非 RANK

## 现象

`mask_fill`（`MaskedFillKernel<T, 4>`，`RANK=4` 模板覆盖 rank∈[0,4] 的所有 tensor）Kernel 编译通过后，在 `operators/mask_fill_package/tests/aclnn` 执行 `./run.sh` 跑 Branch-0 ST。汇总 `==== Branch-0 ST summary: 2/23 cases PASSED ====`：唯二通过的是 `[M-17] PASSED`（bf16 `[2,3,4,5]`，4D）与 `[B-05] PASSED`（f32 rank=4 `[2,3,4,5]`，4D）；其余 21 个用例全 FAILED——含 `[M-01]`（f32 `[2,3]`，2D）、`[M-04]`（int8 `[4,8]`，2D）、`[M-10]`（f32 `[8]`，1D）、`[M-08]`（f16 标量 rank=0）等所有低 rank 形状，输出未写入（精度不匹配）。构建本身无错，最小调用通路 smoke test 亦 PASSED。ST 基线期望为 0 通过（kernel 起始为可编译 stub），故 2/23 不是环境问题而是 kernel 实现侧缺陷。

## 适用场景

- **工程形态**：Kernel 直调，`op_kernel/arch35/mask_fill_kernel.h` 的 `MaskedFillKernel<T, RANK>` 类（一个 `RANK=4` 模板覆盖 rank∈[0,4] 的所有 tensor）。
- **API 路径**：NDDMA `DataCopy<T, ND, cfg>`、`FlatToEffectiveCoord`、`CalcInputOffset`/`CalcOutputOffset`/`CalcTransferCount`——所有按维度循环、或把 rank 作实参传给坐标函数的 kernel 函数。
- **验证平台与 CANN 版本**：Ascend950 / CANN 9.1.0。
- **适用粒度**：`operator_family`——任何「一个模板 RANK 覆盖多个实际 rank、host tiling 把低 rank 零填充到模板 RANK」的 AscendC 算子。
- **路线优先级**：`recommended_fastpath`——维度循环上界与传给坐标函数的 rank 实参一律用 tiling data 的实际 rank 字段（`td_->rank`），模板 `RANK` 只用于数组容量与 `if constexpr` 分叉。
- **不适用或未验证场景**：若算子保证输入 rank 恒等于模板 RANK（如某些算子始终 4D/8D），本坑不触发；a2/a3 未在本 trace 上板验证（见 `## 失效边界`）。

## 判别方法

**触发（tool_observed）**：多 rank 共模板的 AscendC 算子 Kernel 编译通过、smoke test 与最小调用通路 PASSED，但 Branch-0 ST 汇总仅 `2/23 cases PASSED`，且通过的两个用例都是真实 4D shape（`[2,3,4,5]`），所有 1D/2D/3D/rank-0 用例 FAILED（输出未写入/精度不匹配）。

**探针（可执行检查）**：
1. 在 host tiling 源里确认 `PadAndSqueeze` 是否把 `max_bro_shape` 补到模板 `kRank`、对低 rank 在 `[rank..kRank-1]` 位置写 0：`grep -n "max_bro_shape\\|kRank\\|rank" op_host/arch35/mask_fill_tiling_common.cpp`。
2. 在 kernel 源里查所有「维度循环上界」与「传给坐标函数的 rank 实参」是否用模板参数 `RANK` 而非 tiling data 的实际 rank `td_->rank`：`grep -nE "d < RANK|< RANK|FlatToEffectiveCoord.*RANK|CalcInputOffset.*RANK|CalcOutputOffset.*RANK|CalcTransferCount.*RANK" op_kernel/arch35/mask_fill_kernel.h`。命中即疑点。

**原始观测（tool_observed）**：Branch-0 ST 输出（`./run.sh` 实测，逐用例结果）：

```
[M-01] FAILED dtype=float32 y_nelem=6 value=-1e+09  (same-shape f32)
[M-04] FAILED dtype=int8 y_nelem=32 value=-1  (same-shape int8)
[M-10] FAILED dtype=float32 y_nelem=8 value=1  (all-True mask f32)
[M-08] FAILED dtype=float16 y_nelem=1 value=5  (rank-0 scalar f16)
...
Test PASSED: aclnnMaskFill minimal call path (float32/bool/float32)
[M-17] PASSED dtype=bfloat16 y_nelem=120 value=1.5  (4D multi-axis size-1 broadcast bf16)
[B-05] PASSED dtype=float32 y_nelem=120 value=-inf  (rank=4 + value=-inf attention 4D)
==== Branch-0 ST summary: 2/23 cases PASSED ====
```

DESIGN 文档（`docs/mask_fill/DESIGN.md` 与分支设计）tool_observed 读取显示 kernel Process 的 `inner_count` 与坐标计算伪代码用 `rank=4` 硬编码：

```
int64_t inner_count = 1;
    inner_count *= td_->max_bro_shape[d];
...
FlatToEffectiveCoord(flat, td_->max_bro_shape, 4, ...)
```

而 host `PadAndSqueeze` 把 `max_bro_shape` 补到 `kRank=4`、`td_->rank` 为实际有效 rank（1~4）；对 rank=2 的 `[2,3]`，`max_bro_shape = [2, 3, 0, 0]`（`[rank..3]` 零填充），`td_->rank = 2`。

**判别与解释（interpretation，agent 根因分析，非 API 规范主张）**：kernel 用模板 `RANK=4` 作维度循环上界与坐标函数的 rank 实参，而非 tiling data 的实际 rank `td_->rank`。对 rank=2、axis=1 的 `[2,3]`：`inner_count` 循环 `d = axis+1..RANK-1` 即 `d=2..3`，`inner_count *= max_bro_shape[2] = 0` → `inner_count = 0` → `count = a_i_seg * 0 = 0` → 该 tile 不写任何元素。`FlatToEffectiveCoord(flat, max_bro_shape, 4, ...)` 传 `RANK=4` 同理让坐标计算越过实际 rank 进入零填充维。真实 4D shape 的 `max_bro_shape[d>0]` 全非零，所以只有 4D 通过。机制链（不跳步）：模板 `RANK=4` 覆盖多 rank → host `PadAndSqueeze` 零填充 `[rank..RANK-1]` → kernel 用 `RANK` 而非实际 rank 作循环上界 → 零填充维参与 `inner_count` 乘积 → `inner_count=0` → `count=0` → 该 tile 不写任何元素 → 低 rank 用例输出全错。结论：维度循环上界与传给坐标函数的 rank 实参一律改用 `td_->rank`，模板 `RANK` 只保留给数组容量（`max_bro_shape[RANK]`）与 `if constexpr (RANK<=5)` 分叉。

## 验证信号

- Branch-0 ST 汇总 `2/23 cases PASSED`：唯二通过的是 `[M-17]`（bf16 4D `[2,3,4,5]`）与 `[B-05]`（f32 rank=4 `[2,3,4,5]`），所有 1D/2D/3D/rank-0 用例 FAILED、输出未写入；smoke test 与构建无错。（来源平台：950；证据等级：tool_observed）
- DESIGN 文档 `inner_count` 循环与 `FlatToEffectiveCoord(flat, td_->max_bro_shape, 4, ...)` 伪代码用 `rank=4` 硬编码，host `PadAndSqueeze` 把 `max_bro_shape` 零填充到 `kRank=4`、`td_->rank` 为实际有效 rank。（来源平台：950；证据等级：tool_observed）

## 失效边界

- Ascend950 / CANN 9.1.0 实测；a2/a3 平台未在本 trace 上板验证，推断可迁移（rank 零填充是 host tiling 行为，与 NPU arch 无关），标注「未验证，推断可迁移」，不写入 `platforms`。
- 若某算子的 host tiling 不做零填充、而是按实际 rank 动态分配 `max_bro_shape` 数组，则模板 RANK 与实际 rank 混用的坑形态不同，需另行验证。
- 修复后 `23/23 pass` 的 closure 信号在 normalized trace 中未恢复为 `tool_result`（run.sh 输出缺失），仅 agent_message 陈述；after 代码（kernel edit）在 trace 中被 redact 为 `_see below_`，未逐字恢复。

## 坏实践

- 不要用 `memset(max_bro_shape, 0, sizeof)` 之外的方式让零填充维「恰好非零」来绕过——那是掩盖根因。
- 不要把低 rank 用例从 ST 集里删掉让 4D-only 看起来全绿——反投机取巧纪律明令禁止删用例。

## 设计决策

- 诊断侧只锁定「低 rank 全 FAILED、唯 4D PASSED」这一 tool_observed 触发模式与 `RANK` vs `td_->rank` 的机制链；修复配方的 verbatim after 因 redact 不发布，避免凭 agent 叙述重写代码。
- 模板 `RANK` 与实际 rank 的分工主张：`RANK` = 数组容量与 `if constexpr` 分叉；`td_->rank` = 维度语义（循环上界 / 坐标函数 rank 实参）。该分工是 `## 判别方法` interpretation 的结论，属 agent 根因分析，非 API 规范主张。
