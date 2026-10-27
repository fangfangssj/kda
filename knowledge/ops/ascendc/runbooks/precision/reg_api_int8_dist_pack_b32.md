---
type: "Runbook"
title: "Reg API int8 DIST_PACK_B32 交错零"
description: "Ascend950 Reg API 用 StoreAlign<int8_t, DIST_PACK_B32> 配 float 粒度 UpdateMask 做 fp32→int8 量化时输出每隔一字节插零；用小 shape debug 脚本打印 y[0] 与参考逐元素对照即可判别。"
tags: ["task.debugging", "topic.compute.quantization", "topic.implementation.api_layer.basic_api", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T15:55:13Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Reg API int8 DIST_PACK_B32 交错零

## 现象

在 Ascend950（dav-3510）上用 ops-direct-invoke-flash 从骨架优化 `AddRmsNormDynamicQuant`（Add + RmsNorm + AbsMax + 动态量化融合）算子。首版把 int8 量化写进一个 `__simd_vf__` Reg API 函数：对 fp32 buffer 做 `Muls(invScale)` → `Reg::Cast<half,float>` → `Reg::Cast<int8_t,half>` → `Reg::StoreAlign<int8_t, Reg::StoreDist::DIST_PACK_B32>`，mask 用 `Reg::UpdateMask<float>(rem)`（float 元素粒度）。kernel 编译通过、xOut/scaleOut 全对，但 int8 输出 `y` 严重错位：每隔一个有效值插一个 0——`[value, 0, value, 0, ...]`，像是每个 int8 元素被按 4 字节（float 宽度）间距落盘，而不是 1 字节紧挨。

## 适用场景

- **工程形态**：ops-direct-invoke-flash Kernel 直调核函数构建（`op_kernel/*.cpp` 内 `__simd_vf__` Reg API + `asc_vf_call` 调用）。
- **API 路径**：`AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK_B32>` + `AscendC::Reg::UpdateMask<float>` + `AscendC::Reg::Cast<int8_t, half, ...>`；对照路线为经典 `AscendC::Muls/Maxs/Mins` + `AscendC::Cast<half>` + `AscendC::Cast<int8_t>`。
- **验证过的平台与版本**：Ascend950（dav-3510），CANN 版本未在 trace 中显式记录（评测在 device 0 上板跑通）。
- **适用粒度**：`operator_family`——任何在 Ascend950 Reg API 里做 fp32→int8 量化落盘的算子。
- **路线优先级**：`recommended_fastpath`——经典 AscendC Cast 链（verbatim 修复代码未从 trace 恢复）。
- **不适用或未验证场景**：a2/a3 平台未验证，推断可迁移；非 Ascend950 Reg API 编程模型不适用。

## 判别方法

### 触发（trigger）

用 Reg API 链（`Reg::Cast<int8_t, half>` → `Reg::StoreAlign<int8_t, DIST_PACK_B32>`）做 fp32→int8 量化落盘，且整条链路的 mask 与寄存器元素粒度挂在 float 上（`UpdateMask<float>(rem)`、`RegTensor<float>` 输入）。kernel 编译通过、fp32/half 中间结果与 scaleOut 全对，唯独 int8 输出 `y` 大面积错位：mismatch 469/512（91.6%）、max_diff=214，`y[0]` 呈 `[-13, 0, 17, 0, 14, 0, -11, 0, ...]` 的交错零形态。

### 探针（probe）

用小 shape（M=8 N=64）fp16 随机输入跑 debug 脚本，单独打印 int8 输出 `y[0]` 与 torch 参考实现逐元素对照，并统计 mismatch 数与 max_diff。重点看 `y[0]` 是否呈 `[value, 0, value, 0, ...]` 的交错零（每隔一字节插零），以及 mismatch 比例是否远高于舍入边界（>50% 而非个位数）。xOut/scaleOut 全对、仅 int8 `y` 错位即可锁定到 int8 落盘环节。

### 观测（observation）

- 修复前 debug 脚本 stdout：`y[0]: [-13, 0, 17, 0, 14, 0, -11, 0, -24, 0, -2, 0, 75, 0, -7, 0, 32, 0, 6, 0]`，参考 `[-13, 17, 14, -11, -24, -2, 75, -7, 32, 6, ...]`，`y mismatch: 469/512 (91.6%)`，`y max_diff: 214`。每隔一个有效值插一个 0。
- 修复后（改走经典 AscendC Cast 链）debug 脚本 stdout：`y mismatch: 3/512 (0.6%)`，`y max_diff: 1`，`y[0]: [-13, 17, 14, -11, -24, -2, 75, -7, 32, 6, -7, -24, 19, 30, 18, 3, 4, -78, -2, 18]`，与参考逐元素一致；残留 3/512 mismatch、max_diff=1 属 int8 舍入边界差。

### 判读（interpretation）

`StoreAlign<int8_t, DIST_PACK_B32>` 的 pack 分布与整条链路的 mask/寄存器元素粒度（float，4 字节）不匹配：寄存器里 int8 数据没有按 1 字节紧挨 pack 到目标 UB，而是每个 int8 结果值按 float 元素间距（4 字节）写入，相邻 3 字节被零填充，落盘后在按 int8 视角读取时即 `[value, 0, value, 0, ...]` 的交错零。即 **int8 store 的 pack 分布与 mask/寄存器链的 float 元素粒度不匹配**是交错零的根因。注意：trace 未做对照实验隔离"改 mask 为 `UpdateMask<int8_t>`"与"换经典 Cast"两个变量，agent 直接整体替换为经典 AscendC Cast 链后精度恢复，因此 mask 粒度 vs pack 分布谁是主因未单独验证；此处判读为 agent 的根因分析，非单独探针验证的结论。

## 验证信号

- 修复前 debug stdout：`y[0]: [-13,0,17,0,14,0,-11,0,-24,0,-2,0,75,0,-7,0,32,0,6,0]`，参考 `[-13,17,14,-11,-24,-2,75,-7,32,6,...]`，`y mismatch 469/512 (91.6%)`，`max_diff=214`，呈交错零（来源平台：950；证据等级：tool_observed）
- 修复后 debug stdout：`y mismatch 3/512 (0.6%)`，`max_diff=1`，`y[0]` 与参考逐元素一致（来源平台：950；证据等级：tool_observed）
- kernel 编译通过、xOut/scaleOut 全对，仅 int8 输出 `y` 错位（来源平台：950；证据等级：tool_observed）
