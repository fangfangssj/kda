---
type: "Runbook"
title: "奇数 H 致 B 视图 UB 地址非 32B 对齐"
description: "bf16/fp16 路径在 H 为奇数时对 B 视图或 Cast 输出做 VEC 计算会触发 UB 地址未 32 字节对齐的向量核异常；根因是 B 视图起始字节偏移 H×dtype_size 不满足 32 字节对齐。"
tags: ["task.debugging", "topic.alignment", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-18T02:30:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# 奇数 H 致 B 视图 UB 地址非 32B 对齐

## 现象

在 Ascend 910B3（NpuArch `DAV_2201`，CANN 9.0.0，bisheng 编译器）上开发 `dequant_swiglu_quant` 融合算子（Dequant + SwiGLU + 逐 token 动态 int8 量化）。算子输入 x 支持 int32/bfloat16/float16，输出 y:int8 + scale:float32。kernel 把一行 x（长度 `twoH = 2*H`，A、B 两段各 H）整体 `DataCopyPad` 装入一个 UB 缓冲，`Cast` 成 float32 后，用 `LocalTensor<float> B = fp32[H_]` 取后半段做 SwiGLU（Silu × Mul）、Abs、`ReduceMax` 等向量计算。

对齐 shape 的前 6 个 cann-bench 用例（H=128/256/512 等 8 的倍数）全部精确通过。但运行到 case 7（shape `[1023, 4098]`，bf16，H=2049）和 case 8（shape `[255, 8194]`，bf16，H=4097）时，NPU 抛出**向量核异常（vector core exception）**，运行时报错原文为"UB地址被VEC指令访问时未对齐"，并导致设备崩溃、后续用例无法继续。H 为奇数（2049、4097）是对齐用例与崩溃用例的唯一差异。

## 适用场景

- **工程形态**：kernel 直调（cann-bench `csrc/ops/<op>/op_kernel/` bisheng kernel，`register_direct_launch_op` 注册）。
- **API 路径**：`DataCopyPad`（GM→UB）、`Cast`（bf16/fp16→float32）、`Silu`/`Mul`/`Abs`/`ReduceMax` 等 VEC 向量指令。
- **验证过的平台与 CANN 版本**：Ascend 910B3，NpuArch `DAV_2201`，CANN 9.0.0，bisheng 编译器，`--npu-arch=dav-2201`。对应注册平台 `a2`（Atlas A2，Ascend910B1~B4 族）。
- **适用粒度**：`operator_family`——凡是在 UB 内对"一段连续加载的 float32 数据按 H 偏移取后半段/做 Cast"的逐行 kernel 都适用，不限于本算子。
- **路线优先级**：`recommended_fastpath`——把 A、B 拆进两个独立 32 字节对齐 UB 缓冲、分别从 GM 加载，是对齐用例和奇数 H 用例都通过的推荐做法。
- **不适用或未验证场景**：a2 之外的平台（950、a3）未在本 trace 验证，推断可迁移但未实测（见 `## 失效边界`）；int32 路径 `sizeof(int32_t)==4`，A 段偏移 H×4 当 H 为 8 的倍数时天然对齐，本卡不针对 int32 的对齐失败（int32 用例在本 trace 全部通过）。

## 判别方法

**触发条件**：kernel 把 `twoH = 2*H` 个元素整体加载进单个 UB 缓冲，再用形如 `fp32[H_]` 的视图取后半段（B 段）做 VEC 计算，且 H 不为 `32/sizeof(element)` 的整数倍。

**可执行探针**：构造一个 bf16、H 为奇数（如 H=2049，对应 `twoH=4098`）的 cann-bench 用例，运行 cann-bench 评测脚本；观察是否在 VEC 指令处触发向量核异常。

**原始观测**（本 trace cann-bench 首轮评测）：
- 前 6 个用例（bf16/fp16，H 为 8 的倍数）精确通过；
- case 7 `[1023, 4098]` bf16（H=2049）、case 8 `[255, 8194]` bf16（H=4097）触发向量核异常，运行时报错原文："UB地址被VEC指令访问时未对齐"（来源平台：a2；证据等级：tool_observed）。

**机制解释（含算式）**：

VEC 类指令（Silu/Mul/Abs/ReduceMax/Cast 等）要求访问的 UB 地址 32 字节对齐。把 A、B 装进同一段连续 float32 缓冲后，B 视图起始字节偏移 = `H_ × sizeof(float) = H_ × 4`。当 H_=2049 时：`2049 × 4 = 8196`，`8196 / 32 = 256.125`——**不是整数**，故 B 视图起始地址未 32 字节对齐，VEC 指令一访问即触发向量核异常。

bf16 Cast 路径同理：对 `xLocal`（bf16 LocalTensor）做 `Cast(fp32, xLocal, ...)` 时，若先取 `xLocal[H_]` 再处理，`H_=2049` 的字节偏移 = `2049 × 2 = 4098`，`4098 / 32 = 128.0625`，同样未对齐。

推荐做法：A、B 拆进两个独立 UB 缓冲（各自按 `AlignUp(H, 8)` 分配，天然 32 字节对齐），从 GM 分别加载 A 段（`xGm[row*2H]`，长度 H）与 B 段（`xGm[row*2H + H]`，长度 H），后续 VEC 计算各自落在对齐缓冲上。

## 验证信号

本卡验证来自本 trace 的 cann-bench 评测实测：
- 奇数 H 用例（H=2049、H=4097）修复前触发"UB地址被VEC指令访问时未对齐"向量核异常（来源平台：a2；证据等级：tool_observed）。
- 对齐用例（H 为 8 的倍数）首轮即精确通过、未触发异常（来源平台：a2；证据等级：tool_observed）。

## 失效边界

- 本 trace 仅在 a2（910B3/DAV_2201）实测。950、a3 未验证，**推断可迁移**（VEC 32 字节对齐约束是 DAV_2201 微架构级要求，同族 NpuArch 大概率一致），但未在本 trace 上板验证。
- 对齐粒度随 dtype 变化：float32 需 H 为 8 的倍数（32B/4B=8），bf16 需 H 为 16 的倍数（32B/2B=16），int8 需 H 为 32 的倍数。判定时按 `32 / sizeof(element)` 取对齐粒度。
- 若 GM 布局允许，也可不拆缓冲而把 A 段 pad 到对齐再放 B 段；但 pad 会改变搬运长度，需同步调整 `DataCopyPad` 的 `blockLen`/`DataCopyExtParams`，本 trace 未采用此变体。

## 坏实践

- 把 A、B 装进同一段连续 UB 缓冲、用 `fp32[H_]` 视图取后半段做 VEC 计算——当 H 不对齐时必然触发向量核异常。
- 仅用 `AlignUp(H, 8)` 做"缓冲分配对齐"却仍把 VEC 计算的 `count` 传 `H_align_`、地址却落在未对齐的 `H_` 偏移上——对齐的是长度不是地址，不解决问题。
- 用 `H_`（有效长度）而非 `H_align_` 作 VEC `count` 也不能修对齐崩溃，因为崩溃根因是**地址未对齐**而非长度未对齐。

## 误诊

- 本 trace 初判曾怀疑"Silu/Mul 等 API 的 count 对齐要求"或"UB 容量超额"。经核算 UB 总占用约 62KB（远低于 192KB 上限），且对齐用例同样用 `H_align_` 作 count 却通过——排除了容量与 count 对齐假设，定位到**地址**未对齐才是根因。
- 曾怀疑 ReduceMax workspace 不足；但 workspace 按 `H_*sizeof(float)+4096` 已足够，且崩溃发生在 Cast/Silu 阶段而非 ReduceMax，故排除。