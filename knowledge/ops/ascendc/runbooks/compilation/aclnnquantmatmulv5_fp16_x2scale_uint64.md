---
type: "Runbook"
title: "aclnnQuantMatmulV5 FP16 x2Scale 须 UINT64"
description: "Ascend950/DAV_3510 上 aclnnQuantMatmulV4/V5 的 L0C→out 直出路径对 int8 输入要求 x2Scale 为 DT_UINT64/DT_INT64，传 DT_FLOAT 返回 161002；用 ASCEND_SLOG_PRINT_TO_STDOUT 探针可定位到 CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input checker。bf16 输出走 L0C→UB 路径不受此约束。"
tags: ["task.debugging", "topic.compute.quantization", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# aclnnQuantMatmulV5 FP16 x2Scale 须 UINT64

## 现象

在 Ascend950（DAV_3510/arch35，CANN 9.2.0）上为 `bench_no_harness_quant_matmul` 算子开发 L2 API：算子输入为 int8 x1/int8 x2、fp32 scale、可选 offset/bias/pertokenScale、输出 fp16/bf16。自定义 Cube kernel 在该芯片上失败后，改为在 L2 API 层直接委托 CANN 内置 `aclnnQuantMatmulV4`（后改 V5），即 `aclnnBenchNoHarnessQuantMatmulGetWorkspaceSize` 内部调用 `aclnnQuantMatmulV4GetWorkspaceSize`/`aclnnQuantMatmulV5GetWorkspaceSize`。

构建安装后运行 C++ ST（`bash run.sh --real`）首次结果仅 35/101 通过，66 个用例失败，失败信息均为 `GetWorkspaceSize 失败: ret=161002`。失败集中在 FP16 输出 + fp32 scale 的组合；bf16 输出 + fp32 scale 的用例通过。

## 适用场景

- **工程形态**：aclnn L2 API 委托内置算子（op_api 侧 `aclnnXxxGetWorkspaceSize` 转调内置 `aclnnQuantMatmulV4/V5`），非 kernel 直调。
- **API 路径**：`aclnnQuantMatmulV4GetWorkspaceSize` / `aclnnQuantMatmulV5GetWorkspaceSize`，参数 `x2Scale`（即算子的 scale 量化参数）。
- **验证平台与版本**：Ascend950（DAV_3510/arch35），CANN 9.2.0。
- **适用粒度**：`operator_family`（int8 量化 matmul 族，凡走 aclnnQuantMatmulV4/V5 L0C→out 直出路径均适用）。
- **不适用/未验证场景**：a2/a3 未验证；bf16 输出走 L0C→UB 路径，不受此 UINT64 约束（本 trace 内 bf16+fp32 scale 通过）。

## 判别方法

**触发**：L2 API 委托 `aclnnQuantMatmulV4/V5` 后，FP16 输出用例大面积返回 161002（`GetWorkspaceSize 失败: ret=161002`），bf16 输出用例通过。属系统性 dtype 约束错配，非单 case 问题。

**可执行探针**（在算子 ST 目录运行，逐用例打印 SLOG 校验失败原因）：

```bash
cd <operator_dir>/tests/st && ASCEND_SLOG_PRINT_TO_STDOUT=1 bash run.sh --real 2>&1 | grep -i "161002\|ERROR.*quant\|ERR.*param\|not support\|invalid" | head -20
```

**原始观测**（SLOG 输出，逐字）：

```
Parameter x2Scale(scale) of aclnnQuantMatmulV4GetWorkspaceSize has incorrect dtype DT_FLOAT. Reason: the dtype of x2Scale(scale) must be in dtype support list [DT_UINT64,DT_INT64,].
```

错误源自 checker `CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input`（见 SLOG 调用栈）。改用 V5 后同一 checker 同一约束仍然生效：

```
the dtype of x2Scale(scale) must be in dtype support list [DT_UINT64,DT_INT64,]
```

即 V4/V5 在该芯片上对 int8 输入走 L0C→out 直出路径时，强制 x2Scale 必须为 `DT_UINT64` 或 `DT_INT64`，不接受 `DT_FLOAT`。

**解释**：aclnnQuantMatmulV4/V5 内部按输入/输出 dtype 选两条路径——
- **L0C→out 直出**：int8 输入 + FP16 输出（输出 dtype 在 int8 之外）走此路径，要求 scale 为硬件量化格式 UINT64/INT64，fp32 scale 被拒；
- **L0C→UB 向量路径**：bf16 输出走此路径，接受 fp32/bf16 scale，因此 bf16+fp32 scale 用例通过。

这是芯片/路径级硬约束，不是 API bug，换 V5 不能绕过（V4/V5 共用同一 checker）。

## 验证信号

- 委托 V4 首次 ST 35/101 通过、66 个用例返回 161002（来源平台：950；证据等级：tool_observed）。
- `ASCEND_SLOG_PRINT_TO_STDOUT=1` 探针输出 `x2Scale(scale) ... must be in dtype support list [DT_UINT64,DT_INT64,]`、checker `CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input`（来源平台：950；证据等级：tool_observed）。
- 改用 V5 后同一 checker 同一约束仍触发 161002（来源平台：950；证据等级：tool_observed）。
- 将 fp32 scale 转为 uint64 后 ST 升至 71/101，剩余 30 例为 bias/offset 约束（来源平台：950；证据等级：tool_observed）。

## 失效边界

- a2/a3 平台未验证，推断可迁移（V4/V5 为 CANN 内置算子，checker 行为可能随芯片代际不同；上线前须在目标芯片用同一 SLOG 探针复核）。
- bf16 输出走 L0C→UB 路径不受此约束；FP16 输出但用 int64/uint64 scale 传入则不触发（本 trace 已验证 uint64 scale 通过）。
- 转换后剩余 30 例仍失败（bias 必须 int32、offset 须打包、offset 与 pertoken 互斥），属另一组约束，不在此卡范围。

## 坏实践

- 直接把 fp32 scale 传给 V4/V5 的 x2Scale 而不做 dtype 适配，在 950 FP16 输出下必然 161002。
- 误以为换 V5 就能绕过 V4 的 scale dtype 约束——本 trace 实测 V4/V5 共用同一 `CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input` checker，约束一致。

## 误诊

- 初始误判 161002 为"V4 不支持某些参数组合"的笼统参数错误，未区分 L0C→out 与 L0C→UB 路径差异；SLOG 探针后才定位到 x2Scale dtype 与具体 checker。此误诊过程说明：遇到 161002 时不应笼统归因为"参数组合不支持"，须用 SLOG 探针取逐字 checker 名与 dtype 约束原文再判路径。