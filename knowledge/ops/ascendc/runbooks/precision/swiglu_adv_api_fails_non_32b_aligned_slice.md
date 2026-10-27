---
type: "Runbook"
title: "SwiGLU 高级 API 在非32B对齐 slice 上精度失败"
description: "AscendC::SwiGLU 高级 API（adv_api/activation/swiglu.h）在 dav-2201 上对 slice*sizeof(T) 非 32B 整数倍的 case 产生大错值（fp32 slice=509 MARE≈2.5e9；fp16 全零 slice=506 MARE=93/65504），对齐 slice 正常。banned：非对齐 slice 不得用此 API，回退手写 SiLU 链。"
tags: ["task.debugging", "topic.compute.elementwise.activation", "topic.implementation.api_layer.basic_api", "topic.data_movement.memory_access", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# SwiGLU 高级 API 在非32B对齐 slice 上精度失败

## 现象

在 Ascend 910B3 (dav-2201, CANN 9.2.0) 上对 `direct_launch_example/csrc/ops/swi_glu` 做性能优化。优化版 kernel 在修复多块 DataCopyPad 的 dst 对齐与 padded 布局问题后，全 20 case 用手写 SiLU 链精度通过。随后按方案分支③ 启用 SwiGLU 高级 API（`#include "adv_api/activation/swiglu.h"`，`AscendC::SwiGLU(dst, src0, src1, beta, tmpBuffer, calCount)`，fp16/fp32 路径）做性能对照。

单独跑对齐 slice 的 case（case1 fp16 slice=1024/2048B 对齐、case2 fp32 slice=2048/8192B 对齐、case15 fp32 slice=2048 对齐）**通过**。但跑全 20 case 时崩或大错，定位到非 32B 对齐 slice 的 case：

- **case11**（fp32, shape [11,13,16,67], dim=-1, slice=509, `509*4=2036` 字节，非 32B 倍数）：MARE≈2.5e9（量级远超阈值）。
- **case14**（fp16, shape [3,7,11,13,1012], dim=-1, slice=506, value_range [0,0] 全零输入, `506*2=1012` 字节，非 32B 倍数）：MARE=93 / 出现 65504（fp16 max，提示溢出/垃圾）。golden 应为全零（silu(0)*0=0）。
- 对齐 slice 的 case（1/2/4/7/10/15/17）通过。

禁用 adv_api、回退手写 SiLU 链（`Muls(-1)→Exp→Adds(1)→Mul→Div→Cast`）后全 20 case 通过。

## 失败路线

**banned**：slice*sizeof(T) 非 32B 整数倍时禁用 `AscendC::SwiGLU` adv_api，必须回退手写 SiLU 链。

### 尝试（banned 路线，从 edit 碎片恢复）

启用 adv_api 的 edit（verbatim 取自 edit 的 newString）：

```cpp
// 分支③：SwiGLU 高级 API 开关（fp16/fp32 优先使用；编译/精度有问题时注释掉回退手写链）
#define SWI_GLU_USE_ADV_API 1
#if defined(SWI_GLU_USE_ADV_API)
#include "adv_api/activation/swiglu.h"  // 分支③：SwiGLU 高级 API（fp16/fp32）
#endif
```

调用点（reasoning 草稿描述的调用模式）：`AscendC::SwiGLU(out, x0, x1, beta, tmpBuffer, calCount)`，其中 `calCount = N = nBlocks * alignUpS`（padded compute 长度）。

### 失败/回退

- case11 (fp32 slice=509) MARE≈2.5e9，case14 (fp16 全零 slice=506) MARE=93 / 65504。
- 全 20 case 运行时部分 case 崩或大错。
- 回退：注释掉 `#define SWI_GLU_USE_ADV_API 1`，fp16/fp32 改走手写链（与 bf16 同路径）。

```cpp
// 手写 SiLU 链（回退，全 dtype 通用）
Muls(tf, x0f, -1.0f, (uint32_t)N);
Exp(tf, tf, (uint32_t)N);
Adds(tf, tf, 1.0f, (uint32_t)N);
Mul(pf, x0f, x1f, (uint32_t)N);
Div(outF, pf, tf, (uint32_t)N);
```

回退后全 20 case ALL PASS（CANN-Bench overall_score 仅作 provenance，不进正文）。

## 适用场景

- **工程形态**：kernel 直调 (direct_launch_example)，bisheng `-xasc --npu-arch=dav-2201`。
- **API 路径**：`AscendC::SwiGLU(dst, src0, src1, beta, tmpBuffer, calCount)`（adv_api 高级 API，fp16/fp32），内部用 `PopStackBuffer` 分配 tmp + `SetVectorMask` 分块处理。
- **验证过的平台与 CANN 版本**：Ascend 910B3 / dav-2201 / CANN 9.2.0（注册平台 token `a2`）。
- **适用粒度**：`operator_family`——任何调用 `AscendC::SwiGLU` adv_api 的算子路径。
- **路线优先级**：`banned`——slice*sizeof(T) 非 32B 整数倍时禁用此 adv_api，必须回退手写 SiLU 链。
- **不适用或未验证场景**：slice*sizeof(T) 为 32B 整数倍时 adv_api 正常工作（本 trace case1/2/15 验证通过），不在 ban 范围；DAV_3510/Ascend950 未验证。

## 根因

`AscendC::SwiGLU` adv_api 内部按 `SetVectorMask` + `PopStackBuffer` 分块处理 `calCount` 个元素。当 `calCount`（或其内部 stackSize 分块边界）与 32B 对齐边界不一致时——即输入 slice 的 `slice*sizeof(T)` 非 32B 倍数——内部 mask/分块逻辑在 dav-2201 上产生越界或未初始化访问，输出大错值（全零输入却出 65504 = fp16 max，强烈提示读到了未初始化 UB / 溢出）。

具体边界算式：
- case11 fp32 slice=509：`509 * 4 = 2036` 字节，`2036 % 32 = 2036 - 63*32 = 2036-2016 = 20 ≠ 0`。
- case14 fp16 slice=506：`506 * 2 = 1012` 字节，`1012 % 32 = 1012 - 31*32 = 1012-992 = 20 ≠ 0`。
- 通过的 case15 fp32 slice=2048：`2048 * 4 = 8192`，`8192 % 32 = 0` ✓。

> 注：adv_api 内部确切的 mask 错配机制未在本 trace 追到指令层（未读 adv_api 的 `SwiGLUImpl` 源码全貌），仅从外部观测定位到"非 32B 对齐 slice 失败"的边界。

## 验证信号

验证来源：本 trace 内 swi_glu_debug.py / swi_glu_validate.py 在 910B3 device 3 上的实测。

- adv_api 对齐 slice：case1/2/15 单跑通过（来源平台：a2；证据等级: tool_observed）
- adv_api 非对齐 slice：case11 MARE≈2.5e9、case14 MARE=93/65504（来源平台：a2；证据等级: tool_observed）
- 回退手写链：全 20 case ALL PASS（来源平台：a2；证据等级: tool_observed）

## 设计决策

1. **为何保留 adv_api include 与 `#if defined` 开关**：对齐 slice case 上 adv_api 可能更快（融合 SwiGLU 内核），保留开关便于后续在已验证对齐的场景下按 dtype 启用；但默认不启用（注释 `#define`）。
2. **为何不修 adv_api 而是回退**：adv_api 是 CANN 内置头（`/home/developer/Ascend/cann-9.2.0/aarch64-linux/asc/include/adv_api/activation/swiglu.h`），不可由算子侧修改；且手写链在全 20 case 已达精度目标，无性能缺口证据支撑冒险用 adv_api。

## 失效边界

- Ascend 950 / A3 (dav-3510) 未验证——adv_api 对非对齐 slice 的行为可能因架构不同而异，标注"未验证，推断可迁移（ban 可能仍成立）"。
- 本 ban 仅针对 `slice*sizeof(T) % 32 != 0` 的 case；对齐 slice 下 adv_api 可用。
- `calCount` 取 padded 长度（`nBlocks*alignUpS`）还是 valid 长度（`n`）对 adv_api 的影响未做隔离实验——可能 padded 长度放大了 mask 错配。

## 坏实践

- 默认启用 adv_api 跑全量 case 而不做逐 case 精度回归——非对齐 case 的失败被掩盖。
- 见到全零输入出 65504 却不查搬运/mask 而归因"输入分布问题"——实际是 adv_api 内部 mask/分块 bug。

## 误诊

- 初判怀疑 adv_api 崩溃是同 trace build-01 的 DataCopyPad dst 对齐问题所致——修复搬运后 adv_api 仍失败，证明 adv_api 的失败是**独立的内部 mask/分块问题**，与搬运 dst 对齐无关。
- 怀疑全零输入（value_range [0,0]）触发了 silu(0) 数值边界——golden 全零计算正确（手写链过），错在 adv_api 内部。

## 预防

- 用 `AscendC::SwiGLU` adv_api 前先判 `slice*sizeof(T) % 32 == 0`：是→可用；否→回退手写 SiLU 链。
- 任何 adv_api 高级算子（SwiGLU/GeGLU 等）接入新 case 集时，必须跑全量 case 精度回归，不能只跑对齐 case 即放行。
- 全零/极端值输入 case（value_range 单点）必须纳入回归集——易暴露 mask/初始化类 bug。