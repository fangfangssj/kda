---
type: "Runbook"
title: "arch35 分块 Select 掩码偏移非 32B 对齐崩溃 507035"
description: "为绕过 VselImpl 不偏移 sel 指针的问题，尝试分块 Select(SEL_CHUNK=64) 用 bMask[off>>3] 按 bitmask 偏移 mask 张量；第二块起 mask 源地址落 byte 8 非 32B 对齐，set_cmpmask 加载致 AIC runtime error 507035 崩溃。n=64 单块通过、n>=128 多块崩溃。banned 路线——对齐与 cmpmask 宽度约束互斥，无分块粒度可同时满足。"
tags: ["task.debugging", "topic.implementation.masking", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# arch35 分块 Select 掩码偏移非 32B 对齐崩溃 507035

## 现象

在 softplus_v2_grad 精度调试中，为绕过 `VselImpl` 不偏移 `sel` 指针的问题，尝试把单次 `Select(VSEL_CMPMASK_SPR)` 改为分块循环：每块 64 个 fp32 元素（= cmpmask 宽度），用 `bMask[off >> 3]`（bitmask 偏移，每 64 元素 = 8 字节）偏移 mask 张量，`bBx[off]`/`bG[off]`/`bThr[off]`（元素偏移，每 64 元素 = 256 字节）偏移数据张量。构建通过，但运行时 NPU 崩溃报 `507035`（AIC runtime error）。n=64（单块）通过，n>=128（多块）崩溃。

## 适用场景

- **工程形态**：kernel 直调（op_kernel/arch35 ComputeFp32Core 内部 `Select` 调用）
- **API 路径**：`AscendC::Select` + `LocalTensor::operator[]` 偏移 + `set_cmpmask` 内在
- **验证过的平台与 CANN 版本**：Ascend950PR（arch35/dav_c100），CANN 9.0.0-beta.2；`platforms: [950]`
- **适用粒度**：`operator_family`（任何在 arch35 上尝试分块 `Select(VSEL_CMPMASK_SPR)` 的算子）
- **路线优先级**：`banned`（此路线因对齐约束崩溃，不可用）
- **不适用或未验证场景**：a2/a3 未验证；若目标 arch 的 `set_cmpmask` 不要求 32B 对齐则可能可行，未验证

## 根因

`set_cmpmask` 内在从 UB 地址加载 cmpmask 寄存器，要求源地址 **32 字节对齐**（经验推断）。TBuf 起始地址天然 32B 对齐，但分块偏移 `bMask[off >> 3]`：

- off=0：`bMask[0]`，byte 0，32B 对齐（首块通过）
- off=64：`bMask[8]`，byte 8，**非 32B 对齐**（第二块崩溃）
- off=128：`bMask[16]`，byte 16，非 32B 对齐

64 fp32 元素的 packed bitmask 仅 8 字节（64 bits），而 32B 对齐要求偏移必须是 32 字节的倍数 = 256 bits = 256 个 fp32 元素的 mask。但 cmpmask 宽度仅 64~128 元素，256 元素块超出单次 `set_cmpmask`+`vsel` 覆盖范围，后续 repeat 仍用错误 mask。即：**分块粒度（64 元素，8B mask）与对齐要求（32B）不可同时满足**——64 元素 mask 8B 无法 32B 对齐，而 32B 对齐的 256 元素块又超 cmpmask 宽度。此为架构级约束，无分块解法。

## 失败路线

**路线：分块 Select（SEL_CHUNK=64，bMask[off>>3] 偏移）** — banned

- **尝试**：循环 `for (off = 0; off < count_padded; off += SEL_CHUNK)` 调用 `Select(bBx[off], bMask[off>>3], bG[off], bThr[off], VSEL_CMPMASK_SPR, SEL_CHUNK)`，期望每块用各自偏移的 mask 与数据复用同一 cmpmask 宽度。
- **失败**：运行时 `507035`（AIC runtime error），NPU 崩溃；目标（绕过 `VselImpl` 不偏移 sel）未达成。
- **边界**：n=64（单块）通过；n>=128（多块）崩溃——崩溃出现在第二块 `bMask[8]` 起。
- **归因**：`set_cmpmask` 要求 32B 对齐源地址，`bMask[8]`（byte 8）不满足；64 元素 mask（8B）与 32B 对齐不可兼得（推断，未从内在实现层确认）。
- **回退**：弃用 `Select`，改用纯算术掩码构造 `ratio = (mask_f + exp) / (1 + exp)`，不再触碰 `Select`/`set_cmpmask`。

## 验证信号

本卡验证信号来自 softplus_v2_grad 在 Ascend950PR（arch35/dav_c100）上的多轮 kernel 运行（同一开发轨迹）。

- 分块 `Select(SEL_CHUNK=64)` 多块调用崩溃 `507035`（AIC runtime error），NPU 崩溃（来源平台：950；证据等级：tool_observed）。
- n=64 单块通过、n>=128 多块崩溃，崩溃出现在第二块 mask 偏移起（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `platforms: [950]`：仅 Ascend950PR（arch35/dav_c100）实测。
- a2/a3：未验证，推断同样受限（同一 `set_cmpmask` 内在）。
- 若目标 arch 的 `set_cmpmask` 对齐要求不同（如 8B 对齐），分块 64 元素可能可行——需独立验证。

## 坏实践

- 用 `LocalTensor::operator[]` 对 packed bitmask（uint8_t）按字节偏移后再传 `Select`——不检查 `set_cmpmask` 的地址对齐要求。
- 在 `set_cmpmask` 内在的源地址上用非 32B 倍数的元素/字节偏移做分块。

## 设计决策

- **为何放弃分块路线**：64 元素 mask（8B）无法 32B 对齐；256 元素块虽 32B 对齐但超 cmpmask 宽度。对齐与宽度约束互斥，无分块粒度可同时满足。
- **为何 banned 而非 safety_fallback**：此路线在 n>64 时必崩，无安全降级空间。