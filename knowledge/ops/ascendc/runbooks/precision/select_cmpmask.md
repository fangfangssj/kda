---
type: "Runbook"
title: "Select cmpmask 不偏移致分支反转诊断"
description: "arch35/dav_c100 上 Select(VSEL_CMPMASK_SPR) 在 count>128 时 L1 出现约 1.2% element mismatch 与 mask 分支反转；通过 n 规模扫描与 VselImpl 源码读取定位为 repeatRounding 循环中 sel 指针不偏移、后续 repeat 复用首块 cmpmask。"
tags: ["task.debugging", "topic.compute.elementwise.comparison_logic", "topic.implementation.api_layer.basic_api", "topic.implementation.masking", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Select cmpmask 不偏移致分支反转诊断

## 现象

在 Ascend950PR（arch35/dav_c100，CANN 9.0.0-beta.2）上调试 `softplus_v2_grad` 算子精度：L0 全部通过（51/51），但 L1 功能测试 115/578 失败。典型失败用例 L1_008（beta=-781.95，threshold=-0.0052，fp16，1204224 元素）约 1.2% element mismatch：对于 `z = beta*self > threshold` 的元素（应走线性分支返回 grad_output），kernel 返回了 sigmoid 分支值（约 0.5×grad_output）；反之 `z <= threshold` 的元素（应走 sigmoid）返回了 grad_output（线性）。失败率约 1.2%，数据相关（随机种子不同时失败位置变化）。fp32/fp16/bf16 均复现，排除 cast 问题。小规模通过、大规模失败，排除纯公式错误。

## 适用场景

- **工程形态**：kernel 直调（op_kernel/arch35 ComputeFp32Core 内部使用 `AscendC::Compare` + `AscendC::Select` 做二分支选择）。
- **API 路径**：`AscendC::Select<float, uint8_t>(dst, mask, src0, src1, SELMODE::VSEL_CMPMASK_SPR, count)`；底层 `kernel_operator_vec_cmpsel_impl.h` 的 `VselImpl`/`VselIntrinsicsImpl`/`set_cmpmask`。
- **验证过的平台与 CANN 版本**：Ascend950PR（arch35/dav_c100），CANN 9.0.0-beta.2；`platforms: [950]`。
- **适用粒度**：`operator_family`（任何在 arch35/dav_c100 上用 `Select(VSEL_CMPMASK_SPR)` 且 count 可能超过 128 fp32 元素的算子均受影响）。
- **不适用或未验证场景**：a2/a3 未验证；count ≤ 128 fp32 元素时原 Select 可正常工作；非 dav_c100 架构（如支持 VSEL_TENSOR_TENSOR_MODE 的其他 arch）未验证不受此问题影响。

## 判别方法

**触发（trigger，tool_observed）**：arch35/dav_c100 上以 `Select(VSEL_CMPMASK_SPR)` 做逐元素二分支选择的 kernel，L1 功能测试出现约 1.2% element mismatch 且失败位置呈周期性、数据相关；线性/sigmoid 分支返回值互换（mask 分支反转）。fp32/fp16/bf16 均复现，小规模通过、大规模失败。

**探针（probe，可执行检查）**：

1. **规模扫描**：固定一组失败用例的输入分布，按元素数 n 递增跑 L1 单算子用例（n=8/64/128/129/256…），记录 mismatch 数随 n 的变化。
2. **掩码—结果一致性比对**：在 `Compare` 写入的 bMask 张量与 `Select` 输出上各取同一元素下标，逐元素核对 mask bit 与实际选中分支是否一致（PRINTF dump bMask 逐 bit 与 `bx > threshold` 比对，再 dump Select 输出与 bMask 比对）。
3. **框架源码读取**：读 `kernel_operator_vec_cmpsel_impl.h` 的 `VselImpl`（约 line 216-268）与 `TSel.hpp` 的 `cmpmaskLen` 注释，确认 repeatRounding 循环中各指针的偏移情况与 cmpmask 加载宽度。

**观测（observation，tool_observed）**：

1. **规模阈值**：n=8/64/128 通过（0 mismatch），n=129 起 2 mismatches，n>128 持续失败——经验边界 128 fp32 元素。失败位置间距约 64（256B/4B=64 fp32 元素），指向 64 元素对齐边界。
2. **bMask 正确、Select 不一致**：`Compare` 写入的 bMask 张量逐 bit 与 `bx > threshold` 一致（正确）；但 `Select` 结果与 bMask 不一致——位于非首 64 元素块的元素，其 mask bit 与首块对应位置相反时，Select 返回了相反分支的值（mask 分支反转）。
3. **dav_c100 Select 仅支持 `VSEL_CMPMASK_SPR`**：其他模式 `ASCENDC_ASSERT false`，无法切换到从张量读 mask 的模式。
4. **VselImpl 在 repeatRounding 循环中不偏移 `sel` 指针**：每次 `VselIntrinsicsImpl(dst + dstOffset, sel, src0 + src0Offset, src1 + src1Offset, ...)` 中 `dst`/`src0`/`src1` 均按 `dstOffsetCount` 偏移，唯独 `sel` 始终传原始指针；`VselIntrinsicsImpl` 在 `VSEL_CMPMASK_SPR` 模式下调用 `set_cmpmask(sel)` 从该地址加载 cmpmask 寄存器再执行 `vsel`。

**判别（interpretation，根因归因）**：单一根因——`VselImpl` 的 repeatRounding 循环不偏移 `sel` 指针，导致当 count 超过单次 `set_cmpmask`+`vsel` 能正确覆盖的宽度时，后续 repeat 始终从 mask 张量首块加载 cmpmask，非首块元素的分支选择被首块 mask bit 覆盖，从而 mask 分支反转。这与“bMask 正确但 Select 结果不一致”“n 超阈值才失败”“失败间距约 64 元素”三项 tool_observed 观测自洽。判别要点：若 bMask dump 正确而 Select 输出在非首块位置发生分支互换、且 failure 随 n 跨过约 128 元素阈值出现，即可判定为本机制，而非公式错误、cast 问题或广播搬运问题（均已被观测排除）。

## 验证信号

判别所依据的实测信号（来源平台：950）：

- L0 51/51 通过、L1 115/578 失败，典型用例约 1.2% element mismatch（来源平台：950；证据等级：tool_observed）
- 规模扫描：n=8/64/128 通过（0 mismatch）、n=129 起 2 mismatches、n>128 持续失败，失败间距约 64 fp32 元素（来源平台：950；证据等级：tool_observed）
- 框架源码读取：dav_c100 Select 仅支持 `VSEL_CMPMASK_SPR`（其他模式 `ASCENDC_ASSERT false`）（来源平台：950；证据等级：tool_observed）
- 框架源码读取：`VselImpl` repeatRounding 循环中 `sel` 指针不随 repeat 偏移，`dst`/`src0`/`src1` 均按 `dstOffsetCount` 偏移（来源平台：950；证据等级：tool_observed）
- 判别后据此改用纯算术掩码折叠分支，L1 失败从 115/578 降至 58/578（57 个 mask 分支反转用例全部修复、0 回归）、C++ ST 57/57 通过，反证判别正确（来源平台：950；证据等级：tool_observed）

剩余 58 个失败均为极端值边界用例（±FLT_MAX/inf/nan/subnormal threshold），在原始 baseline 同样失败，为独立问题，非本机制。