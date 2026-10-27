---
type: "Runbook"
title: "MulAddDst FMA 单次舍入与 golden Mul+Add 双舍入偏差"
description: "dav-3510 VF MulAddDst 走 FMA（a*b+c 无限精度后一次舍入），golden 的 Mul+Add 是两次独立舍入；当 addend 含 -round(a*b) 抵消项且 x==save_mean 时两者偏差 1 ULP 量级，须把 VF 改 Mul+LoadAlign(dst)+Add 分离舍入匹配 golden。"
tags: ["task.debugging", "topic.precision", "topic.programming_model.execution_unit.vector", "topic.compute.elementwise.arithmetic"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["round_to_nearest", "single_rounding"]
---

# MulAddDst FMA 单次舍入与 golden Mul+Add 双舍入偏差

## 现象

Ascend950（dav-3510，CANN 9.0.0-beta.2）上 `bn_training_update_custom` Branch-4 kernel 修复对齐问题后 15/18 通过，剩 3 个用例精度失败：`y = multiplier * x + addend` 的输出在 `x == save_mean` 的通道上与 golden 偏差 1 个 fp16 ULP 量级，统计输出（fp32）已通过。Phase 2 用 VF 函数 `MulAddDstVF`（`__simd_vf__` + `asc_vf_call`，内部 `AscendC::Reg::MulAddDst`）计算 `y = addend + x * multiplier`。golden（C++ 独立参考）用 `multiplier * x` 再 `+ addend` 两次独立 fp32 舍入。

## 适用场景

- **工程形态**：kernel 直调，dav-3510 Phase 2 broadcast `y = multiplier*x + addend` 用 `__simd_vf__` + `asc_vf_call` 调 `MulAddDst` 系列融合指令。
- **API 路径**：`AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask)`（dst = src0·src1 + dst，FMA 单次舍入）；替代为 `Reg::Mul` + `Reg::LoadAlign(dst)` + `Reg::Add` 分离序列。
- **验证过的平台与 CANN 版本**：Ascend950 / dav-3510 / CANN 9.0.0-beta.2。
- **适用粒度**：`operator_family`——凡 broadcast/stateful 算子 Phase 2 用 `MulAddDst` 且 addend 含 `-round(a*b)` 抵消项（BN update/eval、LayerNorm 等）。
- **路线优先级**：`recommended_fastpath`——addend 含抵消项时用 `Mul+Add` 分离舍入匹配 golden；纯 FMA 仅在算子规范明确接受 FMA 语义时用。
- **不适用或未验证场景**：算子规范本身定义 FMA 语义（单次舍入）时不适用（此时 golden 也应 FMA）；非抵消型 addend（独立常数）时 FMA 与 Mul+Add 通常等价，未验证差异。

## 判别方法

- **触发**：用 `MulAddDst`（FMA）计算 `y = multiplier*x + addend`，且 `addend = offset - round(multiplier * save_mean)` 含抵消项。run.sh 上板输出显示部分用例 FAIL，diff 为 1 ULP 量级的小 max_diff（如 bf16 `0.015625`、fp16 `2.05469` 偏差、fp32 `8e-5`），统计输出（fp32）通过。编译过、无 507035。
- **可执行探针**：核对失败用例是否满足三条件同时成立——(a) kernel 用 `MulAddDst`/FMA 融合指令；(b) `addend` 含 `-round(a*b)` 形态的抵消项；(c) 失败仅出现在 `x == save_mean`（抵消应当精确为零）的通道、统计输出（fp32 中间结果）已通过。三条件齐备即命中本规则。也可临时把 VF 内部从 `MulAddDst` 改为 `Mul`+`LoadAlign(dst)`+`Add` 重跑：若原失败用例转 PASS 即坐实。
- **原始观测**：15/18 通过、3 用例 FAIL（case_03 bf16 3/120 元素 diff=0.015625、case_07 fp32 diff=8e-5、case_19 fp16 120/120 diff=2.05469，统计输出已通过）；把 `MulAddDstVF` 内部由 `MulAddDst` 改为 `Mul`+`LoadAlign(dst)`+`Add` 分离舍入后，case_07 与 case_19 转为 PASS（17/18 通过，剩 case_03 为独立 bf16 Cast 舍入问题另见他卡）。
- **解释（根因归因）**：`MulAddDst` 是 FMA：`dst = src0 * src1 + dst`，乘积 `src0*src1` 在无限精度计算后与 `dst` 相加、一次性舍入到 fp32。golden 的 `Mul` + `Add` 是两次独立舍入。设 `p = multiplier * save_mean`，`addend = offset - round(p)`：
  - 非 FMA（golden）：`y = round(round(p) + offset - round(p)) = round(offset) = offset`，精确抵消。
  - FMA（kernel `MulAddDst`）：`y = round(offset - round(p) + p_inf)`，`p_inf - round(p)` 为 `p` 的舍入残差（非零），FMA 把该残差并入加法后一次舍入 → `y = round(offset + 残差) ≠ offset`，残留 `p - round(p)` 量级偏差。
  修复方向：把 VF 函数体由 `MulAddDst` 改为 `Mul` + `LoadAlign(dst)` + `Add` 三步分离舍入（`Mul` 先舍入乘积、`Add` 再舍入和），匹配 golden 的双舍入语义；`asc_vf_call<MulAddDstVF<float>>` 调用签名不变。

## 预防

- 凡 addend 含 `-round(a*b)` 抵消项且规范要求匹配非-FMA golden，VF 内部用 `Mul` + `LoadAlign(dst)` + `Add`，不用 `MulAddDst`。
- review 时核对 VF 函数体：`MulAddDst` 出现且 addend 含抵消项 → 标记风险。
- golden 若用 C++ `a*b + c`（默认两次舍入），kernel 必须匹配双舍入；golden 若用 `std::fma`，kernel 才能用 `MulAddDst`。

## 失效边界

- 仅 Ascend950（arch 3510）实测；FMA 与双舍入差异是 IEEE-754 通用语义，推断其他 dav 平台可迁移——未验证，推断可迁移。
- 算子规范若明确定义 FMA 语义（如 `std::fma`），本卡不适用，应改 golden 为 FMA 而非改 kernel。

## 坏实践

- 放宽 `y` 输出阈值（fp16 1e-3 → 更宽）掩盖 FMA 残差（反投机取巧）。
- 在 golden 侧也改用 `std::fma` 让两者一致——这改变了 oracle 语义，须先确认算子规范允许 FMA，否则是改期望值捷径。

## 误诊

无。注意区分：case_03 bf16 1-ULP 偏差与本卡 FMA 问题不同源——FMA 修复后 case_03 仍 FAIL，其根因在 golden `roundToBf16` 非标准舍入 vs kernel `CAST_RINT`（另见他卡）。

## 验证信号

本卡验证来源：bn_training_update_custom Branch-4 kernel 调试会话（Ascend950 / CANN 9.0.0-beta.2）。

- run.sh 上板输出 15/18 通过、3 用例 FAIL（case_03 bf16 diff=0.015625、case_07 fp32 diff=8e-5、case_19 fp16 120/120 diff=2.05469），统计输出 fp32 已通过（来源平台：Ascend950；证据等级：tool_observed）。
- 失败仅出现在 `x == save_mean` 通道、统计输出（fp32）已通过——精确定位到 Phase 2 `MulAddDst` 舍入语义（来源平台：Ascend950；证据等级：tool_observed）。
- 把 `MulAddDstVF` 内部由 `MulAddDst` 改为 `Mul`+`LoadAlign(dst)`+`Add` 分离舍向后，case_07 与 case_19 由 FAIL 转 PASS（17/18 通过）（来源平台：Ascend950；证据等级：tool_observed）。