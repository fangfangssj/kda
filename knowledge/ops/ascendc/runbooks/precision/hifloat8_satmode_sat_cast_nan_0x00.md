---
type: "Runbook"
title: "HiFloat8 SatMode::SAT Cast NaN 映射 0x00"
description: "AscendC Cast<hifloat8_t,float,castTraitF32toh8> 用 SatMode::SAT 有限格式把 NaN 输入映射为 0x00；golden 用 en_dtypes .astype(HIF8) 得 NaN/负零码 0x80，字节级比对 FAIL。判别：用全 NaN 输入跑 aclnn 比对，mode 2 FAIL 而 mode 0/1 PASS 即指向此 cast 路径。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion", "topic.compute.quantization"]
status: stable
sources: []
created_at: "2026-08-18T05:07:50Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# HiFloat8 SatMode::SAT Cast NaN 映射 0x00

## 现象

在 Ascend950（`--plat Ascend950PR_957b`）环境对 `indexer_quant_cache` 算子做 NaN 专项精度比对：从 `kvce_indexer/csv/indexer_quant_cache.csv` 中挑 4 种 quant_mode 的代表用例，把输入 `x` 的 `input_data_ranges` 改为 `(nan, nan)`，其余输入保持原样，跑 TTK aclnn golden-vs-device 比对。结果 **mode 2（HiFloat8）FAIL/SUPPRESSED，而同输入下 mode 0（MX-FP8）与 mode 1（Normal）PASS**。这一对照把故障收敛到 HiFloat8 cast 这一条路径——不是通用 NaN 输入问题，而是该路径上 device 与 golden 对 NaN 的 cast 语义分歧。

## 适用场景

- **工程形态**：算子精度比对（aclnn 路径 golden vs device），TTK aclnn 比对模式（`--compare close --golden-mode Enable`）。
- **API 路径**：device kernel `op_kernel/arch35/indexer_quant_cache_base.h` 的 `VFProcessHifp8QuantVF` → `Cast<hifloat8_t, float, castTraitF32toh8>`；golden `ttk/user_defined_modules/op_api/golden_funcs/indexer_quant_cache.py` 的 `_encode_hifloat8_row`。
- **验证过的平台与版本**：Ascend950（`Ascend950PR_957b`）；CANN 版本未在 trace 显式记录（环境为用户既有的 950 自定义算子包环境）。
- **适用粒度**：`operator_family`——任何用 `Cast<hifloat8_t, float, castTraitF32toh8>`（SatMode::SAT）做 fp32→hifloat8 量化的算子 golden 都适用。
- **路线优先级**：`recommended_fastpath`——判别时先跑全 NaN 输入比对，mode 2 FAIL 而 mode 0/1 PASS 即定位到 hifloat8 cast 路径的 NaN 语义分歧。
- **不适用或未验证场景**：非 SatMode::SAT 的 hifloat8 cast 路径未验证；a2/a3 平台未验证（推断可迁移，见 `## 失效边界`）。

## 判别方法

**触发条件（trigger）**：对使用 `Cast<hifloat8_t, float, castTraitF32toh8>`（SatMode::SAT）做 fp32→hifloat8 量化的算子，用全 NaN 输入跑 golden-vs-device 字节级比对，HiFloat8 路径 FAIL/SUPPRESSED，而同输入下其他量化模式（如 MX-FP8、Normal）PASS。

**可执行探针（probe）**：构造 `x` 的 `input_data_ranges=(nan, nan)` 用例（TTK `field_parser.py` 支持 `nan` 字面量，`data.py` 在 `low==high==nan` 时用 nan 填充数组），其余输入保持原样，跑：

```
python3 -m ttk aclnn -i nan_cases.csv -o nan_result.csv --plat Ascend950PR_957b --device-whitelist 0 --pc 1 --compare close --golden-mode Enable
```

对照实验：同一组用例把 quant_mode 切到 0/1 各跑一次。若 mode 2 FAIL 而 mode 0/1 PASS → 指向 hifloat8 cast 路径的 NaN 语义分歧，而非通用 NaN 输入问题。

**原始观测（observation）**：mode 2（HiFloat8，用例 `ice_m2_bf16_d128`，d=128、block=32、cache=uint8、scale 不写出）比对 FAIL/SUPPRESSED；同输入 mode 0（MX-FP8）、mode 1（Normal）PASS（mode 1 kernel 显式 NaN-safe；mode 0 见相关领域）。该对照把故障收敛到 HiFloat8 cast 这一条路径。

**机制解释（interpretation）**：device 的 `Cast<hifloat8_t, float, castTraitF32toh8>` 是**有限格式** cast，其 castTrait 逐字（取自 `op_kernel/arch35/indexer_quant_cache_base.h`）：

```cpp
constexpr static AscendC::Reg::CastTrait castTraitF32toh8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND
};
// 调用点（VFProcessHifp8QuantVF 内）：
//   Muls(xReg, xReg, scales, pregLoop);
//   Cast<hifloat8_t, float, castTraitF32toh8>(tmp, xReg, pregLoop);
```

`SatMode::SAT` 是有限格式语义：把溢出/±inf 饱和到最大有限幅值（±32768），并把 **NaN 映射为 `0x00`（正零）**。而 golden 用 `en_dtypes.hifloat8` 的 `.astype(HIF8)`（`_encode_hifloat8_row` 修复前为单行 `y = (x_f32_row * np.float32(scale_attr)).astype(HIF8).view(np.uint8)`），en_dtypes 对 NaN 取 NaN 传播语义、输出 hifloat8 的 NaN/负零码 `0x80`。同一 NaN 输入两边 cast 出不同字节，导致字节级比对失败。

注意：本卡为 trace_snapshot 模式——证明在本记录的开发会话、Ascend950 平台与步骤下该 probe 产生了该结果；不据此宣称上游 `SatMode::SAT` 规范的固定版本契约。

## 验证信号

- mode 2（HiFloat8，`ice_m2_bf16_d128`）全 NaN 输入 golden-vs-device 比对 FAIL/SUPPRESSED；同输入 mode 0（MX-FP8）/mode 1（Normal）PASS（来源平台：950；证据等级：tool_observed）
- device kernel `castTraitF32toh8 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_ROUND}` 与 `Cast<hifloat8_t, float, castTraitF32toh8>` 调用点逐字读取自 `op_kernel/arch35/indexer_quant_cache_base.h`（来源平台：950；证据等级：artifact_verified）
- golden `_encode_hifloat8_row` 修复前为单行 `(x_f32_row * np.float32(scale_attr)).astype(HIF8).view(np.uint8)`（无 NaN 处理），逐字读取自 golden 文件 `ttk/user_defined_modules/op_api/golden_funcs/indexer_quant_cache.py`（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 推断可迁移：同 castTrait（SatMode::SAT 的 fp32→hifloat8）在 a2/a3 平台应同行为，**未验证，推断可迁移**。
- 未验证：非 SAT 的 hifloat8 cast 路径；HiFP4（quant_mode=4）路径本 trace 未覆盖。
- 本卡只覆盖 NaN 的字节对齐判别；正常值的 hifloat8 舍入语义（CAST_ROUND）不在本卡范围。

## 坏实践

- 直接 `(x * scale).astype(HIF8)` 当 golden——en_dtypes NaN 传播与 device SAT 有限格式不一致。
- 用放宽容差（`--compare` 放宽、调 `precision_tolerances`）绕过 `0x80` vs `0x00` 的字节差异——掩盖 golden 缺陷，本卡不走此路。

## 误诊

- 初判曾怀疑 `(nan, nan)` 未真正生成 NaN（因 mode 0/1 意外 PASS），需 dump 确认数据确实为 NaN——但 mode 2 FAIL 本身已锁定 hifloat8 路径；mode 0/1 PASS 是因各自 kernel 的 NaN 处理恰好与 golden 一致，不代表输入非 NaN（证据等级：agent_reported；记录调试路径，不作为 adopted 机制证据）。

## 预防

- 任何用 `Cast<hifloat8_t, float, castTraitF32toh8>`（SatMode::SAT）的算子，其 golden 不能直接 `astype(en_dtypes.hifloat8)`——须先确认 device cast 的 `SatMode`：SAT=有限格式（NaN→0、inf→max），非 SAT 可能保留 NaN 码。
- 写 hifloat8 golden 前先跑全 NaN 输入 probe 做对照，定位是否命中此 cast 路径。