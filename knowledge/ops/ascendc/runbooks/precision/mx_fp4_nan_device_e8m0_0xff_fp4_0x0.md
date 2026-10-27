---
type: "Runbook"
title: "MX-FP4 NaN 块 device 写 e8m0=0xFF、fp4=0x0"
description: "Ascend950 上 indexer_quant_cache 的 MX-FP4（quant_mode=3）golden 对含 INF/NaN 的 block-32 直接算 e8m0=253 并让 en_dtypes 把 NaN cast 成 fp4 0x8，而 device 的 vfComputeScaleMXFP4/vfComputeDataMXFP4 显式特例把含 NaN block 的 e8m0 置 0xFF、halfScale 置 bf16 NaN、CAST_RINT 把 bf16 NaN cast 成 fp4 0x0；判别方法是跑全 NaN 输入 aclnn 比对并 dump out/golden 字节。"
tags: ["task.debugging", "topic.precision", "topic.compute.quantization", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-18T05:06:28Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# MX-FP4 NaN 块 device 写 e8m0=0xFF、fp4=0x0

## 现象

Ascend950（`Ascend950PR_957b`）环境对 `indexer_quant_cache` 做 NaN 专项精度比对：`quant_mode=3`（MX-FP4，用例 `ice_m3_e2m1_d128`，block=32、cache=uint8/fp4 packed 2/byte、scale=e8m0），把 x 的 `input_data_ranges` 改为 `(nan, nan)`，跑 TTK aclnn 比对（`--compare close --golden-mode Enable --dump out,golden --dump-format npy`）。结果精度 `78.125%, 78.125%` **FAIL**。dump out/golden 字节对比发现：

- **Scale 差异**：device 输出 e8m0 字节 `0xFF`（=255，NaN code），golden 输出 `0xFD`（=253）。
- **Cache 差异**：device 输出 fp4 nibble `0x0`（正零），golden 输出 `0x8`（负零/NaN 码）。

被写入的 14 个有效 slot 行每行 128 元素全不同；slot_mapping 为 -1 的跳过行两边都保持 cache 原值 0（一致）。`quant_mode=0/1` 此时 PASS（mode 2 HiFloat8 另有 NaN 卡）。

## 适用场景

- **工程形态**：算子精度比对（aclnn golden vs device），TTK aclnn 比对模式。
- **API 路径**：device kernel `op_kernel/arch35/indexer_quant_cache_mx_fp4_base.h` 的 `vfComputeScaleMXFP4`（scale/halfScale）与 `vfComputeDataMXFP4`（fp4 cast）；golden `ttk/user_defined_modules/op_api/golden_funcs/indexer_quant_cache.py` 与 `op/golden_funcs/indexer_quant_cache.py` 的 `_encode_mxfp4_row`（bf16 输入）与 `_encode_mxfp4_row_fp16`（fp16 输入）。
- **验证过的平台与 CANN 版本**：Ascend950（`Ascend950PR_957b`）；CANN 版本未在 trace 显式记录。
- **适用粒度**：`operator_family`——任何 MX-FP4（block-32 共享 e8m0 指数、fp4 packed）量化算子的 golden 都适用。
- **路线优先级**：`recommended_fastpath`（golden 须显式镜像 device 的三段 NaN 特例，是字节对齐的正解）。
- **不适用或未验证场景**：MX-FP8（block-32 共享 e8m0 但 cache=fp8）的 NaN 行为见 mode 0（本 trace PASS，未深挖硬件 ReduceDataBlock<MAX> 对 NaN 的语义）；a2/a3 未验证（推断可迁移，见 `## 失效边界`）。

## 判别方法

- **触发**：MX-FP4 量化算子精度比对中，把 x 的 `input_data_ranges` 改为 `(nan, nan)`（TTK 支持 nan 字面量，`data.py` 在 `low==high==nan` 时走 `numpy.full` 生成全 NaN 输入）跑 aclnn 比对，`quant_mode=3` 出现 `78.125%, 78.125%, FAIL`，而 `quant_mode=0/1` PASS。编译过、无崩溃，差异只在 scale 与 cache 字节。
- **可执行探针**：在怀疑的 MX-FP4 用例上跑 `python3 -m ttk aclnn -i nan_cases.csv -o nan_result.csv --plat Ascend950PR_957b --device-whitelist 0 --pc 1 --compare close --golden-mode Enable --dump out,golden --dump-format npy`，再 `np.load` out/golden 的 scale 与 cache npy、`.view(np.uint8)` 取唯一值：若 device scale 出现 `255(0xFF)` 而 golden 出现 `253(0xFD)`、device cache nibble 出现 `0` 而 golden 出现 `8`，即命中本规则。把 x 改回有限值（如 `(-2, 2)`）重跑通过即坐实是 NaN 特例缺失。
- **原始观测**：mode 3 全 NaN 输入下 scale `output unique=[255]` vs `golden unique=[253, 255]`、`device slot 52=[255,255,255,255]` vs `golden slot 52=[253,253,253,253]`；cache `first diff: out=[0,0,0,0,0] gold=[8,8,8,8,8]`。`device 255 在 e8m0 是 NaN code(exp=255)`、`golden 253 在 e8m0 是 exp=253(255-2)`。
- **解释**：device 的 `vfComputeScaleMXFP4` 对 block 内 16 位 `vdMaxExp`（`bits & 0x7f80`）做显式特例——`Compare(cmpResult, vdMaxExp, expMask=0x7f80, CMPMODE::NE)` 标记含 INF/NaN 的 block，`Select(scaleValue, scaleValue, fp8NanRegTensor=0xFF, cmpResult)` 把这些 block 的 e8m0 置 `0xFF`，`Select(halfScale, halfScale, nanRegTensor=0x7f81, cmpResult)` 把 halfScale 置 bf16 NaN；随后 `vfComputeDataMXFP4` 里 `halfScale(NaN) × x = NaN`，`Cast<U, bfloat16_t, castTrait={ZERO, UNKNOWN, ZEROING, CAST_RINT}>` 把 bf16 NaN cast 成 fp4 `0x0`（正零）。而 golden 的 `_encode_mxfp4_row` 用 `e_field.max()` 对 bf16 NaN 的 8 位 exp field 得 255，`e8m0 = max(255 - f4emax_exp(2), 0) = 253(0xFD)` 不做 NaN 特例，`inv_scale = 2^(127-253)` 为有限值、`y_scaled = NaN × inv_scale = NaN`、`y_scaled.astype(fp4_dt).view(np.uint8)` 经 en_dtypes 得 fp4 `0x8`（负零/NaN 码）。故 scale 差 253 vs 255、cache 差 0x8 vs 0x0，单点归因为 device 三段 NaN 特例分支与 golden 的缺失。

## 根因

device 的 `vfComputeScaleMXFP4` 对含 INF/NaN 的 block 做显式三段特例：

1. `expMask = Duplicate(BF16_EMASK_AND_INF_VAL_MAXFP4)`（=0x7f80，bf16 16 位 exp+sign field）。
2. `Compare(cmpResult, vdMaxExp, expMask, CMPMODE::NE)` —— `vdMaxExp` 为 block 内 16 位 maxExp（`bits & 0x7f80`）；当 block 含 INF/NaN 时 `vdMaxExp == 0x7f80`，`cmpResult` 标记这些 block。
3. `Select(scaleValue, scaleValue, fp8NanRegTensor, cmpResult)` —— `fp8NanRegTensor = Duplicate(FP8_E8M0_NAN_VAL_MAXFP4)`（=0xFF），把含 INF/NaN block 的 e8m0 置为 **0xFF（255）**。
4. `Select(halfScale, halfScale, nanRegTensor, cmpResult)` —— `nanRegTensor = Duplicate(BF16_NAN_VAL_MAXFP4)`（=0x7f81，bf16 NaN），把这些 block 的 halfScale 置为 **bf16 NaN**。

随后 `vfComputeDataMXFP4`：`halfScale(NaN) × x = NaN`，`Cast<U, bfloat16_t, castTrait>`（`castTrait = {ZERO, SatMode::UNKNOWN, ZEROING, RoundMode::CAST_RINT}`）把 bf16 NaN cast 成 fp4 **`0x0`（正零）**。golden 缺这三段特例，直接算 `e8m0 = max(e_block - f4emax, 0)` 得 253 并让 en_dtypes 把 NaN cast 成 0x8。

## 验证信号

NaN 专项测试（x 全 NaN，roundScale=1）TTK aclnn 比对：mode 3 修复前 `ice_m3_e2m1_d128` 精度 `78.125%,78.125%,FAIL`，修复后 `100%,100%,PASS`；roundScale=0 一组同样 PASS（来源平台：Ascend950；证据等级: tool_observed）。

mode 3 全 NaN 输入 dump 字节对比：scale `output unique=[255]` vs `golden unique=[253,255]`、`device slot 52=[255,255,255,255]` vs `golden slot 52=[253,253,253,253]`；cache `first diff: out=[0,0,0,0,0] gold=[8,8,8,8,8]`（来源平台：Ascend950；证据等级: tool_observed）。

device kernel NaN 三段特例（`vfComputeScaleMXFP4` 的 `fp8NanRegTensor`/`nanRegTensor` Select、`vfComputeDataMXFP4` 的 `CAST_RINT` castTrait）逐字读自 `op_kernel/arch35/indexer_quant_cache_mx_fp4_base.h:156-268`（来源平台：Ascend950；证据等级: artifact_verified）。

全量回归 184 用例：非 rejection 171/171 PASS、rejection 13/13 正确拒绝、零回归（来源平台：Ascend950；证据等级: tool_observed）。

## 预防

- 写 MX-FP4 golden 时，对 block max exp == INF/NaN 必须镜像 device 的三段特例：e8m0→0xFF、halfScale→bf16 NaN、fp4 nibble→0x0。不能只算 `e8m0 = max(e_block - f4emax, 0)`。
- en_dtypes/ml_dtypes 对 NaN 的 fp4 cast 默认产出 `0x8`（负零/NaN 码），与 device `CAST_RINT` 的 `0x0` 不一致——任何 fp4 cast 后都须显式 force NaN block 为 0x0。
- 注意位宽：device kernel 的 `BF16_EMASK_AND_INF_VAL_MAXFP4=0x7f80` 是 16 位 exp field（`bits & 0x7f80`），而 golden 提取的 `e_field` 是 8 位（`(bits>>7) & 0xFF`），二者不能直接相等比较——golden 须与 `0xFF` 比，不能复用 kernel 的 16 位 `0x7f80`（此位宽不一致曾导致第一版 fixup 不生效）。

## 失效边界

- 推断可迁移：同 `vfComputeScaleMXFP4`/`vfComputeDataMXFP4` NaN 特例在 a2/a3 应同行为，**未验证，推断可迁移**。
- 未验证：MX-FP8（mode 0）的 e8m0 NaN 处理——本 trace mode 0 PASS 但未深挖硬件 `ReduceDataBlock<MAX>` 对 NaN 的语义（可能传播也可能抑制）；如 mode 0 出现 NaN 不一致需单独定位。
- 未验证：HiFP4（quant_mode=4）NaN 路径本 trace 未覆盖。

## 坏实践

- 用 `e8m0 = max(e_block - f4emax, 0)` 直接当 scale、不判 INF/NaN —— 得 253 而非 255。
- 用 en_dtypes `.astype(fp4_dt)` 直接 cast NaN block —— 得 0x8 而非 0x0。
- 放宽 `precision_tolerances` 绕过 0xFD vs 0xFF 的 scale 差异 —— 掩盖 golden 缺陷。
