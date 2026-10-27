---
type: "Runbook"
title: "Select VSEL_CMPMASK_SPR count>64 掩码错位判别"
description: "arch35 Select 的 VSEL_CMPMASK_SPR 对 float 仅载首 repeat 的 selMask 并对所有 repeat 复用；count>64 时后续块套用首 64 元素块掩码，错误位置 pos%64 集中在 44/55/56，双向错取 linear/sigmoid 分支。判别：逐元素比对 golden 与 actual 求 pos%64 分布，n=15360 单完整 tile 复现排除 tile 边界。"
tags: ["task.debugging", "topic.compute.elementwise.comparison_logic", "topic.implementation.api_layer.basic_api", "topic.implementation.masking", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T16:49:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Select VSEL_CMPMASK_SPR count>64 掩码错位判别

## 现象

`softplus_v2_grad`（arch35/Ascend950，CANN 9.0.0-beta.2）在修完 L2 缓存键漏 scalar 之后仍有部分用例失败。逐元素比对 golden 与 NPU actual，观测到**双向错误**：

- `pos=108`（z=-2.1 ≤ threshold）：golden 为 sigmoid 分支值，actual 取 linear 分支——kernel 错取 linear。
- `pos=120`（z=2.56 > threshold）：golden 为 linear，actual 取 sigmoid——kernel 错取 sigmoid。

双向错误说明不是分支条件写反，而是 `Select` 的掩码被污染。把所有错误位置 `pos % 64` 求分布，集中在 `44, 55, 56, 12, 31` 等固定偏移；进一步取 n=15360（单个完整 tile，64 对齐）仍复现，排除 tiling 尾块/广播对齐问题——锁定在 `Select` 调用本身。

## 适用场景

- **工程形态**：arch35 Select 的 VSEL_CMPMASK_SPR 对 float 仅载首 repeat 的 selMask 并对所有 repeat 复用。
- **API 路径**：`Select(dst, selMask, src0, src1, selMode, count)` 公共重载（经 `VselImpl`），`selMode` 取 `VSEL_CMPMASK_SPR`(mode 0) 或 `VSEL_TENSOR_TENSOR_MODE`(mode 2)；配套 `Compare<float, uint8_t>(bMask, bBx, bThr, GT, count)` 产掩码。
- **验证过的平台与 CANN 版本**：Ascend950（arch35），CANN 9.0.0-beta.2（头文件 `cann-9.0.0-beta.2/x86_64-linux/asc/impl/basic_api/dav_c310/kernel_operator_vec_sel_impl.h`）。
- **适用粒度**：`platform_wide`——所有 arch35 上对 count>repeatElm（float 为 64）的 `Select` 调用都受此 API 语义约束。
- **路线优先级**：对 count 大于 GetVecLen/sizeof(T) 的场景，`VSEL_TENSOR_TENSOR_MODE` 是唯一逐 repeat 重载掩码的正确模式。
- **不适用或未验证场景**：count ≤ repeatElm 时 `VSEL_CMPMASK_SPR` 与 mode 2 行为一致，本卡不约束小 count；非 arch35（如 a2/a3）未在本 trace 验证。

## 判别方法

- **触发（tool_observed 失败）**：bf16/大 shape 多 tile 用例部分元素错，逐元素比对观测到双向错误——z≤threshold 处 golden 为 sigmoid 分支而 actual 错取 linear，z>threshold 处 golden 为 linear 而 actual 错取 sigmoid。
- **探针（可执行检查）**：固定随机种子复现失败 case，写探针逐元素比对 golden 与 NPU actual，统计错误位置 pos % 64 分布；再用 n=15360（单个完整 tile，64 对齐，无尾块）单独复现，排除 tiling 尾块/广播对齐归因。
- **观测（tool_observed）**：错误位置 pos % 64 集中于 44, 55, 56, 12, 31 等固定偏移；n=15360 单完整 tile 仍有约 808 处大差异，确认 bug 在 `Select` 调用本身而非多 tile 边界。
- **判读（interpretation，根因分析）**：arch35 `SelectMode0Level2`（对应 `VSEL_CMPMASK_SPR`）对 sizeof(T)==4（float）取 repeatElm = GetVecLen/sizeof(T) = 64、repeatTime = CeilDivision(calCount, 64)；其实现 Reg::LoadAlign + MaskUnPack **仅从 sel buffer 载首 repeat 的掩码一次**，随后对所有 repeatTime 次循环复用同一 selMask。count>64 时元素 64..N 全部套用首 64 元素块的掩码，错误位置自然呈 pos % 64 固定偏移模式。对照 `VSEL_TENSOR_TENSOR_MODE`（mode 2）每个 repeat 重新从 sel buffer 对应偏移重载掩码；`Compare` 本身把完整掩码 per repeat 正确写入 bMask，故错在 `SelectMode0Level2` 只读首段。判据：错误位置 pos%64 呈固定偏移 + 单完整 tile 仍复现 → 命中 `VSEL_CMPMASK_SPR` 掩码复用语义缺陷。

## 验证信号

- 错误位置 pos % 64 集中于 44, 55, 56, 12, 31 等固定偏移，符合“首 64 块掩码套用到全部元素”模式（来源平台：Ascend950；证据等级：tool_observed）。
- n=15360（单完整 tile，64 对齐）仍出错，排除 tile 边界归因（来源平台：Ascend950；证据等级：tool_observed）。
- API 头文件取证：`kernel_operator_vec_sel_impl.h` 的 `SelectMode0Level2` 载一次掩码复用 vs `VSEL_TENSOR_TENSOR_MODE` 逐 repeat 重载（来源平台：Ascend950；证据等级：artifact_verified）。
- 将 `Select` 的 selMode 由 `VSEL_CMPMASK_SPR` 切换为 `VSEL_TENSOR_TENSOR_MODE` 后，同一测试集失败用例数显著下降，验证根因判读正确（来源平台：Ascend950；证据等级：tool_observed）。

## 预防

- arch35 上 `Select` 的 count 大于 GetVecLen/sizeof(T) 时禁用 `VSEL_CMPMASK_SPR`，统一用 `VSEL_TENSOR_TENSOR_MODE`；或在 `Select` 封装处对 count>repeatElm 自动切 mode 2。
- 写 `Select` + `Compare` 组合时，先验证 bMask 长度 = Ceil(count, repeatElm) * repeatElm / 8 字节且 `Compare` per repeat 写满，再选 selMode。

## 失效边界

- 推断 a2/a3 同名 API 行为可能不同（不同 arch 的 `kernel_operator_vec_sel_impl.h` 实现），未在本 trace 验证，可迁移需查对应 arch 头文件。
- count ≤ repeatElm（float 即 ≤64）时两模式行为一致，本卡结论不适用。