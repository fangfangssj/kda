---
type: "Runbook"
title: "AscendC::Abs 规范化 NaN payload 致逐位验收误报"
description: "硬件 AscendC::Abs 把 FP16 NaN 规范化为全 1 payload（0x7fff），numpy.abs 保留原 payload（0x7e00），逐位一致验收误报失败；用位差 511 与 NaN 感知探针判别。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.math_function"]
status: draft
sources: []
created_at: "2026-08-18T08:38:46Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# AscendC::Abs 规范化 NaN payload 致逐位验收误报

## 现象

为 abs（绝对值）Kernel 直调算子做 FP16 精度验收，用例 T5（边界/特殊值，128 元素，含 ±65504/±2^-14/±Inf/NaN）首次运行报失败。验收脚本以 uint16 位模式逐位一致（`np.array_equal`）比对 kernel 输出与 `numpy.abs` golden，结果：

```
========== 用例 T5: 边界/特殊值 (len=128) ==========
--- 精度验证 ---
Verification FAILED!
Mismatch count: 16 / 128
Sample mismatched output bits: [511 511 511 511 511 511 511 511 511 511]
ERROR: 用例 T5 精度验证失败
=== [3/5] 可执行文件通路结果: PASS=4 FAIL=1 ===
```

128 个元素中 16 个不一致，全部为 NaN 元素，位差恒为 511（= `0x7fff ^ 0x7e00`）。非 NaN 元素（±max/±min/±Inf/-0.0→+0.0）全部逐位一致。PyTorch 通路 P5 同样报 `max bits diff=511`。当时代码状态：kernel 用 `AscendC::Abs(yLocal, xLocal, count)` 做逐元素绝对值，`DataCopyPad` 搬运 GM→UB→GM，编译通过、T1-T4（随机正负/零值/负零）全 bit-exact 通过，仅含 NaN 的 T5 失败。

## 适用场景

- **工程形态**：Kernel 直调（ops-direct-invoke）精度验收；FP16 elementwise 算子的位模式验收脚本设计。
- **API 路径**：`AscendC::Abs(dst, src, count)`（vector 连续计算接口，DAV_2201）；`numpy.abs` 作 golden 参考；验收脚本位模式比对。
- **验证过的平台与 CANN 版本**：DAV_2201（SocVersion=ASCEND910B），CANN 9.0.0。按 platform-identification §5，仅凭 `ASCEND910B`/`DAV_2201` 不足以区分 A2 与 A3 产品，本卡 `platforms` 记空表 `[]`（未定位到单一注册平台），不伪造值。
- **适用粒度**（transfer_scope 五档取最窄）：`operator_family`——所有「仅清符号位」的 elementwise 算子（abs/neg 等 sign-clearing 类）在 DAV_2201 上用位模式验收时，适用同一 NaN 规范化行为与同一 NaN 感知判别。
- **路线优先级**：`recommended_fastpath`——对含 NaN 的 sign-clearing elementwise 算子，验收脚本直接采用「非 NaN 逐位一致 + NaN 断言符号位清零」的 NaN 感知比对，而非逐位一致。
- **不适用或未验证场景**：① 非 sign-clearing 类算子（计算型，如 exp/sin）的 NaN 传播行为不同，不适用本判别；② DAV_2201 以外架构（如 950/DAV_3510）未验证；③ A2 与 A3 产品级差异未分别独立验证（仅在一台 DAV_2201 设备实测）。

## 判别方法

触发（tool_observed）：FP16 abs Kernel 直调精度验收，T5（含 NaN 的边界用例）报 `Verification FAILED! Mismatch count: 16 / 128`，10 个采样位差全为 `511`；T1-T4（无 NaN）全 bit-exact 通过。即「无 NaN 用例全过 + 含 NaN 用例全败 + 位差恒定」三连。

探针：① 把 kernel 输出与 golden 以 uint16 位模式读取，计算逐元素 XOR；② 对 golden 做 FP16 NaN 位模式分类（指数段 `0x7C00` 全 1 且尾数段 `0x03FF` 非 0），把元素分为 NaN/非 NaN 两组，分别统计 mismatch；③ 验证「不一致元素是否全部落在 NaN 组且位差是否恒等于 `0x7fff ^ 0x7e00 = 511`」。可执行 NaN 分类探针（uint16 位模式，numpy 数组须用按位 `&`+括号，不可用 Python `and`）：

```python
import numpy as np

def is_nan_bits(bits):
    """uint16 位模式判 FP16 NaN：指数全 1 且尾数非 0"""
    return ((bits & 0x7C00) == 0x7C00) & ((bits & 0x03FF) != 0)

def diagnose(output_path, golden_path):
    output = np.fromfile(output_path, dtype=np.uint16)
    golden = np.fromfile(golden_path, dtype=np.uint16)
    diff = output ^ golden
    nan_mask = is_nan_bits(golden)
    n_nan = int(np.sum(nan_mask))
    nan_diff = diff[nan_mask]
    non_nan_diff = diff[~nan_mask]
    print(f"total={len(golden)} nan={n_nan}")
    print(f"non-NaN mismatch count={int(np.sum(non_nan_diff != 0))}")
    print(f"NaN mismatch count={int(np.sum(nan_diff != 0))} "
          f"unique NaN bit-diff={np.unique(nan_diff).tolist()}")
    # 判据：非 NaN 全一致 + 不一致全在 NaN 组 + NaN 位差恒为 511
    return (int(np.sum(non_nan_diff != 0)) == 0
            and int(np.sum(nan_diff != 0)) == n_nan
            and set(np.unique(nan_diff).tolist()) <= {511})
```

观测（tool_observed）：T5 的 128 元素中 16 个为 NaN，16 个全部 mismatch 且位差恒为 `511`（= `0x7fff ^ 0x7e00`）；非 NaN 的 112 个元素（±65504/±2^-14/±Inf/-0.0→+0.0）全部逐位一致。改用 NaN 感知比对（非 NaN 逐位一致 + NaN 断言 `is_nan_bits` 为真且符号位 `(bits & 0x8000) == 0`）后，T5 输出 `Verification PASSED! (bit-exact for non-NaN, 16 NaN elements sign-cleared)`，T1-T4 保持 bit-exact 通过、未因验收脚本变更而退化。kernel 代码（`AscendC::Abs(yLocal, xLocal, count)`）全程未改动。

解释（interpretation）：硬件 `AscendC::Abs` 指令（DAV_2201 vector 连续计算接口）对 FP16 NaN 输入做规范化——将 NaN 的 payload（尾数）置为全 1，得到 canonical NaN `0x7fff`（符号位 0、指数全 1、尾数全 1）；而 `numpy.abs` 仅清符号位、保留原 payload（如输入 `0x7e00` → 输出 `0x7e00`）。位差演算 `0x7fff ^ 0x7e00 = 0x01ff = 511`，与验收输出 `[511 ...]` 完全吻合。两者都是「符号位清零且仍为 NaN」，满足 IEEE 754 的 `abs` 语义（`abs(x)` 返回符号位清零的 x；NaN 的 payload 为 implementation-defined，允许规范化）。因此 kernel 行为正确，逐位一致验收过严——它要求 NaN payload 也逐位一致，而 IEEE 754 并不要求。根因单一归因：硬件 Abs 规范化 NaN payload + 验收脚本对 NaN payload 逐位一致过严，非 kernel 精度缺陷。

## 预防

- 对「仅清符号位」的 elementwise 算子（abs/neg 等）在 DAV_2201 上做位模式验收时，默认采用 NaN 感知比对：非 NaN 逐位一致 + NaN 断言符号位清零，不要对 NaN payload 做逐位一致要求。
- 验收脚本的 `is_nan_bits` 在 numpy 数组上必须用按位 `&`（带括号），不要用 Python `and`（对 numpy 数组触发 `ValueError: truth value of an array is ambiguous`）。
- 出现「无 NaN 用例全过 + 含 NaN 用例全败 + 位差恒定」三连时，先按本探针分解 NaN/非 NaN 组，别先动 kernel compute 逻辑。

## 验证信号

来源：abs 算子精度验收会话（DAV_2201 设备，CANN 9.0.0，FP16）。修复前 T5 失败、改为 NaN 感知比对后 T5 通过，均在本轮内实测；kernel 代码全程未改。

- 修复前 T5（边界/特殊值，128 元素含 NaN）：`Verification FAILED! Mismatch count: 16 / 128`，16 个不一致元素全为 NaN，位差恒为 `511`（= `0x7fff ^ 0x7e00`），与硬件 Abs 规范化 NaN payload 吻合（来源平台:未定位（ASCEND910B/DAV_2201）；证据等级:tool_observed）。
- 修复前 PyTorch 通路 P5：`max bits diff = 511`，同一 NaN 位差根因（来源平台:未定位（ASCEND910B/DAV_2201）；证据等级:tool_observed）。
- 改用 NaN 感知比对后 T5：`Verification PASSED! (bit-exact for non-NaN, 16 NaN elements sign-cleared)`（来源平台:未定位（ASCEND910B/DAV_2201）；证据等级:tool_observed）。
- 改用 NaN 感知比对后 T1-T4（随机正负/零值/负零）：保持 bit-exact 通过，未因验收脚本变更而退化（来源平台:未定位（ASCEND910B/DAV_2201）；证据等级:tool_observed）。
- kernel 代码（`AscendC::Abs(yLocal, xLocal, count)`）未改动：NaN 规范化是 `AscendC::Abs` 指令的固有行为，kernel 已符合 IEEE 754 abs 语义（来源平台:未定位（ASCEND910B/DAV_2201）；证据等级:artifact_verified）。

## 失效边界

- **DAV_2201 内 A2/A3 未分别验证**：本轮设备 SocVersion=ASCEND910B、NpuArch=DAV_2201，证据不足以区分 A2 与 A3（见 platform-identification §5）。NaN 规范化为 vector 指令级行为，预期 A2/A3 一致，但未在两个产品上分别独立验证。**未验证，推断可迁移**至 A2/A3。
- **DAV_3510（950）未验证**：不同微架构的 vector Abs 指令对 NaN payload 的处理可能不同，未在 950 上验证，**推断不可迁移**，需上板实测后再定。
- **非 sign-clearing 算子不适用**：计算型 elementwise（exp/sin/log 等）对 NaN 的传播语义不同（通常 NaN 输入→NaN 输出但不保证 payload），不适用本卡的「符号位清零」断言。
- **非 FP16 dtype**：本探针 `is_nan_bits` 的掩码（`0x7C00`/`0x03FF`/`0x8000`）针对 FP16（binary16）；FP32/BF16 的 NaN 位模式不同，需相应调整掩码，未验证。

## 坏实践

- **逐位一致验收强加于 NaN payload**：用 `np.array_equal(output, golden)` 要求全部元素（含 NaN）逐位一致，会把硬件正确的规范化 NaN 误报为精度失败，诱导后续 agent 去「修 kernel」（kernel 本无错）或放松容差（掩盖真因）。
- **误诊为「精度不达标」而放宽容差**：若把 T5 失败当作精度问题去调 `rtol/atol`，既掩盖了「硬件行为符合语义」的真因，又留下隐患——本卡探针已证位差恒为 511 且全在 NaN 组，应优先走 NaN 感知判别而非放容差。
