---
type: "Runbook"
title: FP16 NaN 用例丢 NaN：ACL fp16 转换位级规避
description: CANN 9.1.0 host 库 aclFloatToFloat16(NaN) 返回 FP16 MAX(0x7bff)、aclFloat16ToFloat 对 NaN/Inf 编码返回有限值；FP16 特殊值测试数据需 NaN 透传 + 规范 NaN 位模式 + 仅 exp==0x1F 段特判的位级规避。
tags: [task.testing, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T00:54:40Z'
updated_at: '2026-09-17T00:54:40Z'
platforms: ["950"]
---

# FP16 NaN 用例丢 NaN：ACL fp16 转换位级规避

## 现象

softplus_v2_grad 的 C08 NaN 传播 ST 用例（shape `[8]`，向 input_features 的 idx=4 注入 NaN，特殊比较规则要求该位置输出必须为 NaN）在 kernel 已实现、其余用例基本通过的情况下仍然失败，报错非常具有迷惑性：

```text
[MISMATCH] idx=4 actual=0.418213 expected=0.418213
[FAIL] C08_nan_propagate [8] — 1/8 mismatches (first at idx 4)
```

actual 与 expected **显示为同一个有限值**。定位发现失败的是 FP16 dtype 组合的用例（FP32 与 BF16 组合通过）。当时代码状态：测试 harness 用 `roundToFP16`（`aclFloatToFloat16`→`aclFloat16ToFloat` round-trip）生成期望输入、`packFloat`（`aclFloatToFloat16`）打包设备输入、`unpackFloat`（`aclFloat16ToFloat`）读回设备输出；oracle 用 host libm `expf` 独立计算（`expf(NaN)=NaN` 本身正确）。

## 适用场景

- 工程形态：aclnn 测试 harness 的 host 侧数据生成/读回（FP16/BF16 特殊值注入用例）。
- API 路径：`aclFloatToFloat16` / `aclFloat16ToFloat`（`acl_base_rt.h`，cann-9.1.0 x86_64 host 库）。
- 验证过的平台与 CANN 版本：ascend950 实测 + host 侧 cann-9.1.0 x86_64（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（host 库行为，与具体算子无关）。
- 路线优先级：recommended_fastpath（在测试 harness 内规避，勿依赖库修复）。
- 明确不适用/未验证场景：kernel/设备侧 `AscendC::Cast`（设备侧转换保 NaN，本卡不涉及；设备侧 NaN/Inf 被 SatMode::SAT 饱和是另一层根因，见 [Cast SatMode::SAT 饱和致 fp16 NaN/Inf 丢失](cast_satmode_sat_fp16_nan_inf.md)）；其他 CANN 版本的 host 库行为未验证。

## 判别方法

写独立探针程序（按 test CMakeLists 方式链 `ascendcl/nnopbase/cust_opapi`）直接调用转换函数对拍：

```text
aclFloatToFloat16(NAN) raw bits=0x7bff        # FP16 MAX=65504，不是 NaN！
roundToFP16(NAN) = 65504 isnan=0
aclFloat16ToFloat(0x7e00) = 98304 isnan=0     # fp32 bits=0x47c00000

NaN(7e00)    acl=98304(isnan=0)   manual=nan(isnan=1)
Inf+(7c00)   acl=65536(isnan=0)   manual=inf
Inf-(fc00)   acl=-65536(isnan=0)  manual=-inf
NaN-(fe00)   acl=-98304(isnan=0)  manual=-nan
1.0 / 2.0 / +0 / -0               acl 与 manual 一致
```

结论：库只坏 **exp==0x1F 的特殊值段**（NaN/Inf），普通值（含 denormal 交给库的路径）正确。对照探针：host libm `expf(±NaN)=NaN`（oracle 无辜）；BF16 路径用手写位运算 round-trip 天然保 NaN（BF16 用例通过的原因）。

## 根因

CANN 9.1.0 host 侧 ACL 类型转换函数对 IEEE-754 特殊值不保语义——`aclFloatToFloat16(NaN)` 落到 FP16 MAX（0x7bff）、`aclFloat16ToFloat` 对 exp==0x1F 的 FP16 编码（NaN/Inf）返回有限值——使测试 harness 的 FP16 数据生成/读回在 round-trip 中丢失 NaN，注入数据与期望值双双变成有限值（65504 过 threshold 分支后输出 = grad 的随机值 0.418213）。

## 解决配方

对 test_aclnn.cpp 的三个函数做位级处理（before/after 为实际 diff 核心段）：

**1) `roundToDtype` NaN 透传**（NaN 不参与 dtype 舍入）：

```cpp
static float roundToDtype(float f, aclDataType dt)
{
    if (std::isnan(f)) return f;  // NaN 透传 (IEEE-754 语义: NaN 不参与 dtype 舍入)
    if (dt == ACL_FLOAT16) return roundToFP16(f);
    if (dt == ACL_BF16)    return roundToBF16(f);
    return f;  // FP32 — no rounding
}
```

**2) `packFloat` 写规范 NaN 位模式**（FP16 quiet NaN 0x7e00 / BF16 quiet NaN 0x7fc0）：

```cpp
    } else if (dt == ACL_FLOAT16) {
        if (std::isnan(f)) {
            // FP16 quiet NaN: sign=0, exp=11111, mantissa=1000000000 → 0x7e00
            uint16_t nan16 = 0x7e00u;
            std::memcpy(out, &nan16, 2);
        } else {
            aclFloat16 h = aclFloatToFloat16(f);
            std::memcpy(out, &h, 2);
        }
    } else if (dt == ACL_BF16) {
        if (std::isnan(f)) {
            // BF16 quiet NaN: sign=0, exp=11111111, mantissa=1000000 → 0x7fc0
            uint16_t nanbf = 0x7fc0u;
            std::memcpy(out, &nanbf, 2);
        } else {
            /* 原有 round-to-nearest-even 位运算路径 */
        }
    }
```

**3) `unpackFloat` 仅对 exp==0x1F 手动还原 NaN/Inf，其余委托库**：

```cpp
    } else if (dt == ACL_FLOAT16) {
        uint16_t raw;
        std::memcpy(&raw, in, 2);
        if (((raw >> 10) & 0x1Fu) == 0x1Fu) {
            uint16_t man10 = raw & 0x3FFu;
            uint32_t bits = (static_cast<uint32_t>(raw >> 15) << 31) |
                            ((man10 != 0) ? 0x7fc00000u : 0x7f800000u);
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
        aclFloat16 h;
        std::memcpy(&h, in, 2);
        return aclFloat16ToFloat(h);
    }
```

（`0x7fc00000u` = FP32 quiet NaN，`0x7f800000u` = FP32 Inf；符号位从 raw>>15 搬到 bit31。）

## 设计决策

- 只特判库确证坏掉的特殊值段（exp==0x1F），normal/denormal/zero 全部继续走 `aclFloat16ToFloat`——保证与库的普通值精度完全一致，避免整体重写引入新差异（见"坏实践"）。
- Inf 用例（C09/C10）本就通过：Inf 被 clamp 到 65504 后 `z > threshold` 分支结果相同，属于被掩盖而非正确——本次只修 NaN（读回侧 exp==0x1F 特判顺带把 Inf 还原为真 Inf，行为更正确且无回归）。

## 预防

FP16/BF16 特殊值（NaN/Inf）测试数据一律不走 ACL host 转换函数 round-trip；写数据用显式位模式，读数据先判指数段再委托库。

## 验证信号

本验证来自该算子 ST 修复会话的实测：

- 修复前 C08 FP16 用例 `idx=4 actual=0.418213 expected=0.418213`（1/8 mismatch，特殊 NaN 规则判 FAIL）（来源平台：950；证据等级：tool_observed）。
- 探针：`aclFloatToFloat16(NAN)` raw bits=0x7bff、`roundToFP16(NAN)=65504 isnan=0`；`aclFloat16ToFloat(0x7e00)=98304 isnan=0` 及特殊值全表（来源平台：950 host cann-9.1.0 x86_64；证据等级：tool_observed）。
- 中间版（全手动 FP16→FP32 转换器）ST 45/48：C01 56/401408、C05 21/262144、C06 25/401408 错值，actual≈-1.27e-3 vs expected≈-4.9e-6（来源平台：950；证据等级：tool_observed）。
- 最终版（仅 exp==0x1F 特判）ST 48/48、run.sh 退出码 0；C09(+Inf)/C10(-Inf)/C11(全零)/C12(FP16 上溢) 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅验证 cann-9.1.0 x86_64 host 库；其他 CANN 版本（更高版本可能已修）行为未验证。
- `0x7e00/0x7fc0` 是 quiet NaN 规范位模式；signaling NaN 的位模式未单独处理。
- denormal 普通值委托库路径未独立复验（探针显示与手工转换一致，故信任库）。

**未决项**：

- 优先项：复现并追根因——aclFloatToFloat16 为何落到 0x7bff 等错误位模式（库缺陷层根因未追，本卡 harness 侧根因已闭环）。

- 库缺陷根因（`aclFloatToFloat16` 为何落到 0x7bff、`aclFloat16ToFloat` 为何把 0x7e00 读成 98304=1.5×2^16）未追；推测是把 exp=0x1F 按 exp=0x1E 处理。
- 其他 CANN 版本是否已修复未验证（可在新环境重跑本卡探针三行即知）。
## 坏实践

- 第一版 `unpackFloat` 换成了**全手动** FP16→FP32 转换器，结果普通 FP16 数值出错（~256× 量级偏差，56/401408 mismatch）——库的普通值转换本来是对的，整体重写把对的也改坏了。正确姿势：只对探针确证坏掉的指数段做特判。
- 用 `aclFloatToFloat16` 做 NaN 注入数据打包（round-trip 丢 NaN）。

## 误诊

首轮曾把该失败记为"C08 BF16 NaN：Cast(BF16→FP32)/libm expf 不保 NaN"——方向错误：失败用例实为 FP16 组合（用例序 39，kSameDtypeIdx[1]），BF16 用例通过；且设备侧 Cast 与 oracle 均无辜，问题在 host 测试 harness 的转换函数。
