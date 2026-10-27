---
type: "Runbook"
title: "算子属性默认值 stringstream 截断致 FP32 标量误差"
description: "CANN op_build 经 libregister.so 的 GetAttrDefaultVal 用 std::stringstream 默认 6 位有效数字序列化 float 属性默认值，使 1.414213562373f 被截为 \"1.41421\" 写入 op-info JSON，kernel 从 TilingData 读到 0x3FB504D5 而非 0x3FB504F3，产生稳定 ULP 精度误差。判别方法：把 TilingData 标量位模式直写 GM 跳过计算、对比期望全精度位模式即可定位截断发生在 kernel 读之前。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-17T03:30:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 算子属性默认值 stringstream 截断致 FP32 标量误差

## 现象

在 ops-registry-invoke 算子仓（目标 Ascend950PR/arch35，CANN 9.0.0）执行 Kernel TTK 测试时，FP32 路径出现稳定精度误差。算子计算 `y = LeakyRelu(x + bias, negative_slope) * scale`，其中 `scale` 默认值在 OpDef 中定义为 `1.414213562373f`（即 √2，IEEE 754 表示 0x3FB504F3）。用 x=2.0、bias=0.5（正分支，不涉及 slope）单点验证：FP32 正确结果应为 0x40624630，NPU 实际输出 0x4062460A，差 38 ULP；负分支差 36 ULP。从 NPU 输出反推出的隐含 scale = 0x3FB504D5，与全精度 0x3FB504F3 的低 6 mantissa 位不同、高 17 位相同。TilingData 中的 `negativeSlope`（0.2f = 0x3E4CCCCD）未受影响。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发（OpDef `.Attr(...).Float(default)` + host Tiling + arch35 kernel），Kernel TTK（`python3 -m ttk kernel`）从 opp/vendors 已安装算子包加载。
- **API 路径**：`OpAttrDef::Float(float)` → `op_build` 工具调用 `attr.GetAttrDefaultVal("[]")` 生成 INI/JSON → 框架运行时 `attrs->GetFloat()` → `tiling->scale = ...` → TilingData host→device 传输 → kernel `GET_TILING_DATA_WITH_STRUCT` 读取。
- **验证过的平台与 CANN 版本**：Ascend950PR_9579（arch35/dav_c310），CANN 9.0.0。仅 950 有实测。
- **适用粒度**：`operator_family`——适用于任何在 OpDef 用 `.Float(<非整/非简短小数默认值>)` 且 TTK/推理走 defaultValue 路径的算子。
- **路线优先级**：`recommended_fastpath`——TTK CSV 显式传 attributes 绕过 defaultValue；`safety_fallback`——kernel 对默认 scale 用编译期常量字面量（仅适用默认值固定场景，不支持用户自定义）。
- **不适用或未验证场景**：用户在模型推理时显式传入 scale 属性（不读 defaultValue，不截断）；属性默认值是整数或 6 位有效数字内可无损还原的小数（如 0.2、1.0、20.0）；a2/a3 平台未验证（推断可迁移，见失效边界）。

## 判别方法

**触发条件**：FP32 路径出现稳定 ULP 精度误差，且反推出的隐含标量与 OpDef 字面量 float 的 IEEE 754 表示相比"mantissa 高位相同、低位不同"（非 round-to-nearest/truncation/round-up/round-down 的合法舍入结果，也不匹配 FP16/BF16/TF32 降精度）。

**可执行探针**：

探针 1（定位截断发生在 kernel 读之前还是之后）——在 kernel `Init` 把 TilingData 标量字段位模式直写 GM、跳过计算，dump hex 与期望全精度位模式对比：

```cpp
scaleFp32_    = tilingData->scale;
slopeFp32_    = tilingData->negativeSlope;
negSlopeBits_ = *reinterpret_cast<const uint32_t*>(&tilingData->negativeSlope);
scaleBits_    = *reinterpret_cast<const uint32_t*>(&tilingData->scale);
// 把 scaleBits_ 直写 yGm 某 slot，skip 计算，dump hex vs 期望 0x3FB504F3
```

探针 2（排除 stale binary 缓存）——在 `Compute` 末尾追加 `Muls(y, y, (T)3.0, n)` 标记，重编重跑，期望输出 = 原值 ×3：

```cpp
Muls(yLocal, yLocal, (T)3.0, currentNum);  // *3 marker to verify binary update
```

探针 3（排除 float 类型序列化假说）——把同值存为 `uint32_t scaleBits` 字段而非 float 字段，若同样被截为 0x3FB504D5，则排除"float 字段序列化"假说，指向"值在写入 TilingData 前就已被字符串截断"。

探针 4（确认截断源）——读 op-info JSON 的 `attr_scale.defaultValue` 字符串；用 `float("...")` 还原其 hex；用 C++ 模拟 `std::stringstream` 默认（6 位有效数字）vs `std::to_string(float)` vs `std::setprecision(9)` 三种序列化的还原 hex。

**观测**：

- 探针 1：`tilingData->scale` 位模式 = 0x3FB504D5（应为 0x3FB504F3），`tilingData->negativeSlope` = 0x3E4CCCCD（全精度，未截断）。截断发生在 kernel 读之前。
- 探针 2：`*3` 标记后输出 10.6066 = 3.5355×3，证明 device binary 确实更新，排除缓存假说。
- 探针 3：`uint32_t scaleBits` 字段同样被截为 0x3FB504D5，排除 float 类型序列化假说。
- 探针 4：op-info JSON `attr_scale.defaultValue = "1.41421"`（6 位有效数字）；`float("1.41421") = 0x3FB504D5` 与 kernel 读到的截断值一致。

**解释**（根因分析）：CANN `op_build` 工具在解析编译后的 OpDef 动态库生成 op-info INI/JSON 时，调用 `libregister.so` 中 `OpAttrDef::GetAttrDefaultVal()` 将 float 默认值转字符串。该函数内部构造 `std::stringstream` 并执行 `ss << value`（float 先 `cvtss2sd` 提升为 double 再写入），未调用 `std::setprecision`，使用 C++ 默认 6 位有效数字：`1.414213562373f → "1.41421" → float("1.41421") = 0x3FB504D5`。要从一个十进制字符串无损还原一个 FP32 需要 9 位有效数字，6 位不足以唯一确定该 float，被解析为相邻可表示值。截断后值写入 op-info JSON 的 `defaultValue` 字段，框架运行时把它解析回 float 填给 `attrs->GetFloat()`，再写入 TilingData struct 的 `scale` 字段；host→device DMA 是 `GET_TILING_DATA_WITH_STRUCT` 内部纯字节 `memcpy`，不再改变值——所以 kernel 读到的就是 0x3FB504D5。`0.2f` 未截断是因为 `"0.2"` 在 6 位有效数字内无损。

C++ 模拟三方法对比（支撑解释）：

| 序列化方法 | 字符串输出 | float 还原 hex | 截断 |
|------|-----------|---------------|------|
| `stringstream` 默认（GetAttrDefaultVal 行为） | `"1.41421"` | 0x3FB504D5 | YES |
| `std::to_string(float)` | `"1.414214"` | 0x3FB504F7 | YES（截断值不同） |
| `setprecision(9)` | `"1.41421354"` | 0x3FB504F3 | NO |

## 验证信号

问题与验证来自一次 fused_bias_leaky_relu 的 Kernel TTK 调试会话（Ascend950PR_9579，CANN 9.0.0），探针前后均用 `python3 -m ttk kernel -d -c -b release --golden-mode Promote` 跑同一 CSV。

- 修复前 FP32 正分支 38 ULP / 负分支 36 ULP，NPU 输出 0x4062460A（正确 0x40624630）（来源平台：950；证据等级：tool_observed）。
- 探针 1：kernel 读到的 `tilingData->scale` 位模式 = 0x3FB504D5（应为 0x3FB504F3），`tilingData->negativeSlope` = 0x3E4CCCCD 全精度未截断（来源平台：950；证据等级：tool_observed）。
- 探针 2：`*3` 标记后输出 10.6066 = 3.5355×3，确认 device binary 已更新（来源平台：950；证据等级：tool_observed）。
- 探针 3：`uint32_t scaleBits` 字段同样截为 0x3FB504D5，排除 float 类型序列化假说（来源平台：950；证据等级：tool_observed）。
- 探针 4：op-info JSON `attr_scale.defaultValue = "1.41421"`，`float("1.41421") = 0x3FB504D5` 与 kernel 读到的截断值一致；C++ 模拟 `stringstream` 默认输出 `"1.41421"`，`setprecision(9)` 输出 `"1.41421354"` 还原 0x3FB504F3（来源平台：950；证据等级：tool_observed）。
- 同根因影响 fused_bias_leaky_relu_grad 内置算子（scale 经 `asc_vf_call` 作 float 参数还多一层截断风险），实测多分支 ULP 误差（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **未验证，推断可迁移**：a2/a3 平台同样使用 CANN op_build/libregister 与 op-info JSON 机制，截断行为推断一致，但本 trace 未在这些平台上实测。
- **不适用**：用户在模型推理时显式传入 scale 属性（走 `possible_attrs[k]` 路径，不读 defaultValue，不截断）；属性默认值是整数或简短可精确小数（6 位有效数字内无损，如 0.2、1.0、20.0）时不受影响。
- 框架级 `setprecision(9)` 修复未在本 trace 代码中落地，CANN 后续版本是否已修未查证。

## 坏实践

- 在未定位真因前，把"放宽容差/改 golden"当作修复——本 trace 中 TTK `stat_rel_err` 在单元素用例下容差很宽（38 ULP 仍 PASS），若放容差会掩盖该缺陷。
- 把"标量在指令编码时被截为 17 位 mantissa"当作根因——单点数值反推不可靠，且无法解释 `uint32_t` 字段同样被截断。

## 误诊

本次定位前走过大量误诊路径（均被探针实测排除，记录于此防重复踩坑）：

- **vmul/vmadd 乘法器精度不足**：曾推断 arch35 Vector ALU 的 FP32 乘法中间结果低于 IEEE 754。排除依据：编译期常量方案 0 ULP，证明 `Add/LeakyRelu/Muls` 指令本身精度完全正确。
- **Duplicate/Muls 标量立即数编码截断（"17 位 mantissa"）**：曾从单点数值反推"标量在指令编码时被截为 17 位 mantissa"。排除依据：单点拟合不可靠，且 DataCopy GM→UB（DMA 全精度搬运）+ Mul 仍是 38 ULP，而编译期常量 0 ULP。
- **Veltkamp split + Duplicate + Mul**：曾试图用 12-bit 分解标量绕过截断。排除依据：Duplicate 本身截断标量，split 后两部分被分别截断，无效。
- **FusedMulAdd（vmadd）FMA**：曾试图用 FMA 单次舍入绕过。排除依据：输出与 Mul 完全相同。
- **SIMT 引擎标准 FP32 算术**：曾试图用 `Simt::VF_CALL` + `__simt_vf__` 走 SIMT 标量处理器绕过 Vector ALU。排除依据：arch35（3510）的 VF_CALL 实际在 `__VEC_SCOPE__` 中直接调用函数，SIMT 标记不触发独立 SIMT 线程，输出与 Vector 路径一致。
- **TilingData float 序列化截断**：曾怀疑 TilingData struct 的 float 字段在 host→device DMA 时被截断。排除依据：把同值存为 `uint32_t scaleBits` 字段同样被截断，排除了"float 类型序列化"假说，指向"值在写入 TilingData 前就已被字符串截断"。
- **二进制缓存未更新**：三次不同代码路径输出 hex 完全相同曾让人怀疑 build 用了缓存 kernel binary。排除依据：`Muls(y, y, (T)3.0, n)` 标记测试输出 10.6066 = 3.5355×3，证明二进制确实更新。
