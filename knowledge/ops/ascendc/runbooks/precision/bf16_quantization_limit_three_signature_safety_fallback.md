---
type: "Runbook"
title: "bf16 量化极限的 MARE 略超阈三签名归类 — 阈值契约不可改时 safety_fallback 上报"
description: "bf16 输出算子在中间通路已全 fp32、MERE 低于阈值 100~1000 倍（最大裕度 578×）、不匹配元素占比 0.0% 时，8 条用例仅 query 的 MARE 略超阈（1.24e-01 vs 7.81e-02），最大差异集中在极小 golden（~3e-4）处——bf16 尾数 8 位 + fp32 累加顺序差异在量化前的残余 + 末端 bf16 舍入，属 bf16 量化极限。三签名（须同时满足，任一不满足继续按缺陷排查）+ safety_fallback 上报；配套 Mock/Real 判别分流。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# bf16 量化极限的 MARE 略超阈三签名归类 — 阈值契约不可改时 safety_fallback 上报

## 现象

MLA 前处理 bf16 算子验收（CANN 9.2.0）：中间通路已全 fp32，MERE 低于阈值 100~1000 倍（最大裕度 578×）、不匹配元素占比 0.0%（33/1048576）时，仍有 8 条用例仅 query 的 MARE 略超阈（1.24e-01 vs 7.81e-02），最大差异集中在极小 golden 值（~3e-4）处。

## 适用场景

- **工程形态**：bf16 输出算子的精度验收（中间通路全 fp32），MERE/MARE 双阈值。
- **API 路径**：比较器 `rel = |a-g|/(|g|+1e-7)`；Mock 模式 CPU golden 自洽校验。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.2.0。
- **适用粒度**：`operator_family`——bf16 输出且阈值契约不可改的算子；推断 bf16 类量化输出适用。
- **路线优先级**：`safety_fallback`（不放松、不删用例、不无方向追精度）。
- **明确不适用 / 未验证场景**：MERE 也超阈或 mismatch 率显著 >0%；非 bf16 或其他量化输出 dtype（fp16 尾数 10 位，量化步长不同）。

## 根因

bf16 尾数 8 位，量化步长 ulp≈9.5e-07；实测 diff≈40 ulp——是 fp32 累加顺序差异在量化前的残余 + 末端一次 bf16 舍入的组合，属 bf16 输出量化与相对误差放大共同作用的精度极限。

## 解决配方

### 三签名归类（须同时满足，任一不满足即继续按缺陷排查）

1. **MERE 低于阈值 2~3 个数量级**（最大裕度 578×）；
2. **不匹配率 ≈ 0.0%**（33/1048576）；
3. **最大差异位置都在极小 golden 值处**（golden≈2^-12 量级、diff≈40 ulp）。

### 处置

按 `safety_fallback`：报告"功能正确性已验证 + MARE 阈值需评估"，不放宽容差、不删用例、不在 kernel 里无方向追精度（常规手段已穷尽：内积法、跨流水线同步、标量 1/sqrt、fp32 中间 workspace）。

### Mock/Real 判定分流（配套判读规则）

CPU golden 自洽校验（actual = golden 经 fp16 写入读回的 round-trip）中，近零行 MARE 0.03~0.23 属"理想舍入"的指标伪影——Mock 判定以 **MERE < 阈值 且 无 NaN** 为准（MARE 仅参考打印），golden 数值正确性由手算用例独立锚定；Real（NPU vs golden）比对保持严格双指标不放宽。

**反例**：MARE 恒 ≈0.5 + 纯 gather 用例也失败 ⇒ 系统性倍差（查位级转换实现），与"离散 0.03~0.23 且集中在近零行"的指标伪影同貌不同因。

## 验证信号

- R11 触发观测：8 条用例同签名（仅 query MARE 略超阈 + MERE 低于阈值 100~1000 倍 + 不匹配 0.0%），L0-02 最大差异点 golden=3.070831e-04/actual=3.452301e-04；同批通过用例 MERE 0~1.35e-05（裕度最大 578 倍）（来源平台：950；证据等级：tool_observed）。
- Mock 分流：判定分流后 Mock 全量 PASS=6 FAIL=0（MARE 如实打印仅参考）；对照 Real 模式同指标小用例严格双指标通过（来源平台：CPU Mock 及 950 上板；证据等级：tool_observed）。

## 失效边界

- 40 ulp 机制解释（fp32 累加顺序 + 末端舍入）未严格验证（单元素差异确切来源为优先项）。
- 本签名只覆盖"MERE 达标 + 不匹配率≈0 + 差异集中在小值"，其他失败签名（NaN、大 mismatch、MERE 超阈）必须继续按缺陷排查。
- Mock 分流只适用于"actual 与 golden 无实现级差异"的自洽比对；真实比对出现近零 MARE 超阈时先做判别信号分析后再归类。

## 未决事项

- 40 ulp 的确切来源验证（单元素差分确定归属于 fp32 累加顺序还是末端量化）。
- 三签名的统计稳健性：极限情况（MERE 恰好在阈值边缘、mismatch 极少量但分布随机）的判责边界尚未有实证。