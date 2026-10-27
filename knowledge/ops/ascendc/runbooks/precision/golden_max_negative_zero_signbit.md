---
type: "Runbook"
title: golden三目求max对±0返回-0，需双零特判
description: "CPU golden 用 C++ (a>=b)?a:b 实现 max 时 max(-0,+0) 返回首参 -0（-0==+0 为真），与 torch.maximum 的 maxNum 语义（恒 +0）不一致；golden 须对双零做 signbit 特判。"
tags: [task.testing, topic.precision, topic.compute.elementwise.comparison_logic]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# golden三目求max对±0返回-0，需双零特判

## 现象

为逐元素最大值自定义算子（对标 PyTorch `torch.maximum`）开发 C++ ST 测试的 CPU golden 自测时，自测项"测试20: +0/-0 比较 (max(+0,-0)=+0, IEEE 754)"失败，同轮 grep 失败项输出原文：

```
  [FAIL] 测试20: +0/-0 比较 (max(+0,-0)=+0, IEEE 754)
```

当时代码状态：golden 的标量最大值用 `MaxElementDouble` 实现，内部是 `return (a >= b) ? a : b;`（前面已有 NaN 传播分支）。该自测构造三组调用：`max(+0,-0)`、`max(-0,+0)`、`max(+0,+0)`，并用 `!std::signbit(r)` 校验结果必须是 +0。`max(-0,+0)` 时三元表达式返回首参 -0.0，`signbit(-0.0)==true`，校验失败。其余 25 项自测（含 NaN/inf/边界值）全部通过，全量 72 用例中仅此语义点错。

## 适用场景

- 工程形态：算子仓开发的 ST/UT 的 CPU golden 参考实现（C++），算子语义对标 PyTorch/NumPy 的 `maximum`。
- API 路径：自研 `MaxElementDouble`/golden 标量核；对标语义为 `torch.maximum`/`numpy.maximum`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）+ CANN 9.1.0（golden 为 CPU 逻辑；含 ±0 用例的全量 ST 随后在 NPU Real 模式 72/72 验证，与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"逐元素 max/clamp/threshold 类"算子的 CPU golden 实现。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：整数 golden（无 ±0）；NPU 侧 ±0 行为（kernel 输出 ±0 的位级语义未在本闭环单测，仅通过全量比对间接覆盖）；min 类算子（maxNum→minNum 对称处理，未验证）。

## 根因/解决配方

**根因（单句）**：IEEE 754 中 -0 与 +0 数值相等但符号位不同，C++ 三元 `(a>=b)?a:b` 在平局时取首参，导致 `max(-0,+0)→-0`（`-0>=+0` 为真）、`max(+0,-0)→+0`，结果依赖参数顺序；而 `torch.maximum`/`numpy.maximum` 遵循 maxNum 语义对 `max(±0)` 恒返回 +0，golden 未做双零特判故与框架语义不一致。

机制链：`-0 == +0`（IEEE 754 比较）→ `(a>=b)` 为真 → 返回首参 `a=-0.0` → `std::signbit(-0.0)==true` → 与期望的 `+0`（signbit=false）不符。

修复（before/after verbatim 取自本轨迹编辑事件）：

```cpp
// before：三目在 ±0 平局时取首参，max(-0,+0) 返回 -0
double MaxElementDouble(double a, double b) {
    // NaN 传播：若任一为 NaN，返回 NaN
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (a >= b) ? a : b;
}
```

```cpp
// after：双零显式特判，对齐 IEEE 754 maxNum / torch.maximum 语义
double MaxElementDouble(double a, double b) {
    // NaN 传播：若任一为 NaN，返回 NaN
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // IEEE 754 maxNum 语义：max(+0, -0) = +0（与 PyTorch torch.maximum / numpy 一致）
    // C++ (a>=b)?a:b 对 max(-0,+0) 会返回 -0（因 -0==+0 为真取首参），需特殊处理。
    if (a == 0.0 && b == 0.0) {
        return (std::signbit(a) && std::signbit(b)) ? -0.0 : 0.0;
    }
    return (a >= b) ? a : b;
}
```

## 设计决策

- 用 `a == 0.0 && b == 0.0` 捕获 ±0（-0==+0==0 均为真），再以 `signbit` 双负才返回 -0：与 maxNum"任一为 +0 则 +0"的语义精确一致，避免逐分支枚举。
- 不引入 `std::max`/`fmax` 替代：`std::max` 同样是首参语义，`fmax` 的 ±0 行为实现可异，显式特判最可控。
- 当前方案属 recommended_fastpath。

## 预防

- 为对标 torch/NumPy 的极值/裁剪类算子写 CPU golden 时，自测清单固定包含：NaN 传播（含广播）、±inf 同异号、`max(+0,-0)`/`max(-0,+0)` 双方向、整型边界值。
- CR 清单：golden 里出现 `(a>=b)?a:b` 或 `std::max` 处理浮点时，检查 ±0 与 NaN 两个语义点。

## 验证信号

来源说明：问题与验证来源：迭代三 C++ ST 测试工程 golden 自测（修复后重编译重跑同二进制）。

- 修复前：`[FAIL] 测试20: +0/-0 比较 (max(+0,-0)=+0, IEEE 754)`，自测 `PASS=25, FAIL=1`（来源平台：950；证据等级：tool_observed）。
- 修复后：`[PASS] 测试20: +0/-0 比较 (max(+0,-0)=+0, IEEE 754)`，自测 `PASS=26, FAIL=0`，全量 `总计: 72 通过: 72 失败: 0`（Mock）（来源平台：950；证据等级：tool_observed）。
- 含 +0/-0 比对用例（L1-009 无广播 float32 +0/-0 比较）的 Real 模式 NPU 全量 72/72 通过，浮点 MERE=0.00e+00（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论针对 CPU golden 与框架语义的对齐；NPU kernel 对 ±0 输出的位级行为（是否恒 +0）未单独位级比对，仅经全量精度比对间接覆盖。
- 同一三目实现若用于 min 语义（minNum：任一为 -0 则 -0），特判方向需对称反转，未验证。

**未决项**：

- 无。
## 坏实践

- 用 `std::max`/裸三元直接当浮点 golden 的 max（平局取首参，语义依赖参数顺序）。
- 自测只测 `max(+0,-0)` 一个方向（本例正是 `-0` 在首参位时才暴露）。

## 误诊

- 无。
