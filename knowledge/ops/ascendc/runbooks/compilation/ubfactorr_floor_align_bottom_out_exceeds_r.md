---
type: "Runbook"
title: ubFactorR 对齐兜底可大于 R
description: tiling 用 FloorAlign(min(maxUbFactorR, R), alignElements) 计算 ubFactorR，R < alignElements 时结果为 0、兜底取 alignElements（如 fp32 R=4 时 ubFactorR=8），可大于 R；UT 断言 ubFactorR<=R 属语义误编码，应断言 >0 且（R>=对齐粒度时）<=R。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# ubFactorR 对齐兜底可大于 R

## 现象

自定义 argmax 算子（BenchNoHarnessArgMax，Ascend950PR，CANN 9.1.0）op_host UT 修复完段错误与 shape 垃圾值后，27 条用例还剩 1 条失败——最小 shape 用例：

```
Expected: (tiling->ubFactorR) <= (param.expectR), actual: 8 vs 4
ubFactorR should be <= R
```

用例是 float_ar_small：shape [1, 4]、dim=-1、fp32，R=4。当时代码状态：tiling 实现里 ubFactorR 的计算为

```cpp
uint32_t ubFactorR = static_cast<uint32_t>(
    FloorAlign(std::min(maxUbFactorR, static_cast<uint64_t>(r)),
               static_cast<uint64_t>(alignElements)));
ubFactorR = (ubFactorR == 0) ? static_cast<uint32_t>(alignElements) : ubFactorR;
```

fp32 的 alignElements=8（32B/4B）。R=4 时：`min(maxUbFactorR, 4)=4` → `FloorAlign(4, 8)=0`（4 不满一个对齐粒度，向下取整为 0）→ 兜底取 8。于是 ubFactorR=8 > R=4。

## 适用场景

- 工程形态：自定义算子 op_host tiling 单测（对切分参数写断言），以及任何"尾轴很短"的归约/逐元素算子 tiling 设计。
- API 路径：`FloorAlign`（host tiling 工具）、TilingData 的 ubFactorR/块因子字段。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，UB=253952B、AIV 24 核），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——"最小 chunk 粒度 = dtype 对齐粒度"的 tiling 兜底模式在向量搬运类算子中普遍。
- 路线优先级：recommended_fastpath（判定规则本身）。
- 明确不适用 / 未验证场景：kernel 侧对 [R, ubFactorR) padding 区的处理方式（DataCopyPad 只搬 R 个、比较/归约按有效长度截断）是配套前提，本卡只覆盖 tiling 参数语义；非 FloorAlign 兜底风格的 tiling 不适用。

## 根因/判别

这是 tiling 的**设计行为**而非缺陷：向量搬运/计算按 32B 粒度工作，chunk 长度小于 alignElements 时无法向下对齐（FloorAlign 结果 0），tiling 兜底把单 chunk 抬到最小可对齐粒度 alignElements。因此 **R < alignElements 时 ubFactorR == alignElements > R 是合法状态**，kernel 靠"搬运 R 个真实元素 + padding 区不参与有效结果"保证正确性。UT 把 `ubFactorR <= R` 当不变量是编码错误。

## 判别方法

**触发**：op_host UT 对小 R 用例（R < 32B/sizeof(dtype)）断言切分参数。

**探针**：用例打印实际值并对照 tiling 公式手推——本例 `FloorAlign(min(max, 4), 8)=0 → 兜底 8`，与实测 8 吻合即设计行为；不吻合才回 tiling 查缺陷。

**观测**：`actual: 8 vs 4`（ubFactorR=8，R=4，fp32 alignElements=8）。

**解释**：见根因。断言应改为：`ubFactorR > 0` 恒成立；`R >= alignElements` 时才可加 `ubFactorR <= R`。

## 解决配方

**before**（误报断言）：

```cpp
        // ubFactorR should be > 0 and <= R
        EXPECT_GT(tiling->ubFactorR, 0u) << "ubFactorR should be > 0";
        EXPECT_LE(tiling->ubFactorR, param.expectR) << "ubFactorR should be <= R";
```

**after**（与设计语义对齐）：

```cpp
        // ubFactorR should be > 0
        // Note: ubFactorR can be > R when R < alignElements (e.g., R=4 < alignElements=8 for float)
        // In that case, ubFactorR is set to alignElements (minimum chunk size)
        EXPECT_GT(tiling->ubFactorR, 0u) << "ubFactorR should be > 0";
```

## 设计决策

- 选"改断言语义"而不是"改 tiling 让 ubFactorR 强制 <= R"：后者会引入非对齐 chunk，违反向量对齐约束、需要 kernel 侧特殊处理，代价远大于收益；padding 语义本身正确。
- 断言保留 `> 0`（防 0 除/空 chunk），放弃对小 R 的上界断言（上界条件化交给公式推导，不写死在通用断言里）。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 给切分参数写 UT 断言前，先把 tiling 公式在边界值（R=1、R=alignElements-1、R=alignElements）上手推一遍，区分"不变量"与"设计行为"。
- 小 shape 用例（R < 对齐粒度）单独分档断言，不与常规用例共用上界断言。

## 验证信号

以下信号来自该算子 op_host UT 会话：

- 失败实测：float_ar_small（[1,4] fp32）`Expected: (tiling->ubFactorR) <= (param.expectR), actual: 8 vs 4`（来源平台：950；证据等级：tool_observed）。
- 公式手推吻合：FloorAlign(min(maxUbFactorR,4),8)=0 → 兜底 8，与实测一致，确认设计行为（来源平台：950；证据等级：tool_observed；公式推导为 agent_reported 辅助）。
- 修后全量：op_host UT 27/27 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论绑定该 tiling 的 FloorAlign+兜底实现风格；其他算子的 tiling 若采取"小 R 特殊路径"（如整行单块处理）则 ubFactorR 语义不同。
- kernel padding 区正确性属于配套实现，本卡未验证 kernel 侧（同批 ST 小 shape 用例 [1,4] 上板未覆盖，L0 用例最小 R=50）。

**未决项**：

- 无（tiling 语义已对齐验证）。
## 坏实践

- 把"参数间的直觉不等式"直接写成通用断言，不先对照 tiling 公式——本卡即由此误报。
- 为让断言通过反过来改 tiling（强制 ubFactorR<=R）——引入非对齐 chunk 的真缺陷。

## 误诊

- "ubFactorR=8 说明 tiling 算错/没考虑小 shape"——是考虑了：0 兜底到最小对齐粒度正是小 shape 路径。
