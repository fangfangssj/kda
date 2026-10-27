---
type: "Runbook"
title: "Philox offset 须放 counter[2:3] 而非 [0:1]"
description: "自定义 Philox kernel 把 generator offset 放入 PhiloxCounter 的 position 字段 counter[0:1]，原生 aclnnSimThreadExponential 放入 subsequence 字段 counter[2:3]，导致 offset=0 时逐元素匹配、offset!=0 时 0% 匹配；用单核逐字段试错对比判别。"
tags: ["task.debugging", "topic.precision", "topic.compute.random_sampling"]
status: stable
sources: []
created_at: "2026-08-18T08:42:11Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Philox offset 须放 counter[2:3] 而非 [0:1]

## 现象

在修复 Philox `Rounds=10` 与 `philox_engine_inputs` increment=12 两个根因后，自定义 `exponential_` 的 offset 推进已与原生逐次对齐（Native 与 Custom 的 `seed=42 offset=0/12/24/36/48` 完全相同）。但单核小尺寸 `compare_small.py`（`shape=(1,128)` 连续 5 次）仍发散——qrun 输出原文：

```
Single-core test: shape=(1, 128), seed=42, calls=5
--- Native ---
  Native call 1: offset 0->12, first3=[1.8045962 2.4804842 0.6826031]
  Native call 2: offset 12->24, first3=[2.4778473  0.5126816 0.34211105]
  ...
--- Custom ---
  Custom call 1: offset 0->12, first3=[1.8045962 2.4804842 0.6826031]
  Custom call 2: offset 12->24, first3=[2.2260642  0.7394174  0.02328083]
  Custom call 3: offset 24->36, first3=[0.7294245  0.09936458 0.23564452]
  ...
--- Comparison ---
Call 1: match=True
Call 2: match=False
Call 3: match=False
Call 4: match=False
Call 5: match=False
```

关键特征：**offset=0 的 Call 1 匹配，offset=12 的 Call 2 起 0% 匹配**，尽管两边的 offset 数值完全相同（0→12→24→36→48）。这说明问题不在 offset 推进量，而在 kernel 内部把 offset 值放入 PhiloxCounter 的哪个字段。

## 适用场景

- **工程形态**：kernel 直调（自定义 AscendC kernel 内部构造 `PhiloxCounter`/`PhiloxKey` 并调用 `PhiloxRandom`）。
- **API 路径**：`AscendC::PhiloxCounter`（128 位 counter，字段布局 `[position_lo, position_hi, subseq_lo, subseq_hi]`）+ `PhiloxRandom<10, uint32_t>`。
- **验证平台与版本**：Ascend950（`[AIV Block 0/32]` 单核测试）；CANN 版本未从 trace 恢复。
- **适用粒度**：`operator_family`（所有自定义 Philox-based RNG kernel 需与原生 `exponential_`/`uniform` 逐 bit 对齐者）。
- **路线优先级**：`recommended_fastpath`（counter 字段放置是逐 bit 对齐的必要条件）。
- **不适用或未验证场景**：不使用 Philox counter 的 RNG；不需要与原生逐 bit 对齐的场景；a2/a3 未验证（见 ## 失效边界）。

## 判别方法

**触发（trigger, tool_observed）**：`Rounds=10` 与 increment=12 都已修复、offset 数值两边逐次相同（0/12/24/36/48），但单核多次调用仍出现 offset=0 匹配、offset!=0 不匹配。即 `compare_small.py`（`shape=(1,128)`，连续 5 次）输出 `Call 1: match=True`、`Call 2-5: match=False`，Native call2 `first3=[2.4778473 0.5126816 0.34211105]` 与 Custom `[2.2260642 0.7394174 0.02328083]` 完全不同。

**探针（probe）**：写 `compare_small.py` 在单核（`shape=(1,128)`，避免多核切分干扰）下连续调用 5 次 `exponential_`，同时打印 Native/Custom 每次的 offset 变化与 first 3 值；在 kernel 内把 offset 分别放入 `counter[0:1]` 与 `counter[2:3]` 各构建一次 `.so` 跑同一对比（逐个试错对比，因原生 counter 布局无公开文档）。判据：若 offset=0 匹配而 offset!=0 不匹配，且改放 `counter[2:3]` 后全匹配，则命中本判别。

**观测（observation, tool_observed）**：
- offset 放 `counter[0:1]`：Call 1 `match=True`，Call 2-5 `match=False`（Native call2 `first3=[2.4778473 0.5126816 0.34211105]` vs Custom `[2.2260642 0.7394174 0.02328083]`）。
- offset 改放 `counter[2:3]` 后（qrun 输出原文）：

```
--- Comparison ---
Call 1: match=True
Call 2: match=True
Call 3: match=True
Call 4: match=True
Call 5: match=True
```

- 多 shape 多 call 全验证（`verify_all.py`，qrun 输出原文）：

```
--- Shape=(1, 128) (128 elements), 5 calls ---   全 match=True
--- Shape=(1, 1024) (1,024 elements), 5 calls --- 全 match=True
--- Shape=(1, 387840) (387,840 elements), 5 calls --- 全 match=True
STATUS: ALL PASSED
```

**解释（interpretation, agent_reported 分析）**：Philox counter 是 128 位，按 `[position_lo, position_hi, subseq_lo, subseq_hi]` 布局（`position` = counter[0:1]，`subsequence` = counter[2:3]）。自定义 kernel 把 generator 返回的 offset 值放入 `counter[0:1]`（position 字段），再在多核切分时用 `Skip(counter, blockOffSet/4)` 把 position 前进到各核起始 group。但原生 `aclnnSimThreadExponential`（内部走 `StatelessRandomUniformV2`）把 offset 放入 `counter[2:3]`（subsequence 字段）。机制：`position` 字段每生成一个 Philox group（4 个 uint32）自动 +1，而 `subsequence` 字段在一次 `PhiloxRandom` 调用内不自动推进；把 offset 放 position 意味着 offset 会被 group 生成循环与 Skip 反复累加进位，放 subsequence 则让 offset 作为"子序列选择"在整个调用内稳定。offset=0 时两者效果恰好相同（position=0、subsequence=0 都从第 0 个 group 开始），所以 Call 1 匹配；offset=12 时放 position 的 counter 从 position=12 开始、放 subsequence 的从 subsequence=12 开始，两者生成完全不同的 4×32 输出，所以 Call 2 起 0% 匹配。改放 `counter[2:3]` 后 offset=0..48 全部逐元素一致，多 shape 多 call `STATUS: ALL PASSED`。三个根因（Rounds=10、increment=12、counter[2:3]）同时满足后，自定义 `exponential_` 与原生在单核、多核、多次连续调用下均逐 bit 对齐。注意：原生 counter 内部布局无公开文档，`counter[2:3]` 的具体放置方式（仅 counter[2] 还是 counter[2:3] 两槽）系逐个字段试错对比得出。

## 验证信号

来源：自定义 `StatelessRandomUniformV3` kernel 替换 `aten::exponential_` 的精度对齐会话（Ascend950，单核 `[AIV Block 0/32]`）。修复前 offset 放 `counter[0:1]` 发散、改放 `counter[2:3]` 后逐 bit 一致，均在本轮内实测；kernel `.asc` 终版 verbatim 未从 trace 恢复。

- offset 放 `counter[0:1]`：单核 `shape=(1,128)` 5 次调用，Call 1 `match=True`，Call 2-5 `match=False`（Native call2 `first3=[2.4778473 0.5126816 0.34211105]` vs Custom `[2.2260642 0.7394174 0.02328083]`）（来源平台:Ascend950；证据等级:tool_observed）。
- offset 改放 `counter[2:3]` 后：单核 `shape=(1,128)` 5 次调用 Call 1-5 全 `match=True`（来源平台:Ascend950；证据等级:tool_observed）。
- `counter[2:3]` 多 shape 多 call（128/1024/387840 各 5 次）：全 `match=True`，`STATUS: ALL PASSED`（来源平台:Ascend950；证据等级:tool_observed）。
- offset 数值两边逐次相同（Native 与 Custom 均为 `0/12/24/36/48`），排除 offset 推进量根因（来源平台:Ascend950；证据等级:tool_observed）。

## 失效边界

- 本结论在 Ascend950 + 当前 CANN 版本上验证；a2/a3 未实测，**未验证，推断可迁移**（counter 字段布局是 Philox 算法本身的标准，理论上跨平台一致，但原生 L0 op 的内部约定可能随版本变化，需用同款单核对比脚本实测确认）。
- 三个根因（Rounds=10、increment=12、counter[2:3]）须同时满足；仅修 counter[2:3] 而不修 Rounds/increment 不会得到一致输出。
- `counter[2:3]` 的具体放置方式（仅 counter[2] 还是 counter[2:3] 两槽）与 `Skip` 函数实现未从 trace verbatim 恢复，本卡只承载单核对比的 qrun 输出与试错判别，不承载 kernel verbatim after 实现。