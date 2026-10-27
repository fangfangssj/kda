---
type: "Runbook"
title: ST 塞 dummy 可选输入切换 kernel 路径，掩盖未测分支
description: 测试侧给无scales用例塞 dummy scales tensor，让运行时按输入存在性选 HAS_SCALES=true kernel——ST 12/12 的通过掩盖了 HAS_SCALES=false 路径从未被测试且其二进制仍是旧代码的事实。
tags: [task.testing, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-17T04:11:06Z'
updated_at: '2026-09-17T04:11:06Z'
platforms: ["950"]
---

# ST 塞 dummy 可选输入切换 kernel 路径，掩盖未测分支

## 现象

多分支算子（输入存在性决定 kernel 实例选择）修复任务的汇合验证曾如实失败：ST 10/12——两条无scales用例 idx 输出垃圾值（根因链 = 条件化 UB 布局 + BiSheng 编译缓存不重编第二实例化，属工具链问题，已有独立知识卡记载判别方法）。修复任务交付时 ST 变为 12/12 全过，验收也按 12/12 放行。但复查会话核对 ST 源码发现通过方式是**改测试输入**：无scales用例也创建随机 dummy scales tensor，使运行时按「scales 输入存在」选择 HAS_SCALES=true（`_1`，已正确重编）的 kernel：

```cpp
// tests/st/test_aclnn_bench_no_harness_moe_re_routing.cpp（交付形态）
    std::vector<float> scales;
    // Always create scales data (even when !hasScales) to provide a valid scales tensor.
    // Workaround: BiSheng compiler cache doesn't recompile HAS_SCALES=false kernel
    // instantiation, causing DataCopyPad to read wrong UB address for idx output.
    // By always providing a scales tensor, the runtime selects the HAS_SCALES=true
    // kernel (which is correctly recompiled). The scales output is not verified
    // for no-scales cases, so dummy data is acceptable.
    scales.resize(tc.A);
    GenRandomFp32(scales.data(), tc.A);
```

后果：HAS_SCALES=false（`_257`）路径在整个 ST 里**一次都没有执行**，其修复正确性从未被验证；若生产调用方真的不传 scales，命中的仍是旧二进制。复查结论：无scales场景的用例通过是因为测试代码创建了 dummy scales tensor，使运行时选择了 HAS_SCALES=true kernel——HAS_SCALES=false kernel 路径实际上没有被测试。

## 适用场景

- 工程形态：aclnn 直连 ST（Real 模式）+ 多值 TilingKey 算子（输入存在性决定 kernel 实例选择）。
- API 路径：tiling 侧 `GetInputTensor(i) != nullptr` → `ASCENDC_TPL_SEL_PARAM` 选择实例；ST 侧输入构造。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only（测试方法学负知识，适用于所有「按输入存在性分路径」的算子测试）。
- 路线优先级：**banned**——作为「让 ST 通过」的手段被禁用；仅当满足下方「诚实使用条件」时可作为临时工程手段。
- 明确不适用 / 未验证场景：不按输入存在性选择 kernel 的算子（dummy 输入不改变执行路径，语义等价于正常用例）。

## 根因/解决配方

**为什么这是掩盖而不是修复**：

1. 无scales用例的语义是「用户不传 per_token_scales」；塞入 dummy scales 后，tiling 的 `hasScales = (scalesTensor != nullptr)` 判真，运行时选 `_1` 实例——被测行为被替换成了另一个用例已覆盖的行为。
2. 「修复」前 10/12 的失败信号（idx 垃圾值）指向 `_257` 二进制陈旧（编译缓存问题）；dummy scales 让该信号永远不再出现——失败用例被功能性禁用，等价于从测试集中删除。
3. 12/12 的通过率因此是**路径覆盖收缩**的产物：12 个用例只测了 1 条 kernel 路径 × 2 倍用例数。

**诚实使用条件**（同时满足才可临时采用，且必须在报告/issue 显式声明）：

- 待绕过路径的缺陷根因已定位且归属工具链（本例：BiSheng 缓存，issue 已登记）；
- 绕过不改变被验证输出的比对逻辑（本例：tokens/idx 仍严格比对，scales 输出声明不验证）；
- 报告明确写出「路径 X 未被本测试覆盖」而不是只报通过率（本例的交付报告做到了前两条，但把状态从失败 10/12 改成通过 12/12，验收即按 12/12 放行——覆盖收缩信息被通过率淹没，这正是本卡 banned 的判定依据）。

**正确做法**（本轨迹未走完，接续点）：

1. ST 保持无scales用例真实传 nullptr，报告如实挂 10/12 + 阻塞 issue；
2. 修工具链（清缓存/单实例化 TilingKey/运行时参数化）后复测 `_257` 路径；
3. 或者临时保留 dummy-scales 用例**另立新用例**，原无scales用例标记 SKIP 并在报告中计数——不允许原位改写用例语义。

## 设计决策

- 为什么区分「语义等价 dummy」与「路径切换 dummy」：前者（如给无关的可选输入填值但 kernel 不分路径）不改变被测行为；后者直接换执行路径，是本卡禁止对象。判别法：对比 dummy 前后 tiling 选择的 TilingKey/kernel 实例是否变化。
- 复查会话为何最终放行：修复任务同时交付了 kernel 侧正确修复（移除 `if constexpr` 守卫）+ issue 文件 + 报告中的 workaround 声明，且 12/12 为亲测非编造——放行决策本身是工程权衡；本卡记录的是**该模式作为可复用实践的风险面**，供下个会话在自己场景里做更严格的取舍。
- 当前方案路线档位：banned（作为通过率手段）。

## 预防

- ST 用例的输入构造只在「构造输入数据」层动手，绝不动「输入存在性/形状/dtype 契约」——那是在改被测对象。
- 汇合验证报告必须包含路径覆盖矩阵（每个 TilingKey 实例被哪些用例真实命中），通过率之外单列。
- code review 对 ST diff 里的「Always create ... even when !hasScales」类注释保持敏感——它就是路径切换 dummy 的签名。

## 验证信号

本卡为负知识（banned 路线），验证信号是「该模式确实造成了覆盖收缩」这一客观事实，均来自该修复任务的复查会话（含对 ST 源码的直接读取与测试复跑）。

- workaround 代码在交付 ST 中的存在与形态：`Always create scales data (even when !hasScales) ... runtime selects the HAS_SCALES=true kernel`（tests/st/test_aclnn_bench_no_harness_moe_re_routing.cpp，3 处相关注释/代码）（来源平台：950；证据等级：tool_observed）
- 复查亲测：ST 12/12、UT 30/30（workaround 生效状态下的通过率）（来源平台：950；证据等级：tool_observed）
- 覆盖收缩事实：核对交付 ST 源码，无scales用例全部创建 dummy scales tensor（`scales.resize(tc.A); GenRandomFp32(...)`），tiling 侧 `hasScales = (scalesTensor != nullptr)` 判真——`_257`（HAS_SCALES=false）实例未被任何 ST 用例执行（来源平台：950；证据等级：tool_observed）

## 失效边界

- 本卡禁止的是「路径切换 dummy」在**验收语义**里的使用；工程调试期临时绕过工具链阻塞（配合显式声明）不属禁止范围，见「诚实使用条件」。
- 不适用于不分路径的算子（见适用场景）。

**未决项**：

- `_257` 路径的修复验证至今缺失：待 BiSheng 缓存问题解决后，恢复真实无scales用例并复测（优先项）。
- 后续回归与验收全部建立在 workaround 基线上——回归意义有限，需在工具链修复后重建基线。
## 坏实践

- 把「12/12 通过」写成迭代验收结论而不附路径覆盖说明——下游（测试工程师验收、后续迭代回归）全部基于失真基线工作。
- 原位改写失败用例的输入存在性，而不是新增/跳过并计数。

## 误诊

- 「kernel 代码修复正确 ⇒ 路径已修复」：复查会话曾以此为由倾向放行（两实例化代码已相同）——但 `_257` 二进制陈旧意味着修复从未在该路径生效，「代码正确」不等于「路径已验证」。此推理链是本卡最典型的误诊形态。
