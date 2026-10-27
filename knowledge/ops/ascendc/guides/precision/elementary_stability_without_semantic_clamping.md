---
type: Guide
title: 初等函数稳定变换与语义保留
description: 纠正 Exp 范围、无条件 epsilon/截断及 Sinh 平移错误，补充负状态抵消与输出 dtype 的分离诊断。
tags:
- task.design
- topic.precision
- topic.compute.elementwise.math_function
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/common-traps.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/case-studies.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/debug-summary-template.md
  role: supporting
created_at: '2026-09-11T02:40:45Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: []
---

# 初等函数稳定变换与语义保留

浮点范围与有效精度是不同限制。FP16最大有限值65504，ln(65504)约11.09；FP32最大有限值的自然对数约88.72。具体Exp溢出舍入边界由实现决定，但来源“FP16在x>88才溢出”显然混用了类型。log(0)与log(负数)也应按API/算子特殊值合同区分，不能一概只称无定义并任意替换。

对有限且有有效元素的Softmax，减max使Exp输入不为正，可避免该处正向指数溢出；分母仍可能在低精度长行求和时溢出或失去有效位。应按需要使用较宽累加和归一化，不能先在FP32求和又过早cast为half而丢失好处。NaN/Inf、全masked和空行另需定义。LogSumExp必须把max加回；源码half返回类型却只写外部output且无return的骨架不是完整API实现。

普通Sinh=(exp(x)-exp(-x))/2在接近0时会抵消，拓宽关键中间值可降低该误差，但不能恢复输入已丢失信息，也不保证全域准确。原总结模板对Sinh提出“先减最大值再Exp”却没补回缩放，改变了结果；Softmax的共同因子抵消不能直接套到独立Exp/Sinh。大幅值时中间Exp与最终Sinh的可表示范围也不同，需对完整公式处理范围。

sqrt(x+1)-sqrt(x)在实数x>=0可有理化为1/(sqrt(x+1)+sqrt(x))，1-cos(x)可改为2*sin(x/2)^2。它们在目标域的数学等价不保证有限精度、无穷值和溢出下逐位等价；评估实现可用的函数精度及中间范围。把已舍入half a/b拓宽后相减也不能恢复原始实数位数。

Sigmoid的1/(1+exp(-x))在负大x仍会产生大指数，可按符号选等价形式：x>=0使用原形式，x<0使用exp(x)/(1+exp(x))，再按特殊值合同核对。Exp(min(x,10))、分母加epsilon、用max(denom,eps)或小分母直接输出0都会改变原数学语义，尤其负分母；仅在算子定义要求时采用，epsilon还需检查目标dtype能否表示。

Sum/Mean存在累计误差，Max/Min不做加法但仍有NaN、tie/索引、padding和输入量化语义，不能说完全不受精度影响。来源SoftmaxV5短列失败不能证明所有Reduce最少8元素；应核对具体API、count/mask和tail实现。RoundMode的支持组合与语义也按具体接口核验，不把CAST_ODD/HYBRID等的来源概括或“多行相同必为ROUND错误”当普适结论。

验证小量、抵消、大幅值、长行、特殊值与原算子域，保留原有舍入点和完整输出。来源精度改善数字及案例根因没有在本批设备复现；Host数学检查仅确认本文所列数值范围和输入表示反例。

## 负初始状态与输出 dtype 的分离诊断

分母由状态项与新输入相加时，正指数权重不保证分母为正或远离零。一个独立普通 TimeDecayExponentialStabilization reference 直接计算：

```text
p = max(max_state, key + time_first)
e1 = exp(max_state - p)
e2 = exp(key + time_first - p)
y = (e1*num_state + e2*value) / (e1*den_state + e2)
```

该 reference 的输入生成器对浮点状态使用 `randn`，允许负 `den_state`；参考文件 SHA-256 为 `63c858124662bdea7f3e14d0350a6f34b754915ab664d51d5f8cd141b0aa5860`。这只说明负状态属于该参考所生成的输入域，不把任何随机种子或“最坏元素”回填为本次复现。

取 `max_state=key=time_first=0`、`num_state=0`、`value=1`，则 `e1=e2=1`。选 `den_state=-1+2^-24`，原分母为 `2^-24`、结果为 `2^24=16777216`；额外添加 `1e-7` 后，结果约为 `6265481.8186`。初始状态与小分母都可以用 float32 精确表示；这个有理数反例只证明两种表达式语义不同，不模拟设备的每一步舍入。取 `den_state=-1-2^-23` 时分母为负的小量，同样会被 epsilon 显著改变；取 `den_state=1` 是正分母对照，取 `-1` 则原商在零分母处没有有限实数定义。不能因为新公式避开零分母，就认定它满足原合同；特殊值仍要另行界定。

这个任务还提供另一条独立诊断线索：reference 将状态转为 float32，并分配 float32 输出，返回类型不是简单跟随输入类型。若结果 `16777216` 再被多余的 float16 cast 缩窄，会超出 half 有限范围；标准库 binary16 范围检查检测到溢出（探针以 Infinity 标记该结果），而 `65504` 仍有限。应将“分母表达式改变”和“wrapper 新增舍入／溢出”分开检查。它们能同时出现，不能只凭一个极大误差数字判定是哪一项。

归档原始日志记录修改前 8/10 匹配，修改后 10/10 匹配；两份日志 SHA-256 为 `a641fabf6bb6d56730532e5114423a1359806d8b613e96a03060c215ad25a0fa` 与 `2d13dedd346f1a643e34490ccbc6be01fc3489d60e7222d0ac7bdaaa6f68de9f`。最终源码可见无 epsilon 的商与保留 float32 的 wrapper；本次未获得 epsilon 与输出 cast 各自的单因素消融；归档修复记录还涉及相邻同步代码，原始随机输入和历史加载二进制尚未完整绑定。本页不采用摘要中的 50/50，不把 10/10 单独归因于移除 epsilon，也不将静态代码对应关系写成新的设备验证。

本次新证据是原创标准库有理数反例、float32 可表示性及 binary16 范围检查；没有重跑旧工程、PyTorch 或 NPU。可审阅附件 `split_contracts` 提供原件逐行提取绑定、四组商输入／完整输出、四个缩窄值及 `verify_contracts.py`。使用到其他递推式时，须重新固定输入域、状态更新和返回 dtype，不能由本例推导普遍禁止 epsilon 或普遍要求 float32。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/source_snippets.json`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/inputs.json#/quotient_cases`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/outputs.json#/mathematical_quotient_cases`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/outputs.json#/output_dtype_boundary`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/outputs.json#/historical_extract_checks`；`cannbot_non_level4_split_review_20261019/evidence_public/split_contracts/verify_contracts.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。
