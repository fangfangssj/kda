---
type: Guide
title: 推理 RunInfo 初始化与 Query/KV 坐标读取
description: 从 FA 模板的条件初始化与消费路径识别潜伏未初始化字段，按推理语义建立默认值及逐实例验证。
tags:
- task.debugging
- topic.attention
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/patterns/domains/fa_class/templates/op_kernel/wholeport/wp_util_regbase.h
  role: header
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/patterns/domains/fa_class/templates/op_kernel/wholeport/wp_block_cube.h
  role: implementation
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/patterns/domains/fa_class/templates/op_kernel/wholeport/wp_kernel_train.h
  role: implementation
created_at: '2026-09-14T04:24:48Z'
updated_at: '2026-09-14T04:24:48Z'
platforms: []
---

# 推理 RunInfo 初始化与 Query/KV 坐标读取

共享 FA 模板中，训练实例可正常运行，而推理特化首次启用时暴露未初始化状态。固定来源 EC-86 报告过 query/kv 行偏移随干净重建变化的案例；本轮读取其固定代码可确认字段声明和部分消费路径，未运行该模板，也不把历史“+1 行”当作全部未初始化读取的固定症状。

`wp_util_regbase.h` 的 COMMON_RUN_INFO 声明 preTokensPerBatch、nextTokensPerBatch 时没有成员默认值；RunInfo<true> 另有无默认值的 queryLeftPaddingSize、kvLeftPaddingSize。另一个类型 RunParamStr<true> 虽为 pre/next 字段设置了 MAX_PRE_NEXT_TOKENS，并不自动初始化 RunInfo 的同名字段。结构体名相近与内存独立性不能混淆。

固定 `wp_block_cube.h` 中，CalcS1Coord 在 isInfer 分支读取 queryLeftPaddingSize，并对负 nextTokensPerBatch 加相反数；CalcS2Coord 在 isInfer 分支读取 kvLeftPaddingSize。它们不再以 hasAtten 保护这些读取。EC-86 将四个字段都归于 CalcS1Coord 的说法过宽：当前查看的函数并未读取 preTokensPerBatch，KV padding 由 CalcS2Coord 读取；preTokens 的其他消费者需单独追踪。

`wp_kernel_train.h::InitUniqueRunInfo` 只在 hasAtten 且执行于 AIV 时为 runInfo.preTokensPerBatch/nextTokensPerBatch 赋值。这项赋值不能单独证明 AIC 对象、hasAtten=false 或新推理实例已得到同值；也没有在该函数里初始化两个 padding 字段。是否最终发生未初始化读取，还需检查具体对象的构造方式、值初始化、所有实际调用路径和写入覆盖，不能只由声明缺默认值就断言每次运行必错。

为状态字段建立符合语义的初始化不变量，可以使用声明默认值或覆盖全部构造路径的显式初始化。无左 padding 的合同允许默认零；真实带 padding 输入则必须写入有效尺寸。pre/next 表示注意力可见窗口，默认值应符合无 mask 或相应 mask 模式，不能把四个字段机械清零而改变可见范围。每批、每任务重复使用对象时，也需要更新会变化的字段。

仅在某个消费点加 `if constexpr(hasAtten)` 可能改变原本合法的推理坐标语义，并遗漏其他读点。先列出字段在 isInfer/hasAtten、AIC/AIV 及布局组合下的来源与读者，再选择一致的初始化方式。声明处修复减少漏改机会，但是否影响生成代码、布局或性能仍需实际构建验证，不能宣称天然零开销。

验证应实际实例化 isInfer=true，覆盖无/有 mask、无/有左 padding、非零窗口及两类核的消费路径。使用不同 query/kv 行的可识别值检查坐标，比较重复执行与独立干净构建；后一项可暴露构建敏感的未初始化状态，但两次相等也不构成没有未定义行为的证明。原记录中未合并算子的历史状态不代表当前所有用户均触发此问题。
