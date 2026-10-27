---
type: Guide
title: msOpGen 迁移的数值保真与示例冲突
description: 保存运算次序、分区和类型可见性知识，识别文本计数不能证明等价。
tags:
- task.implementation
- topic.compilation
- topic.implementation.operator_registration
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/rules.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/cases.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/examples.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/reference.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/verification-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/build-verify.md
  role: supporting
created_at: '2026-09-11T04:17:28Z'
updated_at: '2026-09-11T04:17:28Z'
platforms: []
---

# msOpGen 迁移的数值保真与示例冲突

保留原Kernel运算顺序、epsilon位置、Cast模式、模板实例、循环边界、UB分配、queue生命周期和DataCopy参数，能减少迁移变量。sum/n+eps与sum/(n+eps)数学不同；1/sqrt和rsqrt即使数学同义也未必舍入一致。同一API调用数、类名与函数集合相同仍可因实参、分支、索引或别名改变而错误，文本计数不是语义完整性证明；rg -c统计匹配行，也不等于调用次数。

案例CANN9.0.0-beta.1某预编译配置拒绝TILING_KEY_IS(SCH_MODE)时，可用数值字面量保留相同key；源案例已说明constexpr并非C++语义必然失败，__builtin_constant_p归因是推测。Host SetTilingKey命名常量和Kernel预编译参数不是同一路径，if constexpr本身也不应普遍禁用。

源码std::min(a,b)→(a<b)?a:b对有限无副作用整数常能保持值，但会重复求值选中表达式，且NaN、相等时引用身份可能不同；std::min通常以b<a选择，a为NaN时原返回a而所示三元返回b。因此“STL脱敏”也可能改变语义，需使用匹配调用域的等价实现，不能把所有std符号都当非法。显式AscendC前缀和bfloat16_t可解决某些声明查找错误，但不是所有版本/头文件永远稳定的证明。

入口参数顺序需按真实注册生成接口核对。来源reference骨架各核都传start0/endtotalRows，若Kernel不再按blockIdx切分会重复写全量；完整examples才显式ComputeRowRange并处理空范围，两者不可混拼。UB按32B对齐要连同dtype、物理padding和有效count考虑；尾部不自动安全。

examples.md与rules.md存在实质矛盾：例子把RmsNormTiling六字段扁平化、自算Tiling并改成基础向量计算，还使用命名constexpr key；规则又要求嵌套SDK Tiling、保持高阶API并用字面量。可抽取其八个基础字段totalRows/rowsPerCore/hLength/hLengthAligned/coreNum/tailRows/epsilon/rstdElements及六个统计字段mainBsLength/mainBshLength/tailBsLength/tailBshLength/loopRound/reciprocalOfHLength的职责，但未给出的UB与分区公式不能补想成已验证实现。

移除dtypeBranch/gammaIsFloat等字段只有在所有消费者已从等价分发恢复信息时成立，不能按名字判“冗余”。所谓3个类/5个分支只描述该例，不要求所有实现照此组织；Source中没有完整Kernel/Host函数体和实测日志，不能称已完成可运行迁移。
