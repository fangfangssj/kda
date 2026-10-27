---
type: Guide
title: CSV 用例转换中的形状与预期值保真
description: 防止旧测试转表格时丢失存储形状、动态逻辑、精度和独立预期。
tags:
- task.testing
- topic.precision
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ops-transformer/legacy-code-extraction.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ops-transformer/csv-refactor-workflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/workflow/step3.md
  role: supporting
created_at: '2026-09-11T03:30:43Z'
updated_at: '2026-09-11T03:30:43Z'
platforms: []
---

# CSV 用例转换中的形状与预期值保真

旧 TensorDescription 中两组 shape 可能分别表达逻辑与存储形状；来源仅提取第一组的规则只在两者可证明等价或另一组被 loader 重建正确时成立。不同 shape、format、stride、offset、实例数或动态列表不能挤进无法表达它们的单列而静默丢失。Optional 缺省也不能从空花括号直接推为 shape=[0]，应保留旧上下文实际语义。

每个有效旧测试的输入、属性、输出期望、失败原因和前后状态都需要在新实例中有对应。循环或条件中的多个具体执行可展开为多行；依赖状态变化、随机生成或多阶段断言的测试可能仍需代码。源码中一个 TEST_F 不必等于一个有效测试场景，宏定义在普通运行时 for 中的示例也不是可编译的通用写法。

浮点属性需保留经目标类型解析后的值；来源 0.04166666666666666f 在旧代码里已经舍入成 float，并非任意精度实数。新 CSV 文本与 stof 的结果应对应旧 float 位值，必要时记录原始位表示，不能仅凭小数位数多断言无损。bool/int/string 也需保留类型及默认处理，移除 f 后缀本身不足以验证等价。

TilingKey/DataHash 从本次被测实现抄回期望，只建立当前行为快照，不能独立证明正确。已有期望值可作回归证据，新增值需要对应可解释的分区、资源与合同；不同编译器或未定义 padding 导致的序列化变化也应分清。失败状态和预期输出应分别保留，不能通过更新期望把错误变为通过。

新旧用例数相同不能证明无丢失，应比较展开后的具体参数、实际运行条目和断言范围。编译不检查运行时 CSV 的缺列或格式错误，第一条通过也不证明剩余行已解析；数据变更后实际加载并执行的集合才是证据。逐五条迁移、备份/删除步骤、子 Agent 分工和强制交互留在 Skill。

来源头文件相对路径表把 arch32/arch35 写成不同层级，不能据此照抄。对于相同深度的 tests/ut/op_host/archXX，两者到算子 op_host 的相对路径应同深度；以实际文件系统计算，避免把构建错误归因于架构差别。
