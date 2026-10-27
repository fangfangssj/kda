---
type: Guide
title: Scalar 九条原则的编译器条件
description: 完整保留九类改写选择，限定别名、常量传播与spill的推断。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/scalar/guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/scalar/coding_principles.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/scalar/conv_scalar.md
  role: supporting
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-11T06:42:38Z'
platforms: []
---

# Scalar 九条原则的编译器条件

Scalar供给不足可让Cube/Vector/MTE出现空档，瓶颈可能是加载/存储、地址算术、分支、依赖或发射；不能预先断定主因总是spill或Load/Store>30%。通过生成代码、真实热点和Task时间验证，再选择下列改写。

1. 动态索引小数组可比较独立标量/模板选择，但合法数组下标不会在C++语义上允许写到结构体其他成员，编译器也不必放弃整个对象常量传播；额外三目选择可能增加分支。
2. 首/主/尾分离、周期load分段可减少热循环控制；必须保持0/1次循环、首次加载、最后flush及周期边界。
3. 显式嵌套for利于看清不变量和遍历，但for不必走硬件回跳、while也可被识别；状态机3–5次mispredict/200行触发I-cache都需测量，不能只按语法判断。
4. 局部值快照可能利于标量替换，但成员不必每次Load，局部也可spill；跨调用若值会合法变化，快照不能保留过期值。
5. 缩短实际live range有助寄存器复用，源码声明位置不是唯一决定，常量可延后物化、优化器也可重排。不能为晚声明改变带副作用表达式执行时刻。
6. 减少依赖指针链可降低加载延迟，值聚合可减少间接访问，但内联/别名已证明时原链也可折叠，值复制可能增加成本或改变共享更新语义。
7. 按阶段拆大结构并只传所需字段有助局部性，64B/128B与10字段是来源经验，不是所有芯片cacheline或spill阈值；大结构未用字段可能被完全消除，按值也不必无成本。
8. 真正编译期值可用constexpr/非类型模板表达，但const对象的已知值同样可能优化；constexpr对象取地址或跨不可见调用也不保证零Load。运行时值不可强行固化，特化爆炸需控制。
9. 外提无副作用对象构造与减少逃逸地址可能有益，但LocalTensor构造不必产生Store、取地址不必强制spill；必须保留实际初始化/析构和API写回副作用，不能把需要输出的指针API改成按值后丢更新。

这些措施可独立或组合评估，不要求所有算子按固定顺序应用。保留同语义输入、数值精度与同步，检查改写前后实际访存/分支/寄存器和完整时间；固定90%→10%或0指令不是本次验证结果。
