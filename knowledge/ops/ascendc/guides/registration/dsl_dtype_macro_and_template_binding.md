---
type: Guide
title: DSL 多 dtype 编译宏与模板绑定
description: 保留按输入名特化和合法分支，限制二进制相同的误诊。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/multi_dtype/multi_dtype_guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/multi_dtype/add_custom_reference.cpp
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# DSL 多 dtype 编译宏与模板绑定

来源构建模型按注册输入名注入DTYPE_<INPUT_NAME_UPPER>，x1才对应DTYPE_X1；query/input/x等需与实际构建宏桥接。float默认值适合编辑器提示，但生产编译缺宏时静默落到float可能把16位GM当32位读取，应验证实际编译命令和预处理类型。一个源编译三次是该工程策略，也可通过已验证的其他模板/分发机制实现。

“FP16/BF16结果与FP32一致”不证明宏未注入，精确可表示输入可能结果相同；object字节相同也可能含相同已优化代码，而字节不同可能只因调试信息。宏、实例类型、加载binary和实际dtype测试联合确认，不能只xxd diff两对象。

I/O以DTYPE_X1类型绑定，计算类型独立。对来源dav_c220的BF16基础Muls/Mul/Add路径，可升FP32计算再按受支持模式降回；不能把这个受限列表扩展成所有BF16 API均无原生能力。FP16直接计算和升FP32计算会有不同中间舍入，Add的alpha先转half再Muls与保持float的BF16分支也不同，正确性依实际算子合同。

if constexpr在非模板函数中不会普遍免除未选分支的语义检查；宏替换后的固定类型不是依赖模板参数。DataCopy<float>(float,half)的冲突不能靠普通函数中if constexpr消失，使用template<typename T>使相关条件和表达式依赖T，才可在实例化时丢弃不适用分支。所有非依赖名字仍须合法；来源“同一DTYPE张量就一定无需模板”不足以证明静态断言/API约束都通过。

仅声明TBuf成员而不InitBuffer通常没有该数据区的UB分配，不能从成员数估算实际UB。Selective upcast与full upcast是两种资源策略；full不是float→float也必须Cast，float分支可保留同类型视图/合法搬运。TBuf不是所有中间量唯一合法选择，若需要表达生产消费和复用，TQue也可实现。
