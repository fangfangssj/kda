---
type: Guide
title: 向量谓词、标量数学与 Axpy 修改语义
description: 保留专用指令选择，区分两种Select极性及浮点运算合同。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_10_vectorized_copy.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_07_data_alignment.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/compare_and_select/select.md
  role: constraint
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-09-11T06:15:28Z'
platforms: []
---

# 向量谓词、标量数学与 Axpy 修改语义

CompareScalar也是向量比较，不是逐元素Scalar ALU。FakeQuant可以Compare生成压缩bitmask再Select成数值0/1，最后两数值mask相乘实现AND；有限且严格0/1时FP16乘法恰好精确，不存在来源所称必然精度损失。任意非0值或NaN不能套此结论，压缩bitmask与bool字节mask也不能直接互当。

此片段使用basic memory-vector的SELMODE::VSEL_TENSOR_TENSOR_MODE，其mask bit1选src0、bit0选src1，因此Select(oneTensor,zeroTensor)产生真值1。不要套用高级bool Select“0选src0”的极性；同名API的mask表示与源顺序不同。>=min且<=max的比较域应是算子要求的量化前/舍入后域，NaN结果及尾mask需明确。

Axpy(dst,src,alpha)修改dst为原dst+alpha*src。ForeachAddList把srcLocal1作为dst，来源文字x+alpha*y若把参数角色反过来会出错；独立dstLocal需在完成后复制，且原srcLocal1被修改不应破坏其他消费者。AddcDiv先tensor2/tensor3再Axpy到tensor1，除0、Cast、饱和/舍入及alpha精度应匹配真实合同；减少一次中间存储可能改变浮点舍入，不保证与独立Mul+Add逐位相同。

Adds可省去为相同scalar Duplicate整段buffer，alpha等参数的预计算仍需正确；Foreach较宽scalar先存低精度T再转float不会恢复损失。RINT须按具体输出类型/接口的ties规则理解，不统一称普通四舍五入。合法dtype/平台和别名必须与实际重载一致，不能把所有INT32或BF16笼统允许/禁止。

AdamW ScalarPow示例把x广播8个FP32再Power，V_S完成后GetValue0。Power的临时空间、isSetMask=false调用约定与当前mask状态需核验，不能只看到Duplicate就认为可调用。beta/step、负底数非整数指数、beta接近1导致1-beta^step消减及step精度均需验证；用向量算单标量不必然更快，powf也不能仅凭函数名归类CPU执行。

DynamicBlockQuant常量tensor与BinaryRepeatParams的src blockStride0可以表示按块广播，前提是该API语义和实际8元素常量容量匹配；通用count型Div不会自动广播一个8元素tensor到任意count。Cast NONE不等于饱和或指定舍入。数学合同先于“纯向量化接近峰值”的宣传，按实际中间值与完整调用比较。
