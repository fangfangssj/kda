---
type: Guide
title: Block 归约压缩的语义与尾部条件
description: 说明何时块归约能替代抽取，保留 dtype、repeat、编码后处理及实际搬运范围。
tags:
- task.design
- topic.compute.reduction
- topic.data_movement.data_copy
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P55_block_reduce_max_optimization.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P55_block_reduce_max_optimization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P55.yaml
  role: supporting
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# Block 归约压缩的语义与尾部条件

从每个32字节块抽取一个指定元素，与对该块求max/min/sum是不同运算。只有所取值已经与对应归约结果等价，例如每块保存相同的复制值且max规则一致，才能以BlockReduce进行压缩；纯下采样、中位数或任意第k项不能直接换成max。Source的playbook已说明这一限制，而card的“实现slice效果”未限定，需以真实数据不变量为准。

FP32每32字节块有8元素，FP16有16元素；256字节向量分别是64/128元素。若每repeat覆盖8个数据块，则R个输出块需要floor(R/8)个完整repeat及R%8个尾块，不能仅用整数除法后忽略余数。R<8不等于没有结果。源参数列表的mask/repeat顺序、dst/src/repeat stride、原地重叠允许性和当前平台支持dtype都要核对实际重载，本卡不认证该片段为可编译调用。

分批处理超过repeat范围的输入时，应同时推进源块、压缩结果目的与剩余计数，不是仅减remaining。普通连续DataCopy写出R个int32还需满足该重载对齐，R非整8时可保留合法非对齐写回路径，不能为连续化多写有效范围之外的GM。

来源DataCopyPad一次取4字节、UB侧srcStride=0仍可按其隐式32字节块跨度推进，因此不能靠“stride字段非零”识别所有抽取。原始源buffer需真实容纳R个物理块，而非仅R个压缩元素。归约可以改变峰值活跃空间；只有确认允许原地计算且无其他旧值消费者，才能说没有新增那份结果buffer。

int32先Cast到float再max并转回不保证整数精确：超过2^24的整数可能失去区分。普通有限值max没有加法累计误差，不能把错误归因于“max累加顺序”，但NaN、符号零、tie及索引语义仍需核对。half/float、int/bfloat16的可用重载不能仅凭此旧策略统一下结论。

来源card归约后执行CAST_ROUND与ShiftLeft(...,23)，playbook主模板却省掉ShiftLeft。该位移可能属于特定指数/编码结果的构造，不能在“搬运优化”时删除或对普通数值统一添加；需保留完整输出编码、整数范围与实际位移接口语义。选择Copy/Gather而非归约时也要匹配真实步长，Adds(0)不自动提供任意元素采样能力。

验证每块非首元素更大、全复制块、NaN/±0、整数精度边界、R=0/1/7/8/9和超过repeat范围、最终输出编码。用对应操作的中性值或有效mask处理尾部，确认依赖和地址，再比较短搬运减少是否抵消新增Vector操作与空间成本。
