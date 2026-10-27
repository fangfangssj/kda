---
type: Guide
title: A2 FA Stage 资源与组件适用范围
description: 保存四段流水和轮转容量，修正平台与管线泛化。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/flash-attention.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-handcraft-pitfalls.md
  role: supporting
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# A2 FA Stage 资源与组件适用范围

来源A2/A3组件方案由AIC QK、AIV online softmax、AIC PV、AIV rescale组成，中间S/P/O经独立GM workspace交接。单块依赖顺序不应违反，但不同块可流水重叠；不能把“禁止改变顺序”理解为四段全局串行。生产完成后ready、消费前wait及最后消费者结束后的release共同保护slot；只在stage入口/出口握手是一种粒度选择，内部更细tile流水也可合法。

preLaunch2对应3个轮转槽，元素offset候选(core*3+slot)*WORKSPACE_BLOCK_SIZE_DB，常量131072必须明确是元素还是字节。来源S/Otemp/Oupdate每核131072*12B、P每核131072*6B，对应三槽FP32或FP16；总计每核5505024B=5.25MiB。按实际row/stack、dtype、核数和阶段深度可重推，不能把该大容量公式当所有FA最小需要。

源资源表UB192KiB/L1512KiB/L0A/B64KiB为所述配置候选，L0C一处写256KiB，与同批其他A2资料128KiB冲突，未核验前不能合并成硬件事实。mask[128,1024]FP16单独已256KiB，大于该UB192KiB，必须分片或其他合法设计；源码示意预算表并非可直接分配的配置。超容量可能编译/静态检查或运行时失败，不是一律编译干净。

M_MTE1八个、FIX_M两个、MTE1_MTE2八个预置信用及出口wait是来源特定Block的事件生命周期，不能给任何FA kernel无条件加18对；组件自身也可能初始化。MTE1不是L0→UB、MTE2不是通用UB运算，需按真实搬运方向建依赖。硬同步基址只用于依赖该协议的launch，系统workspace函数也不能仅因FA名字统一禁用。

L1/L0<128,128,128>、qS128、KV128、qN按分组推导是候选实例，静态K、动态D及尾处理须按组件定义，不能同时宣称静态K128又要求任意D≤128都与K相等。PAGED=false历史设备异常仅约束所测组件版本，恒等页表可作其连续KV适配，不证明所有A2非分页attention不可实现。

FAIQK旧模板与FAQK新模板、参数6/9项及Tail特化差异需固定实际依赖版本核验；GitCode HEAD不保证长期兼容，空DispatchPolicy声明也不证明存在实现。“手搓默认/复用默认/不许读外部源码”等冲突是Skill流程，本卡不将它们作为能力限制。历史cube_util96%、1159/1166us或手写慢3–9倍不能推成必然性能。
