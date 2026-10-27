---
type: Guide
title: CATLASS 线性 Attention 的阶段合同与 Chunk 归属
description: 补充 KDA dAv、物理块、分组 head 与跨阶段数据依赖。
tags:
- task.design
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/attention/linear-attention.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/attention/gdn-kda.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/flash-linear-attention-npu-reference.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/open-source-linear-attention-map.md
  role: supporting
created_at: '2026-09-11T04:33:09Z'
updated_at: '2026-09-11T04:33:09Z'
platforms: []
---

# CATLASS 线性 Attention 的阶段合同与 Chunk 归属

GDN/KDA/retention/RWKV等家族同时包含Matmul、gate/decay、scan、状态递推及跨chunk复用，不能只按书写公式顺序切阶段。数据依赖应包含真实tensor/layout/dtype、生产与消费核型、状态初值、是否跨head/chunk/launch、workspace生命周期和wait/set边界。gate、v或v_new、scale作用位置、clamp和输出dtype的不同会改变数学，参考实现的同族名称不足以证明同合同。

KDA dAv一般只是backward中的dA/dV阶段：dA_raw=dO*V^T后按causal/gate/scale处理，dV=A^T*dO，A究竟来自前向中间量还是经后处理的新量必须明确。若dV确实消费AIV生成量，需生产完成再RunDv；如果数学上独立，仅凭顺序图也不能凭空创造依赖，能否重叠还受共享资源和flag credit约束。不可把该阶段宣称完整KDA backward。

varlen任务数、nt及基址来自cu_seqlens/chunk_indices和每个序列的chunk，而非统一ceil(T/BT)。物理BT块与validRows分别用于矩阵布局及有效mask/writeback，保留full-BT要求实际分配及无效元素初始化安全。来源validRows收敛导致长case分钟级、恢复BT回秒级是历史性能案例，未附本批原始执行证据，不能推导一切尾块收敛都错误。

HK/HV分组均分时hk=hv/(HV/HK)，要求HK>0且HV%HK==0。缓存scope还必须包含gate/beta/scale/状态真正依赖的head；共享输入与梯度累加归属不同。V=256可能使某个GEMM的K轴分段，也可能只是输出N轴扩展，应按每阶段矩阵维度确定累加和Fixpipe完成点，不能把所有V256都强制Split-K。

L2只可描述期望cache命中，显式中间量归GM；L1供Cube复用，L0A/B/C供矩阵块和累加，UB供Vector门控/归约/转换。A2/A3的跨域通路需对应硬件与库证据，不能无证据画物理L0C→UB。窗口或slot复用需free/credit或等价最后消费者保护，GM slot与UB ping/pong并非同一个计数域。

公开Epilogue缺少gate/mask/scan/finalize不表示算法不可实现：可在合法设备位置以自定义Tile/Block表达Vector计算，scan/递推按真实依赖另划stage。接口应覆盖shape/layout、cast模式、计算粒度、UB/L1与事件预算、跨域同步；来源关于必须组件化、禁手写GEMM和Agent阶段规则属于该Skill实现选择，不是Ascend C语言能力限制。

覆盖维度包括BT64/128及1/2/多/尾chunk，K/V128/256（64按支持合同），HK=HV与分组共享，单/多batch/head、任务少于/接近/多于核数、zero gate/high beta/exp饱和/近零输出/状态初值。来源8例representative、建议30例是测试组织约定，不能仅按数量保证充分覆盖。stage的输入golden须有独立上游依据；benchmark缺失/不支持/运行失败与自实现数值失败分别记录，全流程多launch需完整计时。
