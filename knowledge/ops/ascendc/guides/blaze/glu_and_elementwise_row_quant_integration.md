---
type: Guide
title: GLU 与逐元素输出接整行量化的联合约束
description: 保留 BMM/Grouped 行映射、paired metadata、单两遍选择与执行拓扑边界。
tags:
- task.design
- topic.compute.quantization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-development.md
  role: supporting
created_at: '2026-09-11T03:01:48Z'
updated_at: '2026-09-11T03:01:48Z'
platforms: []
---

# GLU 与逐元素输出接整行量化的联合约束

Matmul直接量化、Matmul→GLU→量化、Matmul→逐元素DAG→量化的量化输入不同。直接路径消费最终FP Matmul结果；GLU消费完整K的两路投影并产[R,Q]，只有两路等宽拼成2Q时Q=N/2；同位置逐元素DAG保持[R,N]。Softmax/LayerNorm含跨元素归约，不能当作普通unary后处理。bias、scale、dequant及其顺序只在原数学合同要求时出现。

GLU的N轴weight/bias/scale按同一qOffset/curQ成对切分，M/K轴metadata不随两分支自动复制。小Q时两路独立padding，gate起点由物理act跨度决定；外部逻辑shape、mask和输出仍保留Q。BMM继续使用真实batch解码和offset，Epilogue可消费batch-local view或展开row base；GMM需要自己的group合同，不能把GMM资产强用于MM/BMM。

R是实际逻辑行数：普通MM为M，等形BMM可为B*M，Grouped需由各组有效行和输出布局确定。前段列tile可以跨任务产生，同一量化行必须等全部逻辑列final写入后再分配给行消费者。GLU量化域是Q，其他两路为N；workspace dtype、物理pitch、容量及所有输出y/yScale必须一致，padding不参与absmax。

能直接以Fixpipe生成最终FP workspace时，不必为了形式接量化再加identity Vector。生产者与消费者可以采用已证明的单MIX阶段协议，或在允许的实现合同中采用同stream独立Kernel顺序。GLU/elementwise组合来源指定单entry是其场景实现约束，不能推广为数学上禁止分Kernel；任何拓扑都须证明参与者、初始化、完成、数据可见性与复用，并以同设备测量评估开销。

来源通用CV1/V2组合层用tuple承载两个Epilogue，前段拥有Matmul/GLU/group遍历，后段按完整行切slice；V2不能继承局部N tile当完整行。AIV-only SyncAll仅在AIC到AIV最终交接已闭合时足够，不能替换为本核PipeBarrier。静态single-pass/two-pass量化entry可由Host共享UB布局公式选择，选择器与实际分配一致，不能保留single-pass却删掉更宽行能力。

宽行两遍量化第一遍全行absmax、第二遍重读用同一scale；0行、tiny、饱和、半整数舍入、absmax在首个向量/首tile之后以及容量边界前/等于/后均需覆盖。若MMAD、elementwise和最终INT8有差异，分别比较workspace、yScale和y；相同scale不证明每列原值一致。half-tie分类应使用生成参考整数输出的同一个normalized中间值，不能重新计算后误分边界。两阶段单独通过不能替代联合回归。
