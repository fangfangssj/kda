---
type: Guide
title: Blaze 分阶段诊断需保留正式执行条件
description: 将 C-direct、C-through 与 V-known 的观察点绑定到真实入口和数据表示。
tags:
- task.debugging
- topic.precision
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/kernel-design/glu-precision-diagnosis.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-development.md
  role: supporting
created_at: '2026-09-11T03:01:48Z'
updated_at: '2026-09-11T03:01:48Z'
platforms: []
---

# Blaze 分阶段诊断需保留正式执行条件

诊断模式先要能追到实际Host分支、launch symbol、Kernel实例和Block/Epilogue。名称c-direct却不输出raw C，或verifier未执行，均不构成Cube正确证据。明确每个模式实际执行/跳过阶段、输出shape/dtype/layout/pitch、观察位置与匹配参考。

C-direct观察Matmul结果；C-through-L0C2UB额外覆盖融合搬运；V-zero检查零输入路径但不能覆盖一般公式；V-known以可控C检查激活/量化及写回；Full验证真实联合链。各层通过只缩小可能范围，不形成唯一根因证明。观察模式若改了AIC/AIV任务分配、SplitM、mask、slot、同步或写回布局，可能引入或掩盖错误，不能直接与正式链归因比较。

先用已知正确且能识别故意差异的对照校准诊断输出/验证器，再观察原失败输入。保持相同shape、Tiling、raw输入和Golden；以M/N/K/group分别变化的用例隔离共同因素，不因测试名叫K-tail就断定K轴错。清除调试输出后还需验证正式产物，因为插桩可能改变资源和时序。

逐元素后处理若在producer VF/UB执行，核对其局部布局、mask与workspace store；若通过额外GM→UB→GM pass，须独立核对下一行起点、逻辑pitch、物理padding及Copy重载。N*sizeof(dtype)不对齐时尤其不能只根据Tensor可表示二维shape推断搬运可用。MMAD workspace通过而后续失败时，这条数据路径与公式都仍是候选。

量化将上游微小漂移转成整数边界差异。先比较原始FP workspace、再scale、再相同division/rounding/cast链；不能靠更换所谓等价GELU表达式、缩窄随机分布或重算Golden掩盖上游误差。正式指标、分母、特殊值与阈值来自验收合同，诊断max误差不能自动取代它。本批没有复现来源的具体设备失败。
