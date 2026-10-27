---
type: Guide
title: Ascend C 精度对比的证据与故障隔离
description: 保持输入和数值契约一致，结合误差坐标、计算阶段和结构边界区分数值误差与布局/同步错误。
tags:
- task.debugging
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/data-comparison.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/binary-search-debug.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/best-practices.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/diagnosis-workflow.md
  role: supporting
created_at: '2026-09-10T12:59:46Z'
updated_at: '2026-09-10T12:59:46Z'
platforms: []
---

# Ascend C 精度对比的证据与故障隔离

精度对比首先要求两条路径接收相同逻辑输入，并采用可比较的 shape、布局、属性和输出语义。随机生成两组不同输入，或改变 shape 后直接比较输出，不能隔离 dtype 的影响。低精度输入的参考还应区分“对原始高精度值求值”与“对实际已舍入输入求值”。

最大误差、平均误差及分位数提供不同信息，最差元素还需保留多维坐标、实际值与参考值。整体统计可能掩盖稀疏越界或单个尾 tile 的错误。空输出和 NaN/Inf 要先按契约单独处理，不能无条件调用 max/argmax 或让非有限值混入普通误差统计。

相对误差在参考值接近零时敏感。容差公式、分母保护、特殊值比较和允许失败比例必须来自实际验收语义；诊断脚本中的 `99%` 或固定 rtol/atol 示例不是所有 Ascend C 算子的合格标准。

## 结构信号与计算信号

对齐、非对齐、单/多核、单/多 tile、不同 dtype 可以帮助隔离条件，但任何一个小样本通过都不证明整个算法正确。尤其“FP32 对齐样本通过”不能排除其他 shape 的地址错误或 FP32 数值不稳定。

中间结果可按计算依赖拆分，定位首次出现差异的阶段。对比点应处于同一数学阶段，且数据已经完成生产。若实现有重算、融合或在线递推，不能只根据变量名假设中间值相同。

误差的周期、累积趋势及特定位置可提出布局、同步或舍入假设，但不能直接构成根因。常量替换、退化 shape 或增加屏障也可能同时改变多条路径，应配合对具体地址、数据依赖或数值步骤的检查。

修复证据应覆盖原失败条件及相关结构边界，并使用实际交付的未插桩产物；提高中间精度后还需确认最终舍入、资源占用与性能。算子数学不变量可以补充验证，但例如 Softmax 输出和接近 1 并不能单独证明每个输出正确。
