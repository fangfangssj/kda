---
type: Guide
title: 误差指标与实际表示的测试输入
description: 修正容差公式、空/非有限值统计和数据生成中改变 shape 或退化为零的陷阱。
tags:
- task.testing
- topic.precision
- topic.implementation.debugging
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/tools-reference.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/case-studies.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/debug-summary-template.md
  role: supporting
created_at: '2026-09-11T02:40:45Z'
updated_at: '2026-09-11T02:40:45Z'
platforms: []
---

# 误差指标与实际表示的测试输入

误差统计之前先核对shape、dtype/布局、输入和阶段。不能让NumPy广播悄悄比较不同shape；低精度或整数直接相减可能产生额外舍入或整数溢出，应选择足够精度/范围的比较表示。空数组、NaN、Inf和正负零按验收语义单独处理，再对有限部分计算max/mean/median/std及95/99分位，并保留最差元素多维坐标。

若采用常见加性容差，逐元素条件为abs(pred-truth)<=atol+rtol*abs(truth)。来源使用abs_error<atol或abs_error/(abs(truth)+atol)<rtol，是另一条件，不能标成同一验收。例如truth=10、误差1.5、atol=1、rtol=0.1时，加性条件通过而原条件失败；atol=rtol=0配合严格小于甚至拒绝完全相等。纯整数/索引输出应按算子合同精确比较。

“99%通过”会容许整块尾数据错误，且来源两脚本分别用>99与>=99，边界也不同。是否允许任何失败比例须来自实际验收；固定FP16/FP32/Reduce容差表只是示例，不能以放宽容差修复语义错位。Softmax输出和接近1也不能单独证明逐元素正确。

np.random.rand生成[0,1)数值再cast为int8会全部变0，不能覆盖整数算子的非零/负值。将尾轴向对齐尺寸扩大是在改变测试逻辑shape，保存为.npy也不证明设备基址或实际stride对齐；应分别保存逻辑范围、物理padding与实际分配地址，原非对齐用例仍需保留。

边界数先按目标dtype表示再核验：half中的1e-10可能变0，1e6/1e10可能已变Inf，不能把生成前标签当实际输入范围。FP16的0.1实际为0.0999755859375，1000个此输入的精确和是99.9755859375，最后转half又可能舍入到100。参考值应区分实际输入求值与理想十进制100；来源“误差1.5降到1e-4”等未提供本批可复现结果。

printf格式应匹配接口和类型，普通C格式表不保证每个Device接口全支持；half可按接口要求转换，线程/核/阶段与地址信息应一起保留。脚本退出成功仅说明其实际条件通过。检验包含原失败输入、稀疏大误差、全非有限、空输出、精确整数与容差等号边界，固定目录和批量脚本留于Skill。
