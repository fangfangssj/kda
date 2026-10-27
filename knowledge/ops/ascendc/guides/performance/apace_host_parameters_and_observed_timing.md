---
type: Guide
title: APACE Host 参数接线与实际计时
description: APACE Host 参数接线与实际计时的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/development-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/host-and-testing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/plan-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-to-all-quant-matmul/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/troubleshooting/optimization-playbook.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE Host 参数接线与实际计时

源A2A ST把head_m_size只用于日志，实际headMSize=ceil(usedCoreNum/ceil(N/baseN))*baseM重算；改变CLI但未改变最终CommTilingData不是有效粒度实验。AG固定tileM=min(m,512)且CSV无head列，参数接口不同。应记录最终head/tail、T、baseM/baseN和真实下发字段，避免依据输入参数名判断实现已生效。

<<<>>>异步，Host时钟若只围住launch测的是下发耗时。以同步或event闭合区间可测实际端到端/设备段，两者都可有用，不必宣称只有msprof有效。来源PUT10轮去首轮、AG aclrtEvent20轮、解析脚本另跳3轮并过滤>1.2*min，这些不能混为一个统计合同；msprof采样可能扰动多rank时序，双峰也可能真实，需对照原始Host与设备记录。

cacheFlush与boost输入只分配填充但未被kernel消费，不能算完成L2清理。来源heavy_add模板用Add(v,v,scalar,...)，需核对该SDK是否有此标量重载（常见标量加接口为Adds），不能标为编译已验证。totalLength未参与guard，blockLength*grid越界不能靠名字防护；每block只扫完整tile可漏尾，覆盖足够大地址域是缓存条件化效果的一部分。

模板local-to-local DataCopy和Vector输出之间也需对应队列/事件的真实顺序与允许位置保证，不能从一段伪代码推断零风险。只有heavy kernel记录数正确还不足以证明扫描真实执行、覆盖目标地址或所有cache被驱逐；需容量、访问路径、设备完成及对照。主kernel计时排除还是包含flush要写明，比较路径使用一致政策。
