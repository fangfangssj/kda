---
type: Guide
title: FA 分页与量化操作数合同
description: 补齐物理Head偏移、尾页和反量化尺度。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/flash-attention.md
  role: supporting
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# FA 分页与量化操作数合同

分页先按逻辑token确定table项，再由物理块号与页内位置、KV head、D得到K/V地址。来源table[B*H,maxBlocks]与物理KV[blocks,128,H,D]同时出现时，必须明确定义table是每head独立还是每batch共享；示意phys*128*H*D还缺当前head*D，除非kBlocksPlane已是相应视图。K/V必须使用同一逻辑页对应各自布局，物理页顺序不能替代causal位置。

chunk拆128子块时cols//128会漏非整页尾部，应ceil并传实际有效列。每块S写入packed chunk需要正确行距/列偏移；如果softmax仍按完整chunk归一化，不能同时声称已经每128块更新状态而不修改PV缩放。改变粒度要保持P对应的max、sum和oacc缩放，不能只改bmm1指针。

blockTable身份、范围、尾页valid和共享页规则按实际接口检查。恒等页表可与dense做数学对照，但不同分块/浮点次序不保证位级相同。重复物理页是否是合法逻辑重复输入由合同决定，不宜一概去重。

q8来源量化定义q=round(x*s)，反量化为q/s；若输入scale本来表示dequant乘系数，则应q*scale。必须区分两种命名，s0、溢出饱和、舍入和zero-point也需规定。Golden需依据实际量化整数重建，不能只把原始x除s。

dav_c220来源采用INT8→half再Muls，缺INT8→float某直转重载不意味着所有BF16/FP32输出方案都不可能。MTE2加载后Cast需真实依赖，PIPE_ALL只是较粗候选；AIV反量化发布后AIC才能读取。逐tile生产却只在整batch一次ready，必须保证全部消费数据已写完，否则通知粒度不匹配。

完整GM staging增加INT8读、half写、后续half读及计算，流量按具体缓存/复用计算，不能一律“每元素6B纯带宽”。on-load转换可减少往返但需组件支持和数值合同。q8仅FP16、varlen/sink几乎免费及0.3%/4.3%收益属于来源实现和历史样本，不是功能叠加的性能保证。
