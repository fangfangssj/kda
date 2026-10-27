---
type: Guide
title: GM 访问错峰与真实地址对齐
description: 保持任务置换、输出映射及生产消费布局，区分性能对齐与合法搬运要求。
tags:
- task.optimization
- topic.alignment
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P64_gm_address_conflict_avoidance.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P66_gm_512b_alignment.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P64_gm_address_conflict_avoidance.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P66_gm_512b_alignment.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P64.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P66.yaml
  role: supporting
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# GM 访问错峰与真实地址对齐

不同执行者在相近时刻访问同一GM区域，可能形成访存热点。候选错峰 `newProgress=(i+coreId)%L` 在L>0、i遍历[0,L)时为各核自己的循环置换；源/目的地址、任务metadata与消费者必须均使用对应真实任务坐标。来源循环总写相同dst时，重排会改变最后留下的结果，不能只替换读偏移。存在时序状态、前后依赖或有序归约的任务也不能任意置换。

coreId模L相同的核仍会同步访问，L小于核数时尤其明显；不同任务索引也可能落在同一较大地址区域。反之共享读取可能受益于缓存，打散访问不保证更快。来源“同512B区域内MTE2/MTE3/Scalar一律串行”没有平台和访问类型边界，本文将其作为待定位的热点假设；新增全核屏障甚至可能把访问重新对齐到同一时刻。

512字节对齐可以作为特定带宽场景的候选布局，但应检查真实 `base_address+byte_offset`。仅AlignUp(offset,512)不会修复不对齐基址；将原逻辑rawOffset直接向上取整后读取，会跳过实际数据。Tensor下标通常以元素计，512字节需换算dtype，不能把字节偏移当元素偏移。正确做法是分配符合要求的物理布局，并使生产端、消费端、子视图、tile间距及容量共同使用该布局。

若每tile引入padding，需把padding计入实际地址及总分配，保留有效尾部，不能多读写原GM边界。对齐算式和大小乘法需防溢出。普通DataCopy、非对齐DataCopyPad及特定矩阵加载各有自己的合法性要求，不能把32B称为一切搬运通用下限，或把512B性能建议当正确性门槛。

来源相对带宽100/90/70与固定提升比例没有本批测量证据；即便按同数据量70→100，带宽增加约42.86%，耗时下降30%，两指标不能混写。验证小L、非零基址、不同dtype/stride、尾部和输出坐标，再测地址分布、GM流量与完整时间。
