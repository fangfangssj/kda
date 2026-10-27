---
type: Guide
title: 池化地址对齐守卫与算术反例
description: 区分物理行距、有效尾部和特定快速路径的支持范围。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/alignment-guards.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/precision-patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/pooling-anti-patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/tilelang-translation.md
  role: supporting
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# 池化地址对齐守卫与算术反例

NDHWC整行搬运W*C或OW*C，普通按32B粒度DataCopy要求相应地址/长度符合重载；FP16/BF16两字节时完整行32B倍数对应W*C%16=0及OW*C%16=0。以FP32中间acc[ow*C]和inputFp32[iw*C]逐通道Vector计算时，连续相邻C段起址32B对齐要求C%8=0。这是该布局/指令形式的充分守卫，不是池化数学上无条件C必须8倍数。

来源把Vector寄存器说成256bit、一次仅8个FP32，混淆32B DataBlock与常见256B repeat跨度；地址对齐和count有效mask也不同。C非8倍数可通过物理padding、合法mask/分段或其他布局支持，不能一概必须拒绝。若产品明确只支持该快速路径，可准确校验并报出限制，但不能静默丢尾。

算术反例：C16、W31时W*C=496，496%16=0，来源将它列为拒绝用例是错误。FP16奇数W/OW不一定失败，C为16倍数时仍对齐；C8且W奇数才会使该行长度非16倍数。BF16同为两字节，若走同样普通搬运也不能只检查at::kHalf。data_format及真实轴号应先确定，不能无条件self.size(1)是C。

来源按C个block、每block一个元素并pad到32B槽时，每个block物理起址已按槽对齐；后续是否必须C%16取决于压缩/转换/Vector输出格式，不能只从blockCount=C推导这一必要条件。应按每个实际tensor的物理span和消费者逐项核算，避免把W*C行模式守卫误用于另一种布局。

输出OW计算含ceil末端修正及合法参数，先算出真实维度再校验；所有W*C/OW*C乘积在足够宽且不溢出的域计算。DataCopyPad可支持少于32B有效数据但Local起点/padding需合法，不等于任意地址均可使用。512.0、NaN或UB alignment fault不能唯一定位到某条模数条件，还需检查dtype、偏移、容量与同步。
