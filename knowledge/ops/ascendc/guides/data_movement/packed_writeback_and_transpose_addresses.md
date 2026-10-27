---
type: Guide
title: 合并写回的目标布局与转置地址表
description: 保留短搬运合并和 TransDataTo5HD 地址计算细节，明确连续源不等于连续目的。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.data_movement.data_copy
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P10_vectorized_copy.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P7_data_alignment.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P10_vectorized_copy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P7_data_alignment.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P10.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P7.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-issues-aiv.md
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-14T04:19:37Z'
platforms: []
---

# 合并写回的目标布局与转置地址表

多次短搬运可先整理到 UB 再批量写回，但必须保留目标地址集合。来源把 `dstGm[i*STRIDE]` 的离散写变成 `DataCopy(dstGm,packedUb,...)`，只有目标本来连续或另有合法重排阶段时才等价。把源拼连续不会消除目标 stride；若目的有空洞，需保持 strided copy/分段写，不能覆盖空洞或邻接输出。

16 个 half 结果的有效载荷只有 32 字节；减少为一次搬运并不自动成为 256 字节，更不能为满足吞吐阈值向 GM 额外写无效字节。源中 UB 内每次复制 ELEM_PER_ROW 个元素，还要证明它与原来每次一个 half 的语义一致。用数据搬运替代标量取值的收益取决于实际重载、布局和足够工作量，不是所有 SetValue/GetValue 都禁止。

每份 pack buffer 包括数据跨度、对齐和槽数。写回前需等待真正的拼接生产流水完成，MTE3 完成消费前不能回收/覆盖槽；单个 PIPE_V 不普遍替代跨流水事件。参见[按目标布局搬运](../layouts/strided_placement_and_batched_writeback.md)。

来源 TransposeBase16M8 的地址表在一个组 r、地址索引 i 内使用以下元素偏移表达：

```text
src_offset = r*16*colNum + i*colNum
dst_offset = r*16 + floor(i/2)*rowNum + (i%2)*8
repeatTimes = floor(colNum/8)
```

这些式子描述其 float 特化示例的地址选择；它们不是任意 T 的通用矩阵转置公式。需同时核对 TransDataTo5HD 的16地址语义、repeat stride、lane分组、dtype和真正的输出布局。源 helper 模板外观使用 T，内部却调用 float 形态；FP16 的 Base16M16 需要不同块宽。不能仅把 API 名替换或把所有 row/col 向下取整而遗漏余数。

主块之外的行数非16倍、列数非8/16倍，需单独的尾部处理或合法填充。地址表为物理地址列表，不等于 LocalTensor 对象数组；各地址必须落在已分配区间。涉及 ListTensorDesc 的 foreach 输入还需各 Tensor 自身的 shape/stride/基址，不能从 AutoContiguous 名称假定原始列表本来连续。

用存放行列坐标或单个非零元素的矩阵验证转置映射，检查尾部和输出空洞。通过后比较地址生成、pack/transpose、额外UB、调用数量与总时间；来源4–8倍效率、所有>=256B均无需优化或DataCopyPad恒慢一倍等结论未被确立。

## 不规则读取后的连续结果写回

Gather 类计算的源索引不规则，输出却可能按逻辑行连续。此时可将多次短读取/计算的结果放入同一 UB 输出批次，再用符合目的布局的搬运写回。源地址不连续并不要求目的也逐元素写；但该改动依赖真实输出连续、每行输出位置唯一和整批数据已经就绪。

若每行 UB 数据为了对齐带有 padding，实际源行距与 GM 的逻辑行距不同，应使用所选搬运接口支持的 stride 或先合法压紧。不能把带 padding 的 UB 整段直接复制到没有空洞的 GM 输出，也不能通过增加写回长度覆盖 GM 尾外区域。批次扩大还增加同时存活的结果容量与发布等待；是否有收益取决于请求数量、打包成本、流水交接和整体时延。
