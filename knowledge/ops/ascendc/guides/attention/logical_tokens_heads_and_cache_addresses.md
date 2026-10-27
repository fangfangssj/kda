---
type: Guide
title: Attention 的逻辑 token、head 与物理缓存地址
description: 分开处理 TND batch 边界、GQA head 映射、稀疏 token 选择与 page table，防止布局组合后混用坐标。
tags:
- task.design
- topic.compute.indexing
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/references/page_attention_pattern.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/references/tnd_pattern.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/references/head_sharing_pattern.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/references/topk_kv_sparse_pattern.md
  role: supporting
created_at: '2026-09-10T14:32:33Z'
updated_at: '2026-09-10T14:32:33Z'
platforms: []
---

# Attention 的逻辑 token、head 与物理缓存地址

Ascend C Attention Kernel 组合变长、共享 head、分页和稀疏访问时，应分开建立逻辑任务坐标与物理地址，再把它们接到同一 QK/softmax/PV 数据流。它们不要求统一采用某一条固定流水或强制增加 Gather stage。

对无前导零的累计长度数组 L，batch b 的起点为 `b==0 ? 0 : L[b-1]`，长度为 `L[b]-start`。Q 与 KV 分别恢复边界；若接口提供带前导零的数组或非累计长度，公式随之改变。任务到 batch 的映射不能在变长时使用固定每 batch 块数，尾块也按该 batch 有效长度计算。

对按连续 head 组共享 KV 的 GQA，令 `G=Hq/Hkv`，需要正整数 G，`kv_head=floor(q_head/G)`；MQA 对应 Hkv=1。K 与 V 必须使用一致的映射，输出仍按 query head 排列。Host 显式展开可以作为参考计算方式，但实际 Kernel 无需为了 head 共享而物化完整重复 K/V。

当索引合同明确表示逻辑 token 时，先检查 sentinel、长度和可见性，再按 token 寻址。若索引表示 block、page 或已经编码的物理 offset，则应使用该合同的转换，不能重复套 token 公式。例如原库 [NsaSelectedAttentionInfer](../../operators/aclnn/transformer/attention/nsa_selected_attention_infer.md) 的 selected index 表示区间块，不能当作本卡的逐 token 索引。

对 page 大小 P，逻辑 token t 映射为 `logical_page=t/P`、`row=t%P`、`physical_page=table[b,logical_page]`。在连续 `[num_pages,P,Hkv,D]` 布局下，元素 offset 为 `((physical_page*P+row)*Hkv+kv_head)*D+d`；若实际布局是 `[num_pages,Hkv,P,D]`，head 和 row 的乘法顺序不同。必须分别验证 page table 范围、物理 page 范围及 K/V 各自跨度。

连续逻辑 tile 只有在实际物理段可由当前搬运接口表达时才能直接加载。跨 page、离散选择或复杂布局可拆分搬运，或先 gather 成密集 workspace；并非所有分页都需要 gather，也不能只凭“逻辑连续”假定物理连续。

Gather 的 K/V 行顺序与 score 列顺序必须一致；V 暂存要保留到 PV 消费完成。无效行的零填充保护搬运和矩阵输入，但还需要 score 可见性处理。验证使用不规则 page table、非等长 batch、不同 head 数据、负 sentinel、尾块和 ring 复用，并检查每个逻辑 token 的来源。
