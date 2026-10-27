# sorting

- [Ascend C 多路归并的状态与 Proposal 单位](merge_state_and_proposal_units.md)：区分有序段有效长度、物理间距和归并消耗计数，保证归并推进、输出容量与索引语义一致。

- [多核排序的分层归并与工作区交接](hierarchical_merge_and_workspace_handoff.md)：将 tile 排序、核内归并和跨核归并分开预算，显式传递有序段长度、位置与完成状态。

- [排序聚合同索引更新的排列与所有权](grouped_index_updates_and_permutation.md)：保存小维度稀疏更新中按索引分组的机制，明确原始位置、跨块分组和累计顺序。

- [硬件排序的行边界、索引与 Padding](hardware_sort_rows_indices_and_padding.md)：纠正 Sort32 的元素数和方向，并保存批量小行与跨 tile TopK 的完整条件。

- [TopK 阈值、并列值与二分搜索边界](topk_threshold_ties_and_bisection_limits.md)：保存分段排序和采样语义，限制固定迭代及 GatherMask 原地断言。

- [排序 fallback 阶段、容量与归并缓冲缺口](sort_fallback_stages_capacity_and_missing_merge_buffers.md)：保留分核排序模型及分支B缓冲字段遗漏的可复现反例。
- [TopK 裁剪次序与排序字段合同](topk_pruning_order_and_sort_field_contracts.md)：明确分阶段裁剪条件和稳定排序、索引精度、特殊值边界。

- [Radix mask精度、DMA交接与barrier协议](radix_mask_precision_dma_handoff_and_barrier_protocol.md)：保留计数技巧，纠正padding对齐即可安全和额外GM读免费等结论。
- [Radix 选择的保序key、计数与输出语义](radix_selection_ordered_keys_counts_and_output_semantics.md)：保留多级分桶/整数二分选择，补齐ties、Quantile和输入不可逆修改条件。
