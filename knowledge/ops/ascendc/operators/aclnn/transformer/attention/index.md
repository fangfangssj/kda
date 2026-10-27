# 注意力 ACLNN 算子

本目录保存注意力类 ACLNN 算子知识。

- [AttentionUpdate A2/A3](attention_update.md)：AttentionUpdate 算子在 A2/A3 平台的 ACLNN 实现，用于融合更新多路注意力输出与 LogSumExp 状态。
- [AttentionUpdate 950](attention_update_950.md)：AttentionUpdate 算子在 950 平台的 ACLNN 实现，用于融合更新多路注意力输出与 LogSumExp 状态。
- [BlockSparseAttention A2/A3](block_sparse_attention.md)：BlockSparseAttention 算子在 A2/A3 平台的 ACLNN 实现，用于按块稀疏模式计算注意力。
- [BlockSparseAttention 950](block_sparse_attention_950.md)：BlockSparseAttention 算子在 950 平台的 ACLNN 实现，用于按块稀疏模式计算注意力。
- [BlockSparseAttentionGrad A2/A3](block_sparse_attention_grad.md)：BlockSparseAttentionGrad 算子在 A2/A3 平台的 ACLNN 实现，用于计算块稀疏注意力的反向梯度。
- [BlockSparseAttentionGrad 950](block_sparse_attention_grad_950.md)：BlockSparseAttentionGrad 算子在 950 平台的 ACLNN 实现，用于计算块稀疏注意力的反向梯度。
- [ChunkGatedDeltaRule A2/A3](chunk_gated_delta_rule.md)：ChunkGatedDeltaRule 算子在 A2/A3 平台的 ACLNN 实现，用于实现分块门控 Delta Rule 状态更新。
- [FlashAttentionScore 950](flash_attention_score_950.md)：FlashAttentionScore 算子在 950 平台的 ACLNN 实现，用于融合计算 FlashAttention 前向得分与输出。
- [FlashAttentionScoreGrad A2/A3](flash_attention_score_grad.md)：FlashAttentionScoreGrad 算子在 A2/A3 平台的 ACLNN 实现，用于计算 FlashAttention 的反向梯度。
- [FlashAttentionScoreGrad 950](flash_attention_score_grad_950.md)：FlashAttentionScoreGrad 算子在 950 平台的 ACLNN 实现，用于计算 FlashAttention 的反向梯度。
- [FlashAttentionUnpaddingScoreGrad A2/A3](flash_attention_unpadding_score_grad.md)：FlashAttentionUnpaddingScoreGrad 算子在 A2/A3 平台的 ACLNN 实现，用于计算非填充变长 FlashAttention 的反向梯度。
- [FlashAttentionUnpaddingScoreGrad 950](flash_attention_unpadding_score_grad_950.md)：FlashAttentionUnpaddingScoreGrad 算子在 950 平台的 ACLNN 实现，用于计算非填充变长 FlashAttention 的反向梯度。
- [FlashAttentionVarLenScore A2/A3](flash_attention_var_len_score.md)：FlashAttentionVarLenScore 算子在 A2/A3 平台的 ACLNN 实现，用于面向变长序列计算 FlashAttention 前向结果。
- [FlashAttentionVarLenScore 950](flash_attention_var_len_score_950.md)：FlashAttentionVarLenScore 算子在 950 平台的 ACLNN 实现，用于面向变长序列计算 FlashAttention 前向结果。
- [FlashAttn 950](flash_attn_950.md)：FlashAttn 算子在 950 平台的 ACLNN 实现，用于执行 FlashAttention 及可选 FlashDecode 推理流程。
- [FusedCausalConv1d 950](fused_causal_conv1d_950.md)：FusedCausalConv1d 算子在 950 平台的 ACLNN 实现，用于融合一维因果卷积与状态更新。
- [FusedInferAttentionScore A2/A3](fused_infer_attention_score.md)：FusedInferAttentionScore 算子在 A2/A3 平台的 ACLNN 实现，用于融合增量、Prompt 与 MLA 等推理注意力路线。
- [FusedInferAttentionScore 950](fused_infer_attention_score_950.md)：FusedInferAttentionScore 算子在 950 平台的 ACLNN 实现，用于融合增量、Prompt 与 MLA 等推理注意力路线。
- [IncreFlashAttention A2/A3](incre_flash_attention.md)：IncreFlashAttention 算子在 A2/A3 平台的 ACLNN 实现，用于基于 KV Cache 执行增量 FlashAttention 推理。
- [InplaceFusedCausalConv1d 950](inplace_fused_causal_conv1d_950.md)：InplaceFusedCausalConv1d 算子在 950 平台的 ACLNN 实现，用于原位更新状态并计算一维因果卷积。
- [MlaProlog A2/A3](mla_prolog.md)：MlaProlog 算子在 A2/A3 平台的 ACLNN 实现，用于融合 MLA 前处理中的投影、归一化、RoPE 与缓存写入。
- [MlaProlog 950](mla_prolog_950.md)：MlaProlog 算子在 950 平台的 ACLNN 实现，用于融合 MLA 前处理中的投影、归一化、RoPE 与缓存写入。
- [NsaCompress A2](nsa_compress.md)：NsaCompress 算子在 A2 平台的 ACLNN 实现，用于压缩 NSA 注意力使用的序列表示。
- [NsaCompressAttentionInfer A2/A3](nsa_compress_attention_infer.md)：NsaCompressAttentionInfer 算子在 A2/A3 平台的 ACLNN 实现，用于融合 NSA 压缩注意力的推理计算。
- [NsaSelectedAttention A2](nsa_selected_attention.md)：NsaSelectedAttention 算子在 A2 平台的 ACLNN 实现，用于按选中的稀疏块计算 NSA 注意力。
- [NsaSelectedAttentionInfer A2/A3](nsa_selected_attention_infer.md)：NsaSelectedAttentionInfer 算子在 A2/A3 平台的 ACLNN 实现，用于按选中的稀疏块执行 NSA 注意力推理。
- [QuantLightningIndexer 950](quant_lightning_indexer_950.md)：QuantLightningIndexer 算子在 950 平台的 ACLNN 实现，用于以量化方式计算 Lightning Indexer 的 TopK 索引。
- [RingAttentionUpdate A2/A3](ring_attention_update.md)：RingAttentionUpdate 算子在 A2/A3 平台的 ACLNN 实现，用于融合更新 Ring Attention 输出与归一化状态。
- [RingAttentionUpdate 950](ring_attention_update_950.md)：RingAttentionUpdate 算子在 950 平台的 ACLNN 实现，用于融合更新 Ring Attention 输出与归一化状态。
- [SparseFlashAttention A2/A3](sparse_flash_attention.md)：SparseFlashAttention 算子在 A2/A3 平台的 ACLNN 实现，用于结合稀疏索引执行 FlashAttention。
- [SparseFlashAttention 950](sparse_flash_attention_950.md)：SparseFlashAttention 算子在 950 平台的 ACLNN 实现，用于结合稀疏索引执行 FlashAttention。
