# MoE 直调算子

本目录保存专家路由与数据重排类 Direct Invoke 算子知识。

- [MoeFinalizeRouting 950](moe_finalize_routing_950.md)：MoeFinalizeRouting 算子在 950 平台的 Direct Invoke 实现，用于恢复专家输出的原始 Token 顺序并合并路由结果。
- [MoeGatingTopKSoftmax 950](moe_gating_top_k_softmax_950.md)：MoeGatingTopKSoftmax 算子在 950 平台的 Direct Invoke 实现，用于计算门控概率并选择 TopK 专家。
- [MoeReRouting 950](moe_re_routing_950.md)：MoeReRouting 算子在 950 平台的 Direct Invoke 实现，用于按路由结果重排 Token 与专家数据。
