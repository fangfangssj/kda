# MoE ACLNN 算子

本目录保存专家路由与数据重排类 ACLNN 算子知识。

- [MoeFinalizeRouting 950](moe_finalize_routing_950.md)：MoeFinalizeRouting 算子在 950 平台的 ACLNN 实现，用于恢复专家输出的原始 Token 顺序并合并路由结果。
- [MoeFinalizeRoutingGrad A2/A3](moe_finalize_routing_grad.md)：MoeFinalizeRoutingGrad 算子在 A2/A3 平台的 ACLNN 实现，用于计算专家路由结果还原过程的反向梯度。
- [MoeInitRoutingV3 A2/A3](moe_init_routing.md)：MoeInitRoutingV3 算子在 A2/A3 平台的 ACLNN 实现，用于统计专家 Token 并生成排序与路由映射。
- [MoeInitRoutingV3 950](moe_init_routing_950.md)：MoeInitRoutingV3 算子在 950 平台的 ACLNN 实现，用于统计专家 Token 并生成排序与路由映射。
- [MoeInitRoutingGrad A2/A3](moe_init_routing_grad.md)：MoeInitRoutingGrad 算子在 A2/A3 平台的 ACLNN 实现，用于计算专家初始路由过程的反向梯度。
- [MoeTokenPermute A2/A3](moe_token_permute.md)：MoeTokenPermute 算子在 A2/A3 平台的 ACLNN 实现，用于按专家路由重排输入 Token。
- [MoeTokenPermuteGrad A2/A3](moe_token_permute_grad.md)：MoeTokenPermuteGrad 算子在 A2/A3 平台的 ACLNN 实现，用于计算 Token 专家重排过程的反向梯度。
- [MoeTokenPermuteWithRoutingMap 950](moe_token_permute_with_routing_map_950.md)：MoeTokenPermuteWithRoutingMap 算子在 950 平台的 ACLNN 实现，用于依据显式路由图将 Token 重排到专家顺序。
- [MoeTokenUnpermute A2/A3](moe_token_unpermute.md)：MoeTokenUnpermute 算子在 A2/A3 平台的 ACLNN 实现，用于将专家输出恢复到原始 Token 顺序并按权重合并。
- [MoeTokenUnpermuteGrad A2/A3](moe_token_unpermute_grad.md)：MoeTokenUnpermuteGrad 算子在 A2/A3 平台的 ACLNN 实现，用于计算 Token 顺序恢复与加权合并的反向梯度。
