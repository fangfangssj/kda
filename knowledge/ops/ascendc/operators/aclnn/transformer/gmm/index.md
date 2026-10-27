# 分组矩阵乘 ACLNN 算子

本目录保存分组矩阵乘类 ACLNN 算子知识。

- [GroupedMatmul A2/A3](grouped_matmul.md)：GroupedMatmul 算子在 A2/A3 平台的 ACLNN 实现，用于按专家或分组批量执行矩阵乘。
- [GroupedMatmul 950](grouped_matmul_950.md)：GroupedMatmul 算子在 950 平台的 ACLNN 实现，用于按专家或分组批量执行矩阵乘。
- [GroupedMatmulAdd A2/A3](grouped_matmul_add.md)：GroupedMatmulAdd 算子在 A2/A3 平台的 ACLNN 实现，用于将分组矩阵乘结果原位累加到输出。
- [GroupedMatmulAdd 950](grouped_matmul_add_950.md)：GroupedMatmulAdd 算子在 950 平台的 ACLNN 实现，用于将分组矩阵乘结果原位累加到输出。
- [GroupedMatmulFinalizeRouting A2/A3](grouped_matmul_finalize_routing.md)：GroupedMatmulFinalizeRouting 算子在 A2/A3 平台的 ACLNN 实现，用于融合分组矩阵乘与 MoE 路由结果还原。
- [GroupedMatmulFinalizeRouting 950](grouped_matmul_finalize_routing_950.md)：GroupedMatmulFinalizeRouting 算子在 950 平台的 ACLNN 实现，用于融合分组矩阵乘与 MoE 路由结果还原。
- [GroupedMatmulSwiGluQuant A2/A3](grouped_matmul_swiglu_quant.md)：GroupedMatmulSwiGluQuant 算子在 A2/A3 平台的 ACLNN 实现，用于融合分组矩阵乘、SwiGLU 激活与输出量化。
- [GroupedMatmulSwiGluQuant 950](grouped_matmul_swiglu_quant_950.md)：GroupedMatmulSwiGluQuant 算子在 950 平台的 ACLNN 实现，用于融合分组矩阵乘、SwiGLU 激活与输出量化。
