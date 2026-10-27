# matmul

- [Matmul 高阶 API 的执行模式与 Host 契约](execution_mode_and_host_contracts.md)：将纯 Cube、消息驱动 MIX 和双主模式分别匹配 Tiling、参与核、结果取得方式与数值模式。
- [GroupedMatmul 的组内尺寸、地址与反量化坐标](group_shapes_and_dequant_coordinates.md)：区分独立 tensor 与拼接存储，按每组有效范围分派矩阵块，并对齐 per-token 和 per-channel scale。

- [多阶段 Cube–Vector 流水的任务与槽位生命周期](multistage_cv_pipeline_lifetimes.md)：说明预加载、消费和回收游标，AIC/AIV 分工，以及 Flash Attention 多阶段流水的启动与排空条件。

- [Matmul 的 K 分段与部分和归约](k_partition_and_partial_sum_reduction.md)：区分 MN 任务切分和 K 方向并行，保持部分和初始化、workspace 坐标、归约同步与精度边界一致。

- [Matmul 高阶替换与常驻回调坐标](high_level_replacement_and_resident_callbacks.md)：保留 Host、单核尺寸、输出模式和 L1 自定义回调的增量约束。

- [异步 Matmul 的结果空间与完成等待](async_result_buffers_and_completion.md)：补齐 Iterate 与 IterateAll 的不同调用、workspace 单位及消费前完成边界。
- [IBShare 共享范围与 Split K 初始化](ibshare_scope_and_split_k_initialization.md)：区分同一 Cube 组内复用、K 分片数据身份与全局部分和清零。
- [TSCM 预加载与 L1 所有权](tscm_preload_is_l1_ownership.md)：明确 TSCM 对应 L1 的逻辑空间，避免把预加载理解成额外缓存容量。

- [Matmul 基本块流量模型与输出对齐](base_block_traffic_model_and_alignment.md)：保留有条件的搬运公式及具体算例，区分计算量、时钟周期与真实带宽。
- [Matmul 搬运模板与 K 重排合同](transport_templates_and_k_reorder_contracts.md)：保存 MDL、NBuffer33、预取和 UnitFlag 的不同前提及 Host/Kernel 一致性。

- [MIX 重写的工作映射与资源归属](mix_rewrite_work_mapping_and_resource_ownership.md)：保留 Cube/Vector 分工、参与核循环、矩阵链和中间状态的静态合同。
- [幂零矩阵的有限 Neumann 三角逆](nilpotent_neumann_triangular_inverse.md)：保留乘积恒等式、层数与累加实现，纠正符号和低精度误差的无条件断言。

- [Count 分组 Matmul 的参考语义与 API 模式边界](count_group_reference_and_api_mode_boundaries.md)：保留连续 token 分组、Bias 与输出拼接，限定简化 torch_npu 描述的适用范围。

- [手工 Dot Product 与 Cube 示例的适用条件](manual_dot_product_and_cube_example_contracts.md)：保存K分块累加与激活融合，纠正容量、布局和误差承诺。

- [双 Matmul 共享与卷积两级缓存模式](two_matmul_share_and_convolution_buffer_modes.md)：保存共享注册和模板开关，修正 Q 在第二次矩阵乘中复用的错误解释。

- [Batch/Group Matmul 分区身份与分支顺序](batch_group_matmul_partition_identity_and_branch_order.md)：保存多维任务映射，限定共享K/N和尾块字段的来源假设。
- [Matmul fallback 资源等式与评分单位](matmul_fallback_resource_equations_and_score_units.md)：保存SWAT分块链路，区分L0/L1预算及文档中的量纲错误。
- [Matmul 全载和 StreamK 能力条件](matmul_full_load_and_streamk_capability_conditions.md)：明确全载的驻留范围、切K覆盖与示例筛选阈值。
- [MX Matmul scale 驻留与预取权衡](mx_matmul_scale_residency_and_preload_tradeoffs.md)：保存量化scale块、L1倍率及预取规模的独立预算。

- [Fixpipe 直达 UB 的镜像布局与复用槽合同](fixpipe_ub_mirror_and_slot_reuse_contract.md)：区分完整输出镜像和紧凑复用槽的地址、容量、路由及通知协议，约束 Cube/Vector 融合中的 GM 往返消除。
