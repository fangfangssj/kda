# reductions

- [Reduce 的有效范围、存储布局与临时空间契约](interface_and_storage_contracts.md)：区分逐行 count 型归约与多维 Pattern 归约，明确有效 count、补齐行步长和不同重载的临时空间要求。
- [分块统计归约中的 Welford 状态与跨组合并](streaming_statistics_and_group_merge.md)：用 count、mean、M2 表达分块统计状态，说明流式更新、不同大小分组合并及 workspace 交接需要满足的条件。

- [归约轴映射与全载、分载设计](axis_mapping_and_load_partition.md)：根据真实 stride 将保留轴与归约轴映射到 AR/ARA，区分沿输出切核与沿归约维分片的合并责任。
- [分片极值归约的索引与并列值语义](index_tracking_and_tie_semantics.md)：将片内索引转换到正确坐标，并在值、索引、并列值与非有限值的统一规则下合并部分结果。
- [多输出归约的存活缓冲预算](live_buffer_budget.md)：按输入、转换、累加器、临时空间与输出的真实类型和同时存活数量计算 UB 及跨核 workspace。

- [独立 Softmax 的驻留、重读与在线统计](softmax_storage_and_input_passes.md)：说明输出完整概率时的统计依赖，比较全量驻留、三遍重读与两遍在线统计的资源代价。

- [Softmax 有限域稳定性与 NaN/Inf 处理](softmax_finite_domain_and_special_values.md)：区分减最大值的数学保证、在线缩放、非有限值和算子要求的特殊行为。
- [Welford 方差修正与 Running 统计量的语义](variance_correction_and_running_statistics.md)：区分在线统计算法、总体/样本方差、EMA 系数及归一化消费，纠正把单样本兜底当作无偏方差的误用。

- [私有 Workspace 与固定顺序归并](deterministic_workspace_merge.md)：说明隔离多核写入后实现确定性所需的局部累计、初始化、发布、归并树和空间布局。

- [向量折叠归约与非二次幂长度](vector_folding_and_non_power_of_two_lengths.md)：区分并行 Welford、分段折叠和最终横向归约，保持实际有效长度及归约运算一致。

- [Block 归约压缩的语义与尾部条件](block_compaction_requires_matching_semantics.md)：说明何时块归约能替代抽取，保留 dtype、repeat、编码后处理及实际搬运范围。

- [折叠归约的进度与余项保留](folding_progress_and_remainder_preservation.md)：保存低延迟折叠的适用条件，并指出不缩短长度、尾项丢失和输出步长问题。

- [标量归约替换与原输入保留](scalar_loop_replacement_and_input_preservation.md)：保留 sum/sum_sq 的有效域、输出发布和空间约束，区分 API 名称与具体重载。

- [归约扫描中的目标 logit 保存](inline_target_logit_capture.md)：在已有输入 tile 中读取目标元素，保留原始值、有效索引、跨核归属及后续损失语义。

- [Blaze Pure Reduce 的 partial 状态与模板缺口](blaze_pure_reduce_partial_state.md)：从 Softmax 改写中保留归约单位元、跨核合并、UB 暂存与零长度边界。

- [统计遍历、融合与真实搬运量](statistics_passes_fusion_and_traffic_accounting.md)：保留 Welford、归约和融合方案，纠正单遍归一化与固定内存瓶颈假设。

- [归约算法选择与数值边界](algorithm_routing_and_accuracy_limits.md)：保留 TwoPass、Welford 与分组归约的适用条件。

- [ORI 归约路由、填充值与方差遍历](ori_routes_padding_and_variance_passes.md)：保留分块和数值模型，限制统一对齐与性能禁令。

- [MSE 的 Lane 累加与跨核归约](mse_lane_and_cross_core_reductions.md)：保存两种归约树，补齐短尾、标量写回、工作区和全核到达条件。
- [整行 Softmax 的容量与非有限值策略](softmax_row_capacity_and_nonfinite_policy.md)：保存逐行稳定计算和BF16流水，区分逻辑列数与物理行距。

- [确定性梯度与 Workspace 发布](deterministic_gradients_and_workspace_publication.md)：区分原子正确性、固定次序与双输出暂存。

- [自定义归约树工作量与 Welford 计数](custom_reduction_tree_work_and_welford_counts.md)：保存半区间和并行更新，纠正复杂度与浮点计数边界。

- [特化归约 Mask、Lane 宽度与采样点组](specialized_reduction_masks_lane_widths_and_point_groups.md)：保存CLZ二分、分层平方和及INT64辅助路径。

- [归约 fallback 轴路由与序列化字段遗漏](reduction_fallback_axis_routes_and_abi_omissions.md)：保留AR/ARA建模，指出非连续归约轴被错误合并。
- [归约 fallback 容量单位与分组 workspace](reduction_fallback_capacity_units_and_group_workspace.md)：纠正repeat被当元素数、行数误赋值和强制分组零workspace。
- [Softmax 重算、折叠与统计量归并](softmax_recompute_fold_and_statistic_merge.md)：保存Softmax经验中的可复用状态，区分max、sum、方差和RMS。

- [AR FullLoad归约临时区随批量行数增长](softmax_ar_full_load_binary_scratch_scales_with_rows.md)：保留一次GM载入的计算骨架，定位固定512字节临时区不足。
- [ARA FullLoad的零填充行与四路折叠](softmax_ara_full_load_zero_row_and_four_way_fold.md)：保留8行压缩和Vector存储加载顺序，解释尾偏移哨兵。
- [Softmax 公共Cast视图与DMA步长单位](softmax_base_cast_views_and_dma_stride_units.md)：保存同字节布局重解释的约束，防止把空操作当FP32拷贝。
- [Softmax 公共归约树的有限适用范围](softmax_common_tree_helpers_have_finite_reduction_domains.md)：保存折叠、缓存层级和两类末端归约上限。
- [ARA Online的chunk状态与全掩码初始化](softmax_online_chunk_state_and_all_masked_bootstrap.md)：保存两次GM读取、running状态公式和无效首块处理条件。
- [Softmax Recompute配对、缓存根与缓冲角色](softmax_recompute_pairing_cache_root_and_buffer_roles.md)：保存AR和ARA三遍组织，区分buffer数量及尾部字段。
- [AR SmallR转置预取与padding生命周期](softmax_small_r_transpose_preload_and_padding_lifetimes.md)：保存短轴转置、两份指数和输出复用，补充空核预取条件。
- [Softmax 模板的轴映射与Norm扫描次数](softmax_template_axis_mapping_and_norm_pass_counts.md)：保存六类组织及Norm可复用部分，纠正维度改名和阶段数混淆。
