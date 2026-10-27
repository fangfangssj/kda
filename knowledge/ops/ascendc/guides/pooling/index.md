# pooling

- [池化反向的梯度贡献与 gather/scatter 归属](backward_contribution_ownership.md)：区分按输入收集与按输出分发梯度的所有权、局部合并、跨核累加和实际性能成本。
- [池化的原生布局、行地址和转换成本](native_layout_and_conversion_cost.md)：对齐 NCDHW/NDHWC 接口与内部工作布局，避免重复转换并核对真实行地址及搬运范围。

- [池化地址对齐守卫与算术反例](address_alignment_guards_and_counterexamples.md)：区分物理行距、有效尾部和特定快速路径的支持范围。
- [深度池化 Direct Buffer 路径与同步修订](direct_buffer_depth_path_and_sync_revisions.md)：保存行地址和复用条件，消除双重屏障必需的矛盾。
- [池化窗口除数与有效元素计数修正](window_divisors_and_valid_count_corrections.md)：保存 ceil/padding/override 优先级及前后向一致性。

- [NHWC 池化任务分组与排除 Padding 的平均值](nhwc_task_groups_and_excluded_padding_average.md)：保存真实窗口采样和分组地址，修正一次加载与均匀分核的误述。

- [Pooling NaN、Argmax 与插值边缘](pooling_nan_argmax_and_interpolation_edges.md)：补齐Ceil修正、Padding获胜和仅右角有效的分支。

- [池化 Split-C、Split-W 与 Multi-W 的容量和分发合同](split_c_split_w_multi_w_capacity_contract.md)：依据实际同时存活的缓冲选择池化分块，保留 dtype、尾块、窗口和 Host/Device 分发的一致性。
