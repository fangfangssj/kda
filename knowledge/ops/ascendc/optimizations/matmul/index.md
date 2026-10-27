# matmul

- [Matmul 边缘 tile 的数量不变重分配](edge_tile_size_redistribution.md)：将满块和薄尾块的总范围重新分配给相同数量的tile，维护对齐、尾部正长度、GM前缀和及窗口成本边界。
- [Matmul 的 L1、L0 与 Scale 独立双缓冲](independent_l1_l0_double_buffering.md)：区分 K 覆盖倍数与物理槽数，按三层消费者完成条件组织事件，并用发射时间线判断实际重叠。
- [Matmul 最后一波任务的二维子块分配](last_wave_2d_subdivision.md)：将末波各父tile拆成二维输出子块，明确全局轮次、父子编号、有效范围、空子块与同步参与者。
- [MX Matmul 合并 Scale 加载的周期、偏移与末轮释放](merged_scale_loading.md)：用比数据tile更大的Scale窗口减少小搬运次数，保持量化块坐标、物理布局和所有消费者完成条件一致。
- [Matmul 单侧全载的复用键、Scale 切片与 L1 预算](resident_operand_and_scale_reuse.md)：将不变的矩阵和量化元数据按明确任务范围常驻，推导实际节省字节数，并保留流式对侧的缓冲和完成协议。
- [矩阵分块的蛇形窗口遍历与末窗口映射](serpentine_tile_traversal.md)：通过窗口内M方向并行和窗口间N反向保留访问局部性，明确末窗口高度及线性任务到矩阵坐标的双射。
- [Matmul 软件预取的任务序列、槽位身份与边界](software_prefetch_task_sequence.md)：分离预取、消费和回收游标，处理首轮、K尾段及跨输出tile转换，并区分手工K预取与Matmul高阶API的M/N预加载。
- [Matmul 静态 Tiling、部分常量化与运行期 shape](static_tiling_and_runtime_shapes.md)：区分静态配置上界、基本块和真实矩阵跨度，保留初始化顺序、转置、bias及原子累加语义，并验证实际 Scalar 收益。

- [反量化输出的 Bank Padding 与 L1 搬运步长](dequant_bank_padding_and_l1_copy_stride.md)：将 bank 候选变换落实为完整物理地址关系，纠正固定吞吐与禁止 Tensor API 的外推。
