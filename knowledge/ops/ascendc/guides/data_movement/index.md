# data movement

- [DataCopyPad 多行搬运的有效长度、padding 与物理行距](datacopypad_rows_and_padding.md)：分别推导 GM 与 UB 地址推进、显式 padding 和隐式补齐，检查参数范围、尾行及批量搬运的完整往返。

- [合并写回的目标布局与转置地址表](packed_writeback_and_transpose_addresses.md)：保留短搬运合并和 TransDataTo5HD 地址计算细节，明确连续源不等于连续目的。

- [ND/NZ 随路变换与块位置重排](nd_nz_transform_and_block_placement.md)：保留 Nd2Nz 参数意图和 Attention 分块写回公式，明确实际数据方向与单位。

- [设备 payload 常驻与 Host 索引交换](device_payload_and_host_index_exchange.md)：为路由和重排行为保留索引编码、稳定排序及总线字节模型。

- [批量搬运、LoopMode 与交错 Scale 偏移](batched_copy_loop_mode_and_packed_scale_offsets.md)：保存多维搬运和MX物理行距，补齐参数状态与反向映射。

- [DMA 优化单位、地址对齐与批量输出](dma_optimization_units_alignment_and_batched_outputs.md)：保留搬运融合思路，纠正padding字段单位和向上移动数据起点。

- [DataCopyPad 的版本、重载与尾部真实范围](datacopypad_versions_overloads_and_tail_extent.md)：结合同一来源的失败及后续成功记录限定 DataCopyPad 可用性，避免用填充 workaround 掩盖越界或改变布局。

- [对齐补齐的通道被当作地址使用时是越界风险，不是数值风险](padding_lanes_used_as_addresses.md)：向量按 32B 对齐上取整后，补齐通道若作为 Gather 的偏移被解引用，读的是未初始化字节偏移，属越界访问而非算错。
