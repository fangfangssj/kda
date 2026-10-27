# regbase

- [RegBase 的 GM、UB 与寄存器数据流分层](dataflow_layers.md)：区分 Kernel 搬运与 VF 寄存器计算的对象、地址空间和生命周期，说明 MemBase 迁移时需要保留和重写的边界。
- [RegBase 错误信号与分层诊断假设](failure_triage.md)：依据构建位置和输出分布缩小 RegBase 故障范围，区分接口契约、地址边界、数据交接和数值策略问题。
- [RegBase 同步的作用域与 UB 生产消费依赖](synchronization_layers.md)：区分 VF 内 LocalMemBar、Kernel 队列与流水同步、跨核同步，按实际 UB 读写依赖选择同步范围。
- [RegBase 尾块、掩码与归约精度的联合约束](valid_lengths_and_precision.md)：把有效元素数、存储步长、mask 进度和归约中间精度联系起来，防止 padding、尾块或转换改变数学语义。

- [RegBase API 的操作数、掩码与存储合同](api_families_masks_and_memory_contracts.md)：补全 API 汇总中的模板、有效 lane、分组归约和局部存储约束，区分候选目录与实现证据。

- [Flash VF 的掩码、Cast 分布与标量槽](flash_vf_masks_casts_and_scalar_slots.md)：补全 Reg 实现模板的物理访问和数值合同，限定项目 API 选择与版本假设。

- [AscArange：固定 reference 调用约束](asc_arange_reference_contract.md)：固定 reference 的调用约束与版本边界。以指定起始值生成递增或递减索引序列并保存到矢量数据寄存器，仅在AIV上生效。
- [asc_bfloat162e1m2x2 (废弃)：固定 reference 调用约束](asc_bfloat162e1m2x2_deprecated_reference_contract.md)：固定 reference 的调用约束与版本边界。将bfloat16_t类型转化为fp4x2_e1m2_t类型，并支持多种舍入模式。 关于舍入模式的详细说明，请参见舍入模式。 由于源操作数与目的操作数类型位宽比为4:1，写入数据时需要将一个VL大小的数据分为四部分，根据不同接口选取索引0、索引1、索引2或者索引3。
- [AscCeil：固定 reference 调用约束](asc_ceil_reference_contract.md)：固定 reference 的调用约束与版本边界。将src中的浮点数元素按照CEIL（向正无穷方向舍入）舍入模式截断到整数值，结果仍保持原浮点数据类型写入dst。未被mask筛选的元素置零。
- [AscCopy：固定 reference 调用约束](asc_copy_reference_contract.md)：固定 reference 的调用约束与版本边界。完成矢量寄存器之间、掩码寄存器之间以及矢量寄存器到掩码寄存器的数据搬运。
- [asc_create_iter_reg (废弃)：固定 reference 调用约束](asc_create_iter_reg_deprecated_reference_contract.md)：固定 reference 的调用约束与版本边界。该接口已废弃，请使用asc_update_addr_reg实现此功能。 地址寄存器通过该接口初始化，然后在循环之中使用地址寄存器存储地址偏移量，该重载接口同步支持1-4个offset参数，并支持初始化b8/b16/b32位宽的寄存器，适配1-4维Tensor。底层在多维循环中自动计算地址偏移，消除标量地址运算开销。
- [AscCreateMask：固定 reference 调用约束](asc_create_mask_reference_contract.md)：固定 reference 的调用约束与版本边界。根据pat_mode生成掩码寄存器，支持b8、b16、b32三种位宽模式，指定参与计算的元素位置
- [AscDeintlv：固定 reference 调用约束](asc_deintlv_reference_contract.md)：固定 reference 的调用约束与版本边界。将src0和src1中的元素解交织存入dst0和dst1，提供矢量数据解交织和掩码解交织两种模式，仅在AIV上生效。
- [AscDiv：固定 reference 调用约束](asc_div_reference_contract.md)：固定 reference 的调用约束与版本边界。根据mask对源操作数src0、src1按元素求商，将结果写入目的操作数dst。
- [AscFloat2Half：固定 reference 调用约束](asc_float2half_reference_contract.md)：固定 reference 的调用约束与版本边界。将float类型数据转换成half类型，并支持多种舍入模式与饱和模式。 关于舍入模式和饱和/非饱和模式的详细说明，请参见舍入模式与饱和模式。 由于源操作数与目的操作数类型位宽比为2:1，写入数据时需要将一个VL大小的数据分为两部分，根据不同接口选择数据写入索引为奇数的位置或偶数的位置。
- [AscFloor：固定 reference 调用约束](asc_floor_reference_contract.md)：固定 reference 的调用约束与版本边界。将src中的浮点数元素按照FLOOR（向负无穷方向舍入）舍入模式截断到整数值，结果仍保持原浮点数据类型写入dst。未被mask筛选的元素置零。
- [AscGather：固定 reference 调用约束](asc_gather_reference_contract.md)：固定 reference 的调用约束与版本边界。根据索引位置index将源操作数src按元素收集到目的操作数dst中。
- [AscIntlv：固定 reference 调用约束](asc_intlv_reference_contract.md)：固定 reference 的调用约束与版本边界。将源操作数src0和src1中的元素交织存入目的操作数dst0和dst1中，支持掩码寄存器和矢量数据寄存器两种操作数类型。
- [AscLoad：固定 reference 调用约束](asc_load_reference_contract.md)：固定 reference 的调用约束与版本边界。从UB非32字节对齐的源地址搬运VL大小的数据到矢量数据寄存器。
- [asc_pack (废弃)：固定 reference 调用约束](asc_pack_deprecated_reference_contract.md)：固定 reference 的调用约束与版本边界。该接口已废弃，请使用 asc_pack_to_low 和 asc_pack_to_high 替代。 将源操作数中的元素选取低8位（b16）、低16位（b32）、低32位（b64）写入目的操作数的低半部分或高半部分。 - asc_pack：将源操作数写入目的操作数的低半部分。 - asc_pack_v2：将源操作数写入目的操作数的高半部分。
- [AscReduceMax：固定 reference 调用约束](asc_reduce_max_reference_contract.md)：固定 reference 的调用约束与版本边界。根据mask对源操作数进行归约最大值操作，将最大值及其索引写入目的操作数。
- [AscReduceMin：固定 reference 调用约束](asc_reduce_min_reference_contract.md)：固定 reference 的调用约束与版本边界。根据mask对源操作数进行归约最小值操作，将最小值及其索引写入目的操作数。
- [AscReduceSum：固定 reference 调用约束](asc_reduce_sum_reference_contract.md)：固定 reference 的调用约束与版本边界。根据mask对源操作数进行归约求和操作，将所有参与计算的元素之和保存到目的操作数。
- [AscRint：固定 reference 调用约束](asc_rint_reference_contract.md)：固定 reference 的调用约束与版本边界。将浮点数按RINT（四舍六入五成双）舍入为整数值，结果保持原浮点类型。
- [AscRound：固定 reference 调用约束](asc_round_reference_contract.md)：固定 reference 的调用约束与版本边界。将浮点数按ROUND（四舍五入）舍入为整数值，结果保持原浮点类型。
- [AscSelect：固定 reference 调用约束](asc_select_reference_contract.md)：固定 reference 的调用约束与版本边界。根据mask从源操作数src0、src1中选择元素得到dst，mask比特位为1时选src0对应位置，为0时选src1对应位置。
- [AscSqueezeV2：固定 reference 调用约束](asc_squeeze_deprecated_reference_contract.md)：固定 reference 的调用约束与版本边界。已废弃接口，请使用asc_squeeze_and_storeunalign。将src中被mask选择的有效元素依次复制到dst并连续排列，搭配asc_storeunalign_postupdate可将有效数据大小保存至AR寄存器。
- [AscSqueeze：固定 reference 调用约束](asc_squeeze_reference_contract.md)：固定 reference 的调用约束与版本边界。将src中被mask选择的有效元素依次复制到dst中，有效元素在dst中从低到高连续排列，剩余位置元素置为0。本接口不会将有效数据大小保存至AR寄存器。
- [AscStore：固定 reference 调用约束](asc_store_reference_contract.md)：固定 reference 的调用约束与版本边界。从矢量数据寄存器搬出到 UB，保存 950 来源的 33 个 asc_store 重载、显式 count 与满向量搬运边界。
- [AscStoreUnalign：固定 reference 调用约束](asc_storeunalign_reference_contract.md)：固定 reference 的调用约束与版本边界。从矢量数据寄存器连续非32B对齐搬出到UB，使用uint32_t作为存储偏移量，不改变源操作数地址，每次循环需用户手动更新目的地址。
- [AscTrunc：固定 reference 调用约束](asc_trunc_reference_contract.md)：固定 reference 的调用约束与版本边界。将浮点数按TRUNC（向零方向截断）截断为整数值，结果保持原浮点类型。
- [AscUnsqueeze：固定 reference 调用约束](asc_unsqueeze_reference_contract.md)：固定 reference 的调用约束与版本边界。根据掩码对矢量数据进行解压缩，dst各元素值取决于mask中前序元素为1的数量
- [AscUpdateMask：固定 reference 调用约束](asc_update_mask_reference_contract.md)：固定 reference 的调用约束与版本边界。根据元素个数生成mask并自动扣减当前向量处理单元元素数，支持b8、b16、b32位宽模式
- [比较与选择：固定 reference 调用约束](compare_and_select_reference_contract.md)：固定 reference 的调用约束与版本边界。寄存器侧比较与选择类接口汇总，提供eq、ge、gt、le、lt、ne等逐元素比较及select、squeeze选择能力。
