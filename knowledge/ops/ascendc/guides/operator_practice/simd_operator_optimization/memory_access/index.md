# 内存访问

- [避免同地址访问](avoid_same_address_access.md)：>该性能优化指导适用于如下产品型号：
- [避免UB的bank冲突](avoid_ub_bank_conflict/index.md)：为了提高数据访问的效率和吞吐量，Unified Buffer采用了大小相等的内存模块（bank）结构设计。当多条读写指令同时访问Unified Buffer时，由于硬件资源的限制，这些指令不能同时执行，从而引发bank冲突。在这种情况下，指令需要排队等待资源，无法在一个指令周期内完成。（共 3 张知识卡：避免bank冲突（NPU架构版本2201）、避免bank冲突（NPU架构版本3510））
- [EfficientTransferApi](efficient_transfer_api.md)：使用搬运API时应通过配置搬运控制参数实现连续或固定间隔搬运，避免使用for循环，提升搬运效率。
- [GM地址尽量512B对齐](gm_address_512b_alignment.md)：【描述】由于AI处理器内部设计约束，从GM向Local Memory搬运数据时，保证GM地址512B对齐可以最有效的发挥出带宽的效率。如下图示例，展示了在512B对齐以及32B对齐情况下单核的带宽效率：搬运同等数据量，带宽差距最大的情况，32B对齐场景只能达到512B对齐场景的70%。
- [设置合理的L2 CacheMode](l2_cache_mode_config.md)：>该性能优化指导适用于如下产品型号：
- [L2CacheSplit](l2_cache_split.md)：当输入和输出数据量超过L2 Cache大小时，在Tiling中开启L2 Cache切分策略以提升带宽利用效率。
- [非对齐场景减少无效数据的搬运](reduce_invalid_data_transfer.md)：>该性能优化建议适用于如下型号：
- [非连续搬运场景减少搬运次数](reduce_non_contiguous_transfer.md)：>该性能优化建议适用于如下产品型号：
- [ReduceShapeInfoOptimizeStack](reduce_shapeinfo_optimize_stack.md)：通过缩减Tensor的ShapeInfo维度来减小栈空间占用，减少scalar指令和cache miss，提升算子性能。
- [算子与高阶API共享临时Buffer](shared_temp_buffer.md)：【描述】如果算子使用的高阶API需要传入临时Buffer，如SoftMax，该临时空间会挤占算子其他计算的空间，从而导致单次计算搬运的数据量变少，搬运的次数变多。此场景可通过共享临时Buffer空间，提升单次搬运的数据量，减少搬运的次数，提升内存使用效率。
- [尽量一次搬运较大的数据块](transfer_larger_data_blocks.md)：【描述】搬运不同大小的数据块时，对带宽的利用率（有效带宽/理论带宽）不一样。根据实测经验，单次搬运数据长度16KB以上时，通常能较好地发挥出带宽的最佳性能。因此对于单次搬运，应考虑尽可能的搬运较大的数据块。下图展示了某款AI处理器上实测的不同搬运数据量下带宽的变化图。
- [纯搬运类算子VECIN和VECOUT建议复用](vecin_vecout_reuse.md)：【描述】纯搬运类算子在执行时并不涉及实际vector计算，若存在冗余的vector指令，会导致算子整体执行时间变长。这种场景可以使用Ascend C针对纯搬运类算子提供的TQueBind接口，该接口可以将UB（VECIN）与UB（VECOUT）绑定，省略将数据从UB（VECIN）拷贝到UB（VECOUT）的步骤，从而避免vector的无谓消耗。
