# 协作组

- [binary_partition](binary_partition.md)：根据标签将父组划分为两个子组，标签相同的线程被分配到同一组中。
- [coalescedgroup](coalesced_group/index.md)：共 14 张知识卡（如 CoalescedGroup协作组all、CoalescedGroup协作组any、Ballot 等）。
- [协作组简介](cooperative_groups_intro.md)：协作组（cooperative_groups）是Ascend C SIMT编程模型的一种拓展，用于组织协作线程组。通过协作组能够更细粒度地管理线程，从而实现更高效的线程并行协作。
- [gridgroup](grid_group/index.md)：共 12 张知识卡（如 BlockIndex、BlockRank、DimBlocks 等）。
- [sync（协作组）](sync.md)：同步指定协作组内的所有线程，功能等价于调用该协作组对象的成员函数sync()。
- [threadblock](thread_block/index.md)：共 10 张知识卡（如 DimThreads、GroupDim、GroupIndex 等）。
- [threadblocktile](thread_block_tile/index.md)：共 15 张知识卡（如 ThreadBlockTile协作组all、ThreadBlockTile协作组any、Ballot 等）。
- [tiled_partition](tiled_partition.md)：将线程组划分为多个更小、固定大小的子组，提供模板和非模板版本，分别用于编译时和运行时确定划分大小，模板版本支持跨Warp。

- [thread group](thread_group/index.md)：共 7 张知识卡（如 get_type、num_threads、size 等）。
