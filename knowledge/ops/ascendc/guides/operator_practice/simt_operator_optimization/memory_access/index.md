# memory access

- [避免UB的Bank冲突](avoid_ub_bank_conflict_simt.md)：【描述】SIMT编程模式下，一个Warp内的多个线程可能在同一条UB访问指令中同时访问UB。对于使用UB中转做转置、重排等操作的算子，即使GM读写已经连续，仍需要分析UB访问模式，避免UB内部访问成为新的性能瓶颈。
- [DCacheAccess](dcache_access.md)：SIMT编程模式下通过不同访存函数为不同类型数据指定缓存策略，使热点数据优先驻留DCache，减少GM访问。
- [MemoryAccessMerge](memory_access_merge.md)：SIMT编程模式下让同一Warp内相邻线程访问连续GM地址以实现访存合并，提高GM带宽利用率。
- [ShortVectorEfficiency](short_vector_efficiency.md)：SIMT编程模式下使用短向量类型替代普通数据类型进行数据搬运和计算，覆盖整个Cache Line，提升访存效率。
- [通过类型对齐提升搬运效率](type_alignment_transfer.md)：【描述】SIMT编程模式下，结构体类型对齐会影响Global Memory的访存效率。在数据总量不变时，可通过类型对齐到更大的位宽，提升数据搬运效率。
