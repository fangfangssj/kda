# gridgroup

- [BlockIndex](block_index.md)：获取当前线程所属线程块在Grid中的三维索引。
- [BlockRank](block_rank.md)：获取当前线程所属线程块在Grid内所有线程块中的排名，排名从0开始计数。
- [DimBlocks](dim_blocks.md)：获取当前Grid的线程块维度配置。
- [GridGroupConstructor](grid_group_constructor.md)：grid_group不提供默认的构造函数，用户使用this_grid函数获取当前Grid对应的grid_group对象。
- [grid_group简介](grid_group_intro.md)：grid_group是对一个核函数启动的所有SIMT线程的抽象，线程范围覆盖当前Grid中的所有线程块及其线程。通过grid_group可以查询Grid维度、线程块排名、全局线程排名，并通过sync()在Grid范围内进行跨线程块同步。
- [GroupDim](group_dim.md)：获取当前Grid的线程块维度配置，与接口dim_blocks功能相同。
- [IsValid](is_valid.md)：判断当前grid_group是否可使用sync()接口进行跨线程块同步。grid_group默认支持同步功能，因此接口固定返回true。
- [NumBlocks](num_blocks.md)：获取当前Grid内的线程块总数。
- [NumThreads](num_threads.md)：获取当前Grid内的SIMT线程总数。
- [Grid协作组size](size.md)：获取当前Grid内的SIMT线程总数，与接口num_threads功能相同。
- [Grid协作组sync](sync.md)：同步当前Grid内所有SIMT线程。所有线程块中的所有线程都执行到该同步点位置后，Grid内线程才能继续执行。
- [ThreadRank](thread_rank.md)：获取当前线程在Grid内所有线程中的排名，排名从0开始计数。
