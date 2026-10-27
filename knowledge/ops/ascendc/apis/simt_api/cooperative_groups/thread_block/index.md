# threadblock

- [DimThreads](dim_threads.md)：获取当前线程所属线程块的线程配置，返回值与内置变量blockDim相同。
- [GroupDim](group_dim.md)：获取当前线程所属线程块的线程配置，与接口dim_threads功能相同。
- [GroupIndex](group_index.md)：获取当前线程所属的线程块在网格（grid）中的三维索引，返回值与内置变量blockIdx相同。
- [NumThreads](num_threads.md)：获取当前线程所属线程块内的线程总数。
- [ThreadBlock协作组size](size.md)：获取当前线程所属线程块内的线程总数，与接口num_threads功能相同。
- [ThreadBlock协作组sync](sync.md)：同步线程块内所有线程，所有线程都执行到该同步点位置才能继续执行。
- [thread_block构造函数](thread_block_constructor.md)：thread_block不提供默认的构造函数，用户使用this_thread_block函数获取当前线程所属的thread_block对象。
- [thread_block简介](thread_block_intro.md)：thread_block是对线程块的抽象，它代表了核函数的启动配置（启动的线程块个数，每个线程块内的线程数量等）。
- [ThreadIndex](thread_index.md)：获取当前线程在线程块中的索引，返回值与内置变量threadIdx相同。
- [ThreadRank](thread_rank.md)：获取当前线程在线程块内所有线程中的排名，排名从0开始。
