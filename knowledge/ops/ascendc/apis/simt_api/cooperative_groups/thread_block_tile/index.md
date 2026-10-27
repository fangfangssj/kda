# threadblocktile

- [ThreadBlockTile协作组all](all.md)：判断thread_block_tile组内所有线程的输入是否均不为0。
- [ThreadBlockTile协作组any](any.md)：判断thread_block_tile组内是否有线程的输入不为0。
- [Ballot](ballot.md)：判断thread_block_tile组内每个线程的输入是否非零，返回32位无符号整数表示各线程对应bit位。
- [MetaGroupRank](meta_group_rank.md)：获取当前线程所在的组在从父组划分出的子组集合中的排名，排名从0开始。
- [MetaGroupSize](meta_group_size.md)：获取当前线程所在的协作组其直接父组被划分时创建的子组的数量。
- [NumThreads](num_threads.md)：获取当前线程所属的thread_block_tile组内的线程总数。
- [ThreadBlockTile协作组shfl](shfl.md)：thread_block_tile组内线程的数据交换接口，用于获取组内指定rank线程输入的var值。
- [ShflDown](shfl_down.md)：获取thread_block_tile组内当前线程向后偏移delta的线程的数据。
- [ShflUp](shfl_up.md)：获取thread_block_tile组内当前线程向前偏移delta的线程的数据。
- [ShflXor](shfl_xor.md)：将当前线程的rank与lane_mask进行按位异或运算得到的rank，获取该rank的线程输入的var值。
- [ThreadBlockTile协作组size](size.md)：获取协作组内线程的总数，功能与num_threads函数一致。
- [ThreadBlockTile协作组sync](sync.md)：同步当前线程所属的thread_block_tile组内的所有线程。
- [thread_block_tile构造函数](thread_block_tile_constructor.md)：thread_block_tile不提供默认构造函数，可通过tiled_partition接口从另一个协作组中划分得到。
- [thread_block_tile简介](thread_block_tile_intro.md)：thread_block_tile是一个模板类，用于管理指定大小的线程子组。
- [ThreadRank](thread_rank.md)：获取当前线程在所属的thread_block_tile组内的排名，排名从0开始。
