# coalescedgroup

- [CoalescedGroup协作组all](all.md)：判断coalesced_group组内所有线程的输入是否均不为0。
- [CoalescedGroup协作组any](any.md)：判断coalesced_group组内是否有线程的输入不为0。
- [Ballot](ballot.md)：判断coalesced_group组内每个线程的输入是否非零。
- [coalesced_group构造函数](coalesced_group_constructor.md)：coalesced_group不提供默认的构造函数，用户使用coalesced_threads函数创建coalesced_group对象。
- [coalesced_group简介](coalesced_group_intro.md)：在SIMT架构的硬件层面上，处理器以32个线程为一组（一个Warp）来执行线程。如果核函数代码中的条件分支导致Warp内的线程出现发散（Warp Divergence），那么Warp会串行执行每个分支，在执行某个分支时会屏蔽不在该指令路径上的线程。coalesced_group用于获取当前Warp中实际活跃的线程子集。
- [MetaGroupRank](meta_group_rank.md)：获取当前线程所在的组在其父组划分出的子组集合中的排名，排名从0开始。
- [MetaGroupSize](meta_group_size.md)：获取当前线程所在的协作组其直接父组被划分时创建的子组的数量。
- [NumThreads](num_threads.md)：获取coalesced_group组内线程总数。
- [CoalescedGroup协作组shfl](shfl.md)：coalesced_group组内线程的数据交换接口，不通过共享内存实现直接读取组内指定线程的寄存器中的值。
- [ShflDown](shfl_down.md)：获取coalesced_group组内当前线程向后偏移delta的线程的数据。
- [ShflUp](shfl_up.md)：获取coalesced_group组内当前线程向前偏移delta的线程的数据。
- [CoalescedGroup协作组size](size.md)：获取coalesced_group组内线程总数，与接口num_threads功能相同。
- [CoalescedGroup协作组sync](sync.md)：同步coalesced_group组内所有线程。
- [ThreadRank](thread_rank.md)：获取当前线程在coalesced_group组内的排名，排名从0开始。
