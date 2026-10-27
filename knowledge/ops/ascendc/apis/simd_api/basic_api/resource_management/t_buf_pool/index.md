# TBufPool

- [InitBufPool](init_buf_pool.md)：从Tpipe或TBufPool划分资源池，支持非共享模式和共享模式（复用已有资源池地址）。
- [InitBuffer](init_buffer.md)：为TQue或TBuf进行内存分配，支持指定内存块个数和大小。
- [Reset](reset.md)：切换TBufPool资源池时使用，结束当前资源池相关事件，资源池及Buffer仍然存在。
- [TBufPool](t_buf_pool_constructor.md)：TBufPool对象构造函数，初始化数据成员，支持指定逻辑位置和可分配Buffer数量。
- [TBufPool简介](t_buf_pool_intro.md)：TPipe可以管理全局内存资源，而TBufPool可以手动管理或复用Unified Buffer/L1 Buffer物理内存，主要用于多个stage计算中Unified Buffer/L1 Buffer物理内存不足的场景。
