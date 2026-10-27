# TQueBind

- [AllocTensor](alloc_tensor.md)：从Que中分配Tensor，Tensor所占大小为InitBuffer时设置的每块内存长度。
- [DeQue](de_que.md)：将Tensor从队列中取出，用于后续处理。
- [EnQue](en_que.md)：将Tensor加入队列。
- [FreeAllEvent](free_all_event.md)：释放TQueBind队列中申请的所有同步事件，便于在Buffer数量超过同步事件限制时重新申请队列。
- [FreeTensor](free_tensor.md)：释放TQueBind队列中的指定Tensor。
- [GetTensorCountInQue](get_tensor_count_in_que.md)：查询TQueBind队列中已入队的Tensor数量。
- [HasIdleBuffer](has_idle_buffer.md)：查询TQueBind队列中是否有空闲的内存块。
- [HasTensorInQue](has_tensor_in_que.md)：查询TQueBind队列中目前是否已有入队的Tensor。
- [InitBufHandle](init_buf_handle.md)：为TQue、TBuf对象的内存块进行内存分配操作，包括设置内存块的大小和地址等。
- [InitStartBufHandle](init_start_buf_handle.md)：设置TQue/TBuf的起始内存块指针、内存块个数和每块内存块的大小。
- [TQueBind](t_que_bind_constructor.md)：TQueBind构造函数，用于创建TQueBind对象。
- [TQueBind简介](t_que_bind_intro.md)：TQueBind绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置 、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。Tque是TQueBind的简化模式。通常情况下开发者使用TQue进行编程，TQueBind对外提供一些特殊数据通路的内存管理和同步控制，涉及这些通路时可以直接使用TQueBind。
- [VacantInQue](vacant_in_que.md)：查询TQueBind队列是否已满。
