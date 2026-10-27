# TQue

- [AllocTensor](alloc_tensor.md)：从Que中分配Tensor，Tensor大小为InitBuffer时设置的每块内存长度，支持non-inplace和inplace两种接口。
- [DeQue](de_que.md)：将Tensor从队列中取出用于后续处理，支持non-inplace和inplace两种接口，可通过TQueBind指定源和目的位置。
- [EnQue](en_que.md)：将Tensor加入队列，返回是否入队成功，可通过TQueBind指定源和目的位置实现内存复用。
- [FreeTensor](free_tensor.md)：释放Que中的指定Tensor，需与AllocTensor配对使用。
- [GetTensorCountInQue](get_tensor_count_in_que.md)：查询Que中已入队的Tensor数量，不支持Tensor原地操作场景。
- [HasIdleBuffer](has_idle_buffer.md)：查询Que中是否有空闲的内存块可供AllocTensor分配，不支持Tensor原地操作场景。
- [HasTensorInQue](has_tensor_in_que.md)：查询Que中目前是否已有入队的Tensor，不支持Tensor原地操作场景。
- [TQue简介](t_que_intro.md)：流水任务之间通过队列（Queue）完成任务间通信和同步。TQue是用来执行队列相关操作、管理相关资源的数据结构。TQue继承自TQueBind父类，继承关系如下：
- [VacantInQue](vacant_in_que.md)：查询队列是否已满，返回true表示可继续EnQue，返回false表示已满，不支持Tensor原地操作场景。
