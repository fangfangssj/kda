# TSCM

- [AllocTensor](alloc_tensor.md)：从TSCM中分配tensor，大小为InitBuffer时设置的每块内存长度，分配内容可能为随机值。
- [DeQue](de_que.md)：将tensor从TSCM队列中取出，用于后续处理。
- [EnQue](en_que.md)：将tensor push到TSCM队列中，入队成功返回true，队列已满返回false。
- [FreeTensor](free_tensor.md)：释放TSCM中指定的Tensor资源。
- [TSCM简介](tscm_intro.md)：Vector和Cube之间通过队列（Queue）完成任务间通信和同步。TSCM是数据通路的目的地在TSCM Position时，用来管理执行队列相关操作、相关资源的数据结构。TSCM与TQueBind，TQue属于相同类型结构。TSCM定义如下：
