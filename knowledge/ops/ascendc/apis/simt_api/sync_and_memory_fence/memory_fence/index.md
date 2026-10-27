# 内存栅栏接口

- [AscThreadfence](asc_threadfence.md)：内存栅栏接口，保证调用前的全局和共享内存写操作对其他线程可见，不阻塞线程，仅约束可见性顺序。
- [AscThreadfenceBlock](asc_threadfence_block.md)：Block范围内存栅栏接口，保证调用前的内存读写操作对同一线程块内其他线程可见，不阻塞线程，仅约束Block范围可见性。
