# LocalMemAllocator

- [Alloc](alloc.md)：根据用户指定的逻辑位置、数据类型、数据长度返回对应的LocalTensor对象。
- [GetCurAddr](get_cur_addr.md)：返回当前物理位置空闲的起始地址。
- [LocalMemAllocatorConstructor](local_mem_allocator_constructor.md)：LocalMemAllocator构造函数，用于初始化指定物理位置的内存分配器对象。
- [LocalMemAllocator简介](local_mem_allocator_intro.md)：LocalMemAllocator是在使用静态Tensor编程时用于内存管理的类，用户无需构建TPipe/TQue，而是直接创建LocalTensor对象（也可以直接通过LocalTensor构造函数进行构造）并开发算子，从而减少运行时的开销，实现更优的性能。
