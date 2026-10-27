# 同步与内存栅栏

- [内存栅栏接口](memory_fence/index.md)：共 2 张知识卡（如 AscThreadfence、AscThreadfenceBlock）。
- [同步与内存栅栏简介（同步与内存屏障）](sync_and_memory_fence_intro.md)：SIMT程序中，不同Thread的执行进度和内存访问到达顺序可能不同。当线程之间存在数据依赖时，需要使用同步接口或内存栅栏接口约束执行顺序和内存可见性。
- [同步接口](sync_interface/index.md)：阻塞当前线程块内所有线程直到所有线程到达同步点，确保同步点前的内存操作对块内所有线程可见，常用于数据共享和并行归约。
