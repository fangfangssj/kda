# AI-Core-SIMT编程

- [抽象硬件架构（AI Core SIMT编程）](abstract_hardware_architecture.md)：AI Core上的SIMT（Single Instruction Multiple Thread，单指令多线程）编程允许指令对数据进行独立寻址，从而支持线程级并行计算。这种方式特别适用于离散访问和复杂控制逻辑等场景。SIMT编程能够有效简化复杂算子和不规则控制流的开发，缓解包含分支的发散计算并控制程序复杂度，从而提高硬件利用率和能效。
- [原子操作（AI Core SIMT编程）](atomic_operations.md)：SIMT编程模型中，一个Grid由多个Thread Block组成，每个Thread Block又包含多个Thread。不同Thread可以并发访问同一段Global Memory或Unified Buffer。当多个线程同时对同一地址执行读-改-写操作时，如果没有同步保护，可能出现数据竞争，导致最终结果不符合预期。
- [核函数（AI Core SIMT编程）](kernel_function.md)：正如在概述中所介绍的，能够在AI处理器（NPU）上并行执行、并由主机端（Host）发起调用的函数被称为核函数（Kernel）。核函数被设计为由大量线程（Threads）同时并行运行。每个线程都拥有独立的寄存器和栈，它们协同工作以完成海量的数据计算任务。
- [内存层级（AI Core SIMT编程）](memory_hierarchy.md)：SIMT线程可访问多种内存空间。下表汇总了SIMT编程中常见内存类型的作用域及其生命周期。
- [AiCoreSimtProgrammingOverview](overview.md)：SIMT编程模型适用于离散数据访问和复杂控制流场景，基于Grid-Block-Thread三级线程层级在AIV核上实现线程级并行计算。
- [编程示例（AI Core SIMT编程）](programming_examples.md)：基于SIMT进行算子开发需要使用的内置关键字和API请参见SIMT BuiltIn关键字、SIMT语言扩展层C API。当前SIMT编程暂不支持部分语法结构，相关限制请参考语法限制。
- [同步机制（AI Core SIMT编程）](synchronization.md)：SIMT是一种单指令多线程的编程模型，该编程模型旨在通过多线程并发执行达到隐藏访存延迟，提升计算吞吐量的目的。SIMT编程模型中，一个Thread Block内的多个Thread并行执行，不同Thread的执行进度和内存访问到达顺序可能不同。当线程之间存在数据依赖时，需要使用同步接口或内存栅栏接口约束执行顺序和内存可见性，避免读到未完成写入的数据。同步接口主要分为两类：
- [ThreadArchitecture](thread_architecture.md)：SIMT编程模型采用Grid、Thread Block、Thread分层线程组织结构，每个层次维度可自定义配置，通过Warp调度机制执行。
