# tpipe tque programming

- [TPipe和TQue编程范式](tpipe_tque_paradigm.md)：上一章阐述了TPipe‑TQue编程模型的底层设计原理：基于经典C/C++队列管道（Queue Pipeline）思想，将算子任务拆分为多级流水Stage，依靠TQue队列解决写后读数据依赖、依靠AllocTensor/FreeTensor解决读后写内存复用冲突，通过底层同步指令实现多单元异步并行的时序安全。
- [TPipe-TQue编程原理介绍](tpipe_tque_principles.md)：Ascend C编程模型的核心架构，深度承袭了传统C/C++并发编程中队列管道(Queue Pipeline) 的经典设计理念。在通用并发编程体系中，管道模式的核心价值在于任务解耦与流程结构化：将复杂计算任务拆分为多个独立、可并行执行的处理阶段（Stage），各阶段之间通过线程安全队列完成数据传递与消息同步。
