# 任务间同步

- [SetNextTaskStart](set_next_task_start.md)：在SuperKernel子Kernel中调用，使其后指令与后续子Kernel并行执行，提升整体性能。
- [WaitPreTaskEnd](wait_pre_task_end.md)：算子层面支持SuperKernel特性Early-Start能力的接口，在子Kernel中调用后，调用前的指令可与前序子Kernel并行执行，提升整体性能。
