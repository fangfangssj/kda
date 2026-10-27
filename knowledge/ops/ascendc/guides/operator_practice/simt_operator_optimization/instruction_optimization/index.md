# instruction optimization

- [原子操作指令优化](atomic_instruction_optimization.md)：【描述】SIMT编程模式下，多个线程更新同一个内存地址时需要使用原子操作接口保证更新结果正确。原子操作属于高开销指令，其耗时由两方面决定：一是同一地址上的原子操作只能排队依次执行，二是单条原子指令的处理开销。

- [原地加法指令优化](inplace_add_instruction_optimization.md)：【优先级】中
