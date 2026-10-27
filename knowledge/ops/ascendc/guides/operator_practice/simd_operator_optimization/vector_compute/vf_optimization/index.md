# vf optimization

- [连续非对齐场景优化](continuous_unaligned_optimization.md)：连续非对齐搬运场景的性能优化建议与接口介绍。
- [指令双发优化](dual_issue_optimization.md)：指令双发指的是处理器在同一个时钟周期内，能够同时发射两条指令到执行单元进行处理。这种能力需要满足以下两个条件：
- [VfFusionOptimization](vf_fusion_optimization.md)：将多个VF函数融合成一个VF函数以提升性能，编译器通过Loop Fuse算法自动执行浅度融合、深度融合和VF内自动同步。
- [VfLoopOptimization](vf_loop_optimization.md)：VF函数中循环满足Hardware Loop编码规范时可被编译器优化为硬件级向量循环，提升执行性能，支持成员变量访问、指令分布和地址管理优化。
