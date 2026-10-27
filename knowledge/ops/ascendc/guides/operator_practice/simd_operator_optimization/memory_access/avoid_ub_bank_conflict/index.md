# 避免UB的bank冲突

- [避免bank冲突（NPU架构版本2201）](avoid_bank_conflict_npu_arch_2201.md)：>该性能优化建议适用于如下产品型号：
- [避免bank冲突（NPU架构版本3510）](avoid_bank_conflict_npu_arch_3510.md)：>该性能优化建议适用于如下产品型号：
- [SIMD算子优化概述](overview.md)：为了提高数据访问的效率和吞吐量，Unified Buffer采用了大小相等的内存模块（bank）结构设计。当多条读写指令同时访问Unified Buffer时，由于硬件资源的限制，这些指令不能同时执行，从而引发bank冲突。在这种情况下，指令需要排队等待资源，无法在一个指令周期内完成。
