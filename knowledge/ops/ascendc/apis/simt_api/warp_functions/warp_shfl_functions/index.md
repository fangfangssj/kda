# Warp Shfl类函数

- [AscShfl](asc_shfl.md)：Warp级数据交换接口，按分组宽度获取指定线程持有的变量值，支持数据分发与动态交换场景
- [AscShflDown](asc_shfl_down.md)：获取Warp内当前线程向后偏移delta的线程变量值，支持width参数划分分组
- [AscShflUp](asc_shfl_up.md)：获取Warp内当前线程向前偏移delta的线程变量值，支持width参数划分分组
- [AscShflXor](asc_shfl_xor.md)：获取Warp内当前线程Lane ID与lane_mask异或后对应线程的变量值，支持width分组
