# Lane ID类函数

- [Laneid](laneid.md)：获取线程在其所在Warp内的索引。
- [LanemaskEq](lanemask_eq.md)：获取当前线程的32位掩码，在所属Warp中只有当前线程所在的Lane位被置1，其余为0。
- [LanemaskGe](lanemask_ge.md)：获取当前线程的32位掩码，在所属Warp中将Lane ID大于等于当前线程的线程对应位设为1，其余位为0。
- [LanemaskGt](lanemask_gt.md)：获取当前线程的32位掩码，在所属Warp中将Lane ID严格大于当前线程的线程对应位设为1，其余位为0。
- [LanemaskLe](lanemask_le.md)：获取当前线程的32位掩码，在所属Warp中将Lane ID小于等于当前线程的线程对应位设为1，其余位为0。
- [LanemaskLt](lanemask_lt.md)：获取当前线程的32位掩码，在所属Warp中将Lane ID严格小于当前线程的线程对应位设为1，其余位为0。
