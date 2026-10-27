# advanced kernel programming

- [intra core sync](intra_core_sync/index.md)：当矢量算子的数据搬入、矢量计算和数据搬出等不同流水之间存在数据依赖时，通过Mutex建立流水同步关系，使不同流水中存在数据依赖的操作串行执行，可以避免不同流水并发访问同一资源时产生数据错误。
