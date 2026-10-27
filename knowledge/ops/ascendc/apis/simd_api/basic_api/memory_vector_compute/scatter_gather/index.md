# 离散与聚合

- [Gather](gather.md)：根据基地址和地址偏移张量确定索引，将输入张量中对应元素收集到结果张量中，支持前n个数据和高维切分计算。
- [Gatherb](gatherb_isasi.md)：根据DataBlock偏移张量将输入张量按DataBlock粒度收集到结果张量中，每次迭代完成8个DataBlock的数据收集。
- [Scatter](scatter_isasi.md)：根据目的地址偏移张量将连续输入张量的元素分散到结果张量的指定位置。
