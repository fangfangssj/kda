# 数据填充

- [Brcb](brcb.md)：数据填充接口，每次取输入张量8个数，将每个数填充到结果张量一个DataBlock（32字节）中。
- [CreateVecIndex](create_vec_index.md)：创建指定起始值的向量索引接口，支持前n个数据计算和高维切分计算两种方式。
- [Duplicate](duplicate.md)：将变量或立即数复制多次并填充到向量中，支持前n个数据计算和高维切分计算两种方式。
- [VectorPadding](vector_padding_isasi.md)：根据填充模式和方向对源操作数按datablock进行填充操作，支持邻近数填充和对称填充等模式。
