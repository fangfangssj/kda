# SIMD计算说明

- [连续计算（SIMD计算）](continuous_compute.md)：连续计算API，支持Tensor前n个数据计算。针对源操作数的连续n个数据进行计算并连续写入目的操作数，解决一维Tensor的连续计算问题。
- [高维切分（SIMD计算）](high_dim_split.md)：说明 SIMD 高维切分 API 如何通过 repeatTime、stride 和 mask 控制迭代、地址间隔与掩码。
- [SIMD计算掩码](mask.md)：Mask（掩码）是Ascend C中用于控制数据元素是否参与计算的标识位序列。每一位掩码对应数据中的一个元素的位置，通过有效位和无效位标记，实现对数据操作的精细化开关控制。
- [SIMD计算概述](overview.md)：矢量计算通过Vector计算单元完成，矢量计算的源操作数和目的操作数均通过Unified Buffer（UB）来进行存储。
