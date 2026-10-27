# ReduceAny接口

- [GetReduceAnyMaxMinTmpSize](get_reduce_any_max_min_tmp_size.md)：在host侧获取ReduceAny接口计算所需的最大和最小临时空间大小，用于选择合适的空间作为Tiling参数传递到kernel侧。
- [ReduceAny](reduce_any.md)：对一个多维向量在指定的维度求逻辑或。 定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对shape为\(2, 3\)的二维矩阵进行运算，指定在第一维求逻辑或，输出结果为\[1, 0, 1\]；指定在第二维求逻辑或，输出结果为\[1, 1\]。 图1 ReduceAny按第一个维度计算示例 ! 图2 ReduceAny按最后一个维度计算示例 !
