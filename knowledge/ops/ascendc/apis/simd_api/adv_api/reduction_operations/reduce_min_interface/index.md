# ReduceMin接口

- [GetReduceMinMaxMinTmpSize](get_reduce_min_max_min_tmp_size.md)：在host侧获取ReduceMin接口计算所需的最大和最小临时空间大小，用于选择合适的空间作为Tiling参数传递到kernel侧。
- [ReduceMin](reduce_min.md)：对一个多维向量在指定的维度求最小值。 定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对shape为\(2, 3\)的二维矩阵进行运算，指定在第一维求最小值，输出结果为\[1, 2, 3\]；指定在第二维求最小值，输出结果为\[1, 4\]。 图1 ReduceMin按第一个维度计算示例 ! 图2 ReduceMin按最后一个维度计算示例 !
