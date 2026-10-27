# 排序操作

- [Concat（排序操作）](concat.md)：对数据进行预处理，将要排序的源操作数src一一对应的合入目标数据concat中，数据预处理完成后，可以进行Sort。
- [Extract（排序操作）](extract.md)：处理Sort的结果数据，输出排序后的value和index。
- [GetConcatTmpSize](get_concat_tmp_size.md)：获取Concat接口所需的临时空间大小，单位为字节。
- [GetSortLen](get_sort_len.md)：根据元素数量，获取Sort数据的大小（单位为字节）。
- [GetSortMaxMinTmpSize](get_sort_max_min_tmp_size.md)：在host侧获取带SortConfig模板参数的Sort接口计算所需的最大和最小临时空间大小。
- [GetSortOffset](get_sort_offset.md)：根据元素位置，获取Sort数据中的对应偏移量（单位为字节）。
- [GetSortTmpSize](get_sort_tmp_size.md)：获取Sort接口所需的临时空间大小。
- [MrgSort](mrg_sort.md)：将已排好序的最多4条队列合并排列成1条队列，结果按照score域由大到小排序。
- [Sort](sort.md)：排序函数，按照数值大小进行降序排序，排序结果以score和index结构存储。
- [TopK](top_k.md)：获取最后一个维度的前k个最大值或最小值及其对应的索引，最多支持二维输入。整数倍，所以当n不是32的
- [TopKTiling](top_k_tiling.md)：获取TopK kernel计算时所需的Tiling参数，包括临时空间大小和tiling信息。
