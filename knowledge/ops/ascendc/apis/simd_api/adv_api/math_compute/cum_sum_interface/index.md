# CumSum接口

- [CumSum](cum_sum.md)：对输入二维张量按行或列进行累加和操作，输出每个位置及之前所有行或列元素的累加结果，支持逐行和Sklansky二分两种算法。
- [GetCumSumMaxMinTmpSize](get_cum_sum_max_min_tmp_size.md)：在host侧获取CumSum接口所需的最大和最小临时空间大小，供开发者选择合适的Tiling参数传入kernel侧。
