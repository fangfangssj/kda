# Gelu接口

- [FasterGelu](faster_gelu.md)：GELU激活函数的FastGelu化简版本，通过公式化简大幅提升计算性能，支持高精度和高性能模式。
- [FasterGeluV2](faster_gelu_v2.md)：GELU激活函数FastGeluV2版本实现，通过公式化简降低算力需求，支持高精度和高性能模式。
- [Gelu](gelu.md)：GELU激活函数实现，灵感来源于Relu和Dropout，在激活中引入随机正则思想，支持高精度和高性能模式。
- [GetGeluMaxMinTmpSize](get_gelu_max_min_tmp_size.md)：在host侧获取Gelu系列接口所需最大最小临时空间大小，供开发者选择合适的Tiling参数传递到kernel侧。
