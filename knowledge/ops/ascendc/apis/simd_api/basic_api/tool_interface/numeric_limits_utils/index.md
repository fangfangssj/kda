# numeric limits utils

- [DeNormMin](de_norm_min.md)：返回指定数据类型的最小正的非正规值，支持标量和矢量两种接口，矢量接口为dstLocal前count个元素赋该值。
- [Infinity](infinity.md)：返回指定数据类型的正无穷大值，支持标量和矢量两种接口，矢量接口为dstLocal前count个元素赋正无穷大值。
- [工具接口简介](intro.md)：NumericLimits工具类，用于查询指定数据类型的最大值/最小值等属性。
- [Lowest](lowest.md)：返回指定数据类型的最低有限值，即无其他有限值更小的有限值，支持标量和矢量两种接口。
- [Max](max.md)：返回指定数据类型的最大有限值，支持标量和矢量两种接口，矢量接口为dst前count个元素赋最大有限值。
- [Min](min.md)：返回指定数据类型的最小有限值，浮点类型返回最小规约正数，支持标量和矢量两种接口。
- [NegativeInfinity](negative_infinity.md)：返回指定数据类型的负无穷大值，支持标量和矢量两种接口，矢量接口为dstLocal前count个元素赋负无穷大值。
- [QuietNaN](quiet_na_n.md)：返回指定数据类型的安静NaN值，支持标量和矢量两种接口，矢量接口为dstLocal前count个元素赋安静NaN值。
- [SignalingNaN](signaling_na_n.md)：返回指定数据类型的发信NaN值（浮点尾数最高位为0），提供标量和矢量两种接口。
