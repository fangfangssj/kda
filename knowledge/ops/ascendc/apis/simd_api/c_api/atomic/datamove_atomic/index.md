# 搬运原子操作

- [asc_disable_dma_atomic](asc_disable_dma_atomic.md)：头文件路径为："c_api/atomic/atomic.h"。
- [asc_set_atomic_add](asc_set_atomic_add.md)：原子累加过程：将待搬运到GM的数据和GM上已有数据进行求和，然后将求和结果写入GM。本接口对后续目的地址为GM的数据搬运指令开启原子累加，不同产品支持的数据搬运通路请参考约束说明。
- [asc_set_atomic_max](asc_set_atomic_max.md)：设置对后续的从Unified Buffer（UB）/L0C Buffer/L1 Buffer到GM的数据传输开启原子比较取大操作。数据类型支持int8_t、int16_t、half、bfloat16_t、int32_t、float。 开启原子比较取大后，后续执行搬运操作从UB/L0C Buffer/L1 Buffer到GM时，将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。
- [asc_set_atomic_min](asc_set_atomic_min.md)：设置对后续的从Unified Buffer（UB）/L0C Buffer/L1 Buffer到GM的数据传输开启原子比较取小操作。数据类型支持int8_t、int16_t、half、bfloat16_t、int32_t、float。 开启原子比较取小后，后续执行搬运操作从UB/L0C Buffer/L1 Buffer到GM时，将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。
