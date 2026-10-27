# 原子操作

- [AtomicAdd](atomic_add.md)：对GM地址上的值执行原子加操作，将旧值与输入标量求和后写回，返回修改前的旧值。
- [AtomicCas](atomic_cas.md)：对GM地址执行原子比较交换：旧值与value1相等时写入value2，否则不变，返回旧值。
- [AtomicExch](atomic_exch.md)：对GM地址执行原子交换，将输入标量值写入并返回修改前的旧值。
- [AtomicMax](atomic_max.md)：对GM地址上的值执行原子取大操作，将旧值与输入标量比较后写入较大值，返回旧值。
- [AtomicMin](atomic_min.md)：对GM地址上的值执行原子取小操作，将旧值与输入标量比较后写入较小值，返回旧值。
- [原子操作概述](atomic_operations_overview.md)：数据搬运随路原子操作接口用于对后续目的地址为GM的数据搬运开启原子操作，涉及的接口请参见表1。如下图1的左侧子图所示，未开启原子操作时，写入GM的数据搬运完成后，GM中原始数据将被新搬运数据完全覆盖。如图1右侧子图所示，当数据搬运随路原子操作接口被调用后，系统将为后续写入GM的数据搬运开启原子操作。此时，数据搬运完成后，GM中的最终数据由原始GM数据与新搬运数据共同决定。
- [DisableDmaAtomic](disable_dma_atomic.md)：关闭数据搬运随路原子操作，后续搬运时GM原始数据将被新数据完全覆盖。
- [GetStoreAtomicConfig](get_store_atomic_config_isasi.md)：获取当前原子操作的开启状态和数据类型配置。
- [关键特性说明（原子操作）](key_features.md)：确定性计算是指在相同输入条件下，无论执行次数或执行环境如何变化，总能产生完全一致输出结果的计算过程。确定性计算为系统稳定性和实验可验证性提供保障。
- [SetAtomicAdd](set_atomic_add.md)：对后续搬运到GM的数据开启原子累加，将拷贝内容与GM已有值求和后写入。
- [SetAtomicMax](set_atomic_max_isasi.md)：对后续搬运到GM的数据开启原子取大，将拷贝内容与GM已有值比较后写入最大值。
- [SetAtomicMin](set_atomic_min_isasi.md)：对后续搬运到GM的数据开启原子取小，将拷贝内容与GM已有值比较后写入最小值。
- [SetAtomicType](set_atomic_type.md)：通过模板参数设定后续原子操作的数据类型。
- [SetStoreAtomicConfig](set_store_atomic_config_isasi.md)：设置原子操作的开启位和操作类型，支持多种数据类型配置。
