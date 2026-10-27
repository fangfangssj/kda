# 原子操作

- [原子操作概述](atomic_operations_overview.md)：数据搬运随路原子操作接口用于对后续目的地址为GM的数据搬运指令开启原子操作，涉及的接口请参见表1。如下图1的左侧子图所示，未开启原子操作时，写入GM的数据搬运完成后，GM中原始数据将被新搬运数据完全覆盖。如图1右侧子图所示，当数据搬运随路原子操作接口被调用后，系统将为后续写入GM的数据搬运开启原子操作。此时，数据搬运完成后，GM中的最终数据由原始GM数据与新搬运数据共同决定。
- [搬运原子操作](datamove_atomic/index.md)：共 4 张知识卡（如 asc_disable_dma_atomic、asc_set_atomic_add、asc_set_atomic_max 等）。
- [关键特性说明](key_features.md)：确定性计算是指在相同输入条件下，无论执行次数或执行环境如何变化，总能产生完全一致输出结果的计算过程。确定性计算为系统稳定性和实验可验证性提供保障。
- [标量原子操作](scalar_atomic/index.md)：共 14 张知识卡（如 asc_atomic_add、asc_atomic_and、asc_atomic_cas 等）。
