# 标量原子操作

- [asc_atomic_add](asc_atomic_add.md)：对Global Memory中address指向的单个元素执行原子加操作。读取该元素的旧值old_value，将old_value + val写回该地址，并返回old_value。整个读取、计算和写回过程为原子操作。
- [asc_atomic_and](asc_atomic_and.md)：对Global Memory中address指向的单个元素执行原子按位与操作。读取该元素的旧值old_value，将old_value & val写回该地址，并返回old_value。整个读取、计算和写回过程为原子操作。
- [asc_atomic_cas](asc_atomic_cas.md)：对Global Memory中address指向的单个元素执行原子比较交换操作。读取该元素的旧值old_value，按照以下公式计算并写回新值new_value：
- [asc_atomic_dec](asc_atomic_dec.md)：对Global Memory中address指向的一个uint32_t或uint64_t元素执行原子递减操作。读取该元素的旧值old_value，按照以下公式计算并写回新值new_value：
- [asc_atomic_exch](asc_atomic_exch.md)：对Global Memory中address指向的单个元素执行原子交换操作。读取该元素的旧值old_value，将val写回该地址以替换旧值，并返回old_value。整个读取和写回过程为原子操作。
- [asc_atomic_inc](asc_atomic_inc.md)：对Global Memory中address指向的一个uint32_t或uint64_t元素执行原子递增操作。读取该元素的旧值old_value，按照以下公式计算并写回新值new_value：
- [asc_atomic_max](asc_atomic_max.md)：对Global Memory中address指向的单个元素执行原子取大操作：读取该地址中的旧值old_value，比较old_value与输入标量值val，将较大值new_value写回该地址，并返回old_value。整个读取、比较和写回过程为原子操作。
- [asc_atomic_min](asc_atomic_min.md)：对Global Memory中address指向的单个元素执行原子取小操作：读取该地址中的旧值old_value，比较old_value与输入标量值val，将较小值new_value写回该地址，并返回old_value。整个读取、比较和写回过程为原子操作。
- [asc_atomic_or](asc_atomic_or.md)：对Global Memory中address指向的单个元素执行原子按位或操作：读取该地址中的旧值old_value，将old_value与输入标量值val进行按位或运算，将结果new_value写回该地址，并返回old_value。整个读取、计算和写回过程为原子操作。
- [asc_atomic_sub](asc_atomic_sub.md)：对Global Memory中address指向的单个元素执行原子减操作：读取该地址中的旧值old_value，计算old_value与输入标量值val的差，将结果new_value写回该地址，并返回old_value。整个读取、计算和写回过程为原子操作。
- [asc_atomic_xor](asc_atomic_xor.md)：对Global Memory中address指向的单个元素执行原子按位异或操作：读取该地址中的旧值old_value，将old_value与输入标量值val进行按位异或运算，将结果new_value写回该地址，并返回old_value。整个读取、计算和写回过程为原子操作。
- [asc_get_store_atomic_config](asc_get_store_atomic_config.md)：获取原子操作启用位与原子操作类型的值，可用于校验asc_set_store_atomic_config_v1和asc_set_store_atomic_config_v2（废弃）设置的原子操作启用位和原子操作类型的值是否符合预期。
- [asc_set_store_atomic_config_v1](asc_set_store_atomic_config_v1.md)：设置原子操作启用位与原子操作类型的值。

- [标量原子操作概述](scalar_atomic_overview.md)：标量原子操作接口用于在Global Memory（GM）的指定地址上完成单个元素的原子读-改-写操作：读取该地址的旧值old_value，按接口语义计算新值并写回该地址，同时返回old_value。整个过程由硬件保证为原子操作，其他原子更新无法插入其中，因此多个AI Core或同一AI Core内的多条指令并发访问同一GM地址时不会相互覆盖，最终结果符合预期。不使用原子操作与使用原子操作的效果对比如标量原子累加效果示意图所示。
