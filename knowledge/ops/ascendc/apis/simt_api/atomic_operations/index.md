# 原子操作

- [asc_atomic_add](asc_atomic_add.md)：对Unified Buffer或Global Memory中的数据与指定数据执行原子加操作，即将指定数据累加到这些内存区域的数据中。
- [asc_atomic_and](asc_atomic_and.md)：对Unified Buffer或Global Memory上address的数值与指定数值val进行原子与（&）操作，即将address数值与（&）val的结果赋值到Unified Buffer或Global Memory上。
- [asc_atomic_cas](asc_atomic_cas.md)：对Unified Buffer或Global Memory上address的数值进行原子比较赋值操作，如果address上的数值等于指定数值compare，则对address赋值为指定数值val，否则address的数值不变。
- [asc_atomic_dec](asc_atomic_dec.md)：对Unified Buffer或Global Memory上address的数值进行原子减1操作，如果address上的数值等于0或大于指定数值val，则对address赋值为val，否则将address上数值减1。
- [asc_atomic_exch](asc_atomic_exch.md)：对Unified Buffer或Global Memory地址做原子赋值操作，即将指定数据赋值到Unified Buffer或Global Memory地址中。
- [asc_atomic_inc](asc_atomic_inc.md)：对Unified Buffer或Global Memory上address的数值进行原子加1操作，如果address上的数值大于等于指定数值val，则对address赋值为0，否则将address上数值加1。
- [asc_atomic_max](asc_atomic_max.md)：对Unified Buffer或Global Memory数据做原子求最大值操作，即将Unified Buffer或Global Memory的数据与指定数据中的最大值赋值到Unified Buffer或Global Memory地址中。
- [asc_atomic_min](asc_atomic_min.md)：对Unified Buffer或Global Memory数据做原子求最小值操作，即将Unified Buffer或Global Memory的数据与指定数据中的最小值赋值到Unified Buffer或Global Memory地址中。
- [asc_atomic_or](asc_atomic_or.md)：对Unified Buffer或Global Memory上address的数值与指定数值val进行原子或（ ）操作，即将address数值或（ ）val的结果赋值到Unified Buffer或Global Memory上。
- [asc_atomic_sub](asc_atomic_sub.md)：对Unified Buffer或Global Memory上的数据与指定数据执行原子减操作，即在这些内存区域的数据中减去指定数据。
- [asc_atomic_xor](asc_atomic_xor.md)：对Unified Buffer或Global Memory上address的数值与指定数值val进行原子异或（^）操作，即将address数值异或（^）val的结果赋值到Unified Buffer或Global Memory上。
- [原子操作简介](atomic_operations_intro.md)：SIMT原子操作用于保证对同一地址的一次读-改-写过程具有原子性，可避免多个线程并发更新同一地址时出现更新丢失。原子操作常用于计数、状态抢占、直方图统计、归约计算和共享状态位更新等场景。
