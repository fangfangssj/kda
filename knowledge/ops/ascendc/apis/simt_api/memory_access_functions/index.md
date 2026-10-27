# 访存函数

- [asc_dcci_entire](asc_dcci_entire.md)：该接口用于刷新核内的整个Data Cache，保证数据读取时Cache的一致性。 在SIMT编程中，写入数据时会立即写入Global Memory，使其他核可见，因此不存在核间一致性问题。当从Global Memory读取数据时，该数据可能已被其他核修改，此时应使用dcci接口直接访问Global Memory，以获取最新数据。 请注意，该接口耗时较大，性能敏感的场景慎用。
- [asc_dcci_single](asc_dcci_single.md)：该接口用于刷新指定地址所在的Cache Line，保证数据读取时Cache的一致性。 在SIMT编程中，写入数据时会立即写入Global Memory，使其他核可见，因此不存在核间一致性问题。当从Global Memory读取数据时，该数据可能已被其他核修改，此时应使用dcci接口直接访问Global Memory，以获取最新数据。
- [AscLdca](asc_ldca.md)：从Data Cache加载数据，未命中则尝试L2 Cache，均未命中则从Global Memory读取并缓存。默认访存底层实现与本接口一致。
- [asc_ldcg（内存访问函数）](asc_ldcg.md)：从L2 Cache加载缓存的数据，如果缓存命中，则直接返回数据。L2 Cache与Global Memory之间的数据一致性由硬件保证。若未命中，则从Global Memory地址预加载数据缓存至L2 Cache，并返回数据。
- [asc_nop（内存访问函数）](asc_nop.md)：本接口的功能为生成一条空操作指令，占用当前线程15个时钟周期，不执行任何实际计算与访存操作。
- [asc_stcg（内存访问函数）](asc_stcg.md)：将指定数据存储到Global Memory的地址address中，并缓存到L2 Cache，但不缓存至Data Cache。
- [asc_stwt（内存访问函数）](asc_stwt.md)：将指定数据存储到Global Memory的地址address中，并缓存至Data Cache和L2 Cache。
- [访存函数简介（内存访问函数）](memory_access_functions_intro.md)：SIMT编程通常需要大量Thread并行访问Global Memory。访存函数用于显式控制Global Memory访问时的缓存路径和缓存一致性维护方式，帮助开发者根据数据复用特征、写回方式和跨核可见性要求选择合适的加载、存储或Cache刷新接口。
