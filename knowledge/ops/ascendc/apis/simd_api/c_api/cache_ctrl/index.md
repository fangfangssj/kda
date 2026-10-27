# 缓存控制

- [asc_datacache_preload](asc_datacache_preload.md)：从GM地址预加载数据到Data Cache，每次调用预加载一个Cache Line大小的数据。
- [AscDcci](asc_dcci.md)：刷新Data Cache保证与GM一致性，支持单Cache Line刷新和整片刷新，用于多核数据一致性和Scalar写GM场景。
- [asc_dci（缓存控制）](asc_dci.md)：数据缓存失效，该指令用于使整个数据缓存无效化，且不会对处于“脏”状态的缓存行执行写回操作。换言之，所有缓存行将被直接标记为无效，其修改过但尚未同步至主存的数据将被丢弃。
- [AscGetIcachePreloadStatus](asc_get_icache_preload_status.md)：调试接口，在asc_icache_preload后调用获取ICache预加载状态，返回0表示完成，1表示未完成。
- [asc_get_scalar_load_cache_mode](asc_get_scalar_load_cache_mode.md)：读取当前标量读（load）路径的L2 Cache管理策略，并返回asc_load_l2_cache_mode枚举值。
- [asc_get_scalar_store_cache_mode](asc_get_scalar_store_cache_mode.md)：读取当前标量写（store）路径的L2 Cache管理策略，并返回asc_store_l2_cache_mode枚举值。
- [AscIcachePreload](asc_icache_preload.md)：从指令所在GM地址预加载指令到ICache中，支持配置预取长度。
- [asc_set_scalar_cache_mode](asc_set_scalar_cache_mode.md)：配置标量单元访问Global Memory时的L2 Cache管理策略。接口通过修改CTRL寄存器对应比特域生效：

- [系统缓存概述](system_cache_overview.md)：Cache（缓存）的主要作用是在搬运单元或Scalar单元与外部存储之间提供一层高速缓冲，以降低数据访问延迟并提高带宽利用率。通常情况下，被频繁访问的数据会写入Cache，搬运单元或Scalar单元在执行过程中优先从Cache中读取数据；当Cache未命中时，再从外部存储加载数据并更新到Cache中。
