# 缓存控制

- [DataCacheCleanAndInvalid](data_cache_clean_and_invalid.md)：解决Data Cache与GM一致性问题，将脏数据写回GM并将指定Cache Line标记为无效，确保下次访问从GM重新加载。
- [DataCachePreload](data_cache_preload.md)：从GM地址预加载数据到DCache，每次调用预加载一个Cache Line大小的数据，提升后续访问效率。
- [GetICachePreloadStatus](get_i_cache_preload_status_isasi.md)：调试接口，在ICachePreLoad后调用获取ICache预加载状态，返回0表示完成，返回1表示未完成。
- [ICachePreLoad](i_cache_pre_load_isasi.md)：从指令所在GM地址预加载指令到ICache中，支持配置预取长度，提升指令读取效率。
- [系统缓存概述（缓存控制）](system_cache_overview.md)：Cache（缓存）的主要作用是在搬运单元或Scalar单元与外部存储之间提供一层高速缓冲，以降低数据访问延迟并提高带宽利用率。通常情况下，被频繁访问的数据会写入Cache，搬运单元或Scalar单元在执行过程中优先从Cache中读取数据；当Cache未命中时，再从外部存储加载数据并更新到Cache中。

- [PRELOAD](preload.md)：头文件路径为："utils/base/sys_macros.h"。
