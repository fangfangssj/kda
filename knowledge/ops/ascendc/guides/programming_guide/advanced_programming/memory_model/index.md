# 内存模型

- [缓存一致性](cache_coherence.md)：本文介绍NPU编程中常见的缓存一致性问题，说明DCache、ICache、SIMT DCache与GM的数据一致性。
- [内存一致性](memory_consistency.md)：本文介绍NPU编程中的内存一致性问题，说明不同执行单元、不同AI Core、不同Stream以及Host侧和Device侧之间如何建立可预期的读写顺序。本文重点讨论“操作顺序和可见性”，缓存副本写回、失效等问题请参考缓存一致性。
