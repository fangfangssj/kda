# 自定义TBufPool

- [ExternImplBufpool](extern_impl_bufpool.md)：辅助开发者自定义TBufPool以管理Unified Buffer和L1 Buffer内存的宏，支持不连续内存块和跨TQue共享等自定义分配需求。
- [GetBufHandle](get_buf_handle.md)：根据offset取出自定义TBufPool中管理的内存块，返回TBufHandle类型的内存块句柄。
- [GetCurAddr](get_cur_addr.md)：获取自定义TBufPool当前已分配的内存地址，用户可从该地址向后继续分配内存块。
- [GetCurBufSize](get_cur_buf_size.md)：获取自定义TBufPool当前已分配的内存块个数。
- [Init](init.md)：完成自定义TBufPool资源与eventId等变量的初始化操作，可在自定义构造函数中调用。
- [Reset](reset.md)：完成自定义TBufPool资源的释放与eventId等变量的复位，调用后对应资源池及已分配Buffer不可继续使用。
- [SetCurAddr](set_cur_addr.md)：设置自定义TBufPool已分配完的地址，保证后续内存块从指定地址开始分配。
- [SetCurBufSize](set_cur_buf_size.md)：设置自定义TBufPool当前已分配的内存块个数。
