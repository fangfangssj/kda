# TPipe

- [AllocEventID](alloc_event_id.md)：用于申请HardEvent（硬件类型同步事件）的TEventID，必须与ReleaseEventID搭配使用，调用后占用TEventID直至释放。
- [Destroy](destroy.md)：释放TPipe资源，用于重复申请释放TPipe，创建TPipe对象后可调用Destroy手动释放资源。
- [DestroyWithoutPipeAll](destroy_without_pipe_all.md)：释放TPipe资源，用于重复申请释放TPipe。内部未设置PIPE_ALL同步，需开发者自行保证同步流水正确。
- [FetchEventID](fetch_event_id.md)：根据HardEvent获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。
- [GetBaseAddr](get_base_addr.md)：根据传入的logicPos（逻辑抽象位置）获取该位置的基础地址，只在CPU调试场景下生效。通常用于计算Tensor在logicPos的偏移地址。
- [Init](init.md)：用于内存和同步流水事件EventID的初始化。
- [InitBufPool](init_buf_pool.md)：初始化TBufPool内存资源池，适用于内存资源有限时手动指定UB/L1内存资源复用的场景。
- [InitBuffer](init_buffer.md)：用于为TQue等队列和TBuf分配内存。
- [InitSpmBuffer](init_spm_buffer.md)：初始化SPM Buffer。SPM Buffer是GM上的空间，当UB空间不足时可存放溢出数据。
- [ReadSpmBuffer](read_spm_buffer.md)：从SPM Buffer读回数据到local内存中。
- [ReleaseEventID](release_event_id.md)：用于释放HardEvent硬件同步事件的TEventID，通常与AllocEventID搭配使用。
- [Reset](reset.md)：完成资源的释放与eventId等变量的初始化操作，恢复到TPipe的初始化状态。
- [TPipe](t_pipe_constructor.md)：构造用来管理内存和同步的TPipe对象。
- [TPipe简介](t_pipe_intro.md)：TPipe用于统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象。其主要功能包括：
- [WriteSpmBuffer](write_spm_buffer.md)：将UB空间不足时需要溢出暂存的数据拷贝到SPM Buffer中。
