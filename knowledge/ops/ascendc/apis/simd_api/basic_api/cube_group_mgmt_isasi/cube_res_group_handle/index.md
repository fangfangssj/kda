# CubeResGroupHandle

- [AllocMessage](alloc_message.md)：AIV从消息队列里申请消息空间，用于存放消息结构体，返回当前申请的消息空间的地址。消息队列的深度固定为4，申请消息空间的顺序为自上而下，然后循环。当消息队列指针指向的消息空间为FREE状态时，AllocMessage返回空间的地址，否则循环等待，直到当前空间的状态为FREE。 图1 AllocMessage示意图 !
- [AssignQueue](assign_queue.md)：AssignQueue 接口，用于 AIV 绑定 CubeResGroupHandle 中某一个消息队列的序号。
- [CreateCubeResGroup](create_cube_res_group.md)：快速创建CubeResGroupHandle对象，内部完成消息队列空间和同步事件分配。推荐使用该接口，避免使用CubeResGroupHandle的构造函数创建对象，出现不同对象的消息队列空间冲突、同步事件错误等情况。
- [CubeResGroupHandle构造函数](cube_res_group_handle_constructor.md)：构造CubeResGroupHandle对象，完成组内的AIC和消息队列分配。构造CubeResGroupHandle对象时需要传入模板参数CubeMsgType，CubeMsgType是由用户定义的消息结构体，请参考表1。使用此接口需要用户自主管理地址、同步事件等，因此更推荐使用CreateCubeResGroup接口快速创建CubeResGroupHandle对象。
- [CubeResGroupHandle](cube_res_group_handle_usage.md)：在分离模式下对AI Core计算资源分组，通过Client和Server架构实现AIV与AIC间的消息通信和任务分发。
- [FreeMessage](free_message.md)：在自定义的回调函数逻辑中，完成消息处理后，调用该接口，刷新消息状态为FREE；或者待消息状态为指定状态waitState时，刷新消息状态为FREE。消息状态的介绍可以参考表2中的参数msgState。
- [PostFakeMsg](post_fake_msg.md)：在消息合并场景中由AIV发送假消息，通知AIC跳过重复消息的读取和处理，实现消息合并机制。
- [PostMessage](post_message.md)：通过AllocMessage接口获取到消息空间地址msg后，构造消息结构体CubeMsgType，发送该消息。
- [SetQuit](set_quit.md)：通过AllocMessage接口获取到消息空间地址后，发送退出消息，告知该消息队列对应的AIC无需处理该队列的消息。如下图，Queue5对应的AIV发了退出消息后，Block1将不再处理Queue5的任何消息。 图1 消息队列退出示意图 !
- [SetSkipMsg](set_skip_msg.md)：AIC跳过指定个数假消息的处理，仅在回调函数中调用。下图中Block0通过调用SetSkipMsg跳过三个假消息。 图1 SetSkipMsg示意图 !
- [Cube核间管理Wait](wait.md)：在调用PostMessage或PostFakeMessage后，查询该消息是否已被AIC处理完。
