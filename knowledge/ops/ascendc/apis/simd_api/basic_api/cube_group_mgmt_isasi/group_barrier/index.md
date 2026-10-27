# GroupBarrier

- [Arrive（Cube核间管理）](arrive.md)：通知其他等待的AIV，本AIV已经完成其依赖的任务。
- [GetWorkspaceLen](get_workspace_len.md)：返回当前GroupBarrier所占用的Global Memory消息空间大小。
- [GroupBarrier构造函数](group_barrier_constructor.md)：创建GroupBarrier对象，通过设置构造函数参数，确定Arrive组的Block个数和Wait组的Block个数。
- [GroupBarrier使用说明](group_barrier_usage.md)：当同一个CubeResGroupHandle中的两个AIV任务之间存在依赖关系时，可以使用GroupBarrier控制同步。假设一组AIV A做完任务x以后，另外一组AIV B才可以开始后续业务，称AIV A组为Arrive组，AIV B组为Wait组。
- [Cube核间管理Wait](wait.md)：阻塞该AIV需要等待Arrive组全部完成任务，再开始执行任务。
