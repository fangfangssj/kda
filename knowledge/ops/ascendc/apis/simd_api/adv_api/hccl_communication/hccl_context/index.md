# HCCL Context

- [GetHcclContext](get_hccl_context.md)：获取指定Index通信域的context（消息区）地址。
- [HCCL Context简介](hccl_context_intro.md)：本章节的接口用于在kernel侧设置/获取通算融合算子每个通信域对应的context（消息区）地址。需要同步在host侧调用HcclGroup接口配置通信域名称后，才可以调用GetHcclContext获取对应的context地址。
- [SetHcclContext](set_hccl_context.md)：设置通算融合算子每个通信域对应的context（消息区）地址。
