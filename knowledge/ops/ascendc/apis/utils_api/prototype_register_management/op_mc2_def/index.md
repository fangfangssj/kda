# OpMC2Def

- [HcclGroup（OpMC2Def）](hccl_group.md)：配置通信域名称，每个名称对应一个通信域。配置后在Kernel侧调用GetHcclContext接口可获取通信域对应的context（消息区）地址。
- [HcclServerType](hccl_server_type.md)：配置HCCL的服务端类型。
- [OpMC2Def构造函数](op_mc2_def_constructor.md)：创建OpMC2Def对象时，初始化数据成员。
- [OpMC2Def简介](op_mc2_def_intro.md)：该类用于在host侧配置通算融合算子的通信域名称。配置后在kernel侧可以获取通信域对应的context（消息区）地址。
- [operator=（OpMC2Def）](operator.md)：赋值运算符重载，实现OpMC2Def类对象的深拷贝。
