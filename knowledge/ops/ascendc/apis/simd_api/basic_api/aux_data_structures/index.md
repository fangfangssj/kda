# 辅助数据结构

- [BinaryRepeatParams](binary_repeat_params.md)：BinaryRepeatParams为用于控制操作数地址步长的数据结构。结构体内包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。
- [Coordinate](coordinate/index.md)：共 3 张知识卡（如 Coordinate简介、Crd2Idx、MakeCoord）。
- [Layout](layout/index.md)：共 10 张知识卡（如 GetShape、GetStride、is_layout 等）。
- [ListTensorDesc](list_tensor_desc.md)：ListTensorDesc用来解析符合以下内存排布格式的数据，并在kernel侧根据索引获取储存对应数据的地址及shape信息。 !
- [ShapeInfo（辅助数据结构）](shape_info.md)：ShapeInfo用来存放LocalTensor或GlobalTensor的shape信息。
- [TPosition](t_position.md)：Ascend C管理不同层级物理内存的抽象逻辑位置，代替片上物理存储概念，包括矢量编程和矩阵编程所需的各类存储位置。
- [TensorDesc](tensor_desc/index.md)：共 8 张知识卡（如 构造和析构函数、GetDataObj、GetDataPtr 等）。
- [TensorTrait](tensor_trait/index.md)：共 7 张知识卡（如 GetLayout、is_tensorTrait、MakeTensorTrait 等）。
- [UnaryRepeatParams](unary_repeat_params.md)：UnaryRepeatParams为用于控制操作数地址步长的数据结构。结构体内包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。
