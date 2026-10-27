# tensor

- [make_mem_ptr](make_mem_ptr.md)：头文件路径为：tensor_api/tensor.h。
- [make_tensor](make_tensor.md)：头文件路径为：tensor_api/tensor.h。
- [pointer](pointer.md)：Tensor API通过make_mem_ptr构造带物理存储位置信息的指针适配器。指针适配器保留原始指针或迭代器的访问能力，并通过ptr_pattern标记Global Memory、L1 Buffer、L0A Buffer等物理位置。使用时只需调用对外接口make_mem_ptr，不需要直接构造内部指针适配器类型。
- [slice](slice.md)：头文件路径为：tensor_api/tensor.h。
- [tensor](tensor.md)：头文件路径为：tensor_api/tensor.h。
- [Tensor](tensor_structure.md)：Tensor由Pointer和Layout组成。Pointer描述数据地址及存储位置，Layout描述数据的Shape和Stride。开发者可以通过Tensor统一访问Global Memory或片上Buffer中的数据，并通过切片创建共享底层存储的子Tensor。
- [view_engine](view_engine.md)：view_engine是Tensor API中的视图引擎，负责持有一个指针对象Iterator，通过begin()返回这段视图的起始位置。在local_tensor中，view_engine和layout一起组成一个Tensor的视图。
