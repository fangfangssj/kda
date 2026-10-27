# OpTilingRegistry

- [构造和析构函数](constructor_and_destructor.md)：创建与释放OpTilingRegistry对象。
- [GetTilingFunc](get_tiling_func.md)：根据算子类型OpType获取对应的Tiling函数。
- [Utils API简介](intro.md)：OpTilingRegistry类属于context_ascendc命名空间，主要用于加载Tiling实现的动态库，并获取算子的Tiling函数指针以进行调试和验证。该类通常与ContextBuilder类配合使用，后者用于构造Tiling函数所需的输入参数。具体用法可参考如何进行Tiling调测。
- [LoadTilingLibrary](load_tiling_library.md)：根据输入路径加载Tiling动态库，支持算子包编译或动态库编译方式获取的so文件，返回加载是否成功。
