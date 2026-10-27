# Layout

- [GetShape](get_shape.md)：返回描述张量形状的Shape对象。
- [GetStride](get_stride.md)：返回描述内存访问步长的Stride对象，与Shape的维度信息一一对应。
- [is_layout](is_layout.md)：判断输入的数据结构是否为Layout数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是Layout类型；反之则为非Layout类型。
- [layout](layout1.md)：获取Layout实例化对象。
- [Layout构造函数](layout_constructor.md)：根据输入的Shape和Stride对象，实例化Layout对象。
- [Layout简介](layout_intro.md)：Layout数据结构是描述多维张量内存布局的基础模板类，通过编译时的形状（Shape）和步长（Stride）信息，实现逻辑坐标空间到一维内存地址空间的映射，为复杂张量操作和硬件优化提供基础支持。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。
- [MakeLayout](make_layout.md)：将传入的Shape和Stride数据打包成Layout数据结构。
- [MakeShape](make_shape.md)：将传入的数据打包成Shape数据结构。
- [MakeStride](make_stride.md)：将传入的数据打包成Stride数据结构。
- [运算符重载](operator_overloading.md)：重载运算符\(\)，内部实现为调用Crd2Idx接口，将多维坐标（Coordinate）通过布局（Layout）转换为内存位置索引（Index）。
