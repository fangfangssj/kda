# TensorTrait

- [GetLayout](get_layout.md)：获取Layout实例化对象。
- [is_tensorTrait](is_tensor_trait.md)：判断输入的数据结构是否为TensorTrait数据结构，可通过检查其成员常量value的值来判断。当value为true时，表示输入的数据结构是TensorTrait类型；反之则为非TensorTrait类型。
- [MakeTensorTrait](make_tensor_trait.md)：生成TensorTrait实例化对象。
- [辅助数据结构更多样例](more_examples.md)：矢量计算基础 API 使用 TensorTrait 的样例集合，演示 Adds 等接口结合 TensorTrait 的调用方式。
- [SetLayout](set_layout.md)：设置Layout对象至TensorTrait结构中。
- [TensorTrait构造函数](tensor_trait_constructor.md)：根据输入的Layout对象，实例化TensorTrait对象。
- [TensorTrait简介](tensor_trait_intro.md)：TensorTrait数据结构是描述Tensor相关信息的基础模板类，包含Tensor的数据类型、逻辑位置和Layout内存布局。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。
