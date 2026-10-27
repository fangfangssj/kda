# onnx framework

- [适配插件开发](adaptation_plugin_development.md)：>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，暂不支持ONNX框架算子调用。
- [高级编程调用样例](call_examples.md)：完成了ONNX框架的适配插件开发后，即可实现从ONNX框架调用Ascend C自定义算子。下面以一个仅包含LeakyRelu算子的ONNX框架网络为例（该网络中的LeakyRelu算子通过适配插件映射为自定义的LeakyRelu算子），呈现一个使用推理工具进行推理的过程，目的在于让您快速体验推理场景下网络中自定义算子调用的过程。
