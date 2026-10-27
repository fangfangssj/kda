# OpDef

- [AICore（OpDef）](ai_core.md)：设置AICore的配置信息：包括Tiling处理回调函数、能力检查回调函数、基础的配置信息等。
- [AICPU（OpDef）](aicpu.md)：设置AI CPU的配置信息：包括算子输入输出的名称、支持的数据类型、数据格式等信息。
- [Attr（OpDef）](attr.md)：注册算子属性参数。 当需要设置的参数不参与kernel侧计算时，可以将该参数注册为算子属性参数。
- [Comment](comment.md)：设置算子分组信息及原型注释（简述、约束等），用于自动生成算子原型头文件时同步生成注释和文档说明。
- [EnableFallBack](enable_fall_back.md)：通过本接口启用fallback配置，启用后将自动生成一个fallback函数并注册给GE。fallback函数的核心逻辑是将GE的输入、输出及属性转换为aclnn单算子API所需的参数格式，随后调用aclnn接口。动态图场景下，GE可直接调用fallback函数（函数中调用了aclnn接口），从而简化调度流程。关于fallback下发算子的详细介绍请参考《图开发》。
- [FormatMatchMode](format_match_mode.md)：设置输入输出tensor的format匹配模式。
- [HostCPU（OpDef）](host_cpu.md)：设置Host CPU的配置信息：包括算子输入输出的name、支持的dtype、format等信息。
- [Input（OpDef）](input.md)：注册算子输入，调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输入信息。
- [MC2（OpDef）](mc2.md)：注册该算子是通算融合算子，调用该接口后会返回一个OpMC2Def结构，后续可通过该结构配置通信域名称。
- [Output（OpDef）](output.md)：注册算子输出，调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输出信息。
- [SetInferDataType](set_infer_data_type.md)：使用图模式时，需要调用该接口注册DataType推导函数。
- [SetInferShape](set_infer_shape.md)：使用图模式时，需要调用该接口注册Shape推导函数。
- [SetInferShapeRange](set_infer_shape_range.md)：使用图模式时，需要调用该接口注册ShapeRange推导函数。
