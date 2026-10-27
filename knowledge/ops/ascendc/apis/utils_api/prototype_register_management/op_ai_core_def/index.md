# OpAICoreDef

- [AddConfig（OpAICoreDef）](add_config.md)：注册算子支持的AI处理器型号以及OpAICoreConfig信息。
- [LaunchWithZeroEleOutputTensors](launch_with_zero_ele_output_tensors.md)：在算子输出为全空Tensor时，用户可以配置该算子依旧会进行NPU上板执行。
- [SetCheckSupport](set_check_support.md)：如果您需要在算子融合阶段进行算子参数校验，则可实现算子参数校验回调函数，并通过该接口进行注册。同时，需要将NeedCheckSupportFlag参数配置为true，则算子编译和融合阶段会调用注册的算子参数校验函数进行相关信息的校验。 若算子参数校验函数校验通过，则代表AI Core支持此算子参数，会选择AI Core上相应的算子执行；否则，会继续查询AI CPU算子库然后执行。
- [SetOpSelectFormat](set_op_select_format.md)：如果您需要自行推导算子输入输出所支持的数据类型与格式，则可实现推导回调函数，并通过该接口进行注册。同时需要将DynamicFormatFlag配置为true，则算子融合时会自动调用推导函数进行数据类型与格式的设置，算子原型注册时无需配置输入输出支持的数据类型与格式。 注意，如果算子原型已经注册过数据类型与格式，则以算子原型注册的数据类型与格式为准，即使注册了推导函数也不会执行。
- [SetOpSpecInfo](set_op_spec_info.md)：注册算子额外信息供FE解析。
- [SetOpSupportInfo](set_op_support_info.md)：注册算子切分信息函数。当算子的切分信息属于特殊类型，可以在算子实现文件里实现算子切分信息函数并注册，开发者需要自己按照一定的规则填写算子输入、输出的可以被切分的轴信息。
- [SetParamGeneralize](set_param_generalize.md)：注册算子泛化函数，包含每个参数shape的range。在图模式中，同一range下的shape共用二进制。
- [SetTiling（OpAICoreDef）](set_tiling.md)：注册Tiling函数。Tiling函数的原型是固定的，接受一个TilingContext作为输入，在此context上可以获取到输入、输出的Shape指针等内容。注册的Tiling函数由框架调用，调用时会传入TilingContext参数。
