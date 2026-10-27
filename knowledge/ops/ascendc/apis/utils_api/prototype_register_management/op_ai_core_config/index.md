# OpAICoreConfig

- [DynamicCompileStaticFlag](dynamic_compile_static_flag.md)：用于标识该算子实现是否支持入图时的静态Shape编译。
- [DynamicFormatFlag](dynamic_format_flag.md)：标识是否根据SetOpSelectFormat设置的函数自动推导算子输入输出支持的dtype和format。设置为“true”，则无需在原型注册时配置固定的dtype与format，会调用推导函数来推导算子输入输出支持的dtype和format。
- [DynamicRankSupportFlag](dynamic_rank_support_flag.md)：标识算子是否支持dynamicRank（动态维度）。
- [DynamicShapeSupportFlag](dynamic_shape_support_flag.md)：用于标识该算子是否支持入图时的动态Shape场景。
- [ExtendCfgInfo](extend_cfg_info.md)：用于扩展算子相关参数配置，提供更灵活的参数配置能力。
- [Input（原型注册管理）](input.md)：某些场景下，同一个算子在不同的AI处理器型号上，其支持的原型输入不同。 通过该接口，可针对不同的AI处理器型号注册差异化的算子原型。调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输入信息。
- [NeedCheckSupportFlag](need_check_support_flag.md)：标识是否在算子融合阶段调用算子参数校验函数进行data type与shape的校验。 - 若配置为"true"，框架会调用通过SetCheckSupport设置的算子参数校验函数，检查算子是否支持指定输入，此场景下需要自行实现算子参数校验的回调函数。 - 若配置为"false"，表示不需要进行校验。
- [OpAICoreConfig构造函数](op_ai_core_config_constructor.md)：OpAICoreConfig构造函数。
- [operator =](operator.md)：赋值运算符重载，实现OpAICoreConfig类对象的深拷贝
- [Output（原型注册管理）](output.md)：某些场景下，同一个算子在不同的AI处理器型号上，其支持的原型输出不同。 通过该接口，可针对不同的AI处理器型号注册差异化的算子输出。调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输出信息。
- [PrecisionReduceFlag](precision_reduce_flag.md)：此字段用于进行ATC模型转换或者进行网络调测时，控制算子的精度模式。只有当精度模式("precision_mode")配置为混合精度("allow_mix_precision")前提下生效。
