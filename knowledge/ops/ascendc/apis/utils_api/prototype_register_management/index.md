# 原型注册与管理

- [原型注册接口（OP_ADD）](op_add.md)：注册算子的原型定义，从而确保算子能够被框架正确识别、编译和执行。 算子原型主要描述了算子的输入输出、属性等信息以及算子在AI处理器上相关实现信息，并关联tiling实现等函数。算子原型通过自定义的算子类来承载，该算子类继承自OpDef类。完成算子的原型定义等操作后，需要调用OP_ADD接口，传入算子类型（自定义算子类的类名），进行算子原型注册。详细内容请参考算子原型定义。
- [OpAICoreConfig](op_ai_core_config/index.md)：共 11 张知识卡（如 DynamicCompileStaticFlag、DynamicFormatFlag、DynamicRankSupportFlag 等）。
- [OpAICoreDef](op_ai_core_def/index.md)：共 8 张知识卡（如 AddConfig（OpAICoreDef）、LaunchWithZeroEleOutputTensors、SetCheckSupport 等）。
- [OpAICPUDef](op_aicpu_def/index.md)：共 9 张知识卡（如 ComputeCost、Engine（原型注册管理）、ExtendCfgInfo 等）。
- [OpAttrDef](op_attr_def/index.md)：共 2 张知识卡（如 Comment（原型注册管理）、OpAttrDef）。
- [OpDef](op_def/index.md)：共 13 张知识卡（如 AICore（OpDef）、AICPU（OpDef）、Attr（OpDef） 等）。
- [OpHostCPUDef](op_host_cpu_def/index.md)：共 9 张知识卡（如 ComputeCost、Engine（原型注册管理）、ExtendCfgInfo 等）。
- [OpMC2Def](op_mc2_def/index.md)：共 5 张知识卡（如 HcclGroup（OpMC2Def）、HcclServerType、OpMC2Def构造函数 等）。
- [OpParamDef](op_param_def/index.md)：共 20 张知识卡（如 AutoContiguous、Comment、构造&拷贝构造函数（OpParamDef） 等）。
- [OpAICoreConfig注册接口（REGISTER_OP_AICORE_CONFIG）](register_op_aicore_config.md)：不同的硬件形态算子原型定义不同的情况，可以通过新增OpAICoreConfig的方式，针对不同的AI处理器型号注册差异化的算子原型。REGISTER_OP_AICORE_CONFIG宏在不改变原有注册的基础上，允许单独新增文件来注册算子在不同硬件形态上的差异化信息。 使用该注册宏需要包含以下头文件：
