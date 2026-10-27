---
type: Guide
title: msOpGen 迁移的原型身份与类型映射
description: 保存注册点、类型元组与分发对应关系，明确源码规则的适用边界。
tags:
- task.implementation
- topic.compilation
- topic.implementation.operator_registration
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/rules.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/cases.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/reference.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/examples.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/build-verify.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/verification-checklist.md
  role: supporting
created_at: '2026-09-11T04:17:28Z'
updated_at: '2026-09-11T04:17:28Z'
platforms: []
---

# msOpGen 迁移的原型身份与类型映射

来源采用msOpGen的BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS、GET_TILING_DATA、IMPL_OP_OPTILING/IMPL_OP_INFERSHAPE及OP_ADD链路；其他模板宏被禁止是该Skill选择，不表示GET_TILING_DATA_WITH_STRUCT或ASCENDC_TPL_ARGS_DECL在CANN一概无效。应匹配生成器、头文件、Host序列化和Device解析体系，不混用未经证明兼容的定义。

OpDef、REG_OP、Tiling注册、推导注册、JSON op及实际调用符号必须指向同一算子。opFile.value、文件基名、kernel入口之间要符合实际生成器的绑定规则；PascalCase/snake_case是该模板命名惯例，Shell ${OP_NAME,,}只能把RmsNorm变rmsnorm，不能生成rms_norm。不能依靠小写转换寻找正确文件或删除旧注册单元。

每条输入输出需保留name、required/optional/dynamic基数、dtype/format合法元组、属性类型与默认语义、SoC配置。DataType按列配套时JSON与OpDef应逐列对应，UnknownShapeFormat也要符合机制；DataTypeList等自由组合另有语义，不能一概令列表长度等于TilingKey数。TilingKey可编码dtype之外的模式/shape，同一dtype可能多个key，多dtype也可共享策略；key=JSON列下标只是此例人为约定。

RmsNorm例的五组(x,gamma)是(FP32,FP32)、(FP16,FP16)、(FP16,FP32)、(BF16,BF16)、(BF16,FP32)，y随x、rstd为FP32、epsilon默认1e-6。这是该例合同，不是所有RmsNorm或芯片的固定组合数；gamma形状、reduction轴、rstd shape及精度路径仍需独立定义。C++类型half/bfloat16_t/float、ge::DT_*、ACL枚举和JSON字符串是不同类型域；声明double字符串不证明Kernel支持FP64。

原型JSON顶层数组和input_desc/output_desc/attr格式、dtype/format/attr token需按所用msopgen版本校验。来源列float/float16/bfloat16及整数、bool、double，attr列int/float/bool/str及列表；其列表不是跨版本能力保证。AutoContiguous和DynamicShape/Rank/Format等开关不能替代实现，PrecisionReduceFlag不放宽验收合同，六个flag全部手写也不是所有工程的必需形式。

SoC显示型号、家族名、编译arch及目录名是映射关系，应查实际工具链支持配置；来源Ascend910B3→ascend910b是例子，“一律删尾数字”会误伤ascend910_93/ascend950等有效名称。AddConfig、生成器目标、CMake配置、binary配置应指向同一实际平台，不能从注释或命令字符串认定已支持。

空原型、遗漏dtype/format或错误列关系可导致注册/选型失败，但NotRegistered/GetOpInfoFailed也可能来自未加载包、错误库或缺二进制。undefined reference应检查定义、链接、namespace和ABI，不能只改算子拼写。检查文件中Input/Output/API出现次数只说明文本存在，不能证明完整合同或运行路径。
