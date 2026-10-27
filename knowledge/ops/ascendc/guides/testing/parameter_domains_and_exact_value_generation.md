---
type: Guide
title: ST 参数域与精确值生成
description: 保存 Tensor、列表、标量的因子模型，修正特殊值与低位格式范围示例。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-st-design/references/parameter-definition.md
  role: primary
created_at: '2026-09-11T03:52:52Z'
updated_at: '2026-09-11T03:52:52Z'
platforms: []
---

# ST 参数域与精确值生成

来源测试schema以name/type/required/io_type描述Tensor、TensorList、Int/Float/Bool Array、ScalarList和标量，再给format、dimensions、length_ranges和dtype关联值域。它是测试数据模型，不证明ACL接口支持表中所有类型或组合。默认rank 1–8不能覆盖所有真实接口，rank0标量与shape含0的空Tensor必须按合同分别建模。

TensorList需保存列表长度和每元素的shape/dtype/format关系，不能将异构列表默认为共享一个shape；来源format表用嵌套列表而部分模板扁平，解析器应有确定schema并拒绝歧义。Array的元素类型受实际参数合同固定，不能从通用dtype候选随意换类型。Optional缺席、空列表、空Tensor和标量零是不同状态；输出不生成输入数值的规则不适用于需要初始内容的inout或别名场景。

枚举用dtype_with_values/value，连续候选用value_range；Scalar来源非枚举也嵌于dtype_with_values，不能凭字段名将其当枚举。真实静态支持域与用于采样的区间不同：随机抽区间不保证抽到端点、最小正规、次正规或相邻舍入边界，应明确点采样、邻值与编码校验。整数边界在整数域保留，JSON/浮点中转可能丢失int64/uint64端点。

±0、Inf、NaN在来源用带引号字符串作特殊值标记；必须由类型感知解析器转换并校验位模式，NaN不能当普通有序区间，整数和bool没有与0不同的-0编码。复数需分别指定实虚部分域，complex32等名称还需核对分量宽度，不能统一套FP32最大值。C++ char符号性不能由名字保证int8，int4/uint1是测试或接口类型标签而非普通标准C++内建类型。

浮点范围应来自实际编码、而非直接照抄来源表。来源FP64例子大量使用FP32边界，缺少FP64自身边界；FP8 E4M3/E5M2的若干近零数不对应最小正次正规。对于有正规/次正规编码、指数bias=b、尾数位p的模型，最小正次正规为2^(1-b-p)：若E4M3采用b=7,p=3则为2^-9；若E5M2采用b=15,p=2则为2^-16。须先确认目标格式及FTZ行为，不把名称相似的特殊格式混同。

E8M0的指数整数/存储字节与解码后的正幂二scale是三个域，来源[-127,127]不能直接作为含负数的scale值范围；保留编码、零块和上下溢见[MX范围条件](../precision/mx_scale_range_and_counterexample_evidence.md)。FP4 E2M1/E1M2、HiFloat8需按目标解码表产生可表示值与±0等特殊点；来源片段不足以建立完整格式表。hf32也不能仅凭测试标签当独立存储dtype。

qint16候选65535超出有符号16位数值域，只有明确无符号载荷解释才可能保留该字节模式。量化测试还需要scale、zero_point、饱和/舍入和轴/组关系；单个raw整数区间不足以定义量化语义。归一化算子并非都只支持[0,1]，不要按名称删掉真实支持域的负值。

示例Addbmm的type组合、dimensions和cubeMathType=[0,1,2,3]仅是给定接口资料的输入，需与目标版本公开合同核对。模板中aclnn_name映射与紧随的裸序列不能原样当一份合法YAML，需实际顶层parameters容器；未提供的类型、值域与省略号不能当已完整生成。
