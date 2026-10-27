---
type: Guide
title: Add 注册模板的跨层 shape 与 dtype 合同
description: 核对 L2、L0、OpDef、图推导与 Kernel 的实际支持域，补足广播和空输入检查。
tags:
- task.implementation
- topic.implementation.operator_registration
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/README.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_api/aclnn_add_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_api/aclnn_add_example.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_api/add_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_api/add_example.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_host/add_example_def.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_host/add_example_infershape.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_graph/add_example_proto.h
  role: supporting
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# Add 注册模板的跨层 shape 与 dtype 合同

来源AddExample的L2采用GetWorkspaceSize构建executor和运行阶段CommonOpExecutorRun。第一段检查输入/输出指针、同dtype（FLOAT/INT32）、非private format及rank≤8，再Contiguous输入、L0调度、ViewCopy到用户输出；executor持有实际计算计划，workspaceSize不是仅手写Kernel临时区。C导出和可见性宏确保正确符号入口，仍需匹配完整签名与所有权。

L2的CheckShape只检查rank，没有证明两个输入可广播、输出shape正确或storage足够；workspaceSize/executor输出参数也未显式判空。HasEmptyTensor在广播/输出合同检查前快速成功可能接受本应失败的组合。空输入是否可直接返回取决于完整输出语义，不能以任一输入empty代替合法性验证。Contiguous只改变物理布局，不执行shape广播。

L0调用BroadcastInferShape分配输出，但Host图InferShape仅复制input0，Tiling仅检查三个元素总数相等，Kernel按相同线性offset相加。这些并非完整广播实现：如[2,1]和[1,2]应得到[2,2]却会被等元素数Tiling拒绝；同元素数不同shape也不证明符合逐维关系。应统一承诺等shape还是实现完整广播，并让L2、图推导、Tiling与Kernel同义，不能只修某一层描述。

OpDef的DataType/Format/UnknownShapeFormat组合、AutoContiguous与动态shape/rank等配置表达注册与框架选择，不自动使所有rank、broadcast、非连续输出或dtype组合受支持。source proto注释称NCHW/NHWC仅float，代码却是FLOAT/INT32，README又写“无约束”；应以经过核对的真实合同消除矛盾。PrecisionReduceFlag也不是数值验收豁免。

L0示例限制DAV_2201/DAV_3510且以输入dtype分配输出，AllocTensor返回值仍应在launch前检查。OP_TYPE_REGISTER、OP_INPUT/OUTPUT/ATTR/OUTSHAPE按具体框架宏传递输入、属性和动态输出，不能因注释给出其他算子调用形式就认定当前Add已实现这些模式。图模式与ACLNN模式可能经过不同推导入口，须分别验证。

接口验证应覆盖同shape合法输入、广播候选、元素数相同但shape不同、rank0、空Tensor、非法输出shape、非连续视图、dtype不匹配和private format。标量[]可在内部转为{1}进行元素数计算，但不应改变外部标量输出合同。整数加法的溢出规则由目标接口决定，Host C++有符号溢出不能当golden的默认环绕。
