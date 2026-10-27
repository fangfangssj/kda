---
type: Guide
title: 注册 UT 的 Tiling 断言与 Tensor 工厂边界
description: 抽取断言漏检、原始字节布局、空输入和仅参数校验的证据限制。
tags:
- task.testing
- topic.implementation.tiling
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_case_executor.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_case_executor.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_context_faker.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_context_faker.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_api/test_aclnn_add_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_api/test_add_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_api/test_op_api_main.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_api/test_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_host/test_add_example_infershape.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_host/test_add_example_tiling.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/op_host/test_op_host_main.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/README.md
  role: supporting
created_at: '2026-09-11T04:13:36Z'
updated_at: '2026-09-11T04:13:36Z'
platforms: []
---

# 注册 UT 的 Tiling 断言与 Tensor 工厂边界

Tiling成功应独立验证返回码、blockDim、key、TilingData有效大小与语义字段、workspace条目数和每项字节量。来源ExecuteTestCase没有检查blockDim；只有actual workspace数>0才进入比较，实际0条、期望{0}会漏检；EMPTY_EXPECT_TILING_DATA主动跳过数据比较。成功返回和正确key不能代替这些断言。

来源把TilingData reinterpret为int64数组再转字符串，size/8向下取整会忽略余数，混合uint32/uint8结构也不能这样解释；还涉及对齐、别名及padding。应按真实结构解码字段并检查长度，不能无条件比较未初始化padding。ExecuteTiling反复向tilingInfo.workspaceSizes追加而不清空会保留上次结果，uint64 key转int64也应保留完整位语义。

当前两条Tiling测试均N2048、FLOAT/INT32，mock64核/256KiB UB，期望blockFactor32、ubFactor10912及key256/259。10912来自floor(262144/4/6)再按32元素向下对齐；key来自本模板编码，不是通用常量。用例没有覆盖N0、1024阈值两侧、UB不足、非法shape、多tile尾或所有块分配。InferShape唯一动态shape用例只验证复制input0，README的广播/全面边界覆盖声明超过代码证据。

L2五条测试仅第一阶段的null与dtype不匹配，无有效Kernel执行或数值验证。L0 EXPECT_DEATH(...,".*")只证明进程以某种方式死亡，不区分预期断言和误解引用，GTEST_HAS_DEATH_TEST不支持时是跳过。只有接口定义了nonnull前提才可按该前提解释内部调用，不能把崩溃当公开错误返回合同。

TestTensorFactory用Host new uint8_t[]分配数据，适合不读设备payload的Host参数路径，不是NPU设备存储。shape乘积、连续stride、dtype字节数须检查负维和溢出，rank0循环需要避免size_t减1下溢后窄化。未知dtype映射ACL_DT_UNDEFINED/size0应拒绝，CreateEmptyTensor未验证shape含0，不能把任何null payload都称合法空Tensor。storageDims=null是否适合目标ACL版本须实际确认。

工厂仅管理自己分配的payload，外部tensorData仍由调用方保活；map的mutex保护容器操作，不让重复Destroy或并发使用同一Tensor变安全。Guard禁复制减少双重释放，创建失败仍需断言。释放payload和销毁descriptor的次序应符合API，不能在异步任务仍使用它们时销毁。注释“FLOAT16仅910B支持”不是硬件能力事实，当前模板本身也未注册half。

Host测试Environment若AddSoToRegistry失败只打印并return，不会自动令GTest失败；旧全局registry还可能使测试误用旧函数。应确认实际.so身份、符号注册成功和每个函数指针有效，避免把环境初始化失败统计成通过。CPU UT没有设备执行证据。
