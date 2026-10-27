---
type: Guide
title: 注册 UT Faker 的属性顺序与平台模拟合同
description: 核对动态 IO、属性编码、对象寿命与模拟平台的实际适用范围。
tags:
- task.testing
- topic.implementation.tiling
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/any_value.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/infershape_case_executor.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/infershape_case_executor.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/infershape_context_faker.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/infershape_context_faker.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_case_executor.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_case_executor.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_context_faker.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/tests/ut/common/tiling_context_faker.h
  role: supporting
created_at: '2026-09-11T04:13:36Z'
updated_at: '2026-09-11T04:13:36Z'
platforms: []
---

# 注册 UT Faker 的属性顺序与平台模拟合同

来源AnyValue以枚举标签和shared_ptr<void>保存string、float、bool、int64及相应列表/嵌套int64列表；自定义deleter保留具体类型，但公开标签/指针若不一致，后续reinterpret读取不安全。泛型CreateFrom只有声明不能证明任意类型受支持；遇到未知类型只打印ERROR并继续会构造不完整上下文，应明确失败。字符串转换到AscendString还需确认复制/借用合同。

Faker的Attr(attrName,value)没有使用attrName，而只调用AppendAttr(value)，所以真实注册的属性顺序、类型、可选默认值必须一致，名字正确不能修复顺序错误。NodeIoNum、IOInstanceNum、动态输入输出组的实例数须和实际Tensor描述一致。origin/storage shape应分别保留；同时用同一个format填原始和存储格式，会漏掉格式变换差异。

Faker以unique_ptr保活gert::Tensor等对象，仍须让constValue所指数据及其dtype编码在被消费时有效。构建器接收的平台、compileInfo、TilingData、workspace容器原始指针同样依赖调用期寿命。未出现的可选Tensor、rank0、合法动态负维描述与真实分配的负维是不同状态，不能都用默认Tensor替代。

来源默认64核、262144字节UB、4096字节Tiling缓冲只是mock参数；workspace ContinuousVector<size_t>的4096是条目容量，不是字节。人工PlatFormInfo设置BT0、L2 32MiB、L1 512KiB、L0A/B64KiB、L0C128KiB及各intrinsic标志，fallback又使用不同容量，不是芯片数据表。CORE_NUM和AICore字段也不能替代真实GetCoreNumAiv查询语义。

soc字符串映射表覆盖范围不一致，operator[]可为未知型号插入空字符串；编译BUILD_SOC_VERSION宏决定实际平台，case参数socVersion没有被执行器使用。宏未定义、大小写变化或case写Ascend910B都不能证明测试了期望架构。JSON parse可能在fallback前抛异常，字段类型与数字转换也须验证；cube_freq写字面字符串不会变成有效频率。

InferShape注册空间、OpImpl及infer_shape函数指针均需检查；EXPECT_EQ是非致命断言，实际失败但期望成功时不能继续读取无效输出。只比期望数组中列出的shape不等于检查所有输出及dtype；shape推导不自动验证dtype支持。用DT_FLOAT16推导通过不能证明只注册FLOAT/INT32的当前Add支持半精度执行。
