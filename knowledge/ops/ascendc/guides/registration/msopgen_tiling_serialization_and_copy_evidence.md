---
type: Guide
title: msOpGen Tiling 序列化与拷贝证据
description: 区分嵌套对象所有权、Host/Device视图和上下文接口，限定 double free 归因。
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
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-to-registry-invoke/references/verification-checklist.md
  role: supporting
created_at: '2026-09-11T04:17:28Z'
updated_at: '2026-09-11T04:17:28Z'
platforms: []
---

# msOpGen Tiling 序列化与拷贝证据

宏TilingData的标量/数组setter、嵌套_STRUCT字段、REGISTER_TILING_DATA_CLASS及Device生成视图应采用同一版本定义。自定义struct是否支持不能仅按“官方/非官方”判定；需要实际序列化注册/布局合同。SDK Tiling嵌入可避免手工重复布局，但必须核对字段类型、容量、大小和当前API的读取范围。

Host取得输入、属性、平台、raw tiling和workspace指针后要检查可用性及范围；SaveToBuffer需容量足够并处理真实返回语义，SetDataSize声明有效序列化字节，SetBlockDim/key/workspace应与已编译变体一致。来源“五板斧”是此框架形态，不能推导每个算子都必须恰有一个workspace slot或大小0。临时区查询/GetXxxTilingInfo的参数、返回类型及失败条件必须按实际签名，示例省略的srcShape/stackBufferSize不能视为已初始化。

Case2记录local嵌套Tiling拷贝后Host heap double free，并建议把最终嵌套字段作为OUT引用直接填充。这保留为来源历史案例及一种减少中间所有权的办法；“所有XxxTiling都是序列化buffer视图”“POD拷贝也必然double free”没有类型定义/栈证据支撑，不能当通则。真实类型若拥有指针且错误浅拷贝可能双释放，纯POD值拷贝本身不会调用两次释放；需要检查构造、copy/move、析构和越界写。glibc检测点也不排除更早其他代码破坏Host内存。

来源Host嵌套对象与Device生成POD视图可能同名却拥有不同语义，不能把Host禁拷贝经验套到Kernel字段赋值。局部对象合法性、setter行为以及API是否接受引用由真实类型决定，不应正则禁止一切XxxTiling local={}。

InferShapeContext的GetInputShape在来源返回const gert::Shape*，输出经GetOutputShape后SetDimNum/SetDim或AppendDim；TilingContext提供StorageShape再按原始/存储语义取Shape，两类接口不能互抄。rstd等附加输出不能机械复制x shape/dtype，动态未知维的推导与真实Tiling分配也不同。IMPL_OP与IMPL_OP_INFERSHAPE的注册/链接目标需与执行路线一致，文件存在不证明图推导已加载。

来源头文件统一入口tiling/tiling_api.h是在该安装/构建的建议，缺adv_api头首先反映搜索路径或版本不匹配；不能把一切adv_api/lib/impl前缀都判为无公开能力。Host类型、设备类型、编译单元和真实声明需分别查明。
