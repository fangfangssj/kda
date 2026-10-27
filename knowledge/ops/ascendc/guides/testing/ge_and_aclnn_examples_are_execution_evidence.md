---
type: Guide
title: GE 与 ACLNN 调用示例的证据范围
description: 保存图构造与两阶段调用知识，纠正数据编码、输入索引和无比较成功消息。
tags:
- task.testing
- topic.implementation.operator_registration
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/examples/test_aclnn_add_example.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/examples/test_geir_add_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/example-guide.md
  role: supporting
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# GE 与 ACLNN 调用示例的证据范围

ACLNN示例建立设备/stream、分配存储并创建Tensor描述符，第一阶段取得workspace/executor，第二阶段提交运行，再同步与D2H读取。描述符的shape、stride、offset、dtype和实际host/device字节数须一致；GetShapeSize乘积要防负维/溢出，rank0的stride循环应先转有符号再减2，不能让size_t下溢。aclCreateTensor可能失败，不应无条件返回成功。

Tensor描述符销毁与底层device存储释放分别执行，workspace依实际完成期保活；初始化或中途失败需清理已得资源。PrintOutResult为void，内部D2H失败只打印并返回，main最后0不能证明结果读出成功。打印每元素1+1=2也不是完整精度或支持域验证。

GE示例先GEInitialize、构建Data/Const与自定义Op的输入输出描述、SetInputs/SetOutputs、Session::AddGraph和RunGraph，导出输入输出，再按正确所有权销毁Session并Finalize。每一步返回值/对象与Tensor payload需核对；来源AddGraph未检查便打印成功，成功路径Session未delete且Finalize时仍存活，不能推广为正确资源生命周期。

ADD_INPUT宏将所有Data的attr_index写0，多个运行输入需与输入向量顺序一致；名字表达式'placeholder'+整数在C++是字符串字面量指针偏移，不是追加数字，应使用显式字符串转换。Const的值作为图常量保存，其是否进入Graph输入列表须按图API合同，不能照搬Data路径。

GenOnesData按dtype计算data_len却用int32数组装整数2后reinterpret为字节，即使描述符DT_FLOAT也不是浮点2.0编码；FP16/BF16/64位类型更不匹配，且new int32[data_len]把字节数当元素数。专门的Float32生成函数未被通用宏使用。应按每种dtype真实编码、检查分配和长度，避免uint32字节数截断并明确Tensor构造是否拷贝/拥有payload。

读取argv[1]前需argc验证，文件fopen/fwrite/fclose与Graph dump均应检查真实成功，二进制输出格式/字节序也需合同。源码无golden比较却无条件打印Precision is ok，此消息只是一行文字；输出文件、进程0和调用成功分别是有限证据，不能混为精度通过。
