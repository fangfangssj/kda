---
type: Guide
title: MX Tiling Stub 的字段关系与整数边界
description: 解析示例基块、Scale跨度与缓冲参数，保留 mock 未完成的硬件与整数验证。
tags:
- task.design
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_tiling/matmul_tiling_stub.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_tiling/matmul_tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_tiling/matmul_constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_kernel/matmul_blaze_kernel.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/common/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_guide.md
  role: supporting
created_at: '2026-09-11T02:53:42Z'
updated_at: '2026-09-11T02:53:42Z'
platforms: []
---

# MX Tiling Stub 的字段关系与整数边界

实际stub并非文档所称始终固定128：baseM/N把min(维度,128)向16对齐，baseK把min(K,128)向64对齐，零值另设16/64。合法硬件基形与当前有效尾块不同，向上对齐基形不代表GM有同样多有效元素，也不证明选定Copy支持尾部。stub忽略transpose，需要真实Tiling针对选定类型/Layout和SOC检查约束。

stub以mTiles*nTiles与48取min作为usedCoreNum。48是样例常数，不是动态平台查询。需检查空维度、核数合法范围、tile乘积溢出和Scheduler对超出核数任务的覆盖。各维度从uint64截到uint32，common工具虽定义CheckUint32Shape，入口实际未调用；存在辅助函数不等于输入已验证。字节数M*N*2、M*K等及Scale容量也需独立做乘法溢出检查。

原始step=ceil(K/baseK)，scaleKL1=step*baseK，而存入uint8 stepK的是min(step,4)。Kernel用stepK*baseK作为数据L1跨度，用scaleKL1作为Scale跨度，两者在长K时不同；这可能表达不同复用范围，不能擅自强制相等，也不能在不检查组件消费和容量时认定正确。验证需要逐项确认Scale按何粒度装载、何时重用及最后一段释放。

Tiling的nBufferNum=2没有被当前Kernel读取，Kernel L1Params直接写2；调该字段不会改变实际缓冲数。dbL0c传入QBMM参数，但名字本身不证明它是布尔开关还是数量，应以实际组件定义为准。L1数据/Scale和L0C多缓冲全部计入容量，不能以成功填充POD结构代替预算。

CeilDiv用商余形式避免a+b-1溢出，但来源b=0直接返回a并非有效除法定义；Align再乘b也可能溢出。CLI只接受正十进制uint64，未覆盖后续uint32截断；C++bool只认字面true，而生成器接受1/t等，传同一字符串可能生成与执行不同转置。统一参数语义是正确性条件。

验证小维度、基形边界、K超过4个baseK、接近整数上界和不同转置；只在Host演算通过仍不能替代真实库组件的资源/覆盖与设备检查。
