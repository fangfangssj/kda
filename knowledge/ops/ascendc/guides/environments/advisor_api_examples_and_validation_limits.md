---
type: Guide
title: API 示例清单与真实编译合同
description: 纠正设备修饰符和命名空间的误判，保存Tensor、队列及Mask操作边界。
tags:
- task.optimization
- topic.performance.analysis
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/standards/ascendc_api_validation_reference.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/mask_operations/reset_mask.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simt_api/simd_simt_hybrid_programming_intro/extended_syntax/function_execution_space_qualifier.md
  role: constraint
created_at: '2026-09-11T04:58:58Z'
updated_at: '2026-09-11T04:58:58Z'
platforms: []
---

# API 示例清单与真实编译合同

AscendC大小写命名空间需要与声明一致，但LocalTensor等无显式前缀可经合法using声明/命名空间引入，并非一律语法错误。枚举是否需限定也要看scoped enum及已导入声明，模板参数顺序需对应真实声明；样例清单中出现名字不等于当前编译单元/版本支持该重载。

Ascend C入口可使用__global__ __aicore__，不能把__global__单词本身一概判为CUDA非法语法；设备可调用helper和全局launch入口的角色不同。SIMT扩展中的__device__等还需看编译模式，不能把特定Scalar kernel约束推广到整个Ascend C。threadIdx/__shared__之类也按目标语言/编译模式判定，不能仅扫描关键字决定正确。

Host STL/堆分配器不能不加区分移到AI Core，但设备C++有受限的标准库风格接口，来源禁止所有std::array、std::或printf过宽。日志/打印应按目标版本及执行环境的接口、格式和开销处理；不存在只因.cpp位于op_kernel就能推导所有语言功能集合的规则。

LocalTensor/GlobalTensor是局部/全局typed视图，SetGlobalBuffer绑定GM，TBuf.Get取已配置缓冲；TPipe.InitBuffer参数size按字节配置，Tensor索引/count按重载定义，不能把4096字节统一当4096个float。TQue模板depth与InitBuffer份数分开，Alloc/EnQue/DeQue/Free建立句柄生命周期，重复变量声明片段不是同作用域可直接粘贴程序。TPosition枚举有GM不代表任意TBuf<GM>可合法分配GM。

Abs/Add/Sub/Mul/Div/Adds/Muls/Exp/Log/Sqrt/Reciprocal、DataCopy/DataCopyPad、Mmad等名字跨基础Tensor、寄存器、高阶和SIMT家族有不同签名/dtype/scratch要求。来源Mmad(c,a,b,M,K,N)和DataCopyPad省略参数是示意，不是已编译调用；Log/Ln也需核对具体家族。数值与地址、mask、stride、同步、别名和临时空间是编译外的运行合同。

SetMaskCount选择Counter模式，SetVectorMask<T,Mode>设置相应有效元素；外部mask通常需匹配isSetMask=false等重载。ResetMask恢复mask值，不等于恢复Normal模式，来源“还原掩码状态”漏掉模式维度；正常模式恢复按SetMaskNorm合同。设置mask后立即Reset而无计算的示例只能演示调用，不能证明优化产生了任何结果。

接口名白名单、namespace检查和模板参数检查适合发现局部问题，不能据此标记整段代码VALIDATED。目标编译检查、真实运行、数值与生命周期证据应分别记录，未在清单中的API也可能合法。来源每月维护/Skill输出闸门是流程，不作为知识使用指令。
