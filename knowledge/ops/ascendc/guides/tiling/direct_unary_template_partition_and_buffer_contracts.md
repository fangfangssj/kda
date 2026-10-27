---
type: Guide
title: 直调单目模板的分区、UB 与类型合同
description: 保留 Kernel/Tiling 示例可复用逻辑，纠正固定常量、尾块和同步红线的泛化。
tags:
- task.implementation
- topic.implementation.tiling
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-op-templates/references/operator-template.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-op-templates/references/operator-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-coding-rules/references/red-lines.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-coding-rules/references/red-lines.md
  role: supporting
created_at: '2026-09-11T03:46:00Z'
updated_at: '2026-09-11T03:46:00Z'
platforms: []
---

# 直调单目模板的分区、UB 与类型合同

来源单目模板用C个核处理N元素：启动数min(可用AIV,ceil(N/1024))并至少取1，blockLength=ceil(N/C)，第i核有效长度clamp(N-i*blockLength,0,blockLength)。每核再按tileSize划分完整tile和tail。1024最小粒度、2048 tile是示例参数；N必须非负，C/tileSize有效，ceildiv与offset乘法需要范围证明，不能溢出后再clamp。

N=0时模板仍启动1核且不做数据循环，空Tensor可能无可解引用的指针；无工作核也不应先构造超出对象允许范围的地址。可在合法Host合同处直接返回规定空输出，或使用已验证空Kernel路径，不能仅凭循环未执行认为所有指针绑定合法。

输入/输出TQue各深度2时，仅这两队列就需4*align(tileSize*sizeof(T))字节，再计入TPipe、临时Cast、padding及其他同时存活资源。增加FP32中转改变预算和数值路径。GetCoreNumAiv/GetCoreMemSize来自实际平台对象，固定tile只要有完整目标域资源证明也可合法，不存在所有常量都必须运行时求值的硬件规则。

DataCopyPad通过DataCopyExtParams的字节长度支持对应非对齐场景；count*sizeof(T)转uint32前要检查字段范围和重载支持，物理UB长度、GM边界与计算mask另外证明。DataCopy在满足其真实合同的对齐场景仍是合法选择，GetValue/SetValue也不是只允许调试。来源泛称Sqrt无half原生重载未固定版本/重载，不能据此决定所有half/bfloat16都须转FP32。

CopyIn→EnQue/DeQue→Compute→EnQue/DeQue→CopyOut→Free表达队列生产消费，但源文本Alloc/Free或EnQue/DeQue数量相等不证明分支/迭代生命周期闭合。非队列路径可以用匹配事件或相应同步机制；PipeBarrier是流水作用域排序，不能把PIPE_MTE2或PIPE_V无条件当跨流水完成替代。来源“同pipe永远保序”和只列RAW/WAW遗漏WAR等风险，需用[真实依赖证明](../synchronization/review_rules_require_actual_dependency_proofs.md)。

声明队列深度2并不自动证明所有搬运与计算重叠，热路径更强屏障可降低并发，但删除必须先证明依赖仍成立。标准库命名空间、递归、动态内存或API重载可用性由真实Device编译模型决定，不把评审表的风格/禁令当跨版本编译能力。初始化与资源寿命仍需完整。

模板Kernel<Op>、AscendC::<Op>和头文件占位符须按实际算子替换，复杂归约/Cube/Attention不能仅换一个函数名就得到正确实现。本卡提取分区与数据流合同，未将模板作为可编译验证产物。
