---
type: Guide
title: APACE AllToAll Matmul 的全局与局部坐标
description: APACE AllToAll Matmul 的全局与局部坐标的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/paradigm-mapping.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE AllToAll Matmul 的全局与局部坐标

固定来源描述的AllToAll+MX Matmul有R个rank；用m表示每目标rank输出行数，globalM=R*m，globalK=R*Ka。每source持A[globalM,Ka]的K分量；B为复制的完整[globalK,N]，物理DN布局与逻辑shape分开描述。source s向target t发送A的[t*m,(t+1)*m)行；t收齐后逻辑A为[m,globalK]，分别乘B对应s的Ka段并沿K累加，得到C_t[m,N]。AllToAll只搬运，求和由Matmul执行。

来源不同页将K写成局部Ka或全局K，故B[rankSize*K,N]与B[K,N]未必矛盾，须统一符号后判断。不能把这个K分量累加样例当成所有AllToAll Matmul的定义；注册版permute+一次Matmul可能是另一布局合同。AllGather输入收集后计算与计算后收集输出也不同。

APACE此快照的basic负责张量/内存视图，block通过CRTP组合通信及Blaze计算，kernel负责调度；aiv_compute和部分epilogue是空占位。声明某组件目录不证明已有执行能力。文档将GET=N切、PUT=K切和GET=计算在前泛化过度；切轴、访问方向、先后顺序分别由实际算子决定。

用source/target/行/K坐标分别编码的输入验证发送分片、B偏移与最终sum；仅相同数据填各rank会掩盖重复self、rank排序及拼接与累加混用。
