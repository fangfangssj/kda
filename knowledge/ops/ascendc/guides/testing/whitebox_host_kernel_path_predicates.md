---
type: Guide
title: 白盒 Host–Kernel 路径谓词
description: 保留 key 构造与二级分支，修正笛卡尔积和优先分支的错误简化。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/00-execution-order.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/01-code-analyzer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-a/00-overview.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-a/01-step1-tiling.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-a/02-step2-trace.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-a/03-step3-kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-a/04-path-config-schema.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# 白盒 Host–Kernel 路径谓词

路径由实际 Host 分发条件、TilingKey 构造与后续改写、平台守卫及 Kernel 入口内部条件共同决定。Init、IsMatch、DoTiling、模板实例、fallback 和 workspace 分配失败都可能改变可执行集合。同一 key 可以进入多个 Kernel 二级分支；只枚举 key 或模板名字不能证明覆盖所有计算路径，也不能证明函数体已实现、可编译或正确。

key 组成维度的笛卡尔积仅在各维独立且每种组合都可达时代表完整 key 集。dtype、shape、平台和算法选择常相关，fallback 不是唯一破坏正交性的原因。源码声明但无 Host 产生的 key 是待分析的 orphan；不完整搜索范围不能据此认定死代码。Host 可产生但 Kernel 无处理的 key 应保留为分发缺口，不能通过从清单删掉来消除问题。

优先分支必须保留前驱未命中条件。例如 if x<10，else if x<20，第二条真实范围为10≤x<20，只有 x<20 会重叠。另一方面 ¬(A∧B)=¬A∨¬B，不能拆成 ¬A∧¬B。来源全面禁止记录前驱条件的否定也会丢失必要语义；应保存原布尔结构，再做可证明等价的化简，OR 不能扁平合并成 AND。

纯参数校验可能终止调用，影响路径合法域；纯 TilingData 赋值也可能被 Kernel 条件消费。只因不直接写 key 就排除这些语句会漏条件。未读到的外部 helper 或配置要标为未知依赖；来源 file:line 线索应绑定源码版本，不能用搜索不到充当条件不存在。

path_id、TnKm、fallback 标号与 active/dead 列表是分析模型中的身份，不是硬件协议。原始 key 数字、字符串展示和 dtype 编码的归一化须有无歧义映射，保留原值。模型中的分组是否允许同 key 跨组取决于 group 维度，不能同时假定 key 全局唯一又复制到多个组；默认分支也需要实际条件和可达性。
