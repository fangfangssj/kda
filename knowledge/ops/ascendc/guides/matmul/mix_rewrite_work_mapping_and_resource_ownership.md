---
type: Guide
title: MIX 重写的工作映射与资源归属
description: 保留 Cube/Vector 分工、参与核循环、矩阵链和中间状态的静态合同。
tags:
- task.design
- topic.parallelism
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P102_scalar_matmul_to_cv_fusion_rewrite.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P102.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/mix-cv-fusion-guide.md
  role: supporting
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-11T02:18:54Z'
platforms: []
---

# MIX 重写的工作映射与资源归属

标量点积/矩阵链或Vector逐行Mul+ReduceSum可考虑改由Cube计算矩阵部分，Vector负责elementwise、mask、decay和状态更新。是否采用MIX由实际硬件、矩阵形状、dtype、数据布局及交接成本决定；64×64是来源候选阈值，不是Cube可用下限，更不是唯一能取得数量级收益的架构。

混合编译/执行中各代码路径须与实际核类型及资源能力匹配，资源初始化也应与使用者一致。来源使用ASCEND_IS_AIC/AIV和MIX_AIC_1_1/1_2；宏展开、可访问存储、初始化重载及启动配置按所用SDK核对，不能把“AIC永远不可触碰UB”推广至全部产品/执行模式，亦不能认定错误调用只会以100%占用死锁一种方式表现。

在来源所用1:1/1:2映射中，可用AIC核号、AIV核号除以subblock数量获得配对工作组，但真实GetBlockIdx/GetBlockNum语义、启动数量和组内编号必须共同验证。全体同步参与者若按slots=ceil(work/cores)共同循环，workId=s*cores+groupId，越界者跳过实际工作却继续到达每个规定屏障。cores=0、work=0、乘加溢出及不同分支的动态到达顺序需处理。

SyncAll的模板及覆盖集合按接口合同选择，调用数相等仍不足以证明通知顺序、数据发布与所有参与者一致。GM workspace交接是来源实现的一条合法候选，需按float/half布局、任务/slot、生产发布和消费回收完整定义；目标平台受支持的L1/TSCM或其他CV通路也可用，不能一律禁止片上交接。

矩阵链需保留GM ND行跨度、L1 NZ布局、L0加载参数、Mmad有效M/N/K及cmatrixInitVal对应初始化/累加、Fixpipe输出dtype和行距。srcDValue以元素为单位的示例不能推导所有Fixpipe stride也相同；根据具体NZ→ND重载逐项换算。队列和显式事件是可选管理方式，核心是完整数据依赖，不能要求所有手动Mmad都必须使用同一队列语法。

L0C同一累加链必须保持其所有K贡献，但不同独立结果可以双缓冲，不能一律禁止L0C num=2。被屏障严格串行化的阶段可能没有可重叠流；阶段内部仍可双缓冲。TBuf手动轮转需要独立slot与事件，增加份数不会自动产生流水。

验证每阶段矩阵与完整多chunk状态，尤其strict/inclusive三角mask、跨chunk decay与scale、空核和尾轮；单chunk通过不能覆盖状态递推。来源3650→261微秒约13.98倍及geomean8.85未在本批复现，且元数据baseline=optimized=1与speedup8.85不自洽，不能作为实测记录导入。
