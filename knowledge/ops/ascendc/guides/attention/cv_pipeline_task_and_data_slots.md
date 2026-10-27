---
type: Guide
title: CV 交错流水的任务槽与数据槽
description: 明确 MM1、Vec1、MM2、Vec2 的三期排程、索引范围和跨核生产消费及回收关系。
tags:
- task.design
- topic.parallelism
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P14_cv_pipeline_preload.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P14.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# CV 交错流水的任务槽与数据槽

对任务 j 的依赖链 `MM1(j)→Vec1(j)→MM2(j)→Vec2(j)`，可以考虑把多个独立任务错期开工。来源的轮次安排为：轮 i 发起MM1(i)，处理Vec1(i-1)及MM2(i-1)，处理Vec2(i-2)。Vec1和MM2对同一任务仍有生产消费关系，不能因为列在同一轮就同时无等待地执行。

若任务总数为 L，轮次0到L+1可排空这一特定排程，但每个阶段应显式满足任务索引在[0,L)：MM1用i<L，Vec1/MM2用i>=1且i-1<L，Vec2用i>=2且i-2<L。L=0时不读任何任务信息。来源仅以isValid辅助判断，数组却未完整初始化，短任务及排空轮可能读到未写状态；所有元数据初态必须定义，数值范围守卫也不能省略。

三份RunInfo分别保存i、i-1、i-2，可以用 i%3、(i+2)%3、(i+1)%3定位；这些是任务描述槽。workspace的两份数据槽是另一维度，不能因为有三份描述就证明两份数据足够。MM1、Vec1和MM2的结果应有明确区域、跨度和最后消费者；若MM1(i)与尚待Vec2消费的MM2(i-2)共用物理区，同为i%2会发生覆盖。

每条边都要建立真实完成与可见性，例如Cube完成的结果发布给Vector、Vector产生的数据发布给Cube，以及最后消费者完成后归还槽位的反向通知。数据槽复用不能只依赖元数据isValid=false。参与AIC/AIV、子核映射、flag编号、pipe与API版本必须匹配；来源“先不加跨核同步，精度失败再补”没有正确性依据。

空间按各阶段结果的实际字节数、数据槽数及需要私有副本的核组数量求和。源文`原workspace*2*coreNum`只有当“原”严格指每核单槽且所有区域都翻倍时才成立；全局基数不能重复乘coreNum。元数据缓存与输出running state也需计入各自存储域。缩为单数据槽后必须重新安排等待/复用，不是只改数字即可保持排程。

Vec2包含浮点累计或在线Softmax状态时，还要保证任务顺序与统计基准一致。使用DMA原子模式的局部区域应按实际接口在完成后恢复普通模式，避免污染后续写入，但不能据此要求所有输出更新都用原子。完整流水需验证0/1/2/3/多任务、每槽首次使用、尾部两轮、空工作核与异步搬运结束。

这是静态排程和数据生命期的设计知识，未声明该骨架已经在设备上编译运行，也不承诺30–50%收益。是否值得交错取决于阶段时长、数据共享、资源竞争和任务数量。
