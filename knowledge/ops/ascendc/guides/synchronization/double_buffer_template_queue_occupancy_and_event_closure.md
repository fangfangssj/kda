---
type: Guide
title: 双缓冲模板队列占用与事件闭合
description: 纠正预取必须重写循环、depth1连续入队及不完整手工事件样例。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/elementwise/double_buffer_design.md
  role: primary
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-11T06:42:38Z'
platforms: []
---

# 双缓冲模板队列占用与事件闭合

物理buffer数由InitBuffer num、队列depth由允许的连续入队占用约束表达，二者不同。常规CopyIn→Compute→CopyOut主循环也可能由异步引擎与双槽跨tile重叠，不必显式拆出预取才有收益；单份input/output分别存储也不等于所有阶段绝对串行，compute-bound仍可能有可掩盖搬运。最终看时间线和依赖，不凭循环写法判定。

source显式预取先CopyIn0再CopyIn1才Compute0，若CopyIn内EnQue，则在第一次DeQue前已连续两次EnQue，与声明depth1不符；延后CopyOut还会令输出队列同时排两个结果。应根据确切最大占用选depth/调度并遵守事件配额，不能把depth1推荐套进任意预取顺序。tileNum0还导致无效预取和unsigned tileNum-1下溢；0/1/2/多tile需分别闭合启动与排空。

naive例分配outLocal却从computeLocal写GM，然后对未写outLocal EnQue/Free，没有完整V→MTE3与最后MTE3→input槽归还；其失败不证明单缓冲本身差。手工双TBuf例首次Wait(MTE3_MTE2)没有初始credit，循环只Set(V_MTE3)而无对应等待/CopyOut/释放credit和最终排空，不能当完整实现。

正确手工协议为首次可用槽、MTE2写→Vector读、Vector完成→MTE3读、MTE3完成→同槽下次MTE2覆盖，最后排空所有在途消费者。事件ID按类型/配额/slot合法复用，非每次Set/Wait都必须永远独占一个ID。TBuf配定向事件可以重叠，PIPE_ALL全停才会主动串行；两个/三个独立计数器轮转本身不保证AIC/AIV/MTE已对齐同数据代际。

容量按每区域真实dtype、份数和scratch计算；只有相同队列占满预算且所有其他量不变时，份数翻倍才需要tile约减半。原本有余量可不减，常驻固定区又使总量不是简单整库除2。来源同TPosition最多8或4块及VECIN/VECOUT合计方式依版本事件管理合同，不能只因为都在UB就确定相同配额。Triple/4buffer是候选，Cube+Vector不强制3，TQueBind须满足合法in-place与存活期，节省50%也非所有总UB。
