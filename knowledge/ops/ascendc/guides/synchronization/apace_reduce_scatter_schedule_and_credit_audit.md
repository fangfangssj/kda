---
type: Guide
title: APACE ReduceScatter 时序与通知计数
description: APACE ReduceScatter 时序与通知计数的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/development.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/design.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE ReduceScatter 时序与通知计数

目标错位流水为通信t和归约t-1在同一个阶段并行，再等待二者本地结束、跨rank fence后进入下一轮。来源fusion如此描述，但场景development的§5.1把ReduceSum(t-1)放在本轮fence后的末尾；下一轮开头又全AIV SyncAll，因此通信核会等该归约结束才下发下一轮，不能据这份伪代码宣称两者重叠。其功能还需尾轮Reduce(T-1)完成；性能结构须以时间线证明。

localLast双flag分别代表remote部分完成和全部本地fragment完成。通信核等flagA、归约核等flagB可以提前启动PUT，但每个发送核/接收核的计数必须按实际跨核拓扑配对；若所有AIC每轮都Set两个flag而相应AIV只消费其中一个，未消费通道可能积压。旧单flag骨架不能无修改拼接新双flag协议，必须给不工作核和非所属角色也定义必要drain。

按mPos>=localFragBoundary首次发ready需scheduler具有相应遍历和完成顺序，并确保全部待发送remote fragment已完成FixPipe；不能从行起点越界就推定所有远端片段已产出。没有本地tile的核也需在自己remote工作完成后兜底通知。计数式flags可跨轮复用但仍受未消费积压约束，不能把首尾各发两次解释为无限轮安全。

归约src pingpong在首次复用前无需等未Set的credit，i>=2等旧slot；R1不消费从未Set的slot1。最终MTE3_V和V_MTE2残留依据实际发过次数清理。源“逐行版T>1/批量版T1”与其他批量规则冲突，核心是实际数据依赖和按批管理事件，T>1也可以批量。
