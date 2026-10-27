---
type: Guide
title: APACE 先计算后通信的实现边界与失效信号
description: 保存Fragment映射、归约分工及事件失效经验，限定数值阈值。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/review-checklist.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow_integration.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# APACE 先计算后通信的实现边界与失效信号

先计算后通信时staging是实际通信源，必须等AIC输出写回可见后发布ready，通信完成后归约再读取。FragmentTensor可将多个rank片段映射为一次计算任务以减少R*T次子调用重建参数开销，但A/C fragment地址必须按同一逻辑顺序对应。来源localLast把远端片段排前、本地排后，并在边界提前发flag启动通信；不能只重排A而遗漏C地址，也不能把每轮随意双发flag当等价协议。

source R≤32是该次fragment构建数量上限，与通信轮T不同，需按真实版本数组/接口核验；localMatmul1对应远端splitKNum=R-1而0/2对应R的规则属于相应模式，不可移到不含本地分离的路径。RunLocalMatmul与后续RunMatmul切换报告需要完成屏障，具体应查资源生命周期，不能把PipeBarrier ALL当所有重入唯一解法。

通信对象totalJobs=rankSize与TeamBarrier的totalJobs用途不同。前R核映射可用barrier rankSize，后R通信核映射的自定义设计可能用barrier1加显式跨设备同步，混用会漏参与者/挂死。多核并行PUT可减少串行R倍启动路径，但单核协议并非必然竞态，只是其吞吐和参与映射需要证明。

归约核集合可按连续行q/r分工且每核只写自己范围，2D DataCopyPad批量行通常减少同步/标量次数；单核归约并不自动产生写竞争，它的主要风险是性能和与其他owner的重叠。来源报告3510某strided路径约29–32行出现静默缺行，可作为复现探针/保守fallback候选，未附API硬上限，不能在知识中宣布所有2D DMA最多32行。应按真实描述符限额、版本和shape验证。

BF16→FP32原地加宽若写覆盖尚未读取输入会错，独立float源buffer容易闭合依赖；特定安全分区/处理顺序及API允许的别名条件下不能宣称所有原地转换数学上不可能。双buffer份数按实际重叠而非固定规则。MTE2_V、V_MTE2、V_MTE3、MTE3_V分别表达就绪与slot复用，Set/Wait需要同事件合同闭合并处理首credit和末drain，不要求所有正确流水都同一迭代配对。

T1成功T>1失败重点检查轮次、offset、slot复用和flag计数；staging0/旧值可能是地址不同源、过早ready或多写者，不能无证据直接加dcci。反过来没有检查真实cache访问和事件保证也不能绝对排除cache。507014/507015是故障信号，不唯一诊断。小shape精度不替代多rank、槽绕回、不同FP8入口及大shape性能验证。
