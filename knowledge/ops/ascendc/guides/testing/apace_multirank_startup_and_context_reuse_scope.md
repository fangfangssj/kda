---
type: Guide
title: APACE 多rank启动与 context 复用范围
description: APACE 多rank启动与 context 复用范围的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/host-and-testing.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/plan-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/design-template.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE 多rank启动与 context 复用范围

来源ST每rank独立进程，先选device/建stream，交换HcclRootInfo并创建通信域，再建device context，跨rank Host barrier后launch。rankId=deviceId只是单机例；真实映射可能不同。父进程须收集全部子进程失败/退出并避免只看rank0输出，任一rank缺席会让其他barrier无限等待。

来源msprof包装改变fork时序，非0rank可能在rank0尚未listen时耗尽重试。CPU钉核只影响时序，不替代显式服务就绪握手、有限重试与错误传播。端口占用需定位本任务监听与进程生命周期，按端口盲杀不是技术修复，也不能把TCP rootInfo引导链路问题归为UDMA数据丢包。资源在异步访问完成后释放，context遵循其真实engine所有权。

ctxTag缓存键表示已建链资源，不自动随shape更新。通信域/rank映射、注册地址、布局或engine发生变化而命中旧context会错误；只有shape改变且context所有资源均保持相容时，不必机械强制换tag。即使tag不变，也要保证新shape窗口容量、epoch与所有peer同步。

设备占用探测需区分设备概要与进程列表，不能把npu-smi设备行视为忙进程。编译产物指纹、实际加载库、每rank输入文件和golden绑定可发现旧构建/旧数据污染；独立端口与进程隔离有助复现。队列审批、Reviewer分工及目录归档要求留在Skill。
