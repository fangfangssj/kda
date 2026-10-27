---
type: Guide
title: Transpose 双轴切分的四类任务与区间契约
description: 保留双切分完整地址机制，补充零区间和混合进制边界。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_cut_two_axis.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_cut_two_axis.template
  role: supporting
created_at: '2026-09-11T06:51:44Z'
updated_at: '2026-09-11T06:51:44Z'
platforms: []
---

# Transpose 双轴切分的四类任务与区间契约

CUT_TWICE在rank≤5时把输入切轴与输出切轴映射回同一坐标系，两者独立后产生主块(Fi,Fo)、输入尾(ri,Fo)、输出尾(Fi,ro)、双尾(ri,ro)。其中ri/ro为0时对应类别不存在。outCut大于输入切轴的输出位置表示下标更内侧，来源反向文字不能代替索引关系。四类分别具有UB输入/输出shape及CopyOut长度、对齐行距。

令qi=floor(Di/Fi)、qo=floor(Do/Fo)、L=其他外侧任务数，则四类数量依次qi*qo*L、[ri>0]*qo*L、[ro>0]*qi*L、[ri>0&&ro>0]*L。Host按此顺序给全局任务区间，Kernel从全局loopidx减去此前真实存在类别的数量后做局部混合进制。尾部基址等于对应轴(D-r)*完整GM后缀积；双尾为两轴偏移之和，输入/输出各按perm求对应stride。

每核q/r分配[start,start+count-1]含尾区间，PorcessOffsetRange使用已初始化为类别完整范围的结果对象再裁剪。其包含类别时直接return是保留正确完整交集，并非漏处理；复用成未初始化通用helper会改变语义。更稳妥的通用表达为lo=max(coreStart,catStart)、hi=min(coreEnd,catEnd)、lo<=hi。源码以[0,0]兼作空哨兵、以end!=0或block0决定是否执行主类，要求Host保证主类非空、0任务核不进入、不存在其他合法单任务0类别；否则可能虚构或跳过任务。

GetLoopAddressOffsetImpl为未固定的轴收集正基数及GM步长；尾类跳过该尾切轴。数组实际5项，但DecimalToMixedBase循环上限使用8：合法编号必须在有效基数耗尽前商归0，否则可能模0或越界，不能作为任意整数转换器。每轮坐标数组重置为0，避免复用残留；源目的基数顺序必须对应同一任务坐标，permutation非法时FindPermIndex静默返回j不能当校验。

SetupLoopInfo从完整输入shape推GM stride，从输出tile推UB stride，在更内切轴之前按32B对齐并向外递推，经perm及reverse适配DMA维序。CopyOut合并完整输出轴后用blockLen、blockCount、LoopMode两层及软件最外层完成；UB物理容量要包含主/两单尾/双尾所有布局最大值。GetCopyOutParams每遇非完整轴令idx++，末尾还写loopSize[idx]，Host应证明idx<5；若所有5轴都触发分段将写第6项，不能仅以rank≤5视为充分保护。单绑定队列num1，不具备双缓冲资源。LoopMode恢复、字段范围与四类总覆盖都需验证。
