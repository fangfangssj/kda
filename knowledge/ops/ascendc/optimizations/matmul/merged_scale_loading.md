---
type: Optimization
title: MX Matmul 合并 Scale 加载的周期、偏移与末轮释放
description: 用比数据tile更大的Scale窗口减少小搬运次数，保持量化块坐标、物理布局和所有消费者完成条件一致。
tags:
- task.optimization
- topic.data_movement
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/scale_coalescing_design.md
  role: primary
created_at: '2026-09-11T01:12:01Z'
updated_at: '2026-09-11T01:12:01Z'
platforms: []
---

# MX Matmul 合并 Scale 加载的周期、偏移与末轮释放

当A/B数据按kL1分段加载，而相邻K段的Scale较小且能连续或按合法布局批量搬运时，可让Scale一次覆盖多个数据段。优化减少的是发起次数和相关开销；数据总字节并不因此自动减少。全载相同数据去掉重复加载，与合并不同小块是两个不同机制。

## 覆盖单位与刷新周期

设数据窗口`Kd=kL1`，Scale窗口在原数据K坐标中覆盖`Ks=scaleKL1`，并在此实现中要求`r=Ks/Kd`为正整数。第j个数据段属于Scale窗口`floor(j/r)`，窗口内数据坐标偏移为`(j%r)*Kd`。每r个数据段加载一次新Scale，并在窗口最后一个消费者结束后释放。

来源用`kL1=baseK*L1_BUFFER_NUM`、`scaleKL1=baseK*SCALE_L1_BUFFER_NUM`。若两者分别为2和16倍baseK，r是8，不是16；不能再写成`scaleKL1=SCALE_L1_BUFFER_NUM*kL1`。r=1可作为不合并的合法基线，r≥2才有跨段合并意义。

K总长有限时，每份窗口覆盖`min(Ks,K-windowStart)`，所需窗口数为`ceil(K/Ks)`。尾窗口即使未消费满r轮，也须在其最后实际消费者完成后回收；不能只等`(j+1)%r==0`，否则会漏掉收尾发布或在后续复用中等待不存在的事件。

## Scale坐标不能等同原数据K坐标

L0消费的原数据坐标是`windowInnerOffset+kL0Offset`，还必须按量化group size、Scale维序和打包单位转换。A/B布局可能轴顺序不同。漏掉窗口内偏移会让多个K段重用首段Scale；漏掉group换算或物理stride会错读。该坐标应与实际A/B分片对应，而不是按Scale缓冲字节数倒猜。

简单按group计数要求窗口起点与量化group边界一致；若起点不对齐，应根据实际起止坐标求所需Scale区间及重叠，不能直接套`ceil(length/group)`。来源group_size=32、Scale单字节或多基元因子的数值只适用于相应MX布局。

## 独立资源和事件

L1预算为各A/B数据槽的真实物理字节数，加上各Scale窗口槽的真实物理字节数，再加对齐和其他常驻区。已按kL1计算的数据槽不能重复乘K覆盖倍数；扩大Scale窗口也不能挤掉仍存活的A/B槽。Scale的轮转序号按窗口推进，数据的轮转序号按数据段推进，不能共用一个计数而忽略不同周期。

Scale可写事件应在该窗口所有MTE1消费者完成后发布；就绪事件必须覆盖实际Scale加载。如果复用A/B的MTE2→MTE1通知，需要证明它在相关Scale加载之后发布并正确覆盖消费者，否则应使用独立完成链。只看到两组不同事件ID，不足以证明没有提前释放。

## 连续性与收益都要证明

内容在逻辑K轴相邻，不保证GM或L1物理地址连续。应检查起点、stride、格式、打包和尾段，不能省略这个条件。不同独立只读块可合并搬运；“累加型”并不天然无写竞争，合并搬运不能替代写权和原子/归约协议。

来源把20KB作为小搬运阈值，并报告某 `[128,8192,4096]` 案例MTE2为34.4→31.27μs、Task为39.53→36.76μs。这些是待在对应环境复现的经验，不是所有芯片统一的20KB硬件边界或5%发射开销定律。应比较相同有效数据的搬运数、每次大小、资源峰值和完整时延，而不是只追求窗口尽大。

Bias、LUT、mask或位置编码可按相同思路审视，但不沿用Scale的K/group公式。StreamK、FullLoad或SWAT也不是无条件兼容或互斥；新的每核K区间和缓存键必须重新满足这些条件。验证至少覆盖r=1/2、多窗口、K非Ks倍数、量化块尾部、不同输出tile和多次槽位复用。
