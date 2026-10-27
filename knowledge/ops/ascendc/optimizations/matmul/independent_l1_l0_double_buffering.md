---
type: Optimization
title: Matmul 的 L1、L0 与 Scale 独立双缓冲
description: 区分 K 覆盖倍数与物理槽数，按三层消费者完成条件组织事件，并用发射时间线判断实际重叠。
tags:
- task.optimization
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/pingpong_design.md
  role: primary
created_at: '2026-09-11T01:12:01Z'
updated_at: '2026-09-11T01:12:01Z'
platforms: []
---

# Matmul 的 L1、L0 与 Scale 独立双缓冲

L1双缓冲允许MTE2写下一块时，MTE1读取当前块；L0A/L0B双缓冲允许MTE1写下一块时，Cube消费当前块。两层槽位可独立轮转，Scale还可具有更长的复用周期。L0C是否双缓冲则取决于累加和写出安排，不能从L1用了两槽推断所有层都应相同。

## 覆盖倍数不等于槽数

来源用 `kL1=baseK*L1_BUFFER_NUM` 表示一份L1数据覆盖多少K，并用 `PINGPONG_NUM` 或 `l1BufNum` 表示物理槽数。这些字段的名称容易混淆。每份数据字节若已经含完整kL1，再乘一次K覆盖倍数会重复计费；若只计算baseK，再漏乘也会低估。

在来源的FP4实例中，baseM=baseN=256、kL1=768，假定每32个数据共享1字节scale且布局无额外补齐：A/B各98304B，scaleA/B各6144B，每槽208896B，两槽417792B；kL1=1024时两槽557056B，超过该例524288B预算。这个算例说明K覆盖可以取3而非2的幂，但不证明3对所有dtype/Scale布局最优。选择值须追加真实对齐、Scale打包和保留区，且结合流水开销测量，不能只追求L1占满。

## 每层分别建立两个方向的依赖

| 缓冲 | 数据就绪方向 | 覆盖前需等待 |
|---|---|---|
| L1 A/B及其实际关联Scale | MTE2→MTE1 | MTE1已读完该槽 |
| L0A/L0B及量化输入 | MTE1→Cube(M) | Cube已用完该槽 |
| L0C结果 | Cube(M)→Fixpipe | 对应Fixpipe消费者已完成，再允许重用/覆盖 |

在支持所列HardEvent的实现中，以上分别对应MTE2_MTE1/MTE1_MTE2、MTE1_M/M_MTE1、M_FIX/FIX_M。是否存在某方向、事件ID数量、参与单元和写出通路须按平台核对；L0C→GM不能统一写成“FIXP+MTE3”。首次使用空槽、最后一次消费和对象复用均须与具体事件协议配套，不按出现Set/Wait的文本次数替代状态判断。

`index & (num-1)`仅在num为正的2次幂时等同取模。固定两槽时可以用`index & 1`；K覆盖倍数为3不意味着槽索引也能采用`&2`。L0A和L0B必须分别按实际容量推导半区偏移，不能假设两者永远相同。

## 双地址不能保证流水重叠

SetFlag和WaitFlag在源代码中紧邻，不足以证明整个多迭代时间线串行；两者分开也不自动证明安全或加速。关键是另一槽的独立生产能否在当前消费者等待之前发射，以及复用是否仍受正确依赖保护。应按任务和槽标出GM→L1、L1→L0、MMAD和写出各阶段的发射、完成与阻塞。

来源给出42.4k→30.2k cycles的观察并将差距归于紧邻事件，但缺少完整发射轨迹，不能唯一归因，也不能把50%–60%总时延作为目标保证。验证包括单K段、奇偶段数、多外层tile、不同Scale周期、L1/L0C资源边界，并检查改动的源文件确实进入当前构建产物。
