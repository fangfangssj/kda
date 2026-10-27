---
type: Guide
title: 持久 KV NZ 布局与消费端合同
description: 区分真实预转换、L1/L0格式及页缓存地址，保留未对齐维度和来源性能案例。
tags:
- task.design
- topic.kv_cache
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P60_kv_nz_format_optimization.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P60_kv_nz_format_optimization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P60.yaml
  role: supporting
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# 持久 KV NZ 布局与消费端合同

预先将KV按消费路径需要的NZ物理布局存储，可以减少每次使用时重复ND→NZ处理。前提是生产端确实完成重排，页表与更新路径按同一布局定位，消费端理解相同的分形顺序、dtype和padding。NZ不能仅称为Channel-first，也不等于任意Cube L0A/L0B的具体输入排布；从L1到L0的合法Load路径仍需存在。

来源用[N,D/16,S/16,16,16]描述一种16分形KV布局，N/S符号在不同段落含义不一致。该形式需先明确head/batch、序列和通道轴及内部16×16顺序，不能直接作为所有K/V共享的ABI。逻辑D/S不对齐时应按物理ceil分块分配并维护有效范围，而非整除丢弃尾部；K和V还可能有不同通道宽度和矩阵转置方向。

只有n*d等于实际连续物理payload且布局正确时，源直接DataCopy(n*d)才覆盖所需内容；NZ有padding或所取tile非连续时，要按物理跨度选片段。源把GM参数写成LocalTensor、用未确认的Nd2nzParams/srcND/dstNZ及LoadB辅助接口，属于不完整示意，不能宣称可编译。仅把ND连续复制到L1、再给LoadB写一组NZ形状，不会自动证明执行了格式转换或“零开销对齐”。

512字节对齐是来源无平台边界的要求，不是全部DataCopy的通用正确性门槛；实际基址、块长及格式加载约束按具体接口检查。ND与NZ可以在明确转换边界共存，不必修改所有无关算子；但所有直接读写同一KV缓存的参与者必须遵守该缓存格式。新增转换与cache append代价应计入完整链路。

来源报告无另一配置叠加时250μs→213μs（时延下降14.8%），另组210μs→178μs（下降约15.24%）；未提供本批完整输入、硬件和原始采集，不能当作已复现收益，也不能据“双页表”标签推断与L2的因果关系。验证页边界、不同K/V维度、append更新、非整16尾部和与ND参考结果一致后，再比较包含准备与转换的总耗时。
