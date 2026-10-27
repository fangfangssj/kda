---
type: Guide
title: Ascend C 多路归并的状态与 Proposal 单位
description: 区分有序段有效长度、物理间距和归并消耗计数，保证归并推进、输出容量与索引语义一致。
tags:
- task.design
- topic.compute.ordering
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-mrgsort.md
  role: primary
created_at: '2026-09-10T12:53:40Z'
updated_at: '2026-09-10T12:53:40Z'
platforms: []
---

# Ascend C 多路归并的状态与 Proposal 单位

多路归并的前提是每路输入已经按相同排序规则有序。每路有效元素数、workspace 中相邻段的物理间距、当前读取偏移和本轮搬入量应分别记录；截断有效长度不自动缩短实际分配的段间距。

将仍有数据的路压紧到调用参数时，要保留原路号与实参路号的映射。返回消耗计数必须按具体重载的返回布局映射回原路，不能无条件读取无效路位置。下一轮的偏移及剩余长度都按实际消耗更新；没有进展的循环应作为异常处理，否则可能永久重复同一批数据。

## 单位与边界

Proposal 数、载体 tensor 元素数和字节数是三种单位。若某个 FP32 value/uint32 index 表示确实每项占 8 字节，则 p 项占 8p 字节，在 float 载体上的跨度为 2p 个元素；其他表示应重新计算。

来源对 `GetSortLen`/`GetSortOffset` 的返回单位存在前后不一致，当前知识库对应 API 的描述也不能支持把所有重载统一写成 `2*n`。调用时应按实际版本声明及测试确认单位，再转换到 typed tensor 的索引，避免重复乘以 `sizeof(T)`。

Top-K 截断时，一轮归并可能产生超过剩余 K 个的结果。循环退出条件与写出上限应分别控制，目的区必须能容纳实际写入；仅在写完后检查 `outputCount>=K` 不能防止本轮越界。

每路排序、归并、提取和最终输出必须采用一致的 value/index 对应关系。排序是否稳定、相等值的 tie-break、NaN 与无穷的顺序需要实际算子语义支持，不能从“归并算法”推导所有 API 必然稳定，也不能默认任意参考排序实现稳定。

验证包括一至四路、不连续活跃路、空路、不同段长、K 边界、多轮归并、重复值及特殊值；同时检查有序性、value/index 对应、消耗总数和写出边界。
