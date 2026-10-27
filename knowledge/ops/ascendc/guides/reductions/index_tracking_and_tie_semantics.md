---
type: Guide
title: 分片极值归约的索引与并列值语义
description: 将片内索引转换到正确坐标，并在值、索引、并列值与非有限值的统一规则下合并部分结果。
tags:
- task.design
- topic.compute.reduction
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/with-index.md
  role: primary
created_at: '2026-09-10T13:09:00Z'
updated_at: '2026-09-10T13:09:00Z'
platforms: []
---

# 分片极值归约的索引与并列值语义

极值归约的状态是值与索引的配对。片内接口返回的索引通常相对于本片，合并前必须转换到算子定义的索引坐标；连续切片可加片起点，间接布局则需要显式映射，不能无条件加扁平 GM 偏移。

若契约要求返回首个极值，按索引递增顺序遍历时，遇到相等值应保留旧索引；并行分片以任意顺序合并时，应比较全局索引来解决并列。只用严格大于更新值，不能保证乱序合并仍返回首个位置。NaN、正负零和空归约应按算子定义处理，普通 LE/GE 比较不自动覆盖这些语义。

Compare/Select 可以同时更新值和索引。若 mask=1 表示保留旧状态，则两个 Select 的源顺序都必须与该极性一致；选择值和选择索引使用不同极性会构造不匹配的配对。TENSOR_SCALAR 模式只有在目标重载支持所需 dtype 与语义时，才可能省去整段索引广播。

用浮点存放索引属于需要范围证明的编码方案。FP32 可连续精确表示至 2^24 的整数；half 不能因为有较大的最大有限值就精确表示同一范围内的每个整数。数值 Cast 与按位读取是不同操作，应根据 API 的索引返回格式选择，不能将值/索引打包输出直接视为两个同类型数值。

索引变体会增加索引状态、比较 mask 和转换空间，并可能改变 reduction scratch。对齐要求必须来自选定的 Compare/Select/Reduce 重载，不能把参考中的“所有 Compare count 均需 256 字节对齐”推广到全部接口。

验证应包含跨片并列极值、极值位于最后一个有效元素、不同分片顺序、索引表示范围边界以及非有限值。检查输出 value 等于对应原始输入 index 所指的值，可以发现值与索引更新不同步，但不能替代完整极值语义验证。
