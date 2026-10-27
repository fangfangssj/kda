---
type: Guide
title: 分阶段别名与转置状态常驻
description: 给出 RmsNorm/RoPE 区域范围及迭代转置的实际循环次数，区分物理分区和分时复用。
tags:
- task.design
- topic.memory
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P42_workspace_zone_reuse.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P44_transpose_buffer_resident.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P44.yaml
  role: supporting
created_at: '2026-09-11T01:49:44Z'
updated_at: '2026-09-11T01:49:44Z'
platforms: []
---

# 分阶段别名与转置状态常驻

来源 RmsNorm/RoPE 示例在 float workspace 中用0、rows*headSize、2*rows*headSize作为三个区的元素偏移。若三个区容量都取 rows*headSize，则字节范围分别为[0,4RH)、[4RH,8RH)、[8RH,12RH)。这些是同时存在的物理分区，本身没有减少容量；真正节省来自下一阶段将不再使用的旧区重新赋予其他对象。

RmsNorm 的 square/sum 区可以在完成全部消费者后用于 RoPE cos/sin或临时值，前提是新对象的物理布局与大小容纳于该区。sum 的逻辑长度可能只有rows，并不一定要分配rows*headSize；API广播布局或scratch可能使它变大，需按真实接口推导。不同dtype的视图偏移先转字节，不能直接沿用float索引。

旧数据即使不再被当前Vector指令引用，也可能仍被异步搬运或另一核消费；复用必须等最后消费者完成。互相别名的阶段不能同时写这些区，但无关区域的独立计算仍可并行，不需要据此把整个Kernel的所有阶段串行化。所谓workspace在此例是TBuf中的UB，不能误计为GM/HBM。

迭代归一化梯度可在开始时转置到计算布局，让后续colNormGrad/rowNormGrad直接维护转置状态，最后再转回并写出。来源 buffer 基础量为 `tAlign*n*n*4` 字节，适用于所示FP32布局，还需计两次转置的临时区和其他状态；n增大时平方项可成为容量限制，不能简单缩n而改变原始数学问题，应切分或使用另一实现。

源循环 `for(j=numIters-1;j>0;--j)` 在合法非负整数范围执行max(numIters-1,0)次，是否另一次更新发生在首尾函数中需展开确认；不能按变量名声称执行了numIters次。numIters的类型和减法也需处理0与窄化。每轮行列归一化的轴应跟转置后的实际布局对应，首尾变换不改变目标算子需要的次数和顺序。

此方式省去每轮重复转置/外部搬运的前提是完整需要状态能保留，所有中间算子支持当前布局且生命周期内无必须对外发布的消费者。来源“至少3次才有收益”不是通用阈值；比较首尾变换、迭代计算、资源挤压和总时间。验证0/1/2及多次迭代、n尾部、与非转置实现逐元素对照，以及相邻阶段别名范围。
