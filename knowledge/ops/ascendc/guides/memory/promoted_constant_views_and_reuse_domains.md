---
type: Guide
title: 常驻参数的扩宽视图与复用范围
description: 给出 gamma 后半区暂存的字节范围，区分原始、转换和广播结果的实际生命期。
tags:
- task.optimization
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P34_weight_load_once_reuse.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P41_gamma_scale_resident.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P34.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P41.yaml
  role: supporting
created_at: '2026-09-11T01:49:44Z'
updated_at: '2026-09-11T01:49:44Z'
platforms: []
---

# 常驻参数的扩宽视图与复用范围

weight/gamma/scale/offset 跨 B×S 或 ubLoop 不变时，可在该复用域开始处搬入，必要时转换或广播，后续复用。变更 batch、head、group、参数版本、layout 或数值转换要求时，应重新判定是否仍是不变量；并非任何名为 Process 的循环都共享同一参数。常驻数据也可以保持原始 dtype，不强制转 FP32。

来源 gamma 示例先分配 L 个 float 的 4L 字节区，再建立 KV_DTYPE 视图并从第 L 个窄元素开始加载。若 KV_DTYPE 为两字节，窄输入占 [2L,4L)，FP32 输出占 [0,4L)，二者重叠；容量恰好能容纳两者不代表 Cast 的重叠形式被支持。若 KV_DTYPE 为四字节，源会落在 [4L,8L)，已经超出这份分配，不能对所有模板类型复用同一公式。源起点2L等也需满足实际搬运对齐，尾部 padding 可扩大真实访问范围。

ReinterpretCast 只改变视图，不完成数值转换或扩容。使用共享存储的扩宽 Cast 之前必须确认对应重载、别名及执行顺序；无明确保证时使用已预算的独立源/目的区。来源声称“约2KiB”仅适用于具体 L，普通分配量应按对齐后的 4L 等真实布局计算。

原始参数、转换结果、广播结果不一定整个循环都同时存活。可以在最后消费者结束后回收原始暂存，或按可证明的生命期共享区间；若确实三者同时在用，则逐份计入 UB。广播也不只有单维一种合法形式，需按具体 shape 与接口支持选择，不能依据 source 的一维示例禁用其他实现。

来源 weight 示例分别写 weightLocal/localW0FP32/localW1FP32，却把 weightFp32 传给 Compute，没有建立全部句柄对应关系。实施时须追到实际生产地址、转换长度和使用值，不能因变量名相似认定已经初始化。所有消费者结束后才回收长生命期的队列句柄；消除循环内重复加载的收益还要减去初始化成本及主 tile 缩小的损失。

验证不同参数长度、两字节/四字节源、尾部对齐、跨复用域切换及零循环。比较转换前后结果与总 UB 峰值，不能只用少了 DataCopy/Cast 的调用数确认正确或更快。
