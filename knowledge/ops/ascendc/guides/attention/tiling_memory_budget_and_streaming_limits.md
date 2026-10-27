---
type: Guide
title: Attention Tiling 的完整内存预算与流式边界
description: 按物理布局和同时存活对象核算 UB、L1、L0 与 workspace。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/resources.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/shape.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/execution.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/implementation_ref.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/extension_points.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# Attention Tiling 的完整内存预算与流式边界

UB需逐项列出V1输入/输出、max/sum/rescale统计量、V2输入/输出、更新scratch、mask、量化辅助区及真实buffer份数，并逐项满足对齐。max、sum、rescale是不同状态；若每行按32B存储，slots*m*32只计一类，不能遗漏另外两类或其他副本。queue深度是生产消费容量，未必等于已分配buffer数量，应以InitBuffer和同时存活量为准。

按固定字节容量分行时chunkRows=floor(bytes/(Dpad*computeBytes))可能为0；若单行已超容量，需要列分块及可合并的行统计/更新，或明确不支持，不能宣称固定buffer覆盖任意D。仅切M仍保留D因子，未完全解除大D限制。整块m*D驻留在能容纳时是合法选择；V1/V2独立按各自物理维度预算，所谓“对称分块”不要求相同chunk数。

L1中的A1/B1是API逻辑位置，不能无证据视为完全独立物理容量或绝对端口隔离。Q/P/K/V、scale及双缓冲按真实共享资源与生命周期合计；来源复用区需要前一消费者完成。LoadData多M-fractal读取必须对应完整连续NZ布局，只准备一个fractal却以mStep>1读取会越过有效布局。尾M/N/K的有效区和padding需分别传递。

L0A预算m*k*ABytes*buffers，L0B为k*n*BBytes*buffers；L0C对S和PV取max只在两者生命周期不重叠且Fixpipe完成后复用时成立，否则应相加或采用独立分配。物理fractal与对齐后的形状才是容量依据。

workspace按核数乘slot大小仅适合前一任务所有消费者完成后可复用的区。Split-KV局部结果通常需按query/split身份保留，稀疏metadata也可能按任务索引；不能一概禁止per-task空间。运行期实际shape分配与最大shape保留空间各有取舍，graph capture等场景可能需要稳定容量。

Host预算必须与Kernel所有实际InitBuffer、局部张量/外部workspace偏移逐项相符；只复制文档中的不完整公式不能证明安全。模板已编码的模式未必需要冗余Tiling字段，但Host dispatch必须确保选中匹配变体。所有字节乘积、对齐、slot索引都需防溢出并处理0容量。
