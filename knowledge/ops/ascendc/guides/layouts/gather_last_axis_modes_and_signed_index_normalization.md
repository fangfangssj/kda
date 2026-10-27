---
type: Guide
title: Gather 尾轴模式与有符号索引规范化
description: 保留多行、单行和稀疏载入，限定负索引与取模。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_13_special_algorithms.md
  role: primary
created_at: '2026-09-11T06:02:18Z'
updated_at: '2026-09-11T06:02:18Z'
platforms: []
---

# Gather 尾轴模式与有符号索引规范化

LastDim可在完整x/index行及全部scratch能放UB时多行处理；否则单行、x轴切片或仅加载实际索引的Scalar路径。IndexAxisSizeEqualOne源还要求batchProcess且specialDataMove0，并非index最后轴1就可无条件走相同搬运。DoScalarMode在其他选择后执行时需明确优先级，不能留下互相矛盾的标志。

当x行远大于index行，直接按索引加载可能节省载入全x的流量，代价是随机读与Scalar开销；按实际索引分布和目标缓存测量。多行模式仍须逐行index基址，不能把上一行x或mask复用到不同输入。

负索引是否允许由该算子接口决定，不能仅因Python有负下标就声称所有PyTorch/TensorFlow Gather都接受。若合法域为[-D,D)，可在i<0时加D一次；源Scalar(i+D)%D在D>0且类型/范围合适时可覆盖此域，但也可能把非法大正值回绕成合法，i+D溢出或unsigned参与会改变负值行为，应先校验。

Vector ShiftRight提符号时须是有符号算术右移得到0或-1，乘-D再加原i才把负值加D；逻辑右移得到1则会反向减D。RIGHT_SHIFT_LEN须对应32/64位，D强转int32前验证，不能对大轴或已截断INT64一律套用。处理后仍需验证0≤idx<D，空轴没有合法索引。

clamp非法索引到边界与负索引规范化/拒绝是不同合同，不以“向量化更高效”名义改变语义。Transpose与LastDim路径应对同一合法索引集返回相同值及layout，并覆盖-1、-D、0、D-1及非法边界。
