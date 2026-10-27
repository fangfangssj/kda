---
type: Guide
title: 索引、Pooling 与采样分发合同
description: 保存合轴、索引宽度和采样分组，补齐边界及非法Fallback。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_01_mixed_precision.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_02_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_03_tilingkey_dispatch.md
  role: supporting
created_at: '2026-09-11T05:51:16Z'
updated_at: '2026-09-11T05:51:16Z'
platforms: []
---

# 索引、Pooling 与采样分发合同

| 算子 | 来源Key/策略 | 必须保持的条件 |
|---|---|---|
| LinearIndex | 1/2为INT32/INT64；11/12 dim0合轴，21/22 dim1合轴；MODE0/1/2 | 实际代码varShapeSize==THREE_DIM，D2文字称二维有歧义，按原rank与合轴后rank区分；先初始化key再相加，此处是十进制加法非位OR |
| MaskedScatterWithPosition | 100/101 BA、200/201 AB；100*pattern+wideFlag | wideFlag选择内部uint32/uint64，输入position仍可强制INT64；全部索引/乘积在所选宽度可表示 |
| SparseToDense | 1000000+10*sizeType+isValuesScalar | sizeType由outSize>MAX_INT32_NUM，scalar values用values0否则values[idx]；输出容量大并不保证所有中间坐标乘积已宽化 |
| PoolGrad NHWC | 500 mergeHWC无检查、501带检查、600 mergeWC、700 bigC、802 SIMT等 | 500..804不是连续每值都支持；tileC向下对齐16时C<16不能得0 |
| MaxPool Argmax | SmallC/BigC、SIMT NCHW/NHWC、BigKernel、Gather等 | format、kernel和C决定capability，策略名称不保证峰值 |
| Modulate | 0/1/2分别B/L/D分块 | 参数broadcast与输出归属须随切轴变化，模板继承本身不改变公式 |
| DeformableAttentionGrad | FP320/FP161 | default转FP32仅可在已验证输入集合后使用，未知dtype不可静默重解释 |

MaskedScatter数据源列FLOAT、FLOAT16、DOUBLE、UINT8、INT8、INT16、INT32、INT64、BOOL、BF16十种，mask BOOL、position INT64，x/updates同型。SparseToDense源values集合另含UINT16而无DOUBLE，不能互相套用；indices INT32/INT64、indices rank≤2、output_shape rank1、values rank≤1、default_value单元素只是部分校验，还需shape内容非负、坐标范围、稀疏项数/values长度、重复索引语义和default/output dtype一致。

64位模板参数不会自动使全部offset乘法64位，也不保证>2^32元素路径安全；先在宽域算shape乘积、检查buffer字节数与输出边界再选内部宽度。LinearIndex/Scatter的INT64→INT32转换只在范围已证明后可用，不能因索引存储类型合法就截断所有值。

PoolGrad FilterMask需将0≤h<hMax与0≤w<wMax同原有效lane取交集。链式Compare把同一mask当输入输出时须按真实ZEROING/MERGING行为确认旧false不会重新激活。无padding不单独证明外部argmax合法，NO_CHECK_RANGE仅在完整producer合同与Host验证已保障范围时使用；SIMT是线程并行模型，不能泛称退回单线程标量。

可变形注意力优化key=(embedDims/16)*1000+pointGroup，源仅embedDims16/32及group2/4/8，对应1002/1004/1008/2002/2004/2008。GroupPoints优先整除8再4再2否则1，numPoints0、剩余组和总任务数须另校验；不满足optPoint或embedDims范围走已实现通用key0，不能因整除截断生成错误特化。310P源__CCE_AICORE__200的key1专用入口是另一平台协议。

InferShape源以samplingLocations某轴<32猜Transpose，仅在两种shape域不重叠的明确ABI下可行；不能对任意合法头数/query数自动判布局。片段numHeads=INPUT_LOCAT_DIM_2等赋的是轴常量而非GetDim值，不是完整shape推导。输出dtype随value不免除locations/weights类型、rank和各轴关系检查。

Scatter源列float/half/int32/uint8/int8/BF16共6种，不是文字7种；6×2索引×2模式=24是候选数，key111/112/121等还要对应真实注册分支。C++ long的位宽取决于ABI，应使用int64_t或静态验证，不能以名字long保证INT64输入解码。
