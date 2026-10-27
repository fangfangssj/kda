---
type: Guide
title: 小通道转置的 UB 预算、索引与 repeat 上限
description: 修正 FP32 经 FP16 的精度边界及 4080/32 对齐冲突。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/conversion/patterns.md
  role: primary
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# 小通道转置的 UB 预算、索引与 repeat 上限

来源路线针对C≤16的[C,N]→[N,C]，包含FP32输入转FP16、vnchwconv、Gather及uint8输出。这是带数值转换的转置路线，不能称任意FP32的无损转置；舍入、溢出、饱和和输出类型需符合算子合同。把多维轴摊平成C、N，必须证明真实物理连续性；NCHW通常需保留外层batch，不能直接假定[C,BHW]与原[B,C,H,W]同布局。

来源主要buffer按tileNA计：两份FP32输入2*C*tileNA*4、两份uint8输出2*C*tileNA、FP16中间C*tileNA*2、16通道FP16转置16*tileNA*2、uint32 Gather偏移C*tileNA*4，总计(16*C+32)*tileNA字节，C=3时每个tileNA为80字节。还需逐buffer对齐、scratch和其他常驻区；ub-reserve须先检查非负。

tileNA按32个元素对齐，同时vnchwconv repeats=tileNA/16≤255时，最大可同时满足的tileNA是4064。先alignDown(...,32)再min(...,4080)会产生不满足32对齐的4080；再向上对齐则4096导致repeat=256。上限应对齐后联合求交，而不是独立截断。尾块ceil(curN/16)的repeat还需分配相应物理区域，并避免处理padding产生有效输出。

Gather的uint32字节偏移可按(p*16+c)*2定位转置后FP16，输出逻辑序号为p*C+c；必须核对实际vnchwconv的16通道布局，不能把字节偏移当元素下标。输入GM偏移c*N+tileOffset、输出GM偏移tileOffset*C仅适用于此连续C-major到N-major模型；输出curN*C字节仅因输出为uint8，其他dtype需乘真实字节数。

C/N/core数为0、UB不足一个对齐块时需显式处理，不能发生除零或无符号下溢。按核数再次调整tileN后仍需不超过tileNMax，ceil分配可能存在空核，不证明100%核利用。偏移表由SetValue逐项生成与预生成表DMA搬入是不同实现；前者需保证Scalar写到Vector消费的依赖，后者需检查搬运完成与表生命周期。
