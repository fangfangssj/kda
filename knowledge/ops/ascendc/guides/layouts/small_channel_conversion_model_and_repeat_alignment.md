---
type: Guide
title: 小通道重排模型与 repeat 对齐上限
description: 保留专用16通道中间布局，纠正任意perm规约和末次向上对齐。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/conversion/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/conversion/fallback/index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/conversion/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/conversion/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/conversion/fallback/script/conversion_tiling.py
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# 小通道重排模型与 repeat 对齐上限

小通道方案只在真实重排可表示连续[C,N]→[N,C]且C≤16时适用；其余轴保持顺序、stride和batch边界。fallback collapse_to_channel_flat从所有p!=i的输入维取最小值作C，再用total/C得N，这并不能证明任意perm是二矩阵转置。例如shape[2,3,4],perm[2,1,0]得到C2,N12，但三轴逆序不等于2×12转置；它也不验证perm是完整无重复的0..rank-1排列。

NCHW→NHWC通常须保留batch外层，不能不证明地址合同就把所有非C轴直接合成N。恒等perm、blockShape1或退化尺寸可能是纯拷贝，来源“TENSOR_MOVE永远不适合重排算子”过强。BatchToSpace/SpaceToBatch还可能含crop/padding与元素数变化，不能一律等同元素数守恒的Transpose。

本例UB模型tileNA*(16*C+32)来自两份FP32输入8C、两份uint8输出2C、half中间2C、uint32偏移4C和16列half转置结果32，合计16C+32字节/位置。它描述带计算/round/转换的特定流水，不是任意dtype纯转置通用预算；脚本dtype字段未参与计算。纯转置要保持输入bitpattern，不能擅自经half数值窄化。

tileNA按32元素对齐且repeat=tileNA/16≤255时，合法最大tileNA为floor(4080/32)*32=4064。源码先取min(UB上限,4080)，再把候选向上32取整，可能得到4096/repeat256；CPU例C1,N4080、1核、充足UB确认此问题。每次round后必须重新验证全部上界。

N>0、C>0、core>0、UB-reserved足够一个基本块后才可计算ceil与分核；source小于一块tileNmax0会除0，空轴/负维/非法perm也缺检查。算得tileN后还需确认ubBytes≤budget、repeat合法，tileNA与每轮有效curN分开，最后tile用精确搬运。

offset[p*C+c]=(p*16+c)*2为half转置输出[Naligned,16]的字节偏移，偏移表长度tileNA*C且最大值适配Gather宽度，C≤16。表按相同tileNA布局可跨块复用，尾块只消费curN*C。tilesPerCore=ceil(totalTiles/blockDim)时各核实际终点仍需min(totalTiles)，不能重复或越尾。
