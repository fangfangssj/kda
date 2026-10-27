---
type: Guide
title: Blaze 本地 MX 模板的 L1 尺度窗口与事件
description: Blaze 本地 MX 模板的 L1 尺度窗口与事件的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_mmad_swat.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/utils/constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/policy/dispatch_policy.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/copy_scale_l1_to_l0a.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/copy_scale_l1_to_l0b.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/tile_mmad_mx.h
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze 本地 MX 模板的 L1 尺度窗口与事件

随附本地NO_FULL_LOAD_MODE、stages2实现把A/B数据各双buffer及scaleA/B各双buffer放L1两半。FP4 sizeShift1时A一槽=baseM*Align(kL1,64)/2，B一槽=baseN*kL1/2；FP8不除2，scale一槽分别baseM/baseN*2*ceil(scaleKL1/64)。当前每半至少A+B+scaleA+scaleB，保留对齐/其他资源，不能仅用原始K长度估尾填充空间。混合A/B打包类型时它用AType决定两边sizeShift，不能无条件支持不同位宽。

scaleL1Window=scaleKL1/kL1要求正整数比；AB循环计数按数据段推进，scaleLoopCnt按窗口推进。窗口首段读scale，内部偏移scaleWindowIter*2*ceil(kL1/64)，再按kL0Offset/64进入硬件MX拷贝。仅把Scale窗口增大、不改两级K偏移会重复读首组scale；尾窗口即使没满也要释放。

构造预置MTE1_MTE2 ids0..5和M_MTE1 ids0/1，并SetMMLayoutTransform(true)；每data槽先等复用，再搬GM→L1，MTE2_MTE1后L1→L0，MTE1_M后MMAD。scale专用复用ids4/5，加载后借同MTE2队列后续data ready覆盖，需确认实际队列順序。L1释放只需MTE1不再读该L1，不必等全部MMAD结束；源注释说“每MMAD完成才释放”比实际依赖更强。

析构等待所有预置资源信用并关闭layout transform，异常退出/少发/多发会破坏drain。L0 pingpong ids0/1等前一MMAD不再读再复用，同一数值ID在不同HardEvent域不是必冲突。Scale L0指针数值与数据L0相同涉及MX专用地址解释，不应据数值重叠断言普通内存别名；需匹配intrinsic合同。
