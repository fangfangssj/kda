---
type: Guide
title: MC2 fallback 尾块大小与候选搜索边界
description: 保存整数切分启发式，修正短块比长块大和不足16行的情况。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/mc2/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/mc2/fallback/fallback-algorithm.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/mc2/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/mc2/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/mc2/fallback/script/mc2_tiling.py
  role: supporting
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# MC2 fallback 尾块大小与候选搜索边界

给定M/N、baseM/baseN和C，参考先寻找M的约数作为等长通信块，优先baseM整倍且每块MN tile足以用满核，再放宽到16行对齐，最后用长块+至多一个尾块或降低用核。约数用整数isqrt枚举，正数ceil使用整数运算避免浮点对大整数失真。最大切分数并不保证最快，通信header、launch、tiling切换和尾块效率均可能抵消重叠收益。

设nBlocks=ceil(N/baseN)，要求ceil(longM/baseM)*nBlocks≥C，则q=ceil(C/nBlocks)。source取minLong=q*baseM；当只要求16行对齐且baseM为16倍数时，达到q个M块的最小长度可为(q-1)*baseM+16，并不必q*baseM。沿用原minLong会漏掉合法较小块。所谓full utilization只是几何上可用核数，不是MAC效率或每波负载均衡。

非整除fallback取longSize按16对齐、longCount=floor(M/longSize)、short=M-longCount*longSize。为避免0<short<16，脚本在longCount>1时减少长块数并把移除部分加给short：CPU例M257,N4096,baseM/N128,C8给long128*1+short129。总覆盖成立，但“短块”已比长块长，不再等于M%long；独立short tiling及容量不可沿用较小尾块假设。

M129同配置得到long128+short1，因longCount=1停止调整，仍违反源声称最小16行。正确策略可允许真实短尾并用合法尾Kernel，或重平衡所有块、退回单块；不能既宣称16下界又返回1。M<baseM时单块M也可能非16对齐，alignGranularity16字段应区分候选步长与有效数据长度。

CLI仅检查部分正数，直接Python入口未全面验证M/N/core/base正、baseM的对齐或容量；非法base还可能使longSize>M、longCount0。分支B用核≥0.8便接受，再尝试其他路线，因此结果最多是此分阶段策略的选择，不是全可行域的词典序或全局性能最优；还应计算尾块用核与真实base随M变化。

每候选验证所有块长度正、总和M、offset连续、前后短块顺序一致，long/short/local实际Matmul tiling和launch参与者均有效。M4096,N4096取128×32或M2048,N256取512×4仅是几何例子，不能推断实测加速。通信类型、K、dtype、scale和拓扑未在该简化模型充分描述，应在实际collective能力合同中补齐。
