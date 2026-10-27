---
type: Guide
title: UB 分区字节核算与复制别名
description: 保存索引缓存及分区，纠正分配数与容量混淆。
tags:
- task.optimization
- topic.memory
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_08_ub_memory_mgmt.md
  role: primary
created_at: '2026-09-11T05:55:38Z'
updated_at: '2026-09-11T05:55:38Z'
platforms: []
---

# UB 分区字节核算与复制别名

AdaptiveAvgPool IndexBuffer以六个数组缓存每输出窗口D/H/W起止索引，按indexBufLen批次生成并在ReduceMean复用。必须绑定相同输入/输出shape和当前输出offset，数组至少容纳本批len且宽度可表示，整除/ceil窗口公式不能因缓存改写。生成阶段仍做地址计算，收益来自复用次数，并非所有除法凭空消失。

AdamW用VAR/M/VI/MAX等order*numPerLoop划分一份LocalTensor，是分区而非同byte复用；混合类型须分别换算byte，最大order对应区也在allocation内。固定MAX_ELEMENTS_PER_LOOP换动态numPerLoop便于调tile，不自动减少相同状态的存储量。

RMSQuant calc_buf分配3*num_col_align_f32*4+32B，gamma/sqx/work各占一份，和三份同大小buffer容量基本相同，不能称从3变1就节省三倍。只有不重叠的生命周期共享同一byte才实际节省峰值，仍需API别名支持及producer完成。额外32B用途和全部队列/scratch需计入。

Foreach BF16 maxCastDataCount=inputsTensorUbSize/4，来源COPY_SPACE_MULTIPLE9扩大另一个总Tensor容量；9是该框架特定布局因子，不能把任意UB可用量乘9分配。inputsTensorUbSize单位byte，maxDataCount元素，cast各输入/结果区及每队列份数在Host/Kernel同一预算，不能只验证单个缓冲。

Axpy先修改srcLocal1，再在dst物理起点不同才DataCopy，是UB内原地/复制选择；不等于GM输入输出地址相同，也不自动省GM写回。相同起点不能证明长度/类型相同，部分重叠起点不同仍需接口明确允许。无数值转换的合法copy不引入舍入，省copy不能声称提高精度；源数据若后续仍需旧值则不能就地Axpy。

Gather按availableUb=ubSize-reserved前须检查ubSize≥reserved，unsigned相减不可回绕。Gather/Transpose互斥阶段可以为共享输入/输出池取max，若同时存活则应求和；再计idx两个INT32区、sort/transpose workspace和对齐。cacheline元素数=cachelineBytes/dtypeBytes只是单位换算，不证明每个offset也对齐。

LinearIndex示例max_ub=ubSizePlatForm/max_ub*max_ub/BUFFER_NUM把旧max_ub当对齐粒度/约束再覆盖，必须先明确旧值正数及实际意义；maxSize=.../indicesSize仅在其他缓冲已纳入分母/预留时足够。UB超预算通常不是自动“spill到GM”的受保证行为，不能依赖溢出后有正确fallback。

SoftmaxGrad用GetSoftMaxGradMaxTmpSize取得scratch，再按输入/输出dtype、额外FP32输入、mask与paddedHeadDim算每行候选成本。dataType枚举不能直接与SIZE_2字节常量比较，需真实变量含义；scratch未必线性随行数，按1行预算得maxLinePerLoop后仍需用最终shape重新求API tiling与scratch，零行/最小不可放要处理。多阶段scratch复用须取最大且生命周期不重叠。
