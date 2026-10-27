---
type: Guide
title: RegBase 尾块、掩码与归约精度的联合约束
description: 把有效元素数、存储步长、mask 进度和归约中间精度联系起来，防止 padding、尾块或转换改变数学语义。
tags:
- task.design
- topic.precision
- topic.implementation.masking
- topic.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/pitfalls/precision_guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/pitfalls/common_traps.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/api/arithmetic_and_reduce.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/dev-experience/tiling_review_notes.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/reg_vector_compute/reg_data_load/load_align_continuous.md
  role: constraint
created_at: '2026-09-10T12:22:42Z'
updated_at: '2026-09-10T12:22:42Z'
platforms:
- '950'
---

# RegBase 尾块、掩码与归约精度的联合约束

## 长度的三种含义

有效元素数决定数学参与范围，补齐后的存储长度决定 UB 分配和行间步长，搬运参数还可能使用字节或数据块单位。这些值即使在对齐输入下恰好相等，也不能共用一个含义不明的 count。

CopyIn 对尾块补齐后，VF 的计算和写回仍需遵守有效范围；搬运 padding 不能替代 mask。连续无掩码 LoadAlign 还可能读取整个 VL，因此输出使用 mask 不能反向证明输入地址安全，UB 分配和该重载的读取跨度也必须成立。

使用通过引用修改计数的 `UpdateMask<T>` 重载时，循环偏移不能再基于修改后的局部计数推导本轮步长。先从当前 SDK 确认参数是否可变，再确定由 API 或外层循环推进 remaining；同一进度不能被推进两次。

## 归约与转换

跨 tile 的归约需要定义 partial 的存储位置、合并算子、累计 dtype 和有效 count。sum 的无效 lane 不应贡献数值；mean 的分母应使用有效元素数；max/min 的 padding 应按真实 dtype、输入语义和 NaN/Inf 规则选取中性值，不能固定用零。

输入、内部计算和输出 dtype 可以不同。升高归约或非线性中间计算的精度是一种候选策略；最终转换的舍入、饱和与布局仍须符合算子契约。不能把 `int32_t → float` 视为所有数值都精确可表示的转换。

Compare 生成的条件 mask 与合法访问范围 mask 承担不同职责；Select 的真假分支也必须具有一致的 lane 解释。对持续累加的寄存器，还需检查 inactive lane 是保留旧值还是清零。

## 验证边界

用单 VL、VL+1、多 VL、单核/多核尾块验证覆盖与写回；用正负值、零及契约允许的极值检验 padding 与累计精度。将融合计算同符合相同 dtype、rounding 和数值语义的参考实现比较。

本文不把 MemBase 的 repeat/count 约束套到 RegBase 的全部重载，也不规定所有 Compare 都需同一种 256 字节补齐。具体约束以对应接口为准。已有故障案例见 [跨块归约状态](../../runbooks/precision/regtensor_cross_chunk_state_should_persist_in_ub_slot.md)。
