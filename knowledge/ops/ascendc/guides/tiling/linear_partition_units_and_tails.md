---
type: Guide
title: 线性切块的单位、核数与尾部范围
description: 统一元素数与字节数，按正长度定义核内循环，并把核数选择与资源、同步及实际开销共同验证。
tags:
- task.design
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/elewise/tiling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/elewise/patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/broadcast/onedim.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/common/core_shrink_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/common/tail_block_design.md
  role: supporting
created_at: '2026-09-10T13:12:33Z'
updated_at: '2026-09-10T13:12:33Z'
platforms: []
---

# 线性切块的单位、核数与尾部范围

对连续且逐元素独立的计算，可以把逻辑元素展平后按核、再按 tile 划分。输入输出 shape 相同本身不证明可以这样做：归约、间接访问、非连续视图或多个输出之间的依赖需要额外分析。

元素数、字节数和位数必须分别标注。若 UB 可用 B 字节，每个 tile 元素对应 d 字节的同时存活缓冲，则容量上界为 `floor(B/d)` 个元素；只有分母采用 bit 时才应把 B 乘 8。混合 dtype、额外转换区、双缓冲与固定 scratch 都应进入 d 或单独扣除的固定开销。

以 g 字节为所选对齐粒度，s 字节为元素大小，在 g 可被 s 整除时，元素对齐单位为 g/s。参考中的 512 元素、512 字节、128 字节缓存行与 256 字节 repeat 不是同一单位，也不是所有路径统一的硬要求。应分别核对搬运起点、长度和计算接口的约束。

## 有效工作范围

若某核有效区间为 `[start,end)`，tile 容量为正数 U，则每期有效长度为 `min(U,end-offset)`，仅在 offset<end 时执行。对于正长度 L，循环数为 `ceil(L/U)`，最后一期长度为 `L-(循环数-1)*U`；整除时最后一期长度为 U。L=0 应单独处理，不能套用无符号的 `loopNum-1`。

搬入 padding 的物理空间可以大于有效长度，计算和写回仍需遵守各自的有效区域。不能为解决相邻输出范围冲突就无条件改成 AtomicAdd；写入是覆盖还是累加，必须由算子语义决定。跨核同步中的空工作核是否能提前返回，也取决于参与者协议。

小工作量可尝试减少核数，以降低调度或同步成本，但“每核至少 4 KiB/8 KiB”与“每核至少 8–64 行”只是示例启发式。基于向上取整的核数公式不自动保证每个尾核都达到该下限。大工作量也不能只凭经验要求满核，应比较实际并行收益、访存竞争与资源限制。

标量输入可在接口支持时使用 tensor-scalar 运算以减少展开缓冲，语义边界见 [向量广播与标量替换](../vector/broadcast_addressing_and_scalar_substitution.md)。是否升精度应由数值契约决定，不能仅因半精度加减就固定插入 FP32 转换。

验证覆盖零工作量、不足一 tile、恰好整除、多核尾块及 dtype 分支；性能评估比较同输入、同数学语义的实际产物。修改核数后，还需重算跨核归并与 workspace 映射。
