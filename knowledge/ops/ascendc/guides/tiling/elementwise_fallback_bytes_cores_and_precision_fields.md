---
type: Guide
title: Elementwise fallback 字节、核数与精度字段
description: 提炼脚本分块模型，保留空输入、容量强行回退和dtype别名反例。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/elewise/fallback/tiling-fields.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/elewise/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/elewise/fallback/script/elewise_tiling.py
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# Elementwise fallback 字节、核数与精度字段

相同shape且无跨元素依赖可扁平为dim0，非连续stride仍需布局合同。参考按ceil(dim0*bits/32768)估核数、最多platformCores，blockFormer=AlignUp(ceil(dim0/C),512)，blockNum=ceil(dim0/blockFormer)，blockTail=dim0-(blockNum-1)*blockFormer。32768bit为4KiB，但ceil并不保证每核至少4KiB，最后块尤其可能很短。512元素和256B UB对齐是此策略参数。

core_num估计与对齐后的block_num可能不同，实际launch和Kernel越界guard须一致。脚本dim0=0时blockFormer0导致除0，core_num0又被max(...,1)强制变1，负shape/未知dtype未拒绝；标量空shape乘积1与含零轴Tensor不同。Python大整数到真实TilingData/地址类型需最终范围验证。

脚本按doubleBuffer2计算每元素字节2*buffer_num*elemBytes+K*4，容量floor((UB-extra)/divisor)后向下256/elemBytes对齐。文档却在分母已用字节时再乘8且漏队列双份，不可混用。K是同时活跃FP32区域份数，实际input/output/scratch不同dtype与别名可使此简化系数不适用。

当算得ubFormer≤0时脚本强行设为一个对齐块，会让不足容量变成可用：UB32B、FP16 add默认三双队列+两FP32区域返回128元素，模型占2560B。应报告资源不足或选其他合法路径，不能向上兜底破坏预算。完整块和末块分别保存loop/tail，整除最后tail=ubFormer，零任务单独处理。

check_fp32_cast仅对字符串fp16/bf16的Add/Sub且input_same_magnitude=false启用两份FP32区，dtype映射同时接受float16/bfloat16而该分支不接受别名，产生不一致。统一规范dtype/operation拼写后按实际数值合同选择；相同量级不保证无消减，单次低精度输入加法最后还写回低精度时，升精度也不自动消除舍入造成的小量丢失。

to_dict只输出部分配置，遗漏k_extra_bufs、buffer_num、extra_size等不代表Kernel可以不知真实资源合同。源码中的UB248KiB/64核是默认测试配置，不能声明对应所有平台；本次CPU反例验证公式与分支，未运行NPU。
