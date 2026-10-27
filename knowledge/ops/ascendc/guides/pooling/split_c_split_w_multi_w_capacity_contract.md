---
type: Guide
title: 池化 Split-C、Split-W 与 Multi-W 的容量和分发合同
description: 依据实际同时存活的缓冲选择池化分块，保留 dtype、尾块、窗口和 Host/Device 分发的一致性。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/assets/rules/special_rules/R_POOLING_MULTI_TILING/R_POOLING_MULTI_TILING.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/assets/rules/special_rules/R_POOLING_MULTI_TILING/code_snippets/case1/good_code/good_code.md
  role: supporting
created_at: '2026-09-14T04:19:37Z'
platforms: []
updated_at: '2026-09-14T04:19:37Z'
---

# 池化 Split-C、Split-W 与 Multi-W 的容量和分发合同

池化中 C 方向连续且按 C 向量化时，可以按容量选择三类候选：Split-C 将一次窗口的通道拆成若干块；Split-W 保留完整通道，但分段处理一个输出窗口需要的 W 输入；Multi-W 在一个批次维护多个输出窗口的状态。策略名不是最优性的证明，原生布局、窗口大小、重叠程度和实际复用决定成本。

| 策略 | 核心容量问题 | 必须保留的语义 |
|---|---|---|
| Split-C | 一次可处理的对齐通道数受输入、输出和累加状态共同限制 | 通道偏移、有效 C 尾部及每个输出通道恰好一次 |
| Split-W | 完整 C 可放下，但单窗口的输入宽度不能一次容纳 | 跨输入片段累加同一个窗口，完成后才归一化或发布 |
| Multi-W | 多份窗口输入和输出状态同时可放下 | 各窗口自己的起止索引、累加状态、除数及窗口批次尾部 |

来源以 s 表示输入/输出元素字节数、c 表示是否另需 FP32 Cast 缓冲、A 表示对齐元素数，候选 tileLen 为 floor(U / (2*s + 4*(1+c)) / A)*A，U 必须是这些缓冲实际可用的 UB 字节。还存在 IndexBuffer、队列槽、临时空间或固定常驻对象时，应先扣除或纳入账本；不能把总 UB 直接当净容量。来源的 Split-C 分支使用 `2*alignC > tileLen`，它是该实现的选择条件，不等同于“完整 C 必然放不下”。

这份示例里 FP32 的 s=4,c=0，FP16/BF16 的 s=2,c=1，容量模型的每元素总字节均为 12。因而“输入 dtype 减半就使可用 tile 翻倍”的解释不成立；低精度新增的升精度状态会抵消节省。若真实实现使用不同的缓冲复用关系，应根据实际生命周期重算，而非固定套 12。

Split-W 的 inputTileNum 与 Multi-W 的 windowWNum 先减固定开销再做除法，计算前需证明减数不大于被减数，防止无符号下溢产生极大的伪容量。dimC、outW、对齐数和分母为零应按空输出或非法输入合同单独处理；tileLen/windowWNum 为零不能进入零步长循环。向上对齐、shape 乘积和窗口界限运算使用足够宽且可表示的整数。示例仅用形状估算 maxWindowWLength；实际窗口仍须按算子定义逐个计算并核验估计是上界。

固定示例用 `mode*10+dtypeKey` 编码 3×3 组合：mode 为 1/2/3，dtypeKey 为 BF16=0、FP16=1、FP32=2。这不是跨算子的通用编码。Host 写入的 key、字段初始化、所有 Kernel 模板分支和 fallback 必须一致；片段中计算出的局部 inputTileNum 还需实际写到消费者读取的 TilingData 字段，不能由存在同名字段推定已赋值。核数分配与 UB 策略是两个决策，按输出点均分也不保证不同窗口的计算时间相同。

比较不同策略需覆盖大/小 C、输入与输出 W 比例、非对齐通道、跨片窗口、窗口批次余数、空输出和各 dtype。平均池化必须保持有效计数/填充计数的除数合同；最大池化另需保持 padding 中性值、NaN 和 index/tie 合同。测量包括索引准备、Cast、额外暂存和总时延；多策略不保证固定 20–50% 收益，Tiling Key 特化也不等于没有任何分发开销。

相关容量细节见 [UB 分区与 IndexBuffer](../memory/ub_partition_byte_accounting_and_copy_aliases.md)，行地址见[原生布局](native_layout_and_conversion_cost.md)。本卡来自固定说明及代码片段的静态分析，未宣称完整算子在 SDK/NPU 上验证通过。
