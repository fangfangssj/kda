---
type: "Runbook"
title: "AscendC Muls cyclic-repeat 致分组 scale 错配"
description: "用 AscendC Muls 把 per-group scale 广播到数据时，Muls 对较短张量做 cyclic repeat（st[i % tg]）而非 block repeat，导致分组量化 scale 错位作用于数据；判别方法与机制推导。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.arithmetic", "topic.compute.quantization"]
status: stable
sources: []
created_at: "2026-08-18T08:28:16Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["cyclic_repeat", "block_repeat", "group_quantization", "scale_broadcast"]
---

# AscendC Muls cyclic-repeat 致分组 scale 错配

## 现象

在 deepseek-v4 模型 NPU 推理中替换 SwigluGroupQuant 自定义 AscendC 算子后，推理不崩溃但 decode 输出乱码（`angle` 后大量 `<｜begin▁of▁sentence｜>` 重复 token）。kernel 被正常调度（日志 `[KERNEL-REPLACE] SwigluGroupQuant kernel launched` 大量出现），环境 Ascend950PR × 4。定位到 `op_kernel/swiglu_group_quant_kernel.asc` 的 `Compute` 函数第 128 行——用 AscendC `Muls` 把 per-group scale 张量 `st` 广播乘到数据张量 `ab` 上时，`Muls` 的广播语义与分组量化的 block-broadcast 需求不一致。

## 适用场景

- **工程形态**：AscendC kernel 直调（`.asc`），用 element-wise API `Muls(dst, src, scalarOrTensor, count)` 把 per-group 缩放因子作用到分组数据上。
- **API 路径**：AscendC `Muls<T>`（vector element-wise 标量或张量广播乘）。
- **验证过的平台与 CANN 版本**：Ascend950（Ascend950PR × 4）；CANN 路径 `/home/d00657570/Ascend/cann`（版本号未记录）。
- **适用粒度**：`operator_family` —— 任何需要把 per-group/per-block scale 广播到连续数据块的 AscendC kernel（group quantization / block-wise scaling / per-head scale）。
- **路线优先级**：`recommended_fastpath` —— 当 scale 张量短于数据张量且需要 block repeat 时，必须先展开 scale 为 groupSize 个连续副本再做 Muls，不得直接对短张量调 Muls。
- **不适用或未验证场景**：scale 与数据等长（无需广播）的逐元素乘不涉及；非 AscendC element-wise API（如 SIMT 循环手写乘法）的广播由作者自己控制，不触发本判别；a2/a3 未验证（推断可迁，见 ## 失效边界）。

## 判别方法

**触发信号**（tool_observed）：用 `Muls<T>(dst, srcBig, srcSmall, count)` 把一个长度为 `tg` 的 per-group scale 张量作用到长度为 `cc = tg * groupSize_` 的数据张量上（`count = cc`）后，分组量化类算子输出语义错误（推理乱码 / scale 组错位），但 kernel 不崩溃、能被调度。修复前 `test.log` 工具结果复现 decode 乱码（`angle` 后跟大量 `<｜begin▁of▁sentence｜>` 重复 token）。

**探针**（可执行核对）：读取 kernel 的 scale 应用代码行，核对 `Muls` 两个操作数长度关系与 `count`。本 trace 中为：

```cpp
Muls<float>(yf, ab, st, cc);          // yf = swiglu * scale  (scale = 448/maxAbs)
```

（取自 `op_kernel/swiglu_group_quant_kernel.asc` `Compute` 函数第 128 行；该行经 file_read 工具读取事件触发，但 file_read 的 tool_result 被 markdown_transcript adapter 截断未捕获，行内容仅在后续 reasoning 中一致引述——故该 verbatim 行不进入 ## 验证信号，仅作判别探针引用。）

**原始观测**（机制推导，不跳步）：

设：
- `ctr` = 当前处理的行组数；
- `nGroups_` = 分组数（本例 groupSize_=128、dHalf_=2048 时 nGroups_=16；groupSize_=32 时 nGroups_=64）；
- `tg = ctr * nGroups_`（scale 张量 `st` 的元素数）；
- `cc = tg * groupSize_ = ctr * dHalf_`（数据张量 `ab` 的元素数）；
- `Muls` 的 `repeatTimes = cc`。

AscendC `Muls` 对较短操作数 `st` 做 **cyclic repeat**：对每个 `i ∈ [0, cc)`，结果为 `ab[i] * st[i % tg]`。

代入本例 `nGroups_ = 16`、`groupSize_ = 128`（`tg=16, cc=2048` 时一个 ctr 块内）：
- `st[0]` 作用于 `ab[0], ab[16], ab[32], …`（每隔 16 个元素一次）；
- `st[1]` 作用于 `ab[1], ab[17], ab[33], …`；
- 即每 16 个元素轮换一个 scale 值。

但分组量化正确语义是 **block repeat**：
- `st[0]` 应作用于 `ab[0:128]`（连续 128 个元素共用 st[0]）；
- `st[1]` 应作用于 `ab[128:256]`；
- 即每 `groupSize_=128` 个连续元素共用一个 scale 值。

cyclic（每 16 元素轮换）与 block（每 128 元素共用）的错位，使每个数据组被施加了错误的 scale 值，反量化结果错误 → 推理输出语义崩溃。

**解释**（interpretation，根因分析）：根因是 `Muls` 的 broadcast 语义（cyclic repeat of the shorter operand）与 group-quantization 的 block-broadcast 需求不匹配。正确做法是先把 `st` 的每个 scale 值展开为 `groupSize_` 个连续副本（得到长度 `cc` 的展开 scale 张量），再对两个等长张量做 `Muls`（或等价 block-repeat 广播），实现块重复语义。

## 验证信号

- 推理 decode 输出乱码（`angle` 后大量 `<｜begin▁of▁sentence｜>` 重复 token），修复前 `test.log` 工具结果复现，kernel 被正常调度但不崩溃（来源平台：950；证据等级：tool_observed）。
- 修复后重新构建的 `libcustom_ops.so` 部署到模型 `build_output/`，两处 md5 一致 = `6a2b142f2ef13e5d693e9ca3f00643be`，证明修改已编译入并部署（来源平台：950；证据等级：tool_observed）。

> 独立精度测试（修复后）据 agent 叙述通过，但测试脚本 bash 输出被 markdown_transcript adapter 截断未捕获，数值仅 agent_reported，按证据等级纪律不计入验证信号。

## 性能影响

未测量 block-repeat 展开方案相对 cyclic-repeat 的性能代价（本卡为正确性诊断，非性能路线）。若关注：展开 scale 到 groupSize 副本会增加一次等长 element-wise 写入，相对直接 Muls 多一访存；成对相对代价未在本 trace 测量。

## 预防

- 在 AscendC kernel 中用 `Muls`/`Adds` 等 element-wise API 广播 per-group/per-block 因子到连续数据块时，先核对较短操作数长度是否等于数据长度除以块大小，并明确 API 是 cyclic repeat 还是 block repeat；若 API 为 cyclic repeat，必须先展开因子为块大小个连续副本。
- 分组量化类算子的 scale 应用单元应作为独立判别点：scale 张量 shape `(rows, nGroups)` 与数据 shape `(rows, nGroups*groupSize)` 的广播方向是 block（沿 groupSize 连续复制），不是 cyclic。

## 失效边界

- 本判别的 cyclic-vs-block 结论基于本 trace kernel 的 `Muls(dst, ab, st, cc)` 调用形态（短张量 `st` 作第三参数、`count=cc`）；其他 element-wise API（如 `Muls` 以标量为参数、或 `Mul` 双张量等长情形）不触发本判别。
- a2/a3 平台未验证，推断可迁（`Muls` cyclic-repeat 语义为 AscendC element-wise API 通用行为，跨 950/a3/a2 一致），但需在目标平台复测。

## 坏实践

- 直接用 `Muls<float>(yf, ab, st, cc)` 把 per-group scale 作用到 `tg*groupSize_` 长度的数据上，假设 API 会做 block repeat。
- 仅看 kernel 是否编译通过、是否被调度执行，不核对 scale 与数据的索引对齐方式。
