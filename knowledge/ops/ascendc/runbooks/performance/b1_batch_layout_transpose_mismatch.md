---
type: "Runbook"
title: B=1 批量化公式与 GM 布局互为转置不可直搬
description: "B==1 批量化优化的设计偏移公式 (a/K)*C*K+c*K+b 假设 (C,K) 行序，而 B==1 时实际 GM 布局为 (K,C) 行序，二者互为转置；直接按公式实现会读错数据。DataCopyPad strided gather 与 strided-repeat 指令均无法廉价聚合列向量，唯一可行路线是 TransDataTo5HD 前后转置；本轮保持 isB1=0 走 padded 路径。"
tags: [task.optimization, topic.tiling, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# B=1 批量化公式与 GM 布局互为转置不可直搬

## 现象

bench_no_harness_cummin（cummin 前缀最小值，布局参数 A×C×B，B 为归约维长度）性能优化任务含"B=1 批量化优化"：当 B==1 且 A>1 时，把 K 行（K 为向量宽度：fp16=128、fp32/int32=64）打包成 (C, K) 矩阵，让每条向量指令的有效 lane 占比从 1/K 提到接近 1。实现前核对设计文档 `docs/DESIGN.md` §3.1 给出的批量化 GM 偏移公式：

```
element(a, c, b) = (a / K) * C * K + c * K + b     // b ∈ [0, K)
```

该公式要求批量化后数据呈 (effectiveA, C, K) 行序连续——对固定 (ea, c)，K 个 lane 的元素在 GM 中连续，可作单个 DataCopyPad 块加载。但把设计文档自己的通用偏移公式 `a * C * B + c * B + b` 代入 B==1（此时逻辑 a 对应物理 ea*K+eb）：

```
实际 offset(a=ea*K+eb, c) = (ea*K + eb) * C + c = ea*K*C + eb*C + c
```

即每个 batch 的 K×C 元素块在 GM 中是 **(K, C) 行序**（lane 为慢轴、c 为快轴），与公式要求的 **(C, K) 行序** 互为转置；二者仅在 K=C=1 时相等。也就是说：设计公式描述的是转置后的目标布局而非实际 GM 布局，设计文档未涵盖达到该布局所需的转置步骤。结论：**直接按设计公式实现 Kernel 读取会因 offset 不匹配读到错误数据，ST 必然失败**（此为公式推导结论，本轮未实际写错实现去触发；功能侧未受损——B==1 一直走 FullColumn padded 路径，全量 ST 通过）。

## 适用场景

- 工程形态：AscendC kernel 的向量化批量化优化设计（把逐元素/低向量利用率路径改造成按向量块处理），以及 GM 布局与 tiling 偏移公式的核对。
- API 路径：DataCopyPad（GM→UB 分块搬运）、strided-repeat 向量指令（Compare/Select/Min 的 `BinaryRepeatParams`）、`TransDataTo5HD`（硬件转置）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（arch35，K：fp16=128 / fp32、int32=64）。
- 适用粒度：operator_family——同一"沿最内维归约 + B==1 边界"的算子族（cummin/cummax 类）直接适用；一般向量化批量化设计按布局核对方法迁移。
- 路线优先级：本卡记载的"按设计公式直接 DataCopyPad 块加载"路线 **banned**（布局不匹配）；可行替代路线（TransDataTo5HD 前后转置）为 safety_fallback 候选——功能正确但引入两次转置，相对成本：相对现有 FullColumn padded 路径需额外 2 次 TransDataTo5HD 全量数据转置（未实现、未测量，只有机制推断）。
- 明确不适用 / 未验证场景：B>1 的常规布局（不受本转置问题影响）；TransDataTo5HD 对 int32/bf16 的 dtype 支持需编译验证；尾 batch（A 不整除 K）与尾 C 片的边界处理未实现。

## 根因

（单句归因）设计文档的批量化偏移公式假设了 (C, K) 行序的转置后布局，而 B==1 时 GM 实际布局是 (K, C) 行序，公式与物理布局不一致，属设计缺陷。

机制推导（纪律：点破算式，不跳步）：

1. B==1 时物理布局由通用公式 `offset = a*C*B + c*B + b`（B=1）给出：`offset(a, c) = a*C + c`——c 是最内（最快）维。
2. 批量化把逻辑行 a=ea*K+eb 分组后，第 ea 组内 K 个物理行在 GM 里仍是逐行排列：`offset(ea*K+eb, c) = ea*K*C + eb*C + c`——固定 c 看 b（=eb）方向，步长是 C（跨整行），不是 1。
3. 设计公式要求的方向是：固定 (ea, c) 沿 b 步长为 1（`... + c*K + b`）——只有数据已被转置成 (C, K) 才成立。
4. 因此"直接块加载"读到的 offset 与实际存放位置错位（每个 b 索引偏了 eb*(C-1) 量级），除非先转置。

不可行的廉价聚合机制（逐一排除，依据为 API 语义/头文件事实）：

- **DataCopyPad strided gather**（blockCount=K、blockLen=1 元素）：UB 目标侧 stride 单位是 32 字节数据块（见[DataCopy blockLen 以 32B 块计非字节卡](../precision/datacopyparams_block_len_32b_block_units.md)），1 元素的 block 会被对齐到 32B 槽位，无法聚合出紧凑的连续 K 向量（strided gather 的另一失效形态见[逐元素 DataCopyPad 做 strided gather 输出全零卡](../precision/datacopypad_strided_gather.md)）。
- **strided-repeat 向量指令**（Compare/Select/Min 带 `BinaryRepeatParams`）：BlkStride/RepStride 以 16 元素（fp16）/8 元素（fp32/int32）块为粒度跨步，做不到"逐元素跨 C 行取列"。
- **GetValue/SetValue 逐元素**：生产代码禁用（效率极低）。

唯一可行机制：`TransDataTo5HD` 硬件转置（[16, N]→[N, 16]，16 行块）。

## 解决配方

本轮的**处理决策**（非代码修复，负知识卡）：

1. tiling 侧保持 `isB1 = 0`——B==1 经 FullColumn padded 路径处理（tileBAlign 对齐到 128/64，1 个有效元素 + padding），功能完全正确（全量 ST 通过），仅向量化效率未优化（有效元素占比 1/K）。
2. 把设计缺陷、机制排除过程与正确实现路径写入 issue 文档 `issues/issue_20260715_b1_batch_layout_mismatch_0.md`，供后续设计审批。

tiling 代码中的决策落点（op_host/arch35/bench_no_harness_cummin_tiling.cpp，verbatim 注释段）：

```cpp
    // 10. B=1 批量化检测
    // 设计文档 §3.1 给出的批量化偏移公式 element(a,c,b) = (a/K)*C*K + c*K + b
    // 假设批量化后数据为 (C, K) 行序连续布局; 但实际 B==1 时 GM 布局为 (A, C) 行序
    // (C 沿最内维连续), 即 (K, C) 行序, 二者互为转置, 公式与实际布局不一致。
    // 正确实现需在 UB 内做转置 (TransDataTo5HD), 设计方案未涵盖该步骤, 属设计缺陷,
    // 详见 issues/issue_20260715_b1_batch_layout_mismatch_0.md。
    // 当前保持 isB1=0: B==1 经 FullColumn padded 路径处理 (功能正确, 仅未优化向量化效率)。
    uint32_t isB1 = 0;
    uint64_t effectiveA = A;
    uint64_t effectiveB = B;
```

正确实现路径（issue 中给出的待审批方案，**本轮未实现**）：前向转置 + 复用 FullColumn cummin + 反向转置——C 按 Ctile（16 的倍数，如 128）分块、跨片 carry-over 串接；加载 K 个 lane 的 Ctile 个 c 元素（每 lane 连续）后用 TransDataTo5HD 按 16 lane 一组转置（dstRepStride=K 使 8/4 组写入同一 [Ctile, K] 缓冲）；转置后每行 c 含 K 个连续 lane 元素，直接复用现有 FullColumn 路径（A=1, C=Ctile, B=K）；values（T 类型）与 indices（float）各反向转置一次，indices 随后 Cast float→int64；按 [K, Ctile] 行序 2D DataCopyPad 写回 GM；边界：尾 batch 用 row0 填充转置 srcList 仅写出 effectiveK 行、尾 C 片回退 padded 路径、bf16 可先 Cast 到 float 再转置。

## 设计决策

- 为什么不按设计公式硬写：偏移错位是确定性的（上文算式），写出来必然读错数据，ST 必挂——不值得用一次必然失败的实现去"实证"。
- 为什么不选 strided gather / strided-repeat：见根因中的机制排除（32B 块粒度与 16/8 元素块粒度都做不到逐元素跨行取列）。
- 为什么保持 isB1=0 而不是仓促实现 TransDataTo5HD 方案：转置方案引入 C 分块、carry-over、双 dtype 转置、尾块边界一整套新复杂度，需设计审批与独立验证；padded 路径功能正确，性能优化非正确性阻塞项。
- 当前路线档位：设计公式路线 banned；TransDataTo5HD 路线 safety_fallback 候选（待实现验证）；现状（padded）为功能正确的基线。

## 系统信息

- Ascend950PR，CANN 9.1.0，arch35；向量宽度 K：fp16=128、fp32/int32=64（bf16 经 float 通路）。
- 相关文件：`docs/DESIGN.md` §3.1（批量化设计公式与通用偏移公式）、`op_host/arch35/bench_no_harness_cummin_tiling.cpp`（isB1 决策点）、`issues/issue_20260715_b1_batch_layout_mismatch_0.md`（完整分析）、CANN `x86_64-linux/asc/include/basic_api/kernel_struct_binary.h`（BinaryRepeatParams 粒度事实）、AscendC DataCopyPad 文档（UB stride 32B 块语义）。

## 预防

- 实现任何"批量化/重排布局"优化前，先把设计偏移公式与通用寻址公式在同一边界 case（如 B==1）上对算一遍：两式给出的 offset 序列若不一致即为转置级错位，直接实现必错。
- 向量指令聚合前核对粒度：UB 侧 stride 以 32 字节块计、repeat 参数以 16/8 元素块计——低于块粒度的聚合需求（逐元素跨行）不可用这两类机制，只能转置或逐元素（禁用）。

## 验证信号

来源说明：以下信号来自 B=1 批量化任务的设计核对与回退闭环；注意布局不一致本身是**文档公式交叉推导结论**（静态推导，非上板观测），上板观测的是回退路径的正确性。

- 布局核对（推导依据，artifact 实读）：`docs/DESIGN.md` §3.1 同时给出通用公式 `a*C*B + c*B + b` 与批量化公式 `(a/K)*C*K + c*K + b`，代入 B==1 推得实际 (K,C) 行序与公式 (C,K) 行序互为转置（来源平台：950；证据等级：tool_observed（文档/公式 artifact 交叉推导，非运行期观测））。
- 机制粒度事实：DataCopyPad UB 目标 stride 以 32 字节块为单位；`BinaryRepeatParams` 的 BlkStride/RepStride 以 16 元素（fp16）/8 元素（fp32/int32）块为粒度（CANN 头文件/文档实读）（来源平台：950；证据等级：tool_observed）。
- 回退路径正确性：isB1=0 下 B==1 走 FullColumn padded 路径，L1 全量 600 用例 553 PASS / 0 FAIL / 47 SKIP；NPU 边界 B=1 用例 3/3 PASS（MERE=0）；最终 ALL 级 633 用例 0 FAIL（来源平台：950；证据等级：tool_observed）。
- 决策留痕：tiling 注释 + `issues/issue_20260715_b1_batch_layout_mismatch_0.md` 完整记录公式推导、机制排除与待审批方案（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "直接按公式实现会读错数据"是确定性推导，本轮未写错实现实测触发（若需要运行期证据，须实现后跑 ST 观察错位——未尝试）。
- TransDataTo5HD 方案整体未实现：其正确性、性能收益（相对 padded 路径的净提升需扣除两次全量转置）均未验证；int32/bf16 的 TransDataTo5HD 支持需编译验证。
- K 的取值（128/64）按 arch35 fp16/fp32 给出，其他架构/dtype 组合需重取。

**未决项**：

- （优先项）TransDataTo5HD 前后转置方案的实现与验证：C 分块/carry-over、尾 batch 与尾 C 片边界、indices 的 float→int64 Cast 时序、int32/bf16 dtype 支持——全部待设计审批后实现。
- 批量化的实际性能收益未测量（现状 padded 路径有效 lane 占比 1/K；转置方案的净收益需上板对比）。
## 坏实践

- 拿到设计文档的优化公式不核对物理布局直接实现——本例公式与实际布局互为转置，属"看起来很对、写出来必错"的陷阱。
- 用 strided gather 硬凑跨行取列（32B 块粒度限制下聚合不出紧凑向量，白搭一次实现）。

## 误诊

- 无明显误诊轮次（问题在设计核对阶段即被发现，未进入实现-失败循环）；价值在于把"为什么不能简单实现"的三条机制排除依据留档，避免下一个实现者重新踩一遍 strided gather 的坑。
