---
type: "Runbook"
title: "arch35 Select 超64 cmpmask 残缺判别"
description: "arch35（DAV_3510）上 Compare 一次产生的 cmpmask 有效位限 256B/sizeof(T)=64（fp32）；kernel 把 count_padded（可达 1088+）一次性传给 Compare/Select(VSEL_CMPMASK_SPR) 时，超出 64 的 cmpmask 残缺/陈旧，Select 据残缺掩码对第 65+ 元素取错分支。判别=固定 threshold≈0、放大 count_padded 做边界实验，PASS/FAIL 边界恰在 64 即定位。"
tags: ["task.debugging", "topic.precision", "topic.implementation.masking", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-18T05:29:39Z"
updated_at: "2026-09-16T11:20:30Z"
platforms: ["950"]
---

# arch35 Select 超64 cmpmask 残缺判别

## 现象

`softplus_v2_grad`（arch35/DAV_3510，Ascend950PR）在修好 host 侧 scalar 缓存键后仍有批量非 scalar 用例失败，且**单跑也失败**（排除 tiling 缓存嫌疑），属真实 kernel 精度 bug。代表用例（fp32，threshold≈-1.18e-38 即≈0 负，5D 广播 shape）逐元素比对观测到 Select 取错分支：

- 某 sigmoid 分支元素（bx<threshold，应 mask=false→src1=grad_sig）：实测 actual=gradOutput，即 kernel 走了**线性分支**（mask=true→src0=gradOutput）。
- 某线性分支元素（bx>threshold，应 mask=true→src0=gradOutput）：实测 actual=gradOutput*0.5，即 kernel 走了 **sigmoid 分支**（mask=false→src1=grad_sig）。

合看即 Select 在 mask=true 时返回 src1、mask=false 时返回 src0，与语义完全反了。但 threshold=20.0 时不复现，故不是 Select 语义写反，而是 cmpmask 在大 count 下被错误填充。

## 适用场景

- **工程形态**：kernel 直调（op_kernel），AscendC SIMD 基础 API `Compare`/`Select`。
- **API 路径**：`AscendC::Compare<float, uint8_t>(dst, src0, src1, CMPMODE::GT, count)` + `AscendC::Select<float, uint8_t>(dst, mask, src0, src1, SELMODE::VSEL_CMPMASK_SPR, count)`，arch35（DAV_3510 系）。
- **验证过的平台与 CANN 版本**：Ascend950PR（注册平台 `950`），CANN 9.0.0-beta.2，arch 目录 arch35。
- **适用粒度**：`platform_wide`——arch35 上一切用 `Compare`/`Select(VSEL_CMPMASK_SPR)` 且 count 可能超 64 的 kernel 都受 cmpmask 单 repeat 容量约束。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；分块循环修复的终版 verbatim 代码未从 trace 逐字恢复。
- **明确不适用/未验证场景**：非 arch35 架构（arch22/a2/a3 的 ONE_REPEAT_ELE_NUM 可能不同，未验证）；count 恒 ≤ 64 的 kernel 不触发；不经 `VSEL_CMPMASK_SPR`（其他 SELMODE）的 Select 不适用本根因。

## 根因

arch35 上 `Compare` 一次产生 cmpmask 的容量 = `ONE_REPEAT_ELE_NUM_FP32 = 256B / 4B = 64` 位（一个 vector repeat 的 fp32 元素数）。`Select` 以 `VSEL_CMPMASK_SPR` 模式工作时按 cmpmask 位选分支。本算子 kernel 把 `count_padded`（按 64 对齐、但可达 1088+）**一次性**传给 `Compare`/`Select`：`Compare(count_padded)` 只正确填了前 64 位 cmpmask，**超出 64 的部分 cmpmask 是陈旧/未定义**；`Select(count_padded)` 据这份残缺 cmpmask 选分支，对第 65+ 元素取错 src。threshold≈0 时 mask 高度敏感（bx 正负微小差异即翻 mask），错分支被放大成可见 mismatch；threshold=20.0 时绝大多数 bx ≪ threshold、mask 几乎恒定，即便 cmpmask 残缺也"碰巧"取对，故不复现——这也是为何本 bug 长期潜伏、只在极端 threshold 触发。参照：CANN 内置 `hard_swish_grad_v2` 在同 arch35 上显式按 `ONE_REPEAT_ELE_NUM_FP32=64` 分块循环 Compare 与 Select，即正确写法。

## 判别方法

本规则按"触发 → 探针 → 观测 → 判别"四段组织；触发与观测均为 trace 中 tool_observed 的测试/边界实验输出。

1. **触发（tool_observed 失败症状）**：逐元素比对发现 Select 在 mask=true 时返回 src1、mask=false 时返回 src0（与语义相反），但仅当 threshold≈0 且 count_padded>64 时出现；threshold=20.0 或 count_padded≤64 时不出现。即命中本规则。

2. **探针（可执行检查）**：固定 threshold≈0（构造极小负 threshold，使 mask 对 bx 正负微小差异高度敏感），逐步放大 shape 使 count_padded 跨越 64：分别取 numel=6/count_padded=64、numel=64/count_padded=64、numel=120/count_padded=128、numel=256/count_padded=256，跑同 kernel 比对 actual vs golden。再做 threshold=20.0 对照（绝大多数 bx ≪ threshold、mask 近恒定）。

3. **观测（tool_observed 边界实验，Ascend950PR/arch35）**：
   - count_padded=64（numel=6 或 64）→ **PASS**；
   - count_padded=128（numel=120）→ **FAIL**（出现 mismatch）；
   - count_padded=256（numel=256）→ **FAIL**（mismatch 更多）。
   - 边界恰在 64：count_padded ≤ 64 通过、> 64 失败。threshold=20.0 对照下不复现。

4. **判别（interpretation，根因分析）**：若 PASS/FAIL 边界恰落在 `ONE_REPEAT_ELE_NUM = 256B/sizeof(elem)`（fp32=64），根因是 cmpmask 单 repeat 容量受限——`Compare(count>容量)` 只填前若干位、`Select(VSEL_CMPMASK_SPR)` 据残缺掩码对超容元素取错分支，**而非 Select 语义写反也非 FTZ**。判别核心是 count 边界实验 + threshold 对照：边界在 64 + threshold≈0 才复现 → cmpmask 残缺；若无论 count 都反 → 语义写反；若仅极端值翻转 → 疑 FTZ。正确修复方向是按 `ONE_REPEAT_ELE_NUM` 分块循环 Compare/Select（或改用不依赖 cmpmask 的 SELMODE），而不是交换 src0/src1 强行对齐。

## 预防

- arch35 上凡用 `Compare`/`Select(VSEL_CMPMASK_SPR)` 处理变长数据，**一律**按 `ONE_REPEAT_ELE_NUM_FP32=64`（或对应 dtype 的 `256B/sizeof(elem)`）分块循环，不要把 `count_padded` 一次性传进去。
- 写 kernel 时若 Select 语义"看起来反了"，先别急着交换 src0/src1——先查 cmpmask 是否因 count 超容而残缺；用 threshold=20.0 vs threshold≈0 的对照实验可快速区分"语义写反"与"cmpmask 残缺"。
- 对照 CANN 内置 `hard_swish_grad_v2`（同 arch35）的分块写法作为参照正确实现。

## 验证信号

- 边界实验：threshold≈0 时 count_padded=64 PASS、count_padded=128 FAIL、count_padded=256 FAIL，PASS/FAIL 边界恰在 64（来源平台：950；证据等级：tool_observed）。
- 修复（Compare/Select 按 `ALIGN_ELEMENTS=64` 分块循环）后 PyTorch ST 通过数 498→565/581（来源平台：950；证据等级：tool_observed）。
- 修复后 C++ ST 57/57 通过、无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **平台**：本开发会话只在 Ascend950PR（`950`、arch35）实测。a2/a3（arch22/DAV_2201）的 `ONE_REPEAT_ELE_NUM` 可能不同，**未验证，推断可迁移**——迁移时须按目标 arch 重算 `256B/sizeof(elem)` 后再定块大小，不可直接照搬 64。
- **dtype**：本卡只测 fp32（64 元素/块）。fp16/bf16 的块大小应为 `256B/2B=128`，int8 为 256——未实测，按算式推导。
- **边界**：count 恒 ≤ 块大小的 kernel 不触发本 bug；不经 `VSEL_CMPMASK_SPR`（如用其他 SELMODE）的 Select 不适用本根因。

## 坏实践

- 看到 Select"取反"就直接交换 src0/src1 参数强行对齐——掩盖 cmpmask 残缺真因，且 threshold=20.0 时会从"错"变"对反再反=碰巧对"，回归隐患极大。
- 把 `count_padded` 当成"已对齐就安全"——对齐只保证是 64 的倍数，不保证 ≤ 64；count_padded=128 是 64 的倍数但仍超容。

## 误诊

早期一度怀疑 Select 语义写反或 FTZ 把 threshold flush 到 0 导致 mask 翻转。但算式核验：threshold=-1.18e-38 即便 FTZ 成 0，对 bx=-0.5953 的 mask 仍是 false、与"取反"表现一致但解释不了 count=64 通过/128 失败的边界。真因是 cmpmask 单 repeat 容量 64，非语义反也非 FTZ（来源平台:950；证据等级:agent_reported——此为排除性推理，非实测信号）。

## 后续增量（2026-09-16，并行策展批：256B 对齐断言与"比较计数对齐 + 掩码限扫"模式）

argmax 算子（BenchNoHarnessArgMax，Ascend950PR/__NPU_ARCH__==3510，CANN 9.1.0）迭代二补上本约束族的**编译期硬性形态**与一种应对模式：

**1. 256B 对齐断言（编译期约束，原卡"运行期残缺"的兄弟形态）**：`Compare`/`Compares` 在 3510/5102 上带有硬性对齐断言（`basic_api/impl/kernel_operator_vec_cmpsel_intf_impl.h`，verbatim）：

```cpp
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
    ASCENDC_ASSERT(((count * sizeof(T3)) % ONE_REPEAT_BYTE_SIZE == 0),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check count elements size in Compares, current size "
        "is %u, should be an integer multiple of 256.", count * sizeof(T3));});
#endif
```

即 `count * sizeof(T)` 必须是 256 的整数倍：int32/fp32 的 count 须为 64 的倍数，fp16/bf16 须 128 的倍数。带尾块（count 非 256B 整倍）直接按 chunkLen 调 Compare 会触发断言。

**2. "比较计数对齐 + 掩码限扫"模式（Int32-AR EQ-mask argmax）**：把"比较用计数"与"有效数据长度"解耦——比较计数用 `rAlignCmp = CeilAlign(R, 64)`（int32；fp16/bf16 为 128），buffer 按 rAlignCmp 分配；DataCopyPad 只搬 chunkLen 个真实数据，`[chunkLen, rAlignCmp)` 区间是残留 garbage，garbage 参与 EQ 比较会在 mask 尾部产生随机置位；掩码消费端只扫前 chunkLen 位（`FindFirstSetBit(maskLocal, chunkLen)`），尾部 spurious bit 不进入结果：

```cpp
// EQ 掩码：比较计数用 256B 对齐的 rAlign_（host 侧 CeilAlign(R,64) 下发）
Compares<int32_t, uint8_t>(maskLocal, inData, chunkMax, CMPMODE::EQ, rAlign_);
PipeBarrier<PIPE_V>();
// 掩码只扫有效 chunkLen 位：[chunkLen, rAlign_) 的 garbage 位不进入结果
uint32_t chunkArgIdx = FindFirstSetBit(maskLocal, chunkLen);
```

选"计数对齐 + 掩码限扫"而不是"数据搬完再补零"：补零要额外一次向量写且引入 0 与 chunkMax 相等的假命中风险（chunkMax 本身可能就是 0）。选 `Compares`（张量 vs 标量）而不是 Compare + Duplicate 广播：省掉 rAlignCmp*4 的广播 UB，二者受同一 256B 断言约束。host tiling 须区分"搬运对齐（32B/8 元素 int32）"与"向量比较对齐（256B/64 元素 int32）"两个粒度，buffer 预算按更粗的算。

**3. 与本卡核心发现的交汇边界（重要）**：本卡根因指出"Compare 一次产生的 cmpmask 有效位限 64（fp32/int32）"——因此上述模式中 **rAlignCmp 超过 64、且有效窗口 chunkLen 也超过 64 时，第 65+ 位的掩码可靠性没有保障**（本卡实测 128/256 均残缺）。该模式的 NPU 精度实测在后续联调会话完成（本卡增量验证止于编译 + UT 54/54，含 Int32-AR tiling 分支用例；构建 `[INFO] Build completed successfully!`、5 个 kernel binary、clean build 通过，均为 tool_observed）；若采用该模式且 R > 64，须先按本卡判别方法（count 边界实验）在目标用例上验证掩码超容行为，或直接按 64 分块循环 Compare/Compares。

**4. 相邻提示**：同轨迹的向量化 `Max/Compare/Select` ARA 路径在 A0<64 时整体失败并回退标量（见库内「向量Compare/Select小A0结果错，改标量扫描」卡及其增量），设计带尾块的向量比较类方案时一并核对。