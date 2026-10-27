---
type: "Runbook"
title: 向量Compare/Select小A0结果错，改标量扫描
description: "ARA-scan 向量 Compare/Select 流水线在小 A0（4/25）时值与索引出错（bf16 索引 26 失配、fp16 值 MERE=2.19e-02）；重写为逐 A0 标量处理后全对。复现增量（Reg API 层，CANN 9.0.0-beta.2）：asc_vf_call 调用的 VF 链内 Reg::Compares/Select 掩码分段使输出全 0（11/11 全 got 0，两个独立 VF 版本复现）——减法探针隔离（identity/Cast/Cast+Abs/去 Select 单分支均非零），回退标量 FP32 上溯后 11/11；VF 内 Compares/Select 分段在该栈 banned，机制层（参数序/掩码极性/平台限制）未验证。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.comparison_logic]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# 向量Compare/Select小A0结果错，改标量扫描

## 现象

Cummin 算子 ARA-scan 模式（扫描维非最后一维、A0>1）的向量 Compare/Select 流水线在 Select 模式修复（见同批《Select张量-标量模式索引不更新，须TT模式》卡）之后，仍有两个小 A0 用例失败：

- L0_004（bfloat16，shape=[983,5,1,5]，dim=-4 → A1=1、R=983、A0=25）：values 通过（MARE=0.00e+00），`indices: 26 mismatches out of 24575`；
- L0_005（float16，shape=[13,235,1,1,4]，dim=1 → A1=13、R=235、A0=4）：`values: MERE=2.19e-02 (>= 9.77e-04), MARE=7.98e-01 (>= 9.77e-03)`，`indices: 1481 mismatches out of 12220`——失配值形如 golden=2.095703 → actual=2.041016、golden=9.820312 → actual=2.041016，说明运行最小值取到了错误位置的数据。

而大 A0 用例（L0_002 fp32 A0=2 通过、L0_003 int32 A0=60 通过）不受影响。向量流水线的核心（r=1..R-1 迭代段）当时为：

```cpp
// r=1..R-1: iterative scan
for (uint64_t r = 1; r < tiling_->rSize; r++) {
    // CopyIn current row
    LocalTensor<T> currentBuf = inQueueX.template AllocTensor<T>();
    uint64_t srcOffset = GetOffset(a1, r, a0Start);
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = tileA0Aligned * sizeof(T);
    ...
    DataCopyPad(currentBuf, xGM[srcOffset], copyParams, {false, 0, 0, static_cast<T>(0)});
    inQueueX.EnQue(currentBuf);

    currentBuf = inQueueX.template DeQue<T>();

    // Compare(GE): cmpBuf = (current >= runningMin)
    LocalTensor<T> runMin = runningMinBuf.template AllocTensor<T>();
    LocalTensor<uint8_t> cmpMask = cmpBuf.template AllocTensor<uint8_t>();
    Compare(cmpMask, currentBuf, runMin, CMPMODE::GE, tileA0Aligned);

    // Select(TT): runningMin = cmp ? runningMin : current
    Select(runMin, cmpMask, runMin, currentBuf, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileA0Aligned);
    ...
    // NaN handling（Compare NE 自比较 + Not/And 掩码 + Select）
    ...
}
```

小 A0 时 `tileA0Aligned` 对齐到 128（fp16/bf16），padding 区（A0=4 时 124 个补 0 元素）远大于有效区。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke），kernel 向量流水线按对齐 tile 处理非对齐小维度。
- API 路径：`Compare`（CMPMODE::GE/NE）+ `Select`（VSEL_TENSOR_TENSOR_MODE）+ `Not`/`And` 掩码逻辑 + `DataCopyPad` padding。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR）+ CANN 9.1.0。
- 适用粒度：`operator_family`（小内维 + 向量条件更新流水线的算子）。
- 路线优先级：safety_fallback（标量重写以性能换正确性；向量方案的修复见遗留问题）。fallback 声明：标量逐元素路径每元素一条比较指令，大 tile 下时延显著高于向量管线（本轨迹无性能数据，未量化相对代价）；仅在向量路径正确性无法保障时采用。
- 明确不适用/未验证场景：大 A0 且对齐的输入（向量管线在这些用例上通过）；向量路径在其他 dtype/shape 下的行为未系统枚举。

## 根因/解决配方

根因（部分定位）：向量 Compare/Select 流水线在小 A0（4/25，对齐后 padding 占比 80%~97%）时产生错误的最小值与索引；指令级根因未定位——怀疑方向是 `DataCopyPad` 的 padding 区 0 值参与了后续 Compare/Select（0 是合法的 fp16/bf16 值，会污染掩码与选择结果），也可能是 Select/掩码逻辑在小 tile 下的行为异常。修复采用整体规避：把 `ProcessAraScan()` 重写为逐 A0 标量处理（复用已验证的 AR-scan 标量模式：float 域比较 + 位级类型转换 + 位级 NaN 判定），逐行搬运、逐元素维护 running_min/running_idx：

```cpp
template <typename T>
__aicore__ inline void KernelCummin<T>::ProcessAraScan()
{
    uint64_t a0Size = tiling_->a0Size;
    ...
    for (uint64_t a1 = a1Start_; a1 < a1End_; a1++) {
        ...
        for (uint64_t r = 0; r < R; r++) {
            // Load current row: x[a1, r, 0:a0Size]
            LocalTensor<T> inputBuf = arInputBuf.template AllocTensor<T>();
            uint64_t srcOffset = GetOffset(a1, r, 0);
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = a0Size * sizeof(T);   // 只搬有效长度，不搬对齐 padding
            ...
            DataCopyPad(inputBuf, xGM[srcOffset], copyParams, {false, 0, 0, static_cast<T>(0)});

            if (r == 0) {
                // First row: initialize running_min and running_idx
                for (uint64_t a0 = 0; a0 < a0Size; a0++) {
                    runMinT.SetValue(a0, inputBuf.GetValue(a0));
                    runIdxF.SetValue(a0, 0.0f);
                    valBuf.SetValue(a0, inputBuf.GetValue(a0));
                    idxI64.SetValue(a0, static_cast<int64_t>(0));
                }
            } else {
                // Subsequent rows: scalar compare and update（int32 原生比较 / float 域 NaN-aware）
                ...
            }
            ...  // 写回 values/indices
        }
    }
}
```

关键差异：标量路径只按 `a0Size`（有效长度）搬运和处理，padding 区不再进入任何比较；掩码逻辑整体消失。

## 设计决策

- 为什么选整体标量重写而不是继续定位向量缺陷：两个失败用例的指令级行为无法在 NPU 上直接观测（无 profiling 能力），掩码链（Compare→Not→And→Select×3）组合空间大；标量模式已在 AR-scan 上全量验证，重写是确定性最高的收敛路径。
- 放弃的候选：修复向量 API 用法（根因不明，试错成本不可控）；只处理 padding 掩码（如把 padding 区掩成恒 false——需要额外掩码缓冲与验证，且根因未证实是 padding）。
- 后续演化：标量版先丢掉了 A0 分块（引入 507035 回归，见同批《507035 ARA-scan大A0整块UB分配超限》卡），最终版为"标量 + tileA0Aligned 分块"。
- 当前方案属 safety_fallback（正确性优先，性能代价未量化）。

## 系统信息

- 工具链：CANN 9.1.0；`bash build.sh --soc=ascend950 -j8` + run 包安装 + C++ ST（NPU 实测）。
- 相关文件：`op_kernel/arch35/cummin.h`（ProcessAraScan 向量版→标量版→分块标量版）、`tests/st/test_aclnn_cummin.cpp`（L0_004/L0_005 用例）。

## 验证信号

验证来自同一开发会话的 NPU 前后对照：修复前 L0_004 `indices: 26 mismatches out of 24575`、L0_005 `values: MERE=2.19e-02 (>= 9.77e-04), MARE=7.98e-01 (>= 9.77e-03)`、`indices: 1481 mismatches out of 12220`（来源平台：950；证据等级：tool_observed）；标量重写（清理重建后）L0_004 `[PASS] values MARE=0.00e+00 / [PASS] indices all 24575 match`、L0_005 `[PASS] values MERE=0.00e+00 / [PASS] indices all 12220 match`（来源平台：950；证据等级：tool_observed）；同一实现随后在全量 L0 ST 21/21 中保持通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 向量管线小 A0 失败的指令级根因未定位，本卡是"规避型"修复——若下个版本需要在向量管线上恢复性能，必须先完成根因复现（见遗留问题）。
- 标量路径性能代价未测量（无 profiling 信号）；对时延敏感的算子需另行评估。
- 大 A0 用例（60、43470）在标量路径同样通过（含 A0 分块），但标量路径在大对齐 tile 下的效率劣势未量化。
- 相邻已闭合根因（排查向量 Select 错误时优先对照）：count>64 时掩码按 64 元素复用/不偏移的错位家族见 [arch35 Select VSEL_CMPMASK_SPR 每 64 元素复用卡](arch35_select_vsel_cmpmask_spr_mask_reuse_count_over_64.md)与[Select VSEL_CMPMASK_SPR count>64 掩码错位卡](select_vsel_cmpmask_spr_count_64.md)——本卡 tileA0Aligned=128 亦大于 64，该族根因未被本轨迹排除，是向量路线修复时的首选假设之一；padding 区垃圾掩码污染及 banned 的全量清零处置见 [Duplicate 全量清零 padding 覆盖有效区卡](duplicate_padding_banned.md)。

**未决项**：

- **优先项**：复现并定位向量 Compare/Select 小 A0 失败的指令级根因（最小用例：fp16 A0=4 + 对齐 padding，单步观测 Compare 掩码与 Select 输出的 padding 区行为），据此决定向量管线能否修复回收。
- 标量 vs 向量路径的相对性能代价（成对对比）未测量。
## 坏实践

- 用 `tileA0Aligned`（对齐长度）参与值域比较而只把 `tileCount`（有效长度）用于搬运回写——padding 区一旦进入 Compare/Select 就是合法 0 值，污染最小值/掩码。
- 根因未定位时反复微调向量 API 参数组合（本会话曾沿"Select 模式→缓冲尺寸→掩码字节数"多条假设排查）——不如先用可验证的标量基线锁定正确性。

## 误诊

- 排查期间先后怀疑过 AllocTensor/FreeTensor 循环竞争、GetOffset 偏移算错、掩码缓冲尺寸（`tileA0Aligned / 8 + 32` 字节）不对（均 agent_reported，经代码核对与对照用例排除——大 A0 同代码路径通过，否定这些与 A0 大小无关的假设）。

## 后续增量（2026-09-16，并行策展批：argmax 实例、mismatch-A0 负相关律与"A0=512 无 padding 仍失败"边界）

argmax 算子（BenchNoHarnessArgMax，Ascend950PR/arch35/DAV_3510，CANN 9.1.0）迭代二在同一根因上补充定量规律与新边界（独立 trace，本卡证据自此覆盖两个 trace）：

**1. 定量规律：mismatch 率与 A0 大小负相关**。向量化 `Max<float>` + `Compare(EQ)` + `Select(VSEL_TENSOR_SCALAR_MODE)` 三件套按 `a0Align`（256B 对齐，≥64 元素）整批执行的 ARA 路径（dim 非最内轴），失败率随 A0 增大单调下降：`[512,512,4] dim=1 bf16`（A0=4）mismatch 92.8%（1901/2048）、A0=32 mismatch 5.08%、A0=50 mismatch 0.15%——A0 越小，padding 区 `[curA0Len, a0Align)` 的垃圾数据占比越大、污染越重。失败输出几乎全为索引初值 0（更新路径从未生效）。

**2. 新边界：A0=512（无 padding）仍失败**。`[2,4,8,16,32] dim=2 int32`（A0=512，无 padding）mismatch 3.52%（144/4096）——向量化路径的缺陷不止 padding 一层，还有未定位到机制层的逻辑缺陷（多核切分边界或 Select 掩码跨轮消费方式）。

**3. 三种 padding 修补全部无效（负知识）**：-INFINITY SetValue 填输入缓冲、填 Cast 后 float 缓冲、Duplicate 预填充 + 对齐 Cast count 三种方案后复测均维持 91/101、同 10 条失败不变（详见同域「ARA 向量化 padding 三方案无效须标量回退」卡）——不要按同样思路重试。

**4. 修复（整体标量回退）**：把 ProcessFloatARA/ProcessInt32ARA 整体替换为逐元素标量 argmax（只处理 `[0, curA0Len)` 有效元素，GetValue/SetValue 逐元素，NaN 语义（NaN>inf、首个 NaN 生效）在标量分支显式实现）后 ST Real 101/101（100%）、原 10 条失败用例全部 bitwise-equal，UT op_host 54/54 + op_api 15/15 无回归。判别方法补充：失败用例的 A0 全部（或多数）< 64、失败率随 A0 减小而升高、失败输出恒为初值 0——即判定是"对齐宽度 padding 参与计算"一类问题；再用一条 A0≥64 且无 padding 的用例交叉验证，若仍失败则向量化路径还有逻辑缺陷，直接评估标量回退。

**5. 边界补充**：标量回退的正确性覆盖 A0∈[1,512]、R∈[2,65536]、fp16/fp32/bf16/int32；未覆盖超大 A0（>8192，超出 ubFactorA0 分块）的分块组合行为；float 索引中间量要求 R ≤ 2^24（float 精确整数范围）。性能代价未实测（无 profiling 信号）。

**6. 验证信号**：

- 修复前：ST Real 91/101，10 条 ARA 失败用例 mismatch 率 0.15%~92.8%，失败输出全为 0；A0 分布 4/16/32/42/50（<64）9 条 + A0=512（无 padding）1 条（来源平台：950；证据等级：tool_observed）。
- 三种 padding 修补方案复测均维持 91/101，同 10 条失败不变（来源平台：950；证据等级：tool_observed）。
- 修复后（标量回退替换两个 ARA 函数，重编重装）：ST Real 101/101（执行通过率 100.00%），原 10 条失败用例全部 PASS 且 bitwise-equal（如 `[8,16,32] dim=1` 256 个元素完全一致）；UT 69/69 通过、迭代一 fp32 AR 回归用例全部通过（来源平台：950；证据等级：tool_observed）。


## 后续增量（2026-09-17，并行策展批：Reg API 层实例——VF 链内 Reg::Compares/Select 掩码分段致输出全 0）

另一独立轨迹并入独立 trace（BesselI1e Branch-1 FP16 上溯 FP32，ascend950，CANN 9.0.0-beta.2）：`__simd_vf__` 函数（经 `asc_vf_call` 调用）内 `Compares<float, AscendC::CMPMODE::LE/LT>` 产分段掩码 → Chebyshev 两分支 → `Select` 合并 → 符号恢复 `Select`。编译安装全部成功，白盒 ST 11/11 全部失败且**所有输出元素为 0**（`Mismatch at [1]: got 0, expected -0.19…`），两个独立实现版本（子函数递推/系数全展开）均复现。

**减法隔离探针（同类问题的可复用隔离工具，均编译通过并实际运行）**：

1. identity 直通（LoadAlign→StoreAlign）→ 输出=输入原值，非零；
2. Cast 往返（f16→f32→f16）→ 保值，非零；
3. Cast+Abs → 输出 |x|，非零；
4. 完整链（加入 Compares 掩码 + Select 合并 + 符号恢复 Select）→ **全 0**；
5. 去掉 Compares/Select/分支 B、只留单分支 → 非零（= |期望值|，缺符号与分段）；
6. 恢复完整链 → 又全 0。

结论：LoadAlign/StoreAlign/Cast/Abs/Mul/Div/Sqrt 通路数值行为正常，全 0 由 `Reg::Compares`/`Select` 引入。**回退修复**：FP16 路径放弃 VF 掩码分段，改标量 FP32 上溯逐元素计算（half → float → 标量函数 → half），VF 仅保留 identity 直通（满足工程对 `asc_vf_call` 存在性的 grep 验收）；修复后 11/11 全 PASS、FP32 分支回归 14/14。

复现增量验证信号：

- 失败：完整 VF 链 11/11 FAIL 逐例 `got 0`（两个独立版本复现）（来源平台：950；证据等级：tool_observed）。
- 探针：identity 直通输出=输入（-7.73/1000 等非零）、Cast 往返保值、Cast+Abs 输出 |x|、去 Select 单分支输出非零（来源平台：950；证据等级：tool_observed）。
- 修复：标量 FP32 上溯版 `PASS: 11 | SKIP: 0 | FAIL: 0 | ERROR: 0 | TOTAL: 11`；FP32 分支回归 `PASS: 14`（来源平台：950；证据等级：tool_observed）。

复现增量边界与坏实践：

- 机制未定：不能排除是调用方式错误（Select 的 true/false 槽位序、Compares 掩码极性）而非平台缺陷——按"该写法在本栈实测致零"使用；【优先】以最小 VF 用例（单个 Compares+Select）核对 `kernel_reg_compute_vec_cmpsel_intf.h` 的参数序与掩码极性，确认是 API 用法错误还是版本缺陷。
- 本实例是 Reg API（`AscendC::Reg::Compares/Select`）+ `asc_vf_call` 上下文，与本卡原实例（classic `Compare`/`Select(VSEL_*)` + 小 A0 padding）分属两层 API，共享"向量 Compare/Select 失效回退标量"的处置。
- 坏实践：为满足 `asc_vf_call`/`__simd_vf__` 的 grep 验收在最终代码里保留不做实际计算的 identity VF 直通（掩盖"VF 计算链未真正落地"）——应把 Compares/Select 失效证据上报并调整验收口径；"输出全 0 + 编译运行无报错"先查计算链中新增的掩码/选择类算子，而不是怀疑数据搬运或 golden。