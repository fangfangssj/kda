---
type: "Runbook"
title: "SoftmaxV2 ARA与AR模板R向累加树不同致1ULP分歧"
description: "SoftmaxV2 arch35 的 ARA 与 AR 两套 tiling 模板对同一 fused shape (a1,r,a0=1) 用不同 R 向归约树拓扑，fp32 非结合律下产生 1 ULP bit 级分歧；附环境变量强制模板选择的双跑对比判别方法。"
tags: ["task.debugging", "topic.compute.normalization", "topic.compute.reduction", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# SoftmaxV2 ARA与AR模板R向累加树不同致1ULP分歧

## 现象

在分析 SoftmaxV2（`activation/softmax_v2`，arch35 模板族）算子时，需要确认 ARA 模板（全载 `ara_full_load` / 重计算 `ara_recompute`）与 AR 模板（全载 `ar_full_load` / 重计算 `ar_recompute`）在 R 方向上的累加顺序是否完全一致。

阅读四个模板与 base 类后发现：ARA 与 AR 的 R 方向 reduce 树**拓扑不同**——ARA 全程纯软件树形二分（`TwoRowAddWithTail`/`BinaryAddVF`/`NlastDichotomyAdd`），AR 用 fold + 硬件 `AscendC::ReduceSum` 指令。FP32 加法不满足结合律，因此两套模板对同一组 `exp` 值的 R 向累加路径不同，理论上会产生不同的舍入结果。

为验证这一推断，构造 fused shape `(a1=2, r=256, a0=1)`（即 `shape=[2,256,1], dim=1`，对 256 维做 softmax）。该 shape 命中 `a0==1` 回退逻辑：正常 tiling 下 ARA 模板 `IsCapable` 在 `a0_==1` 时返回 false，回退到 AR 模板。为强制同一 shape 分别走 AR 与 ARA，在 `SoftmaxV2TilingBase` 加环境变量开关 `IsForceAraMode()`，force 时让 AR 模板不 capable、ARA recompute 忽略 `a0==1` 拦截，从而一次编译、两轮运行切换模板。

用 `cannsim`（dav_3510，Ascend950 仿真）跑两轮：轮1 不设环境变量走 `AR_FULL_LOAD`(tiling key=1000)，轮2 设 `SOFTMAX_V2_FORCE_ARA=1` 走 `ARA_RECOMPUTE`(tiling key=20000)，两轮使用同一 `SOFTMAX_INPUT_SEED=3` 生成的字节级一致输入。对比输出 bin：`result[0]` AR=`0x3b908fec`、ARA=`0x3b908fed`（差 1 ULP）；共 196/512 个元素 bit 不同，最大绝对差 9.3e-10 出现在 idx 32（AR=`0x3c7a468c` vs ARA=`0x3c7a468d`）。

## 适用场景

- **工程形态**：aclnn 调用路径（`aclnnSoftmaxGetWorkspaceSize`/`aclnnSoftmax`），通过 op package 安装后由 `cannsim` 仿真执行编译出的真实 kernel binary。
- **API 路径**：`aclnnSoftmax`（`aclnnop/aclnn_softmax.h`）→ host 侧 tiling `IsCapable`/`DoOpTiling` → arch35 四模板之一。涉及模板选择键 `tiling_key`：`1000`=AR_FULL_LOAD、`2000`=AR_RECOMPUTE、`10000`=ARA_FULL_LOAD、`20000`=ARA_RECOMPUTE。
- **验证过的平台与 CANN 版本**：Ascend950（cannsim `dav_3510` 仿真，CANN 9.2.0）。两轮均执行编译出的 `SoftmaxV2_float32_high_precision.o` 中独立符号 `_1000` 与 `_20000`，非 host 侧 stub。
- **适用粒度**：`same_operator`（SoftmaxV2 arch35 模板族，fused shape `a0==1` 触发 ARA/AR 模板二选一的场景）。
- **路线优先级**：`recommended_fastpath`——当需要对 SoftmaxV2 同 shape 不同 tiling 模板做 bit 级一致性核验时，推荐此「环境变量强制 `IsCapable` + 同输入双跑 + bin 对比」判别方法。
- **不适用或未验证场景**：非 arch35 模板族未验证；`a0!=1` 的 fused shape（ARA `IsCapable` 正常 capable，模板选择由优先级与 shape 共同决定，分歧是否同样存在未在本 trace 验证）；上板（真实 NPU）未验证，仅有 cannsim 仿真证据；半精度/双精度未验证。

## 根因

SoftmaxV2 arch35 的 ARA 与 AR 两套 tiling 模板对同一 fused shape `(a1, r, a0=1)` 的 R 方向归约采用**不同累加树拓扑**，FP32 加法不满足结合律导致舍入路径不同，产生 bit 级差异：

- **ARA recompute 块内归约**：`NlastDichotomyAdd` 软件树形二分（`softmax_v2_base.h`），针对 `binAddRLen` 行 R 做跨 A0 的横向二分；ARA full_load 进一步用 `TwoRowAddWithTail`（8 行组 `((v0+v4)+(v1+v5))+((v2+v6)+(v3+v7))`）+ `BinaryAddVF` 4 路并行软件二分。
- **AR full_load 归约**：`SecondNormCompute` → `LastReduceSum`，按 VL_FP32(=64) 元素段做 fold 二分，段内用硬件 `AscendC::ReduceSum<float, Pattern::Reduce::AR, true>` 指令（64 元素平衡树）；AR recompute 块内同样用硬件 `ReduceSum`。

两者对同一组 `exp` 值求和，但加法分组与合并顺序不同。FP32 加法非结合律：`a+(b+c) != (a+b)+c` 在舍入边界上最多差 1 ULP。本例 256 个 `exp` 值在两种树下求和，`sum` 的最后一位舍入方向不同，经 `y=exp(x_i)/sum` 传播到全部输出元素。row0 的中间 `sum` 恰好落在 fp32 舍入边界两侧触发分歧（196/256 元素差 1 ULP），row1 的中间值未触发（0/256 一致）——证明分歧是**数据相关的概率性现象**：只要 reduce 树拓扑不同，总存在某些输入会产生 bit 级差异，但并非每个输入都触发。

算术关系：所有输出差异均为 1 ULP（fp32 单精度末位），最大绝对差 `9.3e-10` 对应 idx 32 处 `AR=0x0152756087(0x3c7a468c)` vs `ARA=0.0152756097(0x3c7a468d)`，`dUlp=1`，即两个相邻 fp32 表示之间的差，与「仅末位舍入方向不同」的机制自洽。AR 值均略小于 ARA（AR 树的舍入路径使 `sum` 偏大致使 `y=exp/sum` 偏小），方向一致。

## 判别方法

**触发条件**：SoftmaxV2 arch35，fused shape 满足 `a0==1` 且 `r>1`（如 `shape=[N, r, 1], dim=1` 合轴后 `(a1=N, r, a0=1)`）。此时正常 tiling 因 ARA `IsCapable` 在 `a0_==1` 返回 false 而走 AR，无法直接对比两模板。

**Probe（强制模板选择 + 同输入双跑 + bin 对比）**：在 `SoftmaxV2TilingBase` 加环境变量开关，force 时反转 AR/ARA 的 capable 判定，使同一 shape 可分别命中两模板。核心改动（`op_host/arch35/`，verbatim 完整函数）：

```cpp
// softmax_v2_tiling.h — SoftmaxV2TilingBase protected 区新增
static bool IsForceAraMode()
{
    const char* env = std::getenv("SOFTMAX_V2_FORCE_ARA");
    return env != nullptr && env[0] == '1';
}
```

```cpp
// softmax_v2_ar_recompute_tiling.cpp — force 时 AR recompute 不 capable
bool SoftmaxV2TilingARRecompute::IsCapable()
{
    // a0_为1的场景走AR模板
    OP_CHECK_IF(a0_ != DIM_NUM_ONE,
                OP_LOGI(context_->GetNodeName(),
                        "AR recompute template is not capable. axes is: %ld, merged shape is (%ld, %ld, %ld).",
                        reduceAxes_, a1_, r_, a0_),
                return false);
    OP_CHECK_IF(IsForceAraMode(),
                OP_LOGI(context_->GetNodeName(),
                        "AR recompute template is not capable. force ara mode, merged shape is (%ld, %ld, %ld).",
                        a1_, r_, a0_),
                return false);
    return true;
}
```

```cpp
// softmax_v2_ara_recompute_tiling.cpp — force 时 ARA recompute 忽略 a0==1 拦截
bool SoftmaxV2ARARecomputeTiling::IsCapable()
{
    // a0_为1的场景走AR模板（force ara 模式除外）
    OP_CHECK_IF(a0_ == DIM_NUM_ONE && !IsForceAraMode(),
                OP_LOGI(context_->GetNodeName(),
                        "ARA recompute template is not capable. merged shape is (%ld, %ld, %ld).", a1_, r_, a0_),
                return false);
    return true;
}
```

注：`ara_full_load` 在 `a0==1, r=256` 时 UB 装不下（`DoOpTiling` 必失败），故 force 模式只让 ARA recompute 承接，`ara_full_load` 的 `IsCapable` 保持 `a0_==1` 拦截不变。同时为 `ar_full_load` 加对称的 force 拦截（逻辑同 `ar_recompute`）。

例程（`examples/arch35/test_aclnn_softmax_v2.cpp`）用 `SOFTMAX_INPUT_SEED` 生成可复现随机输入，`SOFTMAX_INPUT_FILE`/`SOFTMAX_OUTPUT_FILE` dump 输入输出 bin，保证两轮输入字节级一致。两轮运行命令：

```bash
source /home/z/Ascend/ascend-toolkit/set_env.sh
# 轮1 AR_FULL_LOAD（默认）
SOFTMAX_INPUT_SEED=3 \
SOFTMAX_INPUT_FILE=/tmp/opencode/input_ar.bin \
SOFTMAX_OUTPUT_FILE=/tmp/opencode/output_ar.bin \
unset SOFTMAX_V2_FORCE_ARA; \
bash build.sh --run_example softmax_v2 eager cust --vendor_name=customize --soc=ascend950 --simulator
# 轮2 ARA_RECOMPUTE（force）
SOFTMAX_INPUT_SEED=3 \
SOFTMAX_INPUT_FILE=/tmp/opencode/input_ara.bin \
SOFTMAX_OUTPUT_FILE=/tmp/opencode/output_ara.bin \
SOFTMAX_V2_FORCE_ARA=1 \
./build/test_aclnn_softmax_v2
```

**原始观测**（cannsim dav_3510，Ascend950，CANN 9.2.0）：

| 证据 | AR 轮 | ARA 轮 |
|---|---|---|
| tiling debug | `AR_FULL_LOAD, force=0` | `ARA_RECOMPUTE, force=1` |
| tiling key | 1000 | 20000 |
| 输入 md5 | `089b00828bc9909074bb4e5750ccec3b` | 同 |
| kernel 符号(.o) | `_1000 @ 0x0000` | `_20000 @ 0x9300` |
| 仿真器 PC(RV_VMAX) | `0x10d0dc28` | `0x10d1e520` |
| RV_VMAX 指令编码 | `0x84082b82` | `0x80082782` |
| Total tick | 5011 | 8881 |
| result[0] | `0x3b908fec` | `0x3b908fed` |

输出 bin 逐元素对比（512 个 fp32）：row0(idx 0..255) **196/256** bit-diff 全 1 ULP，row1(idx 256..511) **0/256** 一致；最大绝对差 9.3e-10 @ idx 32（`AR=0x3c7a468c` vs `ARA=0x3c7a468d`，dUlp=1）。

**解释（interpretation）**：三重独立证据闭环——(1) 输入字节级一致（md5 相同）排除输入差异；(2) 仿真器 PC/指令编码/tick 不同证明两轮执行了 `SoftmaxV2_float32_high_precision.o` 中两个独立 kernel 符号而非同一函数内分支；(3) 输出 196/512 元素差 1 ULP 与 fp32 非结合律下不同 reduce 树的舍入差异机制一致。分歧非逻辑错误，是两套正确实现的不 bit-exact 特性。reduce 树拓扑差异本身来自模板源码阅读（代码引用支撑），其与 1 ULP 分歧的因果链为根因解释，未在 trace 内单独跑独立验证脚本反算。

## 验证信号

本节信号来自 SoftmaxV2 arch35 cannsim 仿真（dav_3510 模拟 Ascend950，CANN 9.2.0），shape `[2,256,1]` dim=1 seed=3，两轮同输入双跑。

- 两轮输出 196/512 元素 bit 不同，全部差 1 ULP，最大绝对差 9.3e-10 at idx 32（来源平台：Ascend950；证据等级: tool_observed）
- row0 196/256 bit-diff，row1 0/256 完全一致，分歧为数据相关概率性现象（来源平台：Ascend950；证据等级: tool_observed）
- 两轮输入字节级一致，md5 `089b00828bc9909074bb4e5750ccec3b`（来源平台：Ascend950；证据等级: tool_observed）
- 两轮执行不同 kernel 符号：`_1000@0x0000`(AR_FULL_LOAD) vs `_20000@0x9300`(ARA_RECOMPUTE)，仿真器 PC `0x10d0dc28` vs `0x10d1e520`、RV_VMAX 编码 `0x84082b82` vs `0x80082782`、Total tick 5011 vs 8881（来源平台：Ascend950；证据等级: artifact_verified）
- sumRow0 与 sumRow1 均为 `0x3f800000`(1.0)，两模板 softmax 行和都归一，逻辑正确（来源平台：Ascend950；证据等级: tool_observed）

## 预防

- 对 SoftmaxV2 arch35 多模板场景，若下游对 bit-exact 有要求（如蒸馏参考、量化校准基线），应固定 tiling key 或固定 `IsCapable` 选择，避免同一算子在不同 tiling 路径下产出微小差异。
- 评测/对齐脚本若以某一模板输出为 golden，应在文档标注「本 golden 来自 AR_FULL_LOAD/ARA_RECOMPUTE」，不要假定模板间 bit-exact。
- 修改 `IsCapable` 加调试开关后，发布前清理 `DoOpTiling` 中的 `fprintf(stderr, "[TILING_DEBUG]...")` 打印。

## 失效边界

- 仅在 arch35 模板族、Ascend950（cannsim 仿真）验证；其他 SoC（a2/a3）未验证，推断可迁移（arch35 模板若在其他平台存在相同 reduce 树结构，应同样不 bit-exact，但需独立验证）。
- `a0!=1` 的 fused shape 下 ARA `IsCapable` 正常 capable，两模板是否同样不 bit-exact 未在本 trace 验证。
- 真实 NPU 上板未验证（仅有 cannsim 仿真）；cannsim 执行真实编译 kernel binary，与上板行为高度一致，但 bit 级舍入在上板与仿真间是否完全一致未核验。
- 触发分歧的 seed/输入是数据相关概率性事件：seed=3 触发 row0 分歧、row1 一致；其他输入可能全一致或全分歧，不能以「单次一致」推断模板 bit-exact。

## 坏实践

- 把两模板 1 ULP 差异当精度 bug 修：两套模板在数学上都正确（softmax 行和归一为 1.0），差异属 fp32 非结合律固有性质，不应通过改 reduce 树强行对齐某一模板（会牺牲另一模板的性能/UB 占用）。
- 通过放宽容差让两模板输出「通过」对比：掩盖了模板间不 bit-exact 的真实特性，下游若依赖 bit 级一致会踩坑。
