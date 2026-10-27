---
type: "Runbook"
title: 骨架占位输出全 0：标量直算先打通正确性
description: 迭代骨架 kernel 只做 x→y 拷贝（无卷积/bias/filter 计算）时 ST 输出全 0（actual=0.000000、MERE≈1.2）；重写为按 (N,Cout) pair 多核切分的标量直接卷积（GetValue/SetValue + float 累加）后 40/43 通过。
tags: [task.debugging, topic.precision, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# 骨架占位输出全 0：标量直算先打通正确性

## 现象

BenchNoHarnessConv2d 联调验证修好执行链路（GetWorkspaceSize 与 double free）后，43 个 L0 用例**全部**数值失败，失败模式高度一致——输出全 0：

```
[Real] L0-001 - ResNet 3x3 standard block (H_out=56,W_out=56)
  x[1,64,56,56] filter[64,64,3,3] y[1,64,56,56] dtype=float16
  [FAIL] MERE=1.26e+00 (>= 9.77e-04), MARE=2.44e+03 (>= 9.77e-03)
    不匹配 [0]: golden=27.877716, actual=0.000000, rel_err=1.00e+00
    不匹配 [1]: golden=-1.857936, actual=0.000000, rel_err=1.00e+00
...
总计: 43  通过: 0  失败: 43
```

当时 kernel 状态：骨架阶段的 `Process()` 只做"x 前 N 个元素拷贝到 y"的 DataCopyPad 占位（无 bias 加法、不读 filter、无卷积计算），而 x 与 y 形状本就不同（Cin≠Cout、H≠H_out），拷贝语义本身错误，输出停留在 memset 的 0。

## 适用场景

- 工程形态：自定义算子 ST/NPU 精度验证阶段，kernel 为骨架/占位实现；需要先把"正确但未优化"的可验证路径打通。
- API 路径：`GlobalTensor::GetValue/SetValue` 标量 GM 读写；`GetBlockIdx()` 多核切分；`GET_TILING_DATA_WITH_STRUCT` tiling 数据。
- 验证过的平台与 CANN 版本：Ascend950PR（56 AIV），CANN 9.1.0。
- 适用粒度：same_operator（直接卷积方案本身）→ 联调打通方法论层面可迁移。
- 路线优先级：safety_fallback（正确性优先的过渡实现；正式路线是 Load3D+MMAD+Fixpipe Cube 路径）。
- 明确不适用 / 未验证场景：性能敏感场景（标量路径极慢，未测性能）；bf16 dtype（占位分支，标量 float 转换后端不支持）；超大 shape 的耗时上限未评估。

## 根因

骨架 `Process()` 未实现任何卷积计算——既不读 filter，也不做累加，仅把输入前段元素拷到输出；输出缓冲被测试侧 `aclrtMemset` 清 0 后从未被有效写入非零值，与 golden（完整卷积+bias）比对必然全失配（rel_err=1.00e+00、actual=0.000000）。输出未被有效写入的红基线形态判别另见[ST 红基线输出 poison 防 stub 巧合命中卡](st_poison_red_baseline_unwritten_output_signature.md)。

## 解决配方

修复前（骨架 Process，占位拷贝）：

```cpp
template <typename D_T_X, int TILING_MODE>
__aicore__ inline void Conv2d<D_T_X, TILING_MODE>::Process()
{
    // 骨架阶段：简单 DataCopyPad 路径（将 x 前N个元素拷贝到 y）
    // 后续迭代将实现完整的 Load3D + MMAD + Fixpipe + Add Cube 路径
    LocalTensor<D_T_X> inUB = ubInBuf_.Get<D_T_X>();
    LocalTensor<D_T_X> biasUB = ubBiasBuf_.Get<D_T_X>();
    LocalTensor<D_T_X> outUB = ubOutBuf_.Get<D_T_X>();

    // 加载 bias
    DataCopyParams biasCopyParams;
    biasCopyParams.blockCount = 1;
    biasCopyParams.blockLen = static_cast<uint32_t>(COut_ * sizeof(D_T_X));
    biasCopyParams.srcStride = 0;
    biasCopyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(biasUB, biasGM_, biasCopyParams, padParams);

    // 骨架：简单拷贝 x → y（占位实现，后续迭代替换为完整卷积计算）
    int64_t totalOutput = totalElements_;
    int64_t perCore = (totalOutput + usedCoreNum_ - 1) / usedCoreNum_;
    int64_t myStart = blockIdx_ * perCore;
    int64_t myEnd = (myStart + perCore > totalOutput) ? totalOutput : (myStart + perCore);
    int64_t myCount = myEnd - myStart;

    if (myCount <= 0) {
        return;
    }

    // 分块拷贝
    int64_t ubElements = 4096;
    int64_t loopCount = (myCount + ubElements - 1) / ubElements;

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (myCount - ubElements * i) : ubElements;

        // GM → UB
        DataCopyParams inCopyParams;
        inCopyParams.blockCount = 1;
        inCopyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(D_T_X));
        inCopyParams.srcStride = 0;
        inCopyParams.dstStride = 0;
        DataCopyPad(inUB, xGM_[myStart + i * ubElements], inCopyParams, padParams);

        // UB → GM (直接输出，后续迭代添加 bias 加法和卷积计算)
        DataCopyExtParams outExtParams;
        outExtParams.blockCount = 1;
        outExtParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(D_T_X));
        outExtParams.srcStride = 0;
        outExtParams.dstStride = 0;
        outExtParams.rsv = 0;
        DataCopyPad(yGM_[myStart + i * ubElements], inUB, outExtParams);
    }
}
```

修复后（标量直接卷积：Init 补 filterGM_ 绑定；Process 按 (N, C_out) pair 切分逐元素计算；fp16/fp32 走 float 累加，bf16 占位）：

```cpp
template <typename D_T_X, int TILING_MODE>
__aicore__ inline void Conv2d<D_T_X, TILING_MODE>::Init(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, const BenchNoHarnessConv2dTilingData* tilingData)
{
    N_ = tilingData->N;
    CIn_ = tilingData->CIn;
    H_ = tilingData->H;
    W_ = tilingData->W;
    COut_ = tilingData->COut;
    HOut_ = tilingData->HOut;
    WOut_ = tilingData->WOut;
    KH_ = tilingData->KH;
    KW_ = tilingData->KW;
    strideH_ = tilingData->strideH;
    strideW_ = tilingData->strideW;
    padTop_ = tilingData->padTop;
    padLeft_ = tilingData->padLeft;
    dilationH_ = tilingData->dilationH;
    dilationW_ = tilingData->dilationW;
    blockIdx_ = AscendC::GetBlockIdx();
    blockFactor_ = tilingData->blockFactor;
    usedCoreNum_ = tilingData->usedCoreNum;

    totalElements_ = N_ * COut_ * HOut_ * WOut_;

    // 设置 GM buffer
    xGM_.SetGlobalBuffer((__gm__ D_T_X*)x);
    filterGM_.SetGlobalBuffer((__gm__ D_T_X*)filter);
    biasGM_.SetGlobalBuffer((__gm__ D_T_X*)bias);
    yGM_.SetGlobalBuffer((__gm__ D_T_X*)y);

    // UB buffer 预留（后续迭代优化时用于 DataCopyPad 批量搬运）
    pipe_.InitBuffer(ubBuf_, 1024);
}

template <typename D_T_X, int TILING_MODE>
__aicore__ inline void Conv2d<D_T_X, TILING_MODE>::Process()
{
    // 直接卷积实现（标量 GM 读写，正确但未优化）
    // 后续迭代将替换为 Load3D + MMAD + Fixpipe Cube 路径。
    // 多核切分：按 (N, C_out) 二维任务分配给各核
    int64_t totalPairs = N_ * COut_;
    int64_t perCore = (totalPairs + usedCoreNum_ - 1) / usedCoreNum_;
    int64_t myStart = blockIdx_ * perCore;
    int64_t myEnd = (myStart + perCore > totalPairs) ? totalPairs : (myStart + perCore);

    for (int64_t pair = myStart; pair < myEnd; pair++) {
        int64_t n = pair / COut_;
        int64_t co = pair % COut_;

        if constexpr (std::is_same_v<D_T_X, half> || std::is_same_v<D_T_X, float>) {
            // fp16/fp32: 用 float 累加确保精度
            float biasVal = static_cast<float>(biasGM_.GetValue(static_cast<uint64_t>(co)));

            for (int64_t ho = 0; ho < HOut_; ho++) {
                for (int64_t wo = 0; wo < WOut_; wo++) {
                    float sum = biasVal;
                    for (int64_t ci = 0; ci < CIn_; ci++) {
                        for (int64_t kh = 0; kh < KH_; kh++) {
                            int64_t hi = ho * strideH_ - padTop_ + kh * dilationH_;
                            if (hi < 0 || hi >= H_) continue;
                            for (int64_t kw = 0; kw < KW_; kw++) {
                                int64_t wi = wo * strideW_ - padLeft_ + kw * dilationW_;
                                if (wi < 0 || wi >= W_) continue;
                                float xv = static_cast<float>(xGM_.GetValue(static_cast<uint64_t>(
                                    ((n * CIn_ + ci) * H_ + hi) * W_ + wi)));
                                float fv = static_cast<float>(filterGM_.GetValue(static_cast<uint64_t>(
                                    ((co * CIn_ + ci) * KH_ + kh) * KW_ + kw)));
                                sum += xv * fv;
                            }
                        }
                    }
                    yGM_.SetValue(static_cast<uint64_t>(
                        ((n * COut_ + co) * HOut_ + ho) * WOut_ + wo), static_cast<D_T_X>(sum));
                }
            }
        } else {
            // bf16: 标量 float 转换在 kernel 后端不支持，后续迭代用 Cast 向量算子实现
            D_T_X biasVal = biasGM_.GetValue(static_cast<uint64_t>(co));
            for (int64_t ho = 0; ho < HOut_; ho++) {
                for (int64_t wo = 0; wo < WOut_; wo++) {
                    yGM_.SetValue(static_cast<uint64_t>(
                        ((n * COut_ + co) * HOut_ + ho) * WOut_ + wo), biasVal);
                }
            }
        }
    }
}
```

输入索引通式（stride/pad/dilation 通用，越界跳过即隐式 zero-padding）：

```
hi = ho * strideH - padTop + kh * dilationH；wi = wo * strideW - padLeft + kw * dilationW
x 下标 = ((n*CIn + ci)*H + hi)*W + wi；filter 下标 = ((co*CIn + ci)*KH + kh)*KW + kw
y 下标 = ((n*COut + co)*HOut + ho)*WOut + wo
```

## 设计决策

- 为什么选标量 GM 读写（GetValue/SetValue）而非向量流水：联调验证的首要目标是"数值正确可比对"，标量直接卷积最短路径闭环；Load3D+MMAD+Fixpipe 的 Cube 路径在骨架阶段还须处理 SPLIT_CORE 守卫，风险大。属正确性优先的过渡实现。
- 为什么用 float 累加：避免 fp16 逐元素累加的精度损失（也顺带绕开 bf16 标量运算限制）；阈值 MERE<9.77e-04 / MARE<9.77e-03 下 fp16 用例可通过。
- 切分粒度用 `N*COut` 个 (n, co) pair 而非 tiling 的任务数：两者语义不同（tiling 的 usedCoreNum 按 tile 任务数算），但 blockDim=usedCoreNum 与核内 usedCoreNum_ 一致，pair 切分仍完整覆盖且无重叠——该差异是 L0-006 类边界问题的怀疑点之一，见遗留问题。
- 当前方案路线档位：safety_fallback——相对代价：标量路径性能远劣于 Cube 路径（无量化数据，未测性能；仅保证正确性）。

## 预防

- 骨架阶段就以"能通过 ST 数值比对"为验收线，占位拷贝类 Process 不要进联调；至少实现 bias + 直接卷积的正确性版本。
- 输出全 0 + rel_err=1.00 + 编译运行都正常 → 先查 kernel 是否真的写了输出（写点/写值），再查搬运参数。

## 验证信号

来源说明：以下信号来自该算子联调验证会话：重写 kernel 后重新出包安装并跑全量 L0。

- 修复前：43/43 全部 `[FAIL]`，actual=0.000000、MERE≈1.18~1.66、rel_err=1.00e+00（来源平台：950；证据等级：tool_observed）。
- 修复后：ST 40/43 通过（如 L0-001 MERE/MARE 在阈值内），失败仅剩 L0-006/L0-021/L0-024 三例（来源平台：950；证据等级：tool_observed）。
- 复跑稳定：重建重装后再次全量运行仍 40/43，三例失败信号逐位复现（L0-006 MERE=6.02e-04/MARE=1.00e+00；L0-021 MERE=7.49e+00/MARE=5.71e+04；L0-024 MERE=1.76e-04/MARE=8.09e-02）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 复杂度 O(N·Cout·Hout·Wout·Cin·KH·KW)，大 shape（如 L0-024 的 1×64×224×224）耗时显著但可通过；性能不可接受时必须换 Cube 路径（未验证）。
- bf16 dtype 走占位分支（输出 bias），不具数值正确性。
- 残留 3 个失败用例中，L0-006（仅 y[1,0,0,0] 一元素 actual=0）的根因是逐元素标量 GM 写在多核场景的首写丢失，修复见[GlobalTensor::SetValue 写丢失，改 DataCopyPad 卡](globaltensor_setvalue_gm_tail_write_lost.md)；L0-021（dilation=3 全错）与 L0-024（near-zero MARE）不属本卡闭合范围，见遗留问题。

**未决项**：

- L0-021（dilation=3 全错）与 L0-024（near-zero MARE）移交后续迭代追根因。
- tiling 的 usedCoreNum 语义与 kernel 切分粒度的对齐规范未沉淀（后续迭代重构时处理）。
## 坏实践

- 用"x 拷到 y"的占位 Process 期望联调能过（形状不同 + 无计算，必然全 0）。
- 把输出全 0 当作搬运/对齐问题先去调 DataCopyPad 参数（本例写点根本不存在）。

## 误诊

- 无（本卡根因即"未实现"，重写后信号直接翻转）。
