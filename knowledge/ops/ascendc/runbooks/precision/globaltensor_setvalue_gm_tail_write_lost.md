---
type: "Runbook"
title: "GlobalTensor::SetValue 写丢失，改 DataCopyPad"
description: "AIV kernel 用 GlobalTensor::SetValue 逐元素写输出时，kernel 内 GetValue 读回正确、host 侧却读到 0（尾部元素丢失，MARE≈1.0）；PipeBarrier<PIPE_ALL> 无效，须在 UB 收集后用 DataCopyPad（MTE3）批量写出 GM。R2：多核切分下表现为某核首个 (n,co) 输出对的首元素恒 0（MERE 过、MARE=1.00），barrier/dummy 写均无效，同为\"标量 GM 写不落地、换向量管道批量写恢复\"。"
tags: [task.debugging, topic.memory, topic.data_movement.memory_access, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-16T11:11:42Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# GlobalTensor::SetValue 写丢失，改 DataCopyPad

## 现象

在 Ascend950PR（arch35，CANN 9.1.0）上联调自定义 ROIAlign 算子（AIV kernel）的收尾阶段，一个 fp16 用例（x=2x256x64x64，boxes=50，oh=ow=7，共 490 个输出元素）持续失败，报错原文：

```
[FAIL] MERE=1.111e-01 (>= 1.221e-04), MARE=9.960e-01 (raw=9.960e-01, >= 1.221e-03)
```

MARE≈1.0、大部分元素失配，是"输出丢值"而非数值误差形态。当时 kernel 的输出写出方式是三重循环里逐元素 `yGm_.SetValue(...)`（fp32 累加缓冲 `accLocal.GetValue` 取值后 `static_cast<T>` 回写）。为定位问题，在 kernel 内埋 printf 打印写出前后的值，观测到决定性证据链：

1. 累加与 Muls（乘 1/count）全部正确：`[DBG after Muls] acc[48]=0.372551`、`acc[42]=0.104007`；
2. 写出循环在执行且值正确：`[DBG writeout roi=0 oh=6 ow=6] srcOff=48 dstOff=48 val=0.372551`；
3. **SetValue 之后立即用 `yGm_.GetValue` 读回，值正确**：`readback=0.372551`——即 kernel 内视角写入成功；
4. 但测试进程（host 侧）从 GM 读到的输出里这些元素是 `0.000000`。

即：Scalar 写在 kernel 生命周期内可见，kernel 返回后对 host 不可见——尾部若干次 `GlobalTensor::SetValue` 的 GM 写入丢失。

## 适用场景

- 工程形态：aclnn 自定义算子（AscendC AIV kernel + host TilingFunc），kernel 计算正确但 host 侧读到输出丢值/全 0。
- API 路径：`GlobalTensor::SetValue` / `GlobalTensor::GetValue`（标量 GM 访问）；修复涉及 `TBuf`/`LocalTensor::SetValue`、`DataCopyPad`（UB→GM CopyOut）、`PipeBarrier<PIPE_ALL>`、host 侧 TilingFunc 的 UB 预算。
- 验证过的平台与 CANN 版本：Ascend950PR（`--soc=ascend950`，arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该平台上任何在循环内用 `GlobalTensor::SetValue` 写 GM 输出的 AIV kernel，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：输出本来就是大块连续数据、可直接 `DataCopyPad`/`DataCopy` 整块搬出的 kernel（不应逐元素 SetValue，无此问题）；其他 NPU 架构（arch 非 3510）未验证；根因的微架构层解释（Scalar store buffer 未 flush）为本会话推理，未直接观测（见 ## 失效边界）。

## 根因

`GlobalTensor::SetValue` 的标量写经 Scalar store buffer 提交，kernel 返回时该缓冲未保证完全 flush 到 GM，尾部写入丢失；`PipeBarrier<PIPE_ALL>` 只排空各流水线的在途操作，不提交该标量写缓冲，而 MTE3 管线的 `DataCopyPad`（UB→GM）具备 GM 一致性保证。

## 解决配方

修复前（逐元素标量写 GM；曾在其后追加 `PipeBarrier<PIPE_ALL>()` 试图"冲刷"，实测无效，同用例仍 `MARE=9.960e-01`）：

```cpp
            // 标量写出（fp16 路径: static_cast<T> 将 fp32 回写为 half）
            int64_t roiOffset = roiIdx * C_ * outputHeight_ * outputWidth_;
            int64_t channelBase = cStart * outputHeight_ * outputWidth_;

            for (int64_t oh = 0; oh < ohLen; oh++) {
                for (int64_t ow = 0; ow < outputWidth_; ow++) {
                    int64_t srcOff = oh * outputWidth_ * cLen + ow * cLen;
                    int64_t dstOff = roiOffset + channelBase + (ohStart + oh) * outputWidth_ + ow;
                    for (int64_t c = 0; c < cLen; c++) {
                        float val = accLocal.GetValue(srcOff + c);
                        yGm_.SetValue(dstOff + c * outputHeight_ * outputWidth_, static_cast<T>(val));
                    }
                }
            }
            // 确保 scalar SetValue 的 GM 写入在 kernel 返回前完全提交
            // (GlobalTensor::SetValue 走 Scalar store buffer, kernel 返回时
            //  可能未完全 flush 到 GM, 导致最后几次写入丢失)
            PipeBarrier<PIPE_ALL>();
```

修复分三步。

第一步：kernel 增加输出用 UB 缓冲 `outBuf_`（成员声明）：

```cpp
    TPipe pipe;
    TBuf<TPosition::VECCALC> accBuf_;      // 累加缓冲 [ohTile * ow * cTile] fp32
    TBuf<TPosition::VECCALC> rowBufLow_;   // y_low 行缓存 [cTile * rowStride] T
    TBuf<TPosition::VECCALC> rowBufHigh_;  // y_high 行缓存 [cTile * rowStride] T
    TBuf<TPosition::VECCALC> outBuf_;      // 输出缓冲 [ohTile * ow] T (用于 DataCopyPad 写出 GM)
```

第二步：Init 中分配 `outBuf_`（32 字节对齐）：

```cpp
    if (useRowBuf_) {
        // 计算 UB 中每行的元素步长（32 字节对齐）
        int64_t blockLenAligned = (blockLenBytes + 31) / 32 * 32;
        rowStride_ = blockLenAligned / sizeof(T);
        // 分配行缓冲（cTile 通道 × rowStride 元素 × sizeof(T)）
        pipe.InitBuffer(rowBufLow_, cTile_ * rowStride_ * sizeof(T));
        pipe.InitBuffer(rowBufHigh_, cTile_ * rowStride_ * sizeof(T));
    }

    // outBuf: ohTile * outputWidth 个 T (用于 DataCopyPad 写出 GM)
    // 对齐到 32 字节边界
    int64_t outBufBytes = ohTile_ * outputWidth_ * sizeof(T);
    outBufBytes = (outBufBytes + 31) / 32 * 32;
    pipe.InitBuffer(outBuf_, outBufBytes);
}
```

第三步：写出循环改为"UB 收集 + DataCopyPad（MTE3）写 GM"：

```cpp
            // 标量写出 → DataCopyPad (MTE3) 写出 GM
            // 根因修复: GlobalTensor::SetValue 走 Scalar store buffer, kernel 返回时
            //  可能未完全 flush 到 GM, 导致最后几次写入丢失。
            //  修复方案: 用 LocalTensor.SetValue 收集到 UB outBuf, 再用 DataCopyPad
            //  (MTE3 管线) 写出 GM, MTE3 有 GM 一致性保证。
            int64_t roiOffset = roiIdx * C_ * outputHeight_ * outputWidth_;
            int64_t channelBase = cStart * outputHeight_ * outputWidth_;
            auto outLocal = outBuf_.Get<T>();
            int64_t outLen = ohLen * outputWidth_;

            for (int64_t c = 0; c < cLen; c++) {
                // Step 1: 从 accBuf (fp32, [oh, ow, c]) 收集到 outBuf (T, [oh, ow])
                for (int64_t oh = 0; oh < ohLen; oh++) {
                    for (int64_t ow = 0; ow < outputWidth_; ow++) {
                        int64_t srcOff = oh * outputWidth_ * cLen + ow * cLen + c;
                        int64_t dstOff = oh * outputWidth_ + ow;
                        float val = accLocal.GetValue(srcOff);
                        outLocal.SetValue(dstOff, static_cast<T>(val));
                    }
                }
                // 确保 scalar 写入 outBuf 对 MTE3 可见 (V→MTE3 同步)
                PipeBarrier<PIPE_ALL>();

                // Step 2: DataCopyPad outBuf → yGm_ (MTE3, UB→GM)
                int64_t gmOff = roiOffset + (cStart + c) * outputHeight_ * outputWidth_ +
                                ohStart * outputWidth_;
                DataCopyPad(yGm_[gmOff], outLocal,
                            {1, static_cast<uint16_t>(outLen * sizeof(T)), 0, 0});
            }
            // 确保所有 MTE3 写出在 kernel 返回前提交到 GM
            PipeBarrier<PIPE_ALL>();
```

第四步：host 侧 tiling 的 UB 预算同步计入 `outBuf`（否则 cTile 会算大、UB 越界）：

```cpp
    int64_t ubAvail = static_cast<int64_t>(static_cast<float>(ubSize) * UB_USAGE_RATIO);

    // outBuf 大小 (固定, 不随 cTile 缩放), 32 字节对齐
    int64_t outBufBytes = (ohTile * ow * typeSize + 31) / 32 * 32;

    int64_t denom;
    int64_t fixedOverhead;
    if (useRowBuf) {
        // 行缓冲路径：accBuf + 2*rowBuf 系数
        int64_t wAlignedBytes = (blockLenBytes + 31) / 32 * 32;
        denom = ohTile * ow * FLOAT_SIZE + 2 * wAlignedBytes;
        fixedOverhead = BOXES_BUF_SIZE + outBufBytes;
    } else {
        // 回退路径：仅 accBuf + 少量 workBuf 预留
        denom = ohTile * ow * FLOAT_SIZE + 4 * FLOAT_SIZE;
        fixedOverhead = BOXES_BUF_SIZE + outBufBytes;
    }
```

判别 / 定位方法（本会话走通的链路，可复用）：

1. 看失败形态：MERE/MARE 双超、失配元素 actual=0 而 golden 非零、且分布在"尾部/部分"元素 → 疑似写丢失而非计算错。
2. kernel 内 printf 分段证明：打印写出前的累加值（after Muls）——正确则排除计算与 tiling。
3. 写出点旁插读回探针：`SetValue` 后立刻 `yGm_.GetValue` 打印——读回正确而 host 读 0，即锁定"kernel 内可见、返回后丢失"。
4. 对照验证：换 `DataCopyPad` 写出后同样位置读回 + host 比对恢复一致。

## 设计决策

- 为什么不选"SetValue 后加 PipeBarrier"：先实测过 `PipeBarrier<PIPE_ALL>()` 追加在写出循环后，同用例仍失败——屏障不等价于标量写缓冲向 GM 的提交，方案被数据否决。
- 为什么选"UB outBuf 收集 + DataCopyPad"而不是逐元素小段 DataCopyPad：输出布局对 cLen>1 是通道交错的，逐 (oh,ow) 搬运块长 <32 字节（小块搬运已被同算子迭代一证明有问题）；按通道 c 聚集成 [oh, ow] 连续段后一次搬出，块长 = outLen*sizeof(T)（本例 49*2=98 字节），既满足搬运粒度又走 MTE3 一致性路径。
- UB 预算归属：outBuf 不随 cTile 缩放，计入 fixedOverhead 而非 denom；遗漏这一步会让 tiling 算出过大的 cTile 导致 UB 超额。
- 当前方案路线档位：recommended_fastpath（正确性必需；MTE3 写出是本平台上可验证的 GM 一致性路径）。

## 系统信息

- 硬件/软件：Ascend950PR（`npu-smi` OK，`--soc=ascend950`，arch35/DAV_3510），CANN 9.1.0；测试经 `bash build.sh --soc=ascend950 -j8` + `build/custom_opp_ubuntu_x86_64.run --force` 安装后定向单用例复跑（tests/st/run.sh --real）。
- 相关文件：`op_kernel/arch35/<op>.h`（写出循环、outBuf_ 成员与 InitBuffer）、`op_host/arch35/<op>_tiling.cpp`（UB 预算公式）、`tests/st/test_aclnn_<op>.cpp`（MERE/MARE 比对）。

## 验证信号

来源说明：本结论来自该算子迭代三联调会话——目标 fp16 用例修复（SetValue→UB 收集+DataCopyPad 写出）前后同环境实测。

- 修复前目标用例 FAIL：`MERE=1.111e-01 (>= 1.221e-04), MARE=9.960e-01`，写出读回探针显示 kernel 内 `readback=0.372551` 而 host 侧对应元素为 0（来源平台：950；证据等级：tool_observed）。
- 修复后目标用例 PASS：`MERE=2.851e-06, MARE=0.000e+00 (raw=8.736e-05, threshold=1.221e-04, 490 elems, mixed abs+rel abs_floor=1e-05)`（来源平台：950；证据等级：tool_observed）。
- DataCopyPad 后读回与累加值一致：`yGm[42]=0.104007`、`yGm[48]=0.372551`，与 after-Muls 的 acc 值逐项相等（来源平台：950；证据等级：tool_observed）。
- 回归不破坏其他用例：全量 ST Real 36/36 通过（`汇总: 通过 36 / 失败 0 / 总计 36`）、Mock 36/36、UT 80/80（46 op_host + 34 op_api）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论在 CANN 9.1.0 / arch35（Ascend950PR）上验证；其他架构未验证，推断可迁移（机制为管线提交语义，属平台通用陷阱，但需按目标架构复测）。
- 微架构层的"Scalar store buffer 未 flush"解释是机制推理（与"kernel 内读回正确 / host 读 0 / PipeBarrier 无效 / 换 MTE3 恢复"四点观测自洽），未做寄存器级验证；不影响修复配方的有效性。
- 修复仅覆盖"写丢失"根因；若输出丢值由切分/同步（如多核竞态）导致，本配方不适用——须先用读回探针区分。

**未决项**：

- 【优先】复现并追根因到机制层：Scalar 写缓冲在 kernel 返回时的提交语义（为何 PipeBarrier<PIPE_ALL> 不生效、MTE3 为何有保证），可用最小 kernel + 不同屏障组合实验。
- UT 的 `CalcExpectedCTile` 期望公式未同步 tiling 实现新增的 `outBufBytes` 项（当前用例未触发差异；静态检视结论，未实测失败）——后续对齐并补边界用例。
## R2 增量：多核首个 (n,co) 对首元素丢写（直接卷积 kernel，独立 trace，Ascend950PR + CANN 9.1.0）

同一"逐元素 `GlobalTensor::SetValue` 写 GM 不可靠、UB 聚合 + DataCopyPad 批量写恢复"的机制在多核直接卷积 kernel 上的形态：

- 触发指纹：ST 用例 L0-006（x[2,7,13,13] filter[7,7,3,3] y[2,7,13,13] fp16，tiling 按 (N,C_out) 对切 2 核）中 2366 个输出元素**恰好 1 个**错——`不匹配 [1183]: golden=7.670255, actual=0.000000, rel_err=1.00e+00`，MERE=6.02e-04 通过、MARE=1.00e+00 超阈。y[1183] 即 y[1,0,0,0]——core 1 分到的第一个 (n=1, co=0) 输出对的首元素恒为 0（输出 buffer 已 memset 0，等价该写没发生）；其余 2365 个元素（含 core 1 后续写的全部元素）全部正确。失败与用例顺序无关（非 tiling 缓存）。
- 无效手段逐一实测排除（全部不改变 y[1183]=0）：① `Process()` 末尾加 `pipe_barrier(PIPE_ALL)`；② 移除 `Init()` 中未使用的 `pipe_.InitBuffer(ubBuf_, 1024)`；③ 正式计算前对该位置做 dummy `SetValue(0)` + `pipe_barrier(PIPE_ALL)` 预热写管道。barrier 类手段全部无效，与主卡结论一致——不是"写未提交可被冲刷"的时序问题形态。
- 修复（与主卡同构）：把"逐元素标量 GM 写"重构为"按 (n,co) 对在 UB LocalTensor 累加 + 每对一次 DataCopyPad 批量写"。UB 输出缓冲按 `HOut*WOut` 个元素分配、字节数向上对齐到 32B（`(outBytes + 31) & ~31`）；DataCopyPad 的 `blockLen` 按实际字节给（如 169×2=338 字节非 32 倍数），GM 侧由 DataCopyPad 处理非对齐尾块。修复后 L0-006 单独跑 PASS；全量 L0 43/43 PASS。骨架阶段"占位拷贝输出全 0 → 标量直算 40/43"的上下文见[骨架占位输出全 0 卡](skeleton_placeholder_output_zero_scalar_conv.md)。修复后的写出段（fp16 主路径，verbatim 节选）：

```cpp
        if constexpr (std::is_same_v<D_T_X, half> || std::is_same_v<D_T_X, float>) {
            // fp16/fp32: 用 float 累加确保精度，结果写入 UB，再 DataCopyPad 到 GM
            LocalTensor<D_T_X> yLocal = outBuf_.Get<D_T_X>();

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
                    yLocal.SetValue(static_cast<int64_t>(ho * WOut_ + wo), static_cast<D_T_X>(sum));
                }
            }

            // 使用 DataCopyPad 将 UB 结果批量写入 GM（替代逐元素 SetValue）
            // DataCopyPad 使用向量管道，避免标量写入的缓存一致性问题
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.blockLen = static_cast<int32_t>(outElems * sizeof(D_T_X));
            DataCopyPad(yGM_[gmOffset], yLocal, copyParams);
        } else {
            // bf16: 标量 float 转换在 kernel 后端不支持，后续迭代用 Cast 向量算子实现
            LocalTensor<D_T_X> yLocal = outBuf_.Get<D_T_X>();
            D_T_X biasVal = biasGM_.GetValue(static_cast<uint64_t>(co));
            for (int64_t i = 0; i < outElems; i++) {
                yLocal.SetValue(i, biasVal);
            }
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.blockLen = static_cast<int32_t>(outElems * sizeof(D_T_X));
            DataCopyPad(yGM_[gmOffset], yLocal, copyParams);
        }
```
- 机制边界：会话内已验证的客观事实链是"多核场景某核首个 (n,co) 对的首个标量 GM 写不落地、同路径其余元素全对、换向量管道批量写后恢复"；"标量 GM 写经 copy hub 标量管道提交、多核首写可能未正确提交"的微架构假设未进一步验证。不要外推为"所有标量 GM 写都不可靠"——本例仅多核 + 首写位置复现，首写之前/之外的开发调试性 SetValue 未观察到丢失。同平台多核标量 GM 写的另一形态（SyncAll 后其他核读不到小张量/跨步访问的标量写入，最终按数据量强制单核规避）见[多核 S 管线标量写跨核不可见卡](multicore_scalar_gm_write_cross_core_invisible.md)——那边是跨核读可见性问题，与本卡的写落地问题互补。
- 坏实践增补：多核 kernel 用逐元素 `GlobalTensor::SetValue` 写满整个输出——既踩首写丢失，也放弃向量管道吞吐；用"数据看起来大多正确"判断搬运正确性（竞态类错误对时序敏感，小用例可能偶过）。
- 误诊增补：最初怀疑 `Cast(float→int64)` 不被 arch35 支持（文档只列 half↔float、half→int32 等），差点改成 int32 中转；实际该 Cast 工作正常，垃圾值来自缺失的同步/写路径——文档未列出的类型组合不等于不支持，先排除写入路径根因。

R2 验证信号：

- 修复前：L0-006 失配恰 1 个元素（index 1183，golden=7.670255，actual=0.000000，MARE=1.00e+00）（来源平台：950；证据等级：tool_observed）。
- 无效手段实测：末尾 `pipe_barrier(PIPE_ALL)`、移除未用 `InitBuffer`、dummy 写 + barrier——三者之后 L0-006 仍 FAIL（同一元素为 0）（来源平台：950；证据等级：tool_observed）。
- 修复后（UB + DataCopyPad）：L0-006 单独跑 PASS；前 25 用例 25/25 PASS；全量 L0 43/43 PASS（来源平台：950；证据等级：tool_observed）。

## 坏实践

- 在长循环里逐元素 `GlobalTensor::SetValue` 写 GM 输出：本例直接导致尾部写入丢失；应整段收集后用 MTE3 搬出。
- 把 `PipeBarrier<PIPE_ALL>()` 当作"标量写冲刷"使用：实测对本问题无效，不能替代 MTE3 写出路径。

## 误诊

- 最初怀疑累加/Muls 计算错误：kernel 内 printf 显示 after-Muls 累加值全部正确，排除（分析后经工具输出证实）。
- 怀疑写出偏移（dstOff）计算错：printf 打印 srcOff/dstOff 与 val 均正确，排除。
- "Device do not support double dtype"类警告与本问题无关（另一会话torch 适配层场景），不要被带偏。
