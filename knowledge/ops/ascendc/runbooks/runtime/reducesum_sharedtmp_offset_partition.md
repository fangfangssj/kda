---
type: "Runbook"
title: "ReduceSum tmp 与 src 须分区：同缓冲偏移切分"
description: "fp16/bf16 路径先 Cast 到 float 暂存再 ReduceSum 时，ReduceSum 的 sharedTmpBuffer 与 src 不可别名；同一 UB 缓冲内用 floatLocal[rowLenAlign] 起的偏移区段作独立 tmp（对齐边界切分），一次 Cast+Reduce 分区写法覆盖两种窗口路径。"
tags: [task.implementation, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T03:16:41Z'
updated_at: '2026-09-17T03:16:41Z'
platforms: ["950"]
---

# ReduceSum tmp 与 src 须分区：同缓冲偏移切分

## 现象

为一个多 dtype 池化算子写 kernel 的 fp16/bf16 计算路径：窗口数据先 `Cast` 到 float 暂存（`tmpBuf` 里的 `floatLocal`），再逐行 `ReduceSum` 求和。初版代码把同一个 `floatLocal` 同时用作 ReduceSum 的 `src` 和第三参 `tmpBuffer`（sharedTmpBuffer）——即 tmp 与 src 完全别名。对照 Ascend C ReduceSum API 最佳实践文档（api-reduce 指南中 Softmax 示例的 dst/src/reduceTmp 三块来自**不同**缓冲）确认这是错误用法：ReduceSum 实现会向 tmpBuffer 写中间结果，若 tmp 与 src 重叠，中间写入会破坏尚未读取的 src 数据。修复为在同一 `tmpBuf` 内用 `floatLocal[rowLenAlign]` 起的偏移区段作独立 sharedTmpBuffer（Cast 输出占 `[0..rowLen-1]`，tmp 用 `[rowLenAlign..]`，`rowLenAlign >= rowLen` 保证不重叠），两个窗口路径（普通窗口 `ComputeWindow` 与全局池化 `ProcessGlobalAvgPool`）同构修复。

## 适用场景

- 工程形态：Ascend C kernel 开发，UB（Unified Buffer）内做"低精度输入 → float 暂存 → 逐行 ReduceSum"的窗口/行求和路径。
- API 路径：`AscendC::ReduceSum<T, isSetMask=true>(dst, src, tmpBuffer, count)`（dst 1 元素、src count 元素、tmpBuffer 为归约算法临时区，类型与 T 一致）；`AscendC::Cast`；`LocalTensor::operator[]` 偏移取子张量。
- 验证过的平台与 CANN 版本：Ascend 950PR（arch35）+ CANN 9.1.0（fp16/bf16 全量 NPU ST 通过）。
- 适用粒度：`operator_family`（所有用 ReduceSum 二级接口的 kernel 通用）。
- 路线优先级：recommended_fastpath（tmp 独立是 API 正确性要求，不是优化项）。
- 明确不适用/未验证场景：`ReduceSum` 一级接口（不带 tmpBuffer 参数的形态）无此约束；tmpBuffer 的精确最小尺寸要求未在 trace 中量化（本例用缓冲剩余空间，充足）。

## 根因/解决配方

ReduceSum 的 `tmpBuffer` 是归约算法的中间写区，与 `src`（乃至 `dst`）必须不重叠；API 最佳实践的标准形态是三块独立缓冲。当 fp16/bf16 路径已经把 Cast 输出放进 `floatLocal` 时，不需要额外申请新 TBuf——利用同一 `tmpBuf` 的剩余空间，从 `floatLocal[rowLenAlign]` 起切一段作 tmp 即可：`tmpBuf` 容量为 `maxRowsPerBatch * alignedW * sizeof(float)`，Cast 单行只用 `[0..rowLen-1]`（`rowLen <= rowLenAlign <= alignedW`），偏移 `rowLenAlign` 之后的 `(maxRowsPerBatch - 1) * alignedW` 个 float 足够归约 tmp 使用。

修复前（别名写法，两处窗口路径同构）：

```cpp
            // 2. 逐行 ReduceSum（UB 中每行按 rowLenAlign 对齐，需逐行处理避免读入 padding）
            LocalTensor<float> floatLocal = tmpBuf.Get<float>();
            LocalTensor<float> reduceResult = reduceTmpBuf.Get<float>();

            for (uint32_t row = 0U; row < rowsThisBatch; row++) {
                uint32_t rowOffset = row * rowLenAlign;
                if constexpr (std::is_same_v<D_T_X, float>) {
                    ReduceSum<float>(reduceResult, xIn[rowOffset], floatLocal,
                                     static_cast<int32_t>(rowLen));
                } else {
                    Cast(floatLocal, xIn[rowOffset], RoundMode::CAST_NONE, rowLen);
                    ReduceSum<float>(reduceResult, floatLocal, floatLocal,
                                     static_cast<int32_t>(rowLen));
                }
                partialSum += reduceResult.GetValue(0);
            }
```

修复后（偏移分区，`ComputeWindow` 终版；`ProcessGlobalAvgPool` 同构）：

```cpp
            // 2. 逐行 ReduceSum（UB 中每行按 rowLenAlign 对齐，需逐行处理避免读入 padding）
            LocalTensor<float> floatLocal = tmpBuf.Get<float>();
            LocalTensor<float> reduceResult = reduceTmpBuf.Get<float>();

            for (uint32_t row = 0U; row < rowsThisBatch; row++) {
                uint32_t rowOffset = row * rowLenAlign;
                if constexpr (std::is_same_v<D_T_X, float>) {
                    ReduceSum<float>(reduceResult, xIn[rowOffset], floatLocal,
                                     static_cast<int32_t>(rowLen));
                } else {
                    // fp16/bf16: 先 Cast 到 float，再 ReduceSum
                    // Cast 输出放在 floatLocal[0..rowLen-1]
                    // ReduceSum sharedTmpBuffer 使用 floatLocal[rowLenAlign..] 避免与 src 重叠
                    Cast(floatLocal, xIn[rowOffset], RoundMode::CAST_NONE, rowLen);
                    LocalTensor<float> reduceSharedTmp = floatLocal[rowLenAlign];
                    ReduceSum<float>(reduceResult, floatLocal, reduceSharedTmp,
                                     static_cast<int32_t>(rowLen));
                }
                partialSum += reduceResult.GetValue(0);
            }
```

缓冲角色账（本例）：`reduceTmpBuf`（4096B）只承担 ReduceSum 的 dst（1 个 float，32B 对齐即够）；`tmpBuf`（`maxRowsPerBatch * alignedW * sizeof(float)`）承担 Cast 输出（行首区段）+ ReduceSum tmp（`rowLenAlign` 偏移后的剩余区段）；fp32 路径不受影响（其 src 在 `xIn`，`floatLocal` 天然只当 tmp 用）。

## 设计决策

- **为什么用偏移分区而不是新申请一个 TBuf**：`LocalTensor::operator[]` 以元素偏移取子张量，零拷贝零分配；新 TBuf 需要重排 TPipe 管道与内存账，收益相同成本更高。
- **为什么从 `rowLenAlign` 而不是 `rowLen` 偏移**：`rowLenAlign >= rowLen`（对齐粒度），用对齐边界切分保证 tmp 起址满足对齐要求且与 Cast 区段之间留出对齐空隙。
- **输出端复用同一 `tmpBuf`**：行级归约全部完成后 `tmpBuf` 生命周期结束，输出侧 `Duplicate<float>`+`Cast` 回低精度可再次从 `tmpBuf.Get<float>()` 复用（时序上不冲突）。
- 放弃的候选：无（别名写法本身即错误，不存在权衡）。
- 当前方案属 recommended_fastpath。

## 系统信息

- 芯片/架构：Ascend 950PR（DAV_3510，arch35）；CANN 9.1.0。
- 相关文件：`op_kernel/arch35/bench_no_harness_adaptive_avg_pool_3d.h`（`ComputeWindow` 与 `ProcessGlobalAvgPool` 两处循环）；API 参考：Ascend C Reduce API 最佳实践（api-reduce 指南，ReduceSum 签名与 Softmax 三缓冲示例）。
- Kernel 二进制：`BenchNoHarnessAdaptiveAvgPool3d_*_{0,1,27}` 三变体（float/half/bfloat16_t 模板实例）。

## 预防

- 用 `ReduceSum<T>(dst, src, tmp, count)` 前画一遍三块区域的内存区间图，确认 dst/src/tmp 两两不重叠。
- 低精度升 float 暂存的路径，"暂存区"与"归约 tmp"是两个角色，即使物理上同处一个 TBuf 也必须分区。

## 验证信号

来源：一次多 dtype 池化算子 kernel 实现会话（Ascend950PR + CANN 9.1.0）：按 API 最佳实践文档把别名 tmp 改为偏移分区后编译并逐 dtype 上板（API 文档依据见上文根因段）。

- 修复后 `bash build.sh --soc=ascend950 -j8` 编译成功，fp16/bf16 二进制变体（`_1`/`_27`）正常生成（来源平台：950；证据等级：tool_observed）。
- NPU ST Real 分 dtype 全量：L1_fp16 51/51（最大 MARE 8.15e-04 < 阈值 9.77e-04）、L1_bf16 52/52（误差全 0.00e+00）（来源平台：950；证据等级：tool_observed）。
- PyTorch 接入测试（独立 golden 链路）同算子 fp16/bf16 全量 0 失败，交叉印证（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 偏移分区依赖 `tmpBuf` 有富余空间（`maxRowsPerBatch > 1`）；若缓冲只够单行，需换"申请独立 TBuf"或缩小批量。
- `rowLen` 极大（单行占满 `alignedW * maxRowsPerBatch`）时剩余区段不足，需重新做缓冲账；本例未触碰该边界。
- tmp 与 src 别名在本开发会话 中**未实测复现故障**（实现期即按文档纠正），别名致错结论来自 API 文档与实现机制（`ReduceSum` 向 tmp 写中间结果），非本卡的一手失败观测——使用时以此为准。
- ReduceSum tmpBuffer 的最小尺寸公式未量化（文档示例用 32KB 量级缓冲），本卡只保证"不重叠 + 空间充足"。
- 未验证，推断可迁移：其他 NPU 架构上 ReduceSum tmpBuffer 的别名行为未实测。

**未决项**：

- 别名写法在本算子上是否必然产生可观测错误输出（构造对照实验实测）未做；如需机制级证据可专门复现。
- ReduceSum tmpBuffer 最小尺寸的官方公式待查证（当前按"剩余空间充足"使用）。
## 坏实践

- `ReduceSum(reduceResult, floatLocal, floatLocal, rowLen)` 一写了事——tmp 与 src 同一 Tensor，中间写入破坏 src。
- 为图省事把 fp32 路径的 tmp（`floatLocal`）与低精度路径共用同一写法而不画内存区间——fp32 恰好不别名，掩盖了低精度路径的隐患。
