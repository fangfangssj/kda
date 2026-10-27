---
type: "Runbook"
title: arch35 无 Xor/And/ShiftRight 标量重载，骨架改标量
description: CANN 9.1.0 arch35（Ascend950）上 Xor/And/ShiftRight 等向量位运算 API 不存在 (dst, src, scalar, count) 四参简化标量重载，标量形参还必须与 tensor 元素类型 T 完全一致；Compare 枚举是 CMPMODE、ReduceSum 返回 void。radix sort 骨架版改用 GetValue/SetValue 标量循环回退并通过编译。
tags: [task.implementation, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: '2026-09-17T04:12:25Z'
updated_at: '2026-09-17T04:12:25Z'
platforms: ["950"]
---

# arch35 无 Xor/And/ShiftRight 标量重载，骨架改标量

## 现象

为 BenchNoHarnessUnique 算子（去重，radix sort 路线）编写骨架版 kernel 时，按通用 AscendC 文档的常见写法用向量 API 实现位运算：对 `LocalTensor<uint16_t>` 调用四参"标量"形式做符号位翻转与 4-bit 数字提取，用 `Compare + Select + ReduceSum` 做直方图统计。第一次面向 ascend950 编译（`bash build.sh --soc=ascend950 -j8`）时 kernel 共报 17 个错误，报错原文（节选）：

```
op_kernel/arch35/bench_no_harness_unique.h:188:9: error: no matching function for call to 'Xor'
        Xor(keysU16[1], keysU16[1], static_cast<uint16_t>(0x8000), chunkSize);
        ^~~
note: candidate template ignored: could not match 'LocalTensor<T>' against 'uint16_t' (aka 'unsigned short')

op_kernel/arch35/bench_no_harness_unique.h:216:5: error: no matching function for call to 'ShiftRight'
op_kernel/arch35/bench_no_harness_unique.h:217:5: error: no matching function for call to 'And'
op_kernel/arch35/bench_no_harness_unique.h:229:51: error: use of undeclared identifier 'CompareMode'
op_kernel/arch35/bench_no_harness_unique.h:232:9: error: no matching function for call to 'Select'
op_kernel/arch35/bench_no_harness_unique.h:239:22: error: no matching function for call to 'ReduceSum'
op_kernel/arch35/bench_no_harness_unique.h:239:14: error: cannot initialize a variable of type 'half' with an rvalue of type 'void'
```

其中 ShiftRight 的失败 note 为 `deduced conflicting types for parameter 'T' ('unsigned short' vs. 'uint32_t')`——`digitShift` 变量是 `uint32_t`，与 `uint16_t` tensor 的 T 冲突；And 的失败 note 为 `could not match 'LocalTensor<T>' against 'uint16_t'`——第三参根本不接受标量。也就是说：这些 API 在该平台只有张量-张量重载或带 mask/repeatTime/repeatParams 的完整形式，没有简洁的四参标量重载。

## 适用场景

- 工程形态：AscendC kernel 开发（向量编程），在 kernel 内对 UB tensor 做位运算（符号位翻转、掩码提取、移位）与比较/规约。
- API 路径：`AscendC::Xor / And / ShiftRight / Compare / Select / ReduceSum`（`adv_api/math/xor.h`、`impl/basic_api/kernel_operator_vec_binary_scalar_intf_impl.h`、`kernel_operator_vec_cmpsel_intf.h`、`kernel_operator_reduce_intf_impl.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510）、CANN 9.1.0。
- 适用粒度：platform_wide——该平台 + 版本上所有需要向量位运算的 kernel。
- 路线优先级：骨架期 recommended_fastpath（标量回退保正确性闭环）；性能敏感路径属 safety_fallback（标量循环性能远低于向量操作，须成对评估后再用）。
- 明确不适用 / 未验证场景：A2/A3 等其他架构的 API 形态未在本轨迹验证；完整 6 参签名（mask/repeatTime/repeatParams）的正确用法仅完成头文件查阅、未编译验证。

## 根因

CANN 9.1.0 arch35 的 Xor/And/ShiftRight 向量位运算 API 没有 `(dst, src, scalarValue, count)` 四参简化标量重载（仅有张量-张量形式或需 mask/repeatTime/repeatParams 的 6 参完整形式），且存在标量形参时模板类型 T 必须与 tensor 元素类型完全一致（`uint32_t` 移位数配 `uint16_t` tensor 直接推导冲突）；同一轮失败还暴露 Compare 的枚举名是 `CMPMODE`（不是 `CompareMode`）、Select 需 6+ 参数、ReduceSum 返回 void 写入 dst 而非返回值。

## 解决配方

骨架阶段采取"标量回退"：算法逻辑（键编码、直方图、分桶 scatter、去重）全部改用 `LocalTensor::GetValue/SetValue` 标量循环，向量 API 只保留 `DataCopyPad` 搬运，编译一次通过。

修复前（kernel v1，EncodeKeys 用向量 Xor 翻转符号位）：

```cpp
template <typename D_T_X, int HAS_INVERSE>
__aicore__ inline void BenchNoHarnessUnique<D_T_X, HAS_INVERSE>::EncodeKeys()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t gmOffset = blockFactor_ * coreId_;
    int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t chunkOffset = i * ubFactor_;
        int64_t chunkSize = (chunkOffset + ubFactor_ > blockLength_)
                            ? (blockLength_ - chunkOffset) : ubFactor_;

        // GM → UB: 加载 int32 输入数据
        auto inputLocal = inputKeysBuf_.template Get<uint32_t>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = chunkSize * sizeof(D_T_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(inputLocal, inputGMX_[gmOffset + chunkOffset], copyParams,
                    DataCopyPadExtParams{false, 0, 0, 0});

        // 键编码：使用 ReinterpretCast + Xor 翻转符号位
        // ReinterpretCast uint32 → uint16，对 hi word 执行 Xor 0x8000
        auto keysU16 = inputLocal.template ReinterpretCast<uint16_t>();
        // hi word 在 index 1（小端序：lo=0, hi=1）
        // 使用 Xor 翻转 hi word 的 bit15
        Xor(keysU16[1], keysU16[1], static_cast<uint16_t>(0x8000), chunkSize);

        // UB → GM: 写入编码后的键
        DataCopyPad(keysGM_[gmOffset + chunkOffset], inputLocal, copyParams);
    }
}
```

修复前（kernel v1，ComputeHistogram 用 ShiftRight/And/Compare/Select/ReduceSum 统计 16 桶直方图）：

```cpp
template <typename D_T_X, int HAS_INVERSE>
__aicore__ inline void BenchNoHarnessUnique<D_T_X, HAS_INVERSE>::ComputeHistogram(
    uint32_t passIdx, int64_t chunkOffset, int64_t chunkSize, uint32_t* histogram)
{
    auto inputLocal = inputKeysBuf_.template Get<uint32_t>();
    auto keysU16 = inputLocal.template ReinterpretCast<uint16_t>();
    auto digitLocal = digitBuf_.template Get<uint16_t>();
    auto digitHalf = digitHalfBuf_.template Get<half>();
    auto maskLocal = maskBuf_.template Get<uint8_t>();
    auto onesLocal = onesBuf_.template Get<half>();

    // 确定当前趟的 word index 和 shift
    uint32_t wordIdx = passIdx / (RADIX_BITS);  // 0=lo, 1=hi
    uint32_t digitShift = passIdx % RADIX_BITS * RADIX_BITS;  // 0, 4, 8, 12

    // 提取 4-bit 数字: digit = (key[wordIdx] >> digitShift) & 0x000F
    // ShiftRight + And 在 uint16 上操作
    ShiftRight(digitLocal, keysU16[wordIdx], digitShift, chunkSize);
    And(digitLocal, digitLocal, static_cast<uint16_t>(RADIX_MASK), chunkSize);

    // Cast uint16 → half（Compare 需要 half 类型）
    Cast(digitHalf, digitLocal, RoundMode::CAST_NONE, chunkSize);

    // 直方图统计：对每个桶 v，统计匹配元素数
    for (uint32_t v = 0; v < RADIX_BUCKETS; v++) {
        // Duplicate 创建桶值
        auto bucketHalf = digitHalfBuf_.template Get<half>();  // 复用 buffer
        Duplicate(bucketHalf, static_cast<half>(v), chunkSize);

        // Compare: digit == v
        Compare(maskLocal, digitHalf, bucketHalf, CompareMode::EQ, chunkSize);

        // Select: mask ? 1.0 : 0.0
        Select(onesLocal, maskLocal, static_cast<half>(1.0), static_cast<half>(0.0),
               chunkSize);

        // ReduceSum 统计匹配数
        LocalTensor<half> sumTensor = onesBuf_.template Get<half>();
        // ReduceSum 输出到独立的小 buffer
        // 使用 histBuf 的对应位置
        half count = ReduceSum<half>(sumTensor, chunkSize);
        histogram[v] += static_cast<uint32_t>(count);
    }
}
```

修复后（kernel v2，EncodeKeys 改标量循环；EncodeInt32 为 `static_cast<uint32_t>(val) ^ SIGN_FLIP_MASK`）：

```cpp
template <typename D_T_X, int HAS_INVERSE>
__aicore__ inline void BenchNoHarnessUnique<D_T_X, HAS_INVERSE>::EncodeKeys()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t gmOffset = blockFactor_ * coreId_;
    int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;

    for (int64_t i = 0; i < loopCount; i++) {
        int64_t chunkOffset = i * ubFactor_;
        int64_t chunkSize = (chunkOffset + ubFactor_ > blockLength_)
                            ? (blockLength_ - chunkOffset) : ubFactor_;

        // GM → UB: 加载 int32 输入数据
        auto inputLocal = inputKeysBuf_.template Get<int32_t>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = chunkSize * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(inputLocal, inputGMX_[gmOffset + chunkOffset], copyParams,
                    DataCopyPadExtParams{false, 0, 0, 0});

        // 键编码：使用标量操作翻转符号位
        // 后续迭代将使用向量 Xor API
        for (int64_t j = 0; j < chunkSize; j++) {
            int32_t val = inputLocal.GetValue(j);
            uint32_t encoded = EncodeInt32(val);
            inputLocal.SetValue(j, static_cast<int32_t>(encoded));
        }

        // UB → GM: 写入编码后的键
        DataCopyPad(keysGM_[gmOffset + chunkOffset], inputLocal, copyParams);
    }
}
```

修复后（kernel v2，RadixSort 单趟直方图与分拣改标量循环，删去 ComputeHistogram/ScatterChunk 向量路径）：

```cpp
        // 标量统计直方图
        for (int64_t j = 0; j < chunkSize; j++) {
            uint32_t key = static_cast<uint32_t>(inputLocal.GetValue(j));
            uint32_t digit = (key >> digitShift) & RADIX_MASK;
            globalHist[digit]++;
        }

        // ===== Step 3: Scatter =====
        // 偶数趟: keysGM → sortedGM, 奇数趟: sortedGM → keysGM
        ...
            // 按桶分拣：对每个桶，收集匹配元素到 compactBuf，然后写入 GM
            auto compactLocal = compactBuf_.template Get<int32_t>();
            for (uint32_t v = 0; v < RADIX_BUCKETS; v++) {
                uint32_t compactCount = 0;
                for (int64_t j = 0; j < chunkSize; j++) {
                    uint32_t key = static_cast<uint32_t>(inputLocal.GetValue(j));
                    uint32_t digit = (key >> digitShift) & RADIX_MASK;
                    if (digit == v) {
                        compactLocal.SetValue(compactCount, inputLocal.GetValue(j));
                        compactCount++;
                    }
                }
                // 将紧缩后的元素写入 GM
                if (compactCount > 0) {
                    DataCopyParams outParams;
                    outParams.blockCount = 1;
                    outParams.blockLen = compactCount * sizeof(int32_t);
                    outParams.srcStride = 0;
                    outParams.dstStride = 0;
                    if (passIdx % 2 == 0) {
                        DataCopyPad(sortedGM_[bucketOffsets[v]], compactLocal, outParams);
                    } else {
                        DataCopyPad(keysGM_[bucketOffsets[v]], compactLocal, outParams);
                    }
                    bucketOffsets[v] += compactCount;
                }
            }
```

## 设计决策

- 为什么骨架阶段选择标量回退而不是当场补全 6 参向量签名：骨架阶段的首要目标是算法正确性闭环（编码→排序→去重）与工程链路打通（编译/打包/测试脚手架）；`GetValue/SetValue` 标量语义明确、无签名歧义，把 API 风险从关键路径上摘除。代价是性能——标量循环远慢于向量操作，因此只作为骨架期 fastpath、性能敏感路径的 safety_fallback。
- 向量化替代方案已按 API 差异逐条列入 issue 记录（Duplicate 构造标量张量后走张量-张量 Xor/And；ShiftRight 用完整 6 参签名或 Muls+Cast 组合；Compare 用 CMPMODE；ReduceSum 用 void 返回形式并提供 dst/tmp），留待后续向量化迭代验证。
- And/Or/Xor 的 dtype 约束（文档标注仅支持 int16_t/uint16_t）是 32-bit 键拆 uint16 word 处理的设计依据——这也解释了为什么键编码走 `ReinterpretCast<uint16_t>` 而不是直接对 uint32 位运算。

## 系统信息

- 硬件/软件：Ascend950PR（arch35 / DAV_3510）、CANN 9.1.0、ccec kernel 编译（报错指向 `x86_64-linux/asc/impl/basic_api/kernel_operator_vec_binary_scalar_intf_impl.h`、`adv_api/math/xor.h` 等）。
- 相关文件：`op_kernel/arch35/bench_no_harness_unique.h`（v1 向量版→v2 标量版重写）、`issues/issue_20260722_api_signature_mismatch_1.md`（7 项 API 差异表）。
- 该 issue 文件中记录的差异表（verbatim，供逐 API 对照）：

| API | 预期签名 | 实际签名（arch35） | 影响 |
|-----|---------|-------------------|------|
| Xor | `Xor(dst, src, scalar, count)` 4参数标量版 | 仅支持 `Xor(dst, src0, src1, count)` 张量-张量版，无标量版 | 无法直接对 uint16 执行 `^0x8000` |
| And | `And(dst, src, scalar, count)` 4参数标量版 | 仅支持 `And(dst, src0, src1, ...)` 张量-张量版，需6参数 | 无法直接对 uint16 执行 `&0x000F` |
| ShiftRight | `ShiftRight(dst, src, scalar, count)` 4参数标量版 | 标量版需6参数（含 mask, repeatTime, repeatParams） | 无法简洁地提取 4-bit digit |
| Compare | `Compare(dst, src0, src1, CompareMode::EQ, count)` | 使用 `CMPMODE`（非 `CompareMode`）枚举 | 枚举名不同 |
| Select | `Select(dst, mask, scalar0, scalar1, count)` 5参数 | 需6+参数（含 repeatParams 等） | 签名参数数量不同 |
| ReduceSum | `T ReduceSum(src, count)` 返回值 | `void ReduceSum(dst, src, tmp, count)` 写入 dst | 返回值 vs void |
| SyncAll | `SyncAll<HardEvent::MTE3_MTE2>()` | `SyncAll<bool isAIVOnly, SyncAllConfig& config>()` arch35 需2个模板参数 | 模板参数完全不同 |

## 验证信号

以下信号来自该算子骨架阶段的编译闭环：v1 向量版一次编译集中暴露 17 个签名错误，v2 标量版重写后编译通过。

- 修复前：ascend950 kernel 编译报 `error: no matching function for call to 'Xor'/'And'/'ShiftRight'/'Select'/'ReduceSum'`、`error: use of undeclared identifier 'CompareMode'`、`error: cannot initialize a variable of type 'half' with an rvalue of type 'void'`，共 17 errors（来源平台：950；证据等级：tool_observed）。
- 根因核对：CANN 头文件 grep 确认 `Xor`（`adv_api/math/xor.h:74`）候选第三参为 `LocalTensor<T>`；`ShiftRight`（`kernel_operator_vec_binary_scalar_intf_impl.h:925` 附近）标量形参为 `const T&` 且 4 参形式因 `digitShift` 为 `uint32_t` 与 `uint16_t` tensor 推导冲突；`Compare` 使用 `CMPMODE` 枚举（`kernel_reg_compute_vec_cmpsel_intf.h`）；`SyncAll` 为 arch35 专用签名（来源平台：950；证据等级：tool_observed）。
- 修复后：v2 标量版编译通过，生成 kernel 二进制 `BenchNoHarnessUnique_5b5227062859d6097351dcb83f899f41.o` 与算子包 `custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。
- 向量 API（Xor/And/ShiftRight/Compare/Select/ReduceSum）的 6 参正确签名在本轨迹未编译验证（后续向量化迭代事项），issue 文件明确标注"未验证编译"（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限于 CANN 9.1.0 的 arch35 头文件；其他版本/架构的标量重载集合可能不同，以目标平台头文件为准（未验证，机制推断可迁移——报错 note 会直接给出可用候选签名）。
- 标量回退只保证功能正确性与可编译性，不保证性能；大 shape 下逐元素 GetValue/SetValue 循环是明显瓶颈，性能敏感路径须回到向量实现。
- CMPMODE/Select/ReduceSum 的差异属于"枚举名/参数个数/返回形态"，本卡未给出各自可用的完整调用样例（轨迹内未验证）。

**未决项**：

- 后续向量化迭代需逐 API 验证 6 参向量化替代方案（Duplicate+张量-张量 Xor/And、ShiftRight 完整签名、CMPMODE、void ReduceSum）并做性能对比——这是本卡标量回退的接续点。
- arch35 上是否存在其他简化重载（如 `Xor(dst, src, scalar, count, mask...)` 五参形式）未穷举。
## 坏实践

- 按通用文档或 A2/A3 时代示例直接假设 `(dst, src, scalar, count)` 四参标量重载存在、不查目标平台头文件。
- 给 uint16 tensor 的标量形参传 `uint32_t` 移位数/掩码（`deduced conflicting types` 直接失败）——标量必须与元素类型同型（且即便同型，四参形式在该平台也不可用）。
- 骨架期死磕向量签名、阻塞算法正确性闭环——应先标量回退打通链路，向量化为独立迭代。

## 误诊

- 曾把 ShiftRight 失败归因于 `keysU16[wordIdx]` 的动态下标（怀疑 `[]` 运算符要求编译期常量或按字节偏移）——复核报错 note 后确认是 `T` 推导冲突（`'unsigned short' vs. 'uint32_t'` 来自标量 `digitShift`），与下标无关。
