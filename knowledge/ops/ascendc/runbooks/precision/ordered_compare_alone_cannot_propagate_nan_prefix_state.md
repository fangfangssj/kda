---
type: "Runbook"
title: "累计极值 NaN 掩码须先验证平台比较语义"
description: "保留950的self-NE掩码修复；A2/CANN9.1前n Compare实测连NaN的NE也为假，移植前须校准重载。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.comparison_logic, topic.compute.reduction, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: '2026-10-20T12:33:33Z'
platforms: ["950", "a2"]
---

# 累积极值 NaN 掩码须先验证平台比较语义
## 平台分支与路线选择

原有950/CANN9.0.1的显式self-NE掩码实现及其验证信号保持如下；它只在该分支为 `recommended_fastpath`。新增A2/CANN9.1.0的前n Compare微核显示NaN的NE也为假，因此该受测分支上self-NE检测路线为 `banned`。A2没有已验证的完整scan替代修法，须先校准可用NaN检测再验收value/index。不能用新A2反例改写旧950成功，也不能反向外推950成功。


## 现象

一个 Ascend C 累积极小值 kernel 在 Ascend 950 上完成构建和大多数精度检查后，全量评测只剩全 NaN 输入失败：values 报 NaN 位置不匹配，indices 则有大量元素不一致。有限值、全零、Inf、非对齐和多个 dtype 的其它检查都已通过，因此继续调整 tiling、搬运或一般有限值比较方向不能解释这个单一失败面。

失败时 `op_kernel/<op>_kernel.cpp` 沿扫描轴只生成一份 `running < current` 掩码：bit 为 1 时保留 running 和旧索引，否则取 current 和当前索引。这在有限数平局时能实现最后出现语义，却把 NaN 传播和 NaN 索引语义隐含地交给了裸有序比较结果。

目标语义必须区分三种状态：current 是 NaN 时总取 current；running 已是 NaN 而 current 有限时保留 running；两者有限时在 `current <= running` 时取 current。全 NaN 前缀还要求每个位置更新为当前 NaN，使索引保持最后出现位置。

## 适用场景

- 工程形态：Ascend C kernel 直调中的逐 lane 累积极值 Vector scan；以及标量逐元素 RMW 归约（scatter amax/amin 等，见 R4）。
- API 路径：`Compare(..., NE)` 自反比较检测 NaN，`Compare(..., LE)` 处理有限值，`Not` / `And` / `Or` 合成 bit-mask，再由 `Select` 同步更新 value 与 index；标量形态用 `x != x` 自比较与 `__builtin_nanf("")`。
- 验证过的平台与 CANN 版本：Ascend 950，CANN 9.0.1；实测修复路径为 half/float。R4：Ascend950PR（arch35）+ CANN 9.2.0（float/bf16/half 三分支）。
- 适用粒度：`operator_family`，适用于需要显式 NaN 传播且平局取最后出现位置的累计极小/累计极大或相似 scan，以及比较类标量归约（amax/amin）；极大值需相应替换有限值比较方向。
- 路线优先级：`recommended_fastpath`。它是本 trace 中唯一通过全 NaN、混合 NaN、有限平局及全量精度评测的显式谓词路线；没有与其它正确 NaN 实现做性能排序，不外推通用性能收益。
- 不适用 / 未验证：NaN 策略不同、平局取首次出现、dtype 不支持相同 Compare/bit-mask API、或只需要 unordered 比较标志的场景不能直接套用；原950 scan修复不覆盖BF16 NaN、A2、A3；下文新增A2仅验证比较接口，未验证scan修复。

## 根因

有限值选择可由一个有序比较表达，但 NaN 传播是额外的状态机语义。裸 `running < current` 不能同时编码： 1. current 为 NaN 时，无条件更新 value 和 index； 2. running 为 NaN、current 有限时，保持已传播的 NaN 与最近 NaN 索引； 3. 两者有限时，`current <= running` 才更新，以保留平局的最后出现位置。 修复把选择条件显式写成： `take_current = isCurNan OR ((NOT isRunNan) AND (current LE running))` 其中 `isCurNan` 和 `isRunNan` 都由 `Compare(t, t, NE)` 生成。虽然 `LE` 仍对整块执行，但 running 为 NaN 的位置会被 `NOT isRunNan` 屏蔽，current 为 NaN 的位置则由 `isCurNan` 强制取 current，因此最终 `Select` 不依赖 NaN 有序比较位。

标量形态同根因（R4）：C++ 的 `>=`/`<=` 与 NaN 比较恒为 false，`(of >= uf) ? of : uf` 在 `of` 为 NaN 时返回 `uf`——NaN 被静默替换成另一操作数；而 numpy 的 `np.maximum`/`np.minimum` 是 NaN 传播语义（任一侧 NaN → 结果 NaN），精度标准亦明确要求 amax/amin 下 NaN 胜出（本卡 golden 规定的 NaN 传播合同）。

## 解决配方

以下 before 是失败版本的完整 `ScanChunk`，after 是修复写入中的完整 `ScanChunk` 和新 `UpdateRunning`；均由 trace 的文件事件原样恢复。

### before

代码证据：revision = 修复前最近一次完整文件读取快照（无 Git commit，固定事件锚点仅存 manifest）；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbol = `KernelCummin<X>::ScanChunk`；line_range = 函数完整体。

```cpp
    __aicore__ inline void ScanChunk(int64_t m, int64_t n0, int64_t chunkLen)
    {
        // row k 的 GM offset (chunkLen 个连续元素): m*R*N + k*N + n0
        int64_t rowBase = m * R_ * N_ + n0;
        int64_t rowStride = N_;  // 行间步长

        // 加载 row 0 → init running
        CopyInRow(rowBase, chunkLen);
        auto y0 = yQue_.template DeQue<X>();
        AscendC::DataCopy(tmpX_, y0, alignedCount_);
        AscendC::Duplicate(tmpArg_, static_cast<int32_t>(0),
                           static_cast<int32_t>(alignedCount_));
        yQue_.FreeTensor(y0);

        // 写 row 0 输出
        CopyOutRow(m, n0, 0, chunkLen);

        // 顺序 scan row 1..R-1
        for (int64_t k = 1; k < R_; ++k) {
            int64_t offset = rowBase + k * rowStride;
            CopyInRow(offset, chunkLen);
            auto yRow = yQue_.template DeQue<X>();

            // bit=1 当 running < current
            AscendC::Compare(mask_, tmpX_, yRow, AscendC::CMPMODE::LT,
                             static_cast<uint32_t>(alignedCount_));
            // bit=1 保留 running, bit=0 取 current (后出现语义)
            AscendC::Select(minDst_, mask_, tmpX_, yRow,
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(alignedCount_));
            AscendC::DataCopy(tmpX_, minDst_, alignedCount_);

            AscendC::Duplicate(kRow_, static_cast<int32_t>(k),
                               static_cast<int32_t>(alignedCount_));
            AscendC::Select(argDst_, mask_, tmpArg_, kRow_,
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(alignedCount_));
            AscendC::DataCopy(tmpArg_, argDst_, alignedCount_);

            yQue_.FreeTensor(yRow);

            CopyOutRow(m, n0, k, chunkLen);
        }
    }
```

### after

代码证据：revision = NaN 修复的终版文件写入快照（无 Git commit，固定事件锚点仅存 manifest）；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbols = `KernelCummin<X>::ScanChunk`, `KernelCummin<X>::UpdateRunning`；line_range = 两个函数完整体。

```cpp
    __aicore__ inline void ScanChunk(int64_t m, int64_t n0, int64_t chunkLen)
    {
        // row k 的 GM offset (chunkLen 个连续元素): m*R*N + k*N + n0
        int64_t rowBase = m * R_ * N_ + n0;
        int64_t rowStride = N_;  // 行间步长

        // 加载 row 0 → init running
        CopyInRow(rowBase, chunkLen);
        auto y0 = yQue_.template DeQue<X>();
        AscendC::DataCopy(tmpX_, y0, alignedCount_);
        AscendC::Duplicate(tmpArg_, static_cast<int32_t>(0),
                           static_cast<int32_t>(alignedCount_));
        yQue_.FreeTensor(y0);

        // 写 row 0 输出
        CopyOutRow(m, n0, 0, chunkLen);

        // 顺序 scan row 1..R-1
        for (int64_t k = 1; k < R_; ++k) {
            int64_t offset = rowBase + k * rowStride;
            CopyInRow(offset, chunkLen);
            auto yRow = yQue_.template DeQue<X>();

            UpdateRunning(yRow, k);

            yQue_.FreeTensor(yRow);
            CopyOutRow(m, n0, k, chunkLen);
        }
    }

    // NaN-safe 谓词 (half/float) vs 原 LT 谓词 (bf16/int32)
    __aicore__ inline void UpdateRunning(
        const AscendC::LocalTensor<X>& yRow, int64_t k)
    {
        if constexpr (kNanAwareV<X>) {
            // mask2 = isCurNan = self-NE(current)
            AscendC::Compare(mask2_, yRow, yRow, AscendC::CMPMODE::NE,
                            static_cast<uint32_t>(alignedCount_));
            // mask3 = isRunNan = self-NE(running)
            AscendC::Compare(mask3_, tmpX_, tmpX_, AscendC::CMPMODE::NE,
                            static_cast<uint32_t>(alignedCount_));
            // mask4 = LE(current, running)  (NaN 位垃圾, 下面被 AND 屏蔽)
            AscendC::Compare(mask4_, yRow, tmpX_, AscendC::CMPMODE::LE,
                            static_cast<uint32_t>(alignedCount_));
            // mask_ = NOT isRunNan
            AscendC::Not(mask_, mask3_, static_cast<int32_t>(maskBytes_));
            // mask3 = (NOT isRunNan) AND LE   (复用 mask3 作 dst)
            AscendC::And(mask3_, mask_, mask4_, static_cast<int32_t>(maskBytes_));
            // mask_ = take_current = isCurNan OR ((NOT isRunNan) AND LE)
            AscendC::Or(mask_, mask3_, mask2_, static_cast<int32_t>(maskBytes_));

            // bit=1 → 取 current; src0=yRow(current), src1=tmpX_(running)
            AscendC::Select(minDst_, mask_, yRow, tmpX_,
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(alignedCount_));
            AscendC::DataCopy(tmpX_, minDst_, alignedCount_);

            AscendC::Duplicate(kRow_, static_cast<int32_t>(k),
                               static_cast<int32_t>(alignedCount_));
            // bit=1 → 取 k; src0=kRow, src1=tmpArg_
            AscendC::Select(argDst_, mask_, kRow_, tmpArg_,
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(alignedCount_));
            AscendC::DataCopy(tmpArg_, argDst_, alignedCount_);
        } else {
            // 原 LT 谓词: bit=1 当 running<current → 保留 running (后出现语义)
            AscendC::Compare(mask_, tmpX_, yRow, AscendC::CMPMODE::LT,
                             static_cast<uint32_t>(alignedCount_));
            // bit=1 保留 running, bit=0 取 current
            AscendC::Select(minDst_, mask_, tmpX_, yRow,
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(alignedCount_));
            AscendC::DataCopy(tmpX_, minDst_, alignedCount_);

            AscendC::Duplicate(kRow_, static_cast<int32_t>(k),
                               static_cast<int32_t>(alignedCount_));
            AscendC::Select(argDst_, mask_, tmpArg_, kRow_,
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(alignedCount_));
            AscendC::DataCopy(tmpArg_, argDst_, alignedCount_);
        }
    }
```

after 还依赖以下同一终版写入中的类型门禁和 mask 初始化片段。

代码证据：revision = 同一终版文件写入快照；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbol = `kNanAwareV`；line_range = namespace 内类型门禁定义。

```cpp
// half/float 走 NaN-safe 谓词; bf16/int32 走原 LT 谓词.
template <typename X>
constexpr bool kNanAwareV = std::is_same_v<X, half> || std::is_same_v<X, float>;
```

代码证据：revision = 同一终版文件写入快照；path = `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`；symbol = `KernelCummin<X>::Init`；line_range = bit-mask 字节数、四个 mask queue 和 tensor 分配段。

```cpp
        // bit-mask 字节数 (256B 对齐); Or/And/Not 以此为 count.
        maskBytes_ = ALIGN_UP_BYTES((alignedCount_ + 7) / 8, ALIGN_BYTES);

        // UB buffer 规划 (DESIGN §1.5): 单缓冲. 全部用 TQue<VECCALC,1> +
        // 一次性 AllocTensor 持久复用 (TQue LocalTensor 可 cast 到 __ubuf__).
        pipe_.InitBuffer(yQue_,     1, alignedCount_ * sizeof(X));
        pipe_.InitBuffer(tmpXQue_,      1, alignedCount_ * sizeof(X));
        pipe_.InitBuffer(tmpArgQue_,    1, alignedCount_ * sizeof(int32_t));
        pipe_.InitBuffer(minDstQue_,    1, alignedCount_ * sizeof(X));
        pipe_.InitBuffer(argDstQue_,    1, alignedCount_ * sizeof(int32_t));
        pipe_.InitBuffer(kRowQue_,      1, alignedCount_ * sizeof(int32_t));
        // mask 缓冲: mask_(主/LT 或合成最终 take_current) + mask2/3/4 (NaN 合成中间)
        pipe_.InitBuffer(maskQue_,  1, maskBytes_);
        pipe_.InitBuffer(mask2Que_, 1, maskBytes_);
        pipe_.InitBuffer(mask3Que_, 1, maskBytes_);
        pipe_.InitBuffer(mask4Que_, 1, maskBytes_);
        // VECOUT 输出队列 (经 EnQue/DeQue 提供 V→MTE3 同步)
        pipe_.InitBuffer(valuesOutQue_,  1, alignedCount_ * sizeof(X));
        pipe_.InitBuffer(indicesOutQue_, 1, alignedCount_ * sizeof(int64_t));

        // 一次性分配持久 LocalTensor (compute-only scratch, 跨行复用, 不 FreeTensor)
        tmpX_    = tmpXQue_.template AllocTensor<X>();
        tmpArg_  = tmpArgQue_.template AllocTensor<int32_t>();
        minDst_  = minDstQue_.template AllocTensor<X>();
        argDst_  = argDstQue_.template AllocTensor<int32_t>();
        kRow_    = kRowQue_.template AllocTensor<int32_t>();
        mask_    = maskQue_.template AllocTensor<uint8_t>();
        mask2_   = mask2Que_.template AllocTensor<uint8_t>();
        mask3_   = mask3Que_.template AllocTensor<uint8_t>();
        mask4_   = mask4Que_.template AllocTensor<uint8_t>();
    }
```

### 标量形态（R4）：三目比较 RMW 的 NaN 检查

scatter-reduce 的标量归约函数（逐元素读-改-写同一 `y[target]`）用三目比较实现 max/min 时踩同一根因：data 含 NaN 的 amax/amin/multiply 用例报 `NaN/Inf mismatch`（如 fp32 amax data 全 NaN、bf16 amax data 含 NaN）——golden（numpy）输出 NaN，kernel 输出 update 值（非 NaN）。

修复前（verbatim，float/bf16/half 三个分支共用该三目模式）：

```cpp
    } else if constexpr (std::is_same_v<T, float>) {
        switch (mode) {
            case 1: return oldVal + updVal;
            case 2: return oldVal * updVal;
            case 3: return (oldVal >= updVal) ? oldVal : updVal;
            case 4: return (oldVal <= updVal) ? oldVal : updVal;
            default: return updVal;
        }
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        float of = Bf16ToFloat(oldVal);
        float uf = Bf16ToFloat(updVal);
        float rf;
        switch (mode) {
            case 1: rf = of + uf; break;
            case 2: rf = of * uf; break;
            case 3: rf = (of >= uf) ? of : uf; break;
            case 4: rf = (of <= uf) ? of : uf; break;
            default: return updVal;
        }
        return FloatToBf16(rf);
    } else {
        // half (float16) — fp32 intermediate
        float of = HalfToFloat(oldVal);
        float uf = HalfToFloat(updVal);
        float rf;
        switch (mode) {
            case 1: rf = of + uf; break;
            case 2: rf = of * uf; break;
            case 3: rf = (of >= uf) ? of : uf; break;
            case 4: rf = (of <= uf) ? of : uf; break;
            default: return updVal;
        }
        return FloatToHalf(rf);
    }
```

修复后（显式 NaN 检查，`x != x` 判 NaN，返回 quiet NaN；verbatim 取自会话中修正版）：

```cpp
    } else if constexpr (std::is_same_v<T, float>) {
        switch (mode) {
            case 1: return oldVal + updVal;
            case 2: return oldVal * updVal;
            // IEEE 754 max/min: NaN propagates (np.maximum/minimum semantics)
            case 3: return (oldVal != oldVal || updVal != updVal) ? __builtin_nanf("")
                        : ((oldVal >= updVal) ? oldVal : updVal);
            case 4: return (oldVal != oldVal || updVal != updVal) ? __builtin_nanf("")
                        : ((oldVal <= updVal) ? oldVal : updVal);
            default: return updVal;
        }
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        float of = Bf16ToFloat(oldVal);
        float uf = Bf16ToFloat(updVal);
        float rf;
        switch (mode) {
            case 1: rf = of + uf; break;
            case 2: rf = of * uf; break;
            case 3: rf = (of != of || uf != uf) ? __builtin_nanf("") : ((of >= uf) ? of : uf); break;
            case 4: rf = (of != of || uf != uf) ? __builtin_nanf("") : ((of <= uf) ? of : uf); break;
            default: return updVal;
        }
        return FloatToBf16(rf);
    } else {
        // half (float16) — fp32 intermediate
        float of = HalfToFloat(oldVal);
        float uf = HalfToFloat(updVal);
        float rf;
        switch (mode) {
            case 1: rf = of + uf; break;
            case 2: rf = of * uf; break;
            case 3: rf = (of != of || uf != uf) ? __builtin_nanf("") : ((of >= uf) ? of : uf); break;
            case 4: rf = (of != of || uf != uf) ? __builtin_nanf("") : ((of <= uf) ? of : uf); break;
            default: return updVal;
        }
        return FloatToHalf(rf);
    }
```

标量形态的设计要点：用 `x != x` 而不是 `std::isnan`（kernel 编译环境对 `<cmath>` 可用性不确定，自比较零依赖且语义等价于 quiet NaN 检测）；返回 `__builtin_nanf("")`（quiet NaN）而不是原 NaN 操作数——golden 语义只要求"结果是 NaN"，这里只要求与 golden 的 NaN 类别一致，不保证 payload、符号或逐位相同。

## 设计决策

1. 选择 self-NE 加位掩码合成，因为它把 NaN 状态机直接编码进 `Select` 谓词，value 与 index 共用同一 take-current 位，避免两个输出分叉。
2. 放弃裸 `running < current`：它只足以表达有限值和后出现平局，已经在全 NaN 输入上被设备评测拒绝。轨迹没有验证额外 scalar `has_nan` 状态或更高阶 IsNaN API 的完整替代实现，因此不把它们列为并行正路线。
3. 当前路线为 `recommended_fastpath`：这是本 trace 唯一通过目标 NaN 语义和全量精度评测的 kernel 路线；未进行正确方案之间的性能排名。

## 系统信息

- 目标：Ascend 950（设备名探测为 Ascend950PR_957c）。
- 工具链：CANN 9.0.1。
- 工程：kernel 直调；修改文件角色为 `example/csrc/ops/<op>/op_kernel/<op>_kernel.cpp`。
- 实测修复分支：half/float；BF16 和 int32 保留原有限值 LT 分支。
- R4（标量形态）：Ascend950PR 单卡，CANN 9.2.0；相关文件 `op_kernel/arch35/bench_harness_scatter_kernel.h`（ReduceScalar）。

## 预防

- 写累计极值前先把 NaN 传播与平局索引规则展开为显式真值表，不要从一个有限值比较式外推。
- value 与 index 必须使用同一个最终选择掩码，否则值正确时索引仍可能漂移。
- 用全 NaN、NaN 后有限值、有限值后 NaN、重复 NaN和有限平局分别验证状态转移。
- bit-mask 逻辑 API 的 count 单位要按接口契约计算，并把额外 mask buffer 纳入 UB 预算。
- 移植 numpy/PyTorch 归约前先对齐四类特殊值语义：NaN 传播（max/min）、±Inf 传播（add/multiply）、溢出不得饱和（IEEE 754 返回 ±Inf）、整数溢出与转换按各自 reference 合同核对，不能默认 C++ 有符号溢出合法回绕；评测用例的 `input_data_ranges` 出现 `(nan, nan)`/`(-inf, -inf)` 即是特殊值专项用例，先查 kernel 的特殊值路径再查数值精度。
- add/multiply 的 NaN 由浮点算术自然传播（NaN±x、NaN×x 均为 NaN），不需要本卡补丁；整数模式无 NaN；若 golden 定义为 maxNum/minNum（非传播）语义则方向相反，先读 golden。

## 验证信号

来源是一次 Ascend C 累积极小值 kernel 的全量精度验收和随后设备端 NaN 语义探针；失败输出、函数改写和复评结果均来自工具事件。

- 修复前全量评测为 19/20，唯一失败是全 NaN 输入的 values NaN 位置不匹配与 indices 大量不匹配（来源平台：950；证据等级：tool_observed）。
- 谓词修复后远程构建成功，目标全 NaN 用例通过，有限 half 与 BF16 Inf 代表用例无回归（来源平台：950；证据等级：tool_observed）。
- 设备探针逐项显示自定义输出与 golden 在混合 NaN、前导 NaN、重复 NaN、全 NaN和有限平局上的 values/indices 一致（来源平台：950；证据等级：tool_observed）。
- 最终全量精度复评为 20/20（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：NaN 谓词修复后远程构建成功，目标全 NaN 用例通过，有限值与 Inf 代表用例无回归。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：strict-LT 分支被替换为 GE 普通选择加 current-is-NaN 覆盖；目标重新构建成功，原测试不退化。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3：以 curr_nan OR (runMin_finite AND curr<=runMin) 构造共享的 values/indices 更新 mask 后，mixed-NaN 探针通过，完整评测由 19/20 提升为 20/20，后续 pytest 81/81 通过。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4（BenchHarnessScatter 标量形态，CANN 9.2.0）：修复前 7 个失败用例中 3 个 reason 为 `NaN/Inf mismatch`（fp32 amax data 全 NaN、bf16 multiply data 含 NaN、bf16 amax data 含 NaN）；显式 NaN 检查修复后同批回归 7→3 失败，3 个 NaN 语义用例全部转 PASS（另一例 fp16 amin 同时转 PASS，其随数据在阈值附近摆动，最终由次正规修复稳定，见 [FloatToHalf 次正规阈值卡](floattohalf_subnormal_threshold_exp102.md)）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本谓词绑定“NaN 传播 + NaN/有限平局都取最后出现位置”的语义；首次出现或忽略 NaN 的算子必须重写真值表。
- 终版只让 half/float 进入 NaN-aware 分支。BF16 NaN 语义未验证，不能从本卡推断正确。
- `Not` / `And` / `Or` 的 count 在该实现中使用对齐后的 mask 字节数；不同 API 版本若 count 单位不同，必须按目标 CANN 文档调整。
- 原 950 完整 scan 修复尚未在 A2/A3 验证；新增 A2 证据仅覆盖下文 Compare 探针，不能据此认证完整修复。
- R4 的转 PASS 是同批三处修改（NaN 检查 + FloatToBf16 inf/nan 截断保护 + 拷贝后屏障）的共同结果，无法逐例拆分各自贡献（如实说明）；向量 API `AscendC::Max/Min` 的 NaN 语义未在 R4 会话验证，若改回向量路径需另行确认。

## 坏实践

把 `Compare(running, current, LT)` 当成完整累计极值状态机是被设备结果否定的路线。它没有显式区分 current NaN、running NaN 与两者有限三种状态，即使有限值和平局用例全部通过，也不能据此宣称 NaN 语义正确。

标量形态同理：用 `(a>=b)?a:b` 一行实现“最大值”而不问 NaN 语义——它与 np.maximum 在 NaN 输入下行为相反；只用正常范围随机数据测归约——NaN/Inf 用例是独立代码路径。

## 误诊

- “只有一个极端输入失败，所以是评测器容差或随机误差”不成立：失败同时包含 NaN 位置与整数索引的大量不匹配，是离散语义错误。
- 普通相等比较会让两个同位 NaN 被判不等；诊断脚本可能显示聚合布尔失败，但这不能覆盖设备输出逐项显示相同的事实，也不能作为放宽评测的理由。

## A2/CANN9.1 独立新增探针

### 现象

在校准累计极值候选使用的AscendC比较语义时，CPU常见的`x!=x` NaN检测不能直接移植到目标向量接口。以`AscendC::Compare(dst, a, b, mode, count)`前n个元素重载做真实AIV微核，half与float各256组操作数；只要任一侧是NaN，EQ、NE、LT、LE、GT、GE六种输出位全部为0。NE的行为与CPU IEEE比较oracle不同。

这是一份独立的新平台/版本/重载真值表。旧会话原始输出没有恢复；不能把这次结果称为旧修复复现，也不能覆盖另一平台已有self-NE有效的记录。

### 适用场景

- 工程形态：AscendC AIV_ONLY微核直调及NaN掩码校准。
- API路径：Compare的tensor前n元素、运行时CMPMODE重载；dtype为float32/float16，count=256。
- 验证平台与版本：公开结果记录 a2/Ascend910B4 与 CANN9.1.0 版本标记，并保留源码/模块hash；完整同PID runtime库路径/hash未随该公开投影提供。
- 适用粒度：`exact_task`，仅该接口/版本的实测语义。
- 路线优先级：`banned`，禁止在该受测路径上把self-NE当NaN检测。
- 未验证：其他重载、BF16、其他SoC/CANN版本、异常状态标志，以及完整scan的value/index状态机。

### 判别方法

输入bit模式为float32的`00000000/80000000/3f800000/c0000000/7f800000/ff800000/7fc00001/ffc00001/7f800001`，half对应`0000/8000/3c00/c000/7c00/fc00/7e01/fe01/7c01`，分别表示正负零、+1、-2、正负Inf、正负qNaN与正sNaN。取全部9×9有序对，重复后截为256。每个模式输出32字节packed mask，以little bitorder解包；先验证CPU→NPU→CPU传输的原始bits未改变。

以下为实际执行的完整Compare函数和两个入口，逐字恢复自构建源码：

```cpp
#include "kernel_operator.h"

template <typename T>
__aicore__ inline void CompareTruthTable(GM_ADDR a, GM_ADDR b, GM_ADDR out)
{
    constexpr uint32_t count = 256;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qa, qb;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qo;
    pipe.InitBuffer(qa, 1, count * sizeof(T));
    pipe.InitBuffer(qb, 1, count * sizeof(T));
    pipe.InitBuffer(qo, 1, count / 8);
    AscendC::GlobalTensor<T> ga, gb;
    AscendC::GlobalTensor<uint8_t> go;
    ga.SetGlobalBuffer((__gm__ T *)a, count);
    gb.SetGlobalBuffer((__gm__ T *)b, count);
    go.SetGlobalBuffer((__gm__ uint8_t *)out, 6 * count / 8);
    auto la = qa.AllocTensor<T>();
    auto lb = qb.AllocTensor<T>();
    AscendC::DataCopy(la, ga, count);
    AscendC::DataCopy(lb, gb, count);
    qa.EnQue(la);
    qb.EnQue(lb);
    la = qa.DeQue<T>();
    lb = qb.DeQue<T>();
    const AscendC::CMPMODE modes[6] = {
        AscendC::CMPMODE::EQ, AscendC::CMPMODE::NE,
        AscendC::CMPMODE::LT, AscendC::CMPMODE::LE,
        AscendC::CMPMODE::GT, AscendC::CMPMODE::GE};
    for (uint32_t m = 0; m < 6; ++m) {
        auto lo = qo.AllocTensor<uint8_t>();
        AscendC::Compare(lo, la, lb, modes[m], count);
        qo.EnQue(lo);
        lo = qo.DeQue<uint8_t>();
        AscendC::DataCopy(go[m * count / 8], lo, count / 8);
        qo.FreeTensor(lo);
    }
    qa.FreeTensor(la);
    qb.FreeTensor(lb);
}

extern "C" __global__ __aicore__ void compare_float(GM_ADDR a, GM_ADDR b, GM_ADDR out)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CompareTruthTable<float>(a, b, out);
}
extern "C" __global__ __aicore__ void compare_half(GM_ADDR a, GM_ADDR b, GM_ADDR out)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CompareTruthTable<half>(a, b, out);
}
extern "C" __global__ __aicore__ void rsqrt_paths(GM_ADDR input, GM_ADDR output, uint32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t n = 256;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qi;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp, ones;
    pipe.InitBuffer(qi, 1, n * sizeof(float));
    pipe.InitBuffer(qo, 1, n * sizeof(float));
    pipe.InitBuffer(temp, n * sizeof(float));
    pipe.InitBuffer(ones, n * sizeof(float));
    AscendC::GlobalTensor<float> gi, go;
    gi.SetGlobalBuffer((__gm__ float *)input, n);
    go.SetGlobalBuffer((__gm__ float *)output, n);
    auto x = qi.AllocTensor<float>();
    AscendC::DataCopy(x, gi, n);
    qi.EnQue(x);
    x = qi.DeQue<float>();
    auto y = qo.AllocTensor<float>();
    auto t = temp.Get<float>();
    auto one = ones.Get<float>();
    if (mode == 0) {
        AscendC::Rsqrt(y, x, n);
    } else {
        AscendC::Sqrt(t, x, n);
        if (mode == 1) {
            AscendC::Duplicate(one, 1.0f, n);
            AscendC::Div(y, one, t, n);
        } else {
            AscendC::Reciprocal(y, t, n);
        }
    }
    qo.EnQue(y);
    qi.FreeTensor(x);
    y = qo.DeQue<float>();
    AscendC::DataCopy(go, y, n);
    qo.FreeTensor(y);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void compare_float_do(uint32_t n, void *s, uint8_t *a, uint8_t *b, uint8_t *out)
{
    compare_float<<<n, nullptr, s>>>(a, b, out);
}
extern "C" void compare_half_do(uint32_t n, void *s, uint8_t *a, uint8_t *b, uint8_t *out)
{
    compare_half<<<n, nullptr, s>>>(a, b, out);
}
extern "C" void rsqrt_paths_do(uint32_t n, void *s, uint8_t *input, uint8_t *output, uint32_t mode)
{
    rsqrt_paths<<<n, nullptr, s>>>(input, output, mode);
}
#endif
```

对应实际宿主张量检查和launch也逐字保留；该模块不修改框架参考。

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
extern "C" void compare_float_do(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void compare_half_do(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rsqrt_paths_do(uint32_t, void *, uint8_t *, uint8_t *, uint32_t);
static uint8_t *ptr(const at::Tensor &t)
{
    return static_cast<uint8_t *>(const_cast<void *>(t.data_ptr()));
}
at::Tensor compare_modes(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.numel() == 256 && b.numel() == 256, "probe needs 256 elements");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.device() == b.device(), "matching inputs required");
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
    auto out = at::empty({6, 32}, a.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    if (a.scalar_type() == at::kFloat) compare_float_do(1, stream, ptr(a), ptr(b), ptr(out));
    else {
        TORCH_CHECK(a.scalar_type() == at::kHalf, "float32/float16 only");
        compare_half_do(1, stream, ptr(a), ptr(b), ptr(out));
    }
    return out;
}
at::Tensor inverse_sqrt(const at::Tensor &a, uint32_t mode)
{
    TORCH_CHECK(a.numel() == 256 && a.is_contiguous() && a.scalar_type() == at::kFloat, "256 contiguous float32 elements");
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1 && mode <= 2, "NPU input and mode 0..2 required");
    auto out = at::empty_like(a);
    rsqrt_paths_do(1, c10_npu::getCurrentNPUStream().stream(false), ptr(a), ptr(out), mode);
    return out;
}
PYBIND11_MODULE(_kb_compare_nan_probe, m)
{
    m.def("compare_modes", &compare_modes);
    m.def("inverse_sqrt", &inverse_sqrt);
}
```

实际可执行命令为`python3 build_ascendc.py . --soc-version Ascend910B4`与`python3 run_probe.py`，其公开可审材料保留完整driver、81个唯一有序对的逐模式输出、packed bytes、输入hash与二进制模块hash。输入sNaN bits的往返不等于验证了异常状态或是否quiet化后的算术语义。

#### 本轮原生探针的完整输入、解码与误差统计 driver

以下为实际运行的 run_probe.py 全文。本文件同时运行 Compare 与倒数平方根两个探针；本页只采用对应分支的观测，另一分支保留为同文件上下文，不增加本页结论范围。

```python
"""Measure the actual Ascend C Compare overload, preserving input and mask bits."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "kernel/build"))
import _kb_compare_nan_probe as ext


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.npu.set_device(0)
    rows = []
    tensors = {}
    modes = ["EQ", "NE", "LT", "LE", "GT", "GE"]
    bits_by_dtype = [
        ("float32", np.uint32, np.float32,
         [0x00000000, 0x80000000, 0x3f800000, 0xc0000000, 0x7f800000,
          0xff800000, 0x7fc00001, 0xffc00001, 0x7f800001]),
        ("float16", np.uint16, np.float16,
         [0x0000, 0x8000, 0x3c00, 0xc000, 0x7c00, 0xfc00, 0x7e01, 0xfe01, 0x7c01]),
    ]
    for dtype, uint, floating, patterns in bits_by_dtype:
        labels = ["+0", "-0", "+1", "-2", "+inf", "-inf", "+qnan", "-qnan", "+snan"]
        pairs = [(i, j) for i in range(len(patterns)) for j in range(len(patterns))]
        pairs = (pairs * 4)[:256]
        ba = np.asarray([patterns[i] for i, _ in pairs], dtype=uint)
        bb = np.asarray([patterns[j] for _, j in pairs], dtype=uint)
        a = torch.from_numpy(ba.view(floating).copy())
        b = torch.from_numpy(bb.view(floating).copy())
        an, bn = a.to("npu:0"), b.to("npu:0")
        torch.npu.synchronize()
        a_roundtrip = an.cpu().numpy().view(uint)
        b_roundtrip = bn.cpu().numpy().view(uint)
        packed = ext.compare_modes(an, bn)
        torch.npu.synchronize()
        cpu_packed = packed.cpu()
        unpacked = np.unpackbits(cpu_packed.numpy(), axis=1, bitorder="little")
        with np.errstate(invalid="ignore"):
            av, bv = ba.view(floating), bb.view(floating)
            expected = np.stack([av == bv, av != bv, av < bv, av <= bv, av > bv, av >= bv])
        nan = np.isnan(av) | np.isnan(bv)
        finite = np.isfinite(av) & np.isfinite(bv)
        observed_cases = []
        for k, (i, j) in enumerate(pairs[:81]):
            observed_cases.append({
                "a_label": labels[i], "b_label": labels[j],
                "a_bits": hex(int(ba[k])), "b_bits": hex(int(bb[k])),
                "actual": {m: int(unpacked[n, k]) for n, m in enumerate(modes)},
                "cpu_ieee_oracle": {m: int(expected[n, k]) for n, m in enumerate(modes)},
            })
        rows.append({
            "dtype": dtype, "elements": 256, "mode_order": modes,
            "input_a_sha256": hashlib.sha256(ba.tobytes()).hexdigest(),
            "input_b_sha256": hashlib.sha256(bb.tobytes()).hexdigest(),
            "roundtrip_a_bits_equal": bool(np.array_equal(ba, a_roundtrip)),
            "roundtrip_b_bits_equal": bool(np.array_equal(bb, b_roundtrip)),
            "packed_output_hex": cpu_packed.numpy().tobytes().hex(),
            "finite_mismatches_per_mode": {m: int(np.sum(unpacked[n, finite] != expected[n, finite])) for n, m in enumerate(modes)},
            "nan_pairs": int(nan.sum()),
            "nan_pair_true_counts": {m: int(unpacked[n, nan].sum()) for n, m in enumerate(modes)},
            "cpu_ieee_mismatches_per_mode": {m: int(np.sum(unpacked[n] != expected[n])) for n, m in enumerate(modes)},
            "unique_cases": observed_cases,
        })
        tensors[dtype] = {"a_bits": torch.from_numpy(ba.copy()), "b_bits": torch.from_numpy(bb.copy()), "mask": cpu_packed}
    # Separate C066 experiment: basic Ascend C Rsqrt versus two explicit paths.
    positive = np.asarray([np.float32(2.0 ** ((i - 120) / 2)) for i in range(240)], dtype=np.float32)
    special_bits = np.asarray([0, 0x80000000, 1, 0x007fffff, 0x00800000, 0xbf800000,
                               0x7f800000, 0x7fc00001, 0x3f800001, 0x3f7fffff,
                               0x7f7fffff, 0x3f000000, 0x40000000, 0x40400000,
                               0x3dcccccd, 0x41200000], dtype=np.uint32)
    rx = torch.from_numpy(np.concatenate([positive, special_bits.view(np.float32)]))
    rn = rx.to("npu:0")
    root_outputs = []
    for m in range(3):
        y = ext.inverse_sqrt(rn, m)
        torch.npu.synchronize()
        root_outputs.append(y.cpu())
    framework = torch.rsqrt(rn)
    torch.npu.synchronize()
    root_outputs.append(framework.cpu())
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        oracle = np.asarray(1.0 / np.sqrt(rx.numpy().astype(np.float64)), dtype=np.float32)
    valid = np.isfinite(oracle) & (oracle != 0)
    rsqrt = {"input_bits_hex": rx.numpy().view(np.uint32).tobytes().hex(),
             "input_sha256": hashlib.sha256(rx.numpy().tobytes()).hexdigest(),
             "oracle": "CPU binary64 sqrt/divide, cast to binary32; not legacy LayerNorm inv_std",
             "mode_order": ["AscendC_Rsqrt", "AscendC_Sqrt_Div", "AscendC_Sqrt_Reciprocal", "torch_npu_rsqrt"],
             "observations": []}
    for name, out in zip(rsqrt["mode_order"], root_outputs):
        y = out.numpy()
        rsqrt["observations"].append({"path": name,
            "output_bits_hex": y.view(np.uint32).tobytes().hex(),
            "reference_finite_nonzero_count": int(valid.sum()),
            "max_relative_error_on_reference_finite_nonzero": float(np.max(np.abs((y[valid].astype(np.float64) - oracle[valid]) / oracle[valid]))),
            "exact_output_bits_vs_binary32_oracle": int(np.sum(y.view(np.uint32) == oracle.view(np.uint32))),
            "nan_count": int(np.isnan(y).sum()), "inf_count": int(np.isinf(y).sum())})
        tensors[name] = out
    tensors["rsqrt_input"] = rx
    tensors["rsqrt_oracle"] = torch.from_numpy(oracle.copy())
    torch.save(tensors, ROOT / "input_output_bits.pt")
    header_roots = [Path("/usr/local/Ascend/cann-9.1.0/include"), Path("/usr/local/Ascend/cann-9.1.0/ascendc/include")]
    headers = []
    for base in header_roots:
        for path in base.glob("**/kernel_operator_vec_cmpsel_intf.h"):
            headers.append({"path": str(path), "realpath": str(path.resolve()), "sha256": digest(path)})
    result = {
        "schema_version": "compare_nan_new_probe.v1", "observed_at_unix": time.time(),
        "device_logical": 0, "device_name": torch.npu.get_device_name(0),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__, "python": sys.version,
        "platform": platform.platform(), "cann_latest_realpath": str(Path("/usr/local/Ascend/ascend-toolkit/latest").resolve()),
        "module_sha256": digest(ext.__file__), "module_path": ext.__file__,
        "script_sha256": digest(__file__), "headers": headers,
        "source_sha256": {p.name: digest(p) for p in sorted((ROOT / "kernel").glob("*.cpp"))},
        "tensor_archive_sha256": digest(ROOT / "input_output_bits.pt"),
        "loaded_ascend_libraries": [v.rstrip() for v in Path("/proc/self/maps").read_text().splitlines() if "/Ascend/" in v],
        "environment": {k: os.environ.get(k) for k in ["ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH"]},
        "historical_session_recovered": False, "rows": rows, "C066_native": rsqrt,
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(ROOT / "result.json"), "device": result["device_name"],
                      "summary": [{k: v for k, v in r.items() if k in ["dtype", "finite_mismatches_per_mode", "nan_pair_true_counts", "roundtrip_a_bits_equal", "roundtrip_b_bits_equal"]} for r in rows]}))


if __name__ == "__main__":
    main()
```

### 设计决策

比较模式共用同一输入与同一count，观察的是原始packed mask，不通过后续Select猜测比较结果。保留Inf、signed zero、qNaN和sNaN控制，避免把NaN检测差异误判为一般比较方向错误。此次只给出被否定路线与诊断依据，没有编写并声称通过完整scan修复。

### 系统信息

目标设备是Ascend910B4（a2），CANN9.1.0，torch2.9.0+cpu和torch_npu2.9.0。模块通过现有`build_ascendc.py`构建为AIV_ONLY直调，host实际加载模块SHA为7bdf5104b977b07cdd8591a11c25a08c89afd144dee4dd1821de49dc64b7d875。相关文件为`probe.cpp`、`pybind11.cpp`、`run_probe.py`与构建脚本；没有改旧算子、参考、容差或测试。实际自动同步选项未单独显式设置，其边界以完整构建脚本/日志为准，不能外推其他编译配置。

原始结果的`headers`列表因首次metadata查找路径错误而为空，原文件保持原样。随后从真实编译`.o.d`依赖取出头文件路径，并读取同容器原件核对：`aarch64-linux/asc/include/basic_api/kernel_operator_vec_cmpsel_intf.h` SHA3445867ae1ef92a1143cead32a0d727858e1037b501672a86ae37747c00343bf；`kernel_operator_vec_unary_intf.h` SHAed7f6e44c18d15f7eed22eede71ba2cda92f1e3e50b4d0b286adb4ee107987be。该补证没有回写原始result。

### 验证信号

本次实际微核的完整raw输出经独立按bits重构输入并解包复核。

- float32：256对中138对含NaN，六模式的NaN true count全部为0；118对非NaN（含Inf）对CPU oracle的六模式mismatch全部为0；两个输入往返bits全等。（来源平台：a2；证据等级：tool_observed）

- float16：256对中138对含NaN，六模式的NaN true count全部为0；118对非NaN（含Inf）对CPU oracle的六模式mismatch全部为0；两个输入往返bits全等。（来源平台：a2；证据等级：tool_observed）

### 失效边界

结论绑定上面的Compare重载、两个dtype、256元素及实际CANN9.1.0头文件/二进制。不能据此断言所有AscendC比较都具有同一NaN语义。没有执行NaN掩码替代实现或端到端累计极值value/index验收；下一步如需修复，应按算子NaN传播和平局索引契约单独验证。

### 坏实践

把CPU自反NE规则直接当成向量接口合同；把另一SoC上的self-NE成功记录扩展到当前平台。

### 性能影响

未测性能，无性能优越性声称。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/evidence/result.json#/rows`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/evidence/result.json#/projection`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/kernel/probe.cpp`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/kernel/pybind11.cpp`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/run_probe.py`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/native/build_ascendc.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 公开dynamic投影保留数值、设备名、源码/模块或脚本身份及9.1.0版本标记；不包含完整同PID runtime库路径/hash，不能据此独立认证全部加载库。
