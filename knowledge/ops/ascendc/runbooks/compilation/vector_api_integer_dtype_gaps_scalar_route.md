---
type: "Runbook"
title: vector API 整型缺位走标量 S 管线
description: DAV_3510 上 vector 计算类 API 对整型支持残缺（Abs/Div 仅 half/float、Sub/Min/Max 无 int64、And 仅 int16、Select 无整数），int64 除搬运外无任何 vector 运算；整数迭代算法走 DataCopyPad 搬运 + LocalTensor GetValue/SetValue 标量 S 管线 + TQue 双缓冲，上板 bitwise 通过。复现增量：归约索引类（argmax）补充缺口——ReduceMax(calIndex) T 仅 half/float 且 half 索引上限 65535、bf16 无 Max；附 Float/Int32/Int64 三类分路替代设计（Float 类已编译验证，Int32 Max+EQ 掩码与 Int64 hi/lo 两阶段为设计态）。复现增量：argmax int64 实际交付改用标量原生 > 比较（AR chunk 循环 + ARA 行迭代，max/idx/输出全程 int64），替代 hi/lo 设计，42/44 bitwise-equal 通过（2 条失败为框架 int64Mode 限制）。
tags: [task.design, topic.implementation.api_layer.simt_api, topic.compute.elementwise.math_function, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# vector API 整型缺位走标量 S 管线

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.1.0）上设计整数最大公约数算子（BenchNoHarnessGcd，int16/int32/int64 逐元素，bitwise 精度）。设计阶段对本地 asc-devkit API 文档逐个查证整型支持，结果如下（A3/950 系列口径）：

| API | 支持的 dtype | int16 | int32 | int64 |
|-----|-------------|-------|-------|-------|
| Abs | half, float | ❌ | ❌ | ❌ |
| Div | half, float | ❌ | ❌ | ❌ |
| And | int16_t, uint16_t | ✅ | ❌（需 ReinterpretCast） | ❌ |
| ShiftRight | int16/int32/uint16/uint32 | ✅ | ✅ | ❌ |
| Sub / Min / Max / Adds / Mins / Maxs | half, int16_t, int32_t, float | ✅ | ✅ | ❌ |
| Select | half/float | ❌ | ❌ | ❌ |
| Compare | 整型仅 int32_t + CMPMODE::EQ | — | 仅 EQ | ❌ |
| Duplicate | int16/uint16/int32/uint32/half/bf16/float | ✅ | ✅ | ❌ |
| DataCopyPad | half/bf16/int16/uint16/float/int32/uint32/int8/uint8/**int64**/uint64/double | ✅ | ✅ | ✅ |

关键结论：**除搬运类 DataCopyPad 外，没有任何 vector 计算类 API 支持 int64_t**；连"取绝对值"（Abs）对全部整型都不支持。欧几里得（要整型取模）与 Stein（要 And/ShiftRight，且 int64 缺位）都无法用 vector API 覆盖三个 dtype。最终整条算法走**标量 S 管线**：DataCopyPad(GM→UB)[MTE2] → `LocalTensor.GetValue` + 标量 GCD + `SetValue`[S] → DataCopyPad(UB→GM)[MTE3]，配 TQue 双缓冲与多核 remainder 切分；上板后无广播 int16/int32 用例逐位通过（最大 104 万元素/用例）。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel（`__aicore__` 侧）的整型逐元素迭代算法实现（gcd/lcm/逐元素模运算/整型迭代状态更新等）。
- API 路径：`AscendC::DataCopyPad`（GM↔UB，blockLen 单位字节、blockCount∈[1,4095]、blockLen∈[1,2097151]）；`LocalTensor::GetValue/SetValue`（`__inout_pipe__(S)` 标量访问，仅 VECIN/VECCALC/VECOUT 位置）；`TQue`/`TPipe::InitBuffer`；`GlobalTensor::GetValue`（广播输入直读 GM）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950/arch35/DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——DAV_3510 上所有含 int64 或整型迭代计算的逐元素算子。
- 路线优先级：recommended_fastpath（int64 场景下是当前唯一可行路线；int16/int32 若可完全 vector 化的简单算术除外）。
- 明确不适用 / 未验证场景：浮点 dtype；纯 int16/int32 且能被 Sub/Min/Max 组合覆盖的简单算术（vector 路线更优，本卡不涉及）；性能敏感场景的标量循环吞吐上限未测（见失效边界）；其他架构代际（arch22 等）的 dtype 支持表未查证。

## 解决配方

（即本路线的完整实现方式，见下"完整代码"。）

### 增量（独立 trace：归约索引类算子的补充缺口与替代路线）

归约含索引跟踪类算子（argmax/argmin/max-min-with-index）的补充缺口口径（A3 产品系列文档行，CANN 9.1.0，适用于 Ascend950PR/arch35）：

- **ReduceMax（Level 2, calIndex=true）**：模板 T 仅 half/float；half 输入索引上限 65535（uint16 能表示的最大值）；多最大值返回第一个；索引按 dst 类型位存储，需 reinterpret_cast 读。
- **Compare**：half/float 支持全部 CMPMODE；int32_t 只支持 CMPMODE::EQ；int64 不支持——整数"大于"比较没有直接 API。
- **Max（元素级）**：half、int16_t、int32_t、float；无 int64、无 bfloat16。
- **Select**：dst/src0/src1 仅 half/float；掩码 uint8_t~uint64_t——索引不能以 int32 存储参与 Select 更新。
- **And**：int16_t/uint16_t；uint32/int32 需 ReinterpretCast 后用。
- **Cast**：half→float、bfloat16_t→float 均 CAST_NONE（无损）；int32→float 有损。
- **DataCopyPad(ISASI)**：含 int64；blockCount∈[1,4095]、blockLen∈[1,2097151] 字节——int64 结果可直接搬运出 GM。

据此的 argmax 三类分路设计（输入覆盖 fp16/fp32/bf16/int32/int64，输出 int64，bitwise-equal 判定）：

1. **Float 类（fp16/fp32/bf16）**：全部 Cast→float（fp16/bf16→float 无损）后走 `ReduceMax(calIndex=true)`——同时解除 fp16 索引 ≤65535 限制（float 索引按 uint32 位模式读）；ARA 模式走 `Max<float> + Compare(EQ) + Select(float 索引)` 行迭代。fp32-AR 路径（`ReduceMax<float>(calIndex=true)` + `Cast<int64_t,float>` + DataCopyPad 8 字节输出）已完整编译验证。
2. **Int32 类（设计，未实现）**：禁止 Cast→float 做比较——float 尾数 24 位，|v|>2^24 的两个不同 int32 可舍入到同一 float，argmax 可能取错首个最大值、破坏 bitwise-equal。替代：`Max<int32_t>` 树状/行迭代求 maxVal → `Compare<int32_t>(EQ, data, maxVal)` 生成掩码（EQ 是 int32 唯一可用模式），掩码首个置位位即首个最大值位置；ARA 索引更新用 `Select`，索引以 float 存储（R ≤ 2^21 < 2^24 故 float 索引精确）。
3. **Int64 类（设计，未实现）**：无任何 int64 向量比较/取大 API。小端序下把 int64[N] 重解释为 int32[2N]（lo 在低地址、hi 在高地址）；hi 按有符号 int32 比较（`Max<int32>`），lo 按无符号比较（翻转符号位 XOR 0x80000000 后用 `Max<int32>`，保序映射 [0,2^32)→[-2^31,2^31)）；两阶段比较（先 hi 后 lo tie-break），掩码组合用 `And`（uint8 掩码 ReinterpretCast 到 uint16 后按位与）。

## 设计决策

- **为何放弃 vector 路线**：dtype 支持表是硬约束——Abs/Div 整型全缺、int64 无任何 vector 运算；"Max(x, 0-x) 组合绝对值 + 辗转相减"的 vector 化方案在 int64 上无解，且辗转相减需要 mask 跟踪而 Select/Compare 不支持整数，彻底不可行。
- **为何放弃 Stein（二进制 gcd）**：Stein 需要 And（仅 int16）与 ShiftRight（无 int64）；在标量单元上欧几里得（取模）每元素 O(log min(a,b)) 次迭代已足够，无需引入位运算特化。
- **搬运选 DataCopyPad + DataCopyParams**：blockLen 以字节为单位、对 32 字节块粒度不敏感，尾块（currentNum < tileLength）安全；三个目标 dtype 全支持（含 int64）。`copyParams = {blockCount=1, blockLen=currentNum*sizeof(T), srcStride=0, dstStride=0}`。
- **双缓冲**：`TQue<QuePosition::VECIN, 2>` ×2 输入 + `VECOUT, 2` ×1 输出，靠 EnQue/DeQue 自动同步 MTE2/S/MTE3 管线，标量循环与搬运重叠。
- **多核切分用 remainder 方式**：`coreLength_ = min(totalLength - blockIdx*blockLength, blockLength)`，负值钳 0，免尾核标志位。
- **性能边界自知**：API 文档明确警告"不要大量使用 SetValue 对 LocalTensor 赋值（性能下降）"——本路线接受该代价换取 int64 可行性与 bitwise 正确性；标量循环的元素吞吐上限未测。
- 当前方案路线档位：recommended_fastpath（int64 场景）。

## 系统信息

- 宿主 x86_64（CANN 9.1.0 安装内只有 `x86_64-linux` 目录）；NPU Ascend950PR（ascend950/arch35/DAV_3510）。
- 目录布局：`op_kernel/arch35/bench_no_harness_gcd.h`（kernel 类，本卡代码来源）、`op_kernel/bench_no_harness_gcd_arch35.cpp`（kernel 入口/模板实例化）、`op_kernel/arch35/bench_no_harness_gcd_tiling_data.h` / `..._tiling_key.h`、`op_host/arch35/bench_no_harness_gcd_tiling.cpp`（tiling：GetCoreNumAiv/GetCoreMemSize(UB) 动态取平台信息，tileLength = ubSize/(3 缓冲×2 双缓冲) 按 128 字节对齐 / dtypeSize）。
- 验证载体：`tests/st/test_aclnn_bench_no_harness_gcd.cpp`（上板验证）。

## 完整代码

以下为 kernel 终版（verbatim，`op_kernel/arch35/bench_no_harness_gcd.h`；`BROADCAST_MODE=0` 为无广播主路径，本卡上板验证信号即覆盖该路径；mode 1/2 广播分支随修复轮次加入，其上板验证信号不在本卡）：

```cpp
template <typename T>
__aicore__ inline T GcdScalar(T a, T b)
{
    if constexpr (sizeof(T) == 2) {
        // int16: 提升为 int32，用 uint32 计算
        int32_t a32 = static_cast<int32_t>(a);
        int32_t b32 = static_cast<int32_t>(b);
        uint32_t ua = (a32 < 0) ? (static_cast<uint32_t>(-(a32 + 1)) + 1U) : static_cast<uint32_t>(a32);
        uint32_t ub = (b32 < 0) ? (static_cast<uint32_t>(-(b32 + 1)) + 1U) : static_cast<uint32_t>(b32);
        while (ub != 0) {
            uint32_t r = ua % ub;
            ua = ub;
            ub = r;
        }
        return static_cast<int16_t>(static_cast<int32_t>(ua));
    } else if constexpr (sizeof(T) == 4) {
        // int32: 用 uint32 计算
        uint32_t ua = (a < 0) ? (static_cast<uint32_t>(-(a + 1)) + 1U) : static_cast<uint32_t>(a);
        uint32_t ub = (b < 0) ? (static_cast<uint32_t>(-(b + 1)) + 1U) : static_cast<uint32_t>(b);
        while (ub != 0) {
            uint32_t r = ua % ub;
            ua = ub;
            ub = r;
        }
        return static_cast<T>(ua);
    } else {
        // int64: 用 uint64 计算
        uint64_t ua = (a < 0) ? (static_cast<uint64_t>(-(a + 1)) + 1ULL) : static_cast<uint64_t>(a);
        uint64_t ub = (b < 0) ? (static_cast<uint64_t>(-(b + 1)) + 1ULL) : static_cast<uint64_t>(b);
        while (ub != 0) {
            uint64_t r = ua % ub;
            ua = ub;
            ub = r;
        }
        return static_cast<T>(ua);
    }
}

template <typename T, int BROADCAST_MODE>
class BenchNoHarnessGcd {
    // Double Buffer：3 个 queue 各 2 个 buffer
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr uint32_t MAX_DIMS = 8;

public:
    __aicore__ inline BenchNoHarnessGcd(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const BenchNoHarnessGcdTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline uint64_t BroadcastIndex(uint64_t outLinearIdx,
                                               const uint64_t* strides) const;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;
    uint64_t totalLength_ = 0;   // 输出总元素数
    uint32_t tileLength_ = 0;    // 单次 UB 处理元素数
    int64_t blockLength_ = 0;    // 每核处理元素数
    int64_t coreLength_ = 0;     // 本核实际处理元素数
    int64_t myOffset_ = 0;       // 本核输出起始偏移
    uint32_t scalarFlag_ = 0;    // bit0=x1标量, bit1=x2标量
    T scalarValX1_ = 0;
    T scalarValX2_ = 0;
    uint32_t dimNum_ = 0;
    uint32_t outDims_[MAX_DIMS] = {0};
    uint64_t x1Strides_[MAX_DIMS] = {0};
    uint64_t x2Strides_[MAX_DIMS] = {0};
};

template <typename T, int BROADCAST_MODE>
__aicore__ inline void BenchNoHarnessGcd<T, BROADCAST_MODE>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const BenchNoHarnessGcdTilingData* tilingData)
{
    totalLength_ = tilingData->totalLength;
    tileLength_ = tilingData->tileLength;
    blockLength_ = static_cast<int64_t>(tilingData->blockLength);
    scalarFlag_ = tilingData->scalarFlag;

    // 本核处理范围（remainder 方式，健壮处理尾核）
    int64_t blockIdx = GetBlockIdx();
    myOffset_ = blockLength_ * blockIdx;
    int64_t remainder = static_cast<int64_t>(totalLength_) - myOffset_;
    coreLength_ = (remainder > blockLength_) ? blockLength_ : remainder;
    if (coreLength_ < 0) { coreLength_ = 0; }

    if constexpr (BROADCAST_MODE == 0) {
        // 无广播：x1/x2/y 均按本核范围设置
        inputGMX1.SetGlobalBuffer((__gm__ T*)x1 + myOffset_, coreLength_);
        inputGMX2.SetGlobalBuffer((__gm__ T*)x2 + myOffset_, coreLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + myOffset_, coreLength_);
    } else if constexpr (BROADCAST_MODE == 1) {
        // 标量广播：标量输入读 1 个元素，非标量输入按本核范围设置
        if (scalarFlag_ & 0x1U) {
            GlobalTensor<T> x1GM;
            x1GM.SetGlobalBuffer((__gm__ T*)x1, 1);
            scalarValX1_ = x1GM.GetValue(0);
            inputGMX2.SetGlobalBuffer((__gm__ T*)x2 + myOffset_, coreLength_);
        } else {
            GlobalTensor<T> x2GM;
            x2GM.SetGlobalBuffer((__gm__ T*)x2, 1);
            scalarValX2_ = x2GM.GetValue(0);
            inputGMX1.SetGlobalBuffer((__gm__ T*)x1 + myOffset_, coreLength_);
        }
        outputGMY.SetGlobalBuffer((__gm__ T*)y + myOffset_, coreLength_);
    } else {
        // 维度广播：x1/x2 全量访问（stride 索引），y 按本核范围
        dimNum_ = tilingData->dimNum;
        for (uint32_t i = 0; i < dimNum_ && i < MAX_DIMS; i++) {
            outDims_[i] = tilingData->outDims[i];
            x1Strides_[i] = tilingData->x1Strides[i];
            x2Strides_[i] = tilingData->x2Strides[i];
        }
        inputGMX1.SetGlobalBuffer((__gm__ T*)x1, tilingData->x1ElemCount);
        inputGMX2.SetGlobalBuffer((__gm__ T*)x2, tilingData->x2ElemCount);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + myOffset_, coreLength_);
    }

    // 初始化 UB buffer（Double Buffer：每个 queue 2 个 tileLength 大小的 buffer）
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength_ * sizeof(T));
}

template <typename T, int BROADCAST_MODE>
__aicore__ inline uint64_t BenchNoHarnessGcd<T, BROADCAST_MODE>::BroadcastIndex(
    uint64_t outLinearIdx, const uint64_t* strides) const
{
    uint64_t inIdx = 0;
    uint64_t rem = outLinearIdx;
    for (int64_t i = static_cast<int64_t>(dimNum_) - 1; i >= 0; i--) {
        uint64_t m = rem % static_cast<uint64_t>(outDims_[i]);
        rem /= static_cast<uint64_t>(outDims_[i]);
        inIdx += m * strides[i];
    }
    return inIdx;
}

template <typename T, int BROADCAST_MODE>
__aicore__ inline void BenchNoHarnessGcd<T, BROADCAST_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    if constexpr (BROADCAST_MODE == 2) {
        // 维度广播：无需 CopyIn，Compute 直接从 GM 读取
        return;
    }

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    if constexpr (BROADCAST_MODE == 0) {
        LocalTensor<T> x1Local = inQueueX1.template AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.template AllocTensor<T>();
        DataCopyPad(x1Local, inputGMX1[progress * tileLength_], copyParams, {false, 0, 0, 0});
        DataCopyPad(x2Local, inputGMX2[progress * tileLength_], copyParams, {false, 0, 0, 0});
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    } else {
        // 标量广播：只搬运非标量输入
        if (scalarFlag_ & 0x1U) {
            LocalTensor<T> x2Local = inQueueX2.template AllocTensor<T>();
            DataCopyPad(x2Local, inputGMX2[progress * tileLength_], copyParams, {false, 0, 0, 0});
            inQueueX2.EnQue(x2Local);
        } else {
            LocalTensor<T> x1Local = inQueueX1.template AllocTensor<T>();
            DataCopyPad(x1Local, inputGMX1[progress * tileLength_], copyParams, {false, 0, 0, 0});
            inQueueX1.EnQue(x1Local);
        }
    }
}

template <typename T, int BROADCAST_MODE>
__aicore__ inline void BenchNoHarnessGcd<T, BROADCAST_MODE>::Compute(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();

    if constexpr (BROADCAST_MODE == 0) {
        LocalTensor<T> x1Local = inQueueX1.template DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.template DeQue<T>();

        for (int64_t j = 0; j < currentNum; j++) {
            T a = x1Local.GetValue(static_cast<uint32_t>(j));
            T b = x2Local.GetValue(static_cast<uint32_t>(j));
            T g = GcdScalar(a, b);
            yLocal.SetValue(static_cast<uint32_t>(j), g);
        }

        outQueueY.template EnQue<T>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    } else if constexpr (BROADCAST_MODE == 1) {
        if (scalarFlag_ & 0x1U) {
            LocalTensor<T> x2Local = inQueueX2.template DeQue<T>();
            T a = scalarValX1_;
            for (int64_t j = 0; j < currentNum; j++) {
                T b = x2Local.GetValue(static_cast<uint32_t>(j));
                T g = GcdScalar(a, b);
                yLocal.SetValue(static_cast<uint32_t>(j), g);
            }
            outQueueY.template EnQue<T>(yLocal);
            inQueueX2.FreeTensor(x2Local);
        } else {
            LocalTensor<T> x1Local = inQueueX1.template DeQue<T>();
            T b = scalarValX2_;
            for (int64_t j = 0; j < currentNum; j++) {
                T a = x1Local.GetValue(static_cast<uint32_t>(j));
                T g = GcdScalar(a, b);
                yLocal.SetValue(static_cast<uint32_t>(j), g);
            }
            outQueueY.template EnQue<T>(yLocal);
            inQueueX1.FreeTensor(x1Local);
        }
    } else {
        // 维度广播：从 GM 直接读取（stride 索引），只使用输出 UB
        uint64_t outBase = static_cast<uint64_t>(myOffset_ + progress * static_cast<int64_t>(tileLength_));
        for (int64_t j = 0; j < currentNum; j++) {
            uint64_t outIdx = outBase + static_cast<uint64_t>(j);
            uint64_t x1Idx = BroadcastIndex(outIdx, x1Strides_);
            uint64_t x2Idx = BroadcastIndex(outIdx, x2Strides_);
            T a = inputGMX1.GetValue(x1Idx);
            T b = inputGMX2.GetValue(x2Idx);
            T g = GcdScalar(a, b);
            yLocal.SetValue(static_cast<uint32_t>(j), g);
        }
        outQueueY.template EnQue<T>(yLocal);
    }
}

template <typename T, int BROADCAST_MODE>
__aicore__ inline void BenchNoHarnessGcd<T, BROADCAST_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outputGMY[progress * tileLength_], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

template <typename T, int BROADCAST_MODE>
__aicore__ inline void BenchNoHarnessGcd<T, BROADCAST_MODE>::Process()
{
    if (coreLength_ <= 0) { return; }

    int64_t loopCount = (coreLength_ + tileLength_ - 1) / tileLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (coreLength_ - tileLength_ * i) : tileLength_;
        CopyIn(i, currentNum);
        Compute(i, currentNum);
        CopyOut(i, currentNum);
    }
}
```

tiling 侧要点（`op_host/arch35/bench_no_harness_gcd_tiling.cpp`）：`GetCoreNumAiv()` 与 `GetCoreMemSize(UB)` 动态取平台信息（禁止硬编码核数）；`tileLength = (ubSize / (3 缓冲 × 2 双缓冲) / 128) * 128 / dtypeSize`（128 字节缓存行对齐）；`blockLength` 对齐 tileLength 均分总元素。

## 性能影响

- 标量 S 管线相对 vector 路线的成对相对代价未在本会话测量（无同算子 vector 基线——vector 路线在本 dtype 约束下不可实现）；文档侧信号"大量 SetValue 有性能下降"提示该路线的吞吐上限受标量单元约束，性能敏感的 int16/int32 算子应优先尝试 vector 组合。无绝对分数可报。

## 预防

- 设计整数算子先查 dtype 支持表（本地 asc-devkit API 文档 `docs/api/context/`），尤其确认 int64；不要默认 vector API 有整型重载。
- `LocalTensor.GetValue/SetValue` 仅在 VECIN/VECCALC/VECOUT 位置的 tensor 上可用（`__inout_pipe__(S)`），别对任意 LocalTensor 使用。

## 验证信号

以下信号来自该算子设计期 API 查证与迭代一上板联调会话：

- API dtype 表：Abs/Div/And/ShiftRight/Sub/Min/Max/Select/Compare/Duplicate/DataCopyPad 的整型支持逐项查自本地 API 文档（Abs.md/Div.md/And.md/ShiftRight.md/DataCopyPad(ISASI).md 等），结论如上表（来源平台：950（DAV_3510 API 文档口径）；证据等级：tool_observed）。
- 编译验证：该标量路线 kernel 三 dtype 编译通过并生成二进制（`[ascend950] Generating BenchNoHarnessGcd_*.o Done`，3 个 kernel）（来源平台：950；证据等级：tool_observed）。
- 上板功能验证：ST Real 首轮 L0 运行中，无广播 int16 用例（含 1,048,576 元素大用例、4D/6D 形状）与 int32 用例（含边界值域）全部 bitwise 一致；int64 无广播边界用例因取绝对值溢出缺陷独立失败（同路线、算法边界问题，另见 gcd golden INT_MIN 回绕相关卡）（来源平台：950；证据等级：tool_observed）。
- Mock 侧：CPU golden 自测 24/24（含三 dtype 广播与边界）（来源平台：950；证据等级：tool_observed）。

增量信号（独立 trace，argmax 算子方案设计阶段的系统性 API 核查与实现，Ascend950PR + CANN 9.1.0）：

- 归约索引类补充缺口：ReduceMax（T 仅 half/float、half 索引 ≤65535、返回第一个最大值、索引需 reinterpret 读）、Compare int32 仅 EQ/int64 不支持、Max 无 int64/bf16、Select 仅 half/float（掩码 uint8~uint64）、And 仅 int16/uint16、Cast half/bf16→float 无损与 int32→float 有损、DataCopyPad 参数域——逐条查自官方 API 文档"模板参数说明/参数说明"表的 A3 产品系列行（来源平台：A3 系列文档行（适用于 Ascend950PR/arch35）；证据等级：tool_observed）。
- Float 类子集编译验证：fp32-AR 路径使用 `ReduceMax<float>(calIndex=true)` + `Cast<int64_t,float>` + DataCopyPad 8 字节输出，`--soc=ascend950` 完整编译通过并生成 5 个 kernel 二进制（来源平台：950；证据等级：tool_observed）。
int32（Max+EQ 掩码 + float 索引）与 int64（hi/lo 两阶段）替代路线仅完成设计与文档化，未实现、未编译、未上板（设计结论，不作为实测信号）。

## 失效边界

- dtype 支持表为 CANN 9.1.0 / DAV_3510（A3/950 系）口径；其他代际（arch22/A2 等）需重查（未验证，推断 int64 同样受限但以文档为准）。
- 标量循环的绝对吞吐上限未测；元素量极大且算法迭代深的算子需另行评估（文档警告大量 SetValue 性能下降）。
- 广播分支（mode 1/2）的上板验证信号在后续修复轮次产生，不在本卡验证信号内。

**未决项**：

- 标量路线与潜在 vector 组合路线（仅 int16/int32）的成对相对性能对比未测。
- 标量循环在更深迭代算法（每元素 >30 次取模）下的吞吐曲线未测。
## 坏实践

- 设计时假设"vector API 一定有 int 重载"直接开写，到编译/联调才发现 Abs/Div 不支持整型——本算子若走该路会在实现中段返工。
- 用 `And + ReinterpretCast` 把 int32/int64 拆成 int16 块拼 Stein 算法——文档口径的迂回用法，复杂度高且 int64 无解，不如标量路线直接。
- （独立 trace 补充）按 torch/NumPy 心智默认"所有 dtype 都有比较/归约 API"，不做文档核查直接写整数比较路径；用 int32→float Cast 绕过整数比较追求"先跑通"——bitwise-equal 场景下大值随机数据必错且错误隐蔽（仅特定值域触发）；索引用 int32 存储参与 Select 更新——Select 不支持，编译即失败。

## 误诊

- "标量循环肯定跑不动/结果不可信"——上板 104 万元素逐位通过证明该路线功能正确；性能需另行测量，不能凭直觉否决（int64 场景它本是唯一可行路线）。

## 增量（2026-09-17，并行策展批：int64 argmax 的标量原生比较落地——替代 hi/lo 设计）

argmax 算子（BenchNoHarnessArgMax，Ascend950PR / DAV_3510 / arch35，CANN 9.1.0，5 dtype × AR/ARA 共 10 TilingKey）的 int64 路径**实际交付**没有走 复现增量的 Int64 hi/lo 两阶段设计（DESIGN.md §3.7.2），而是两个路径全部用**标量原生 `>` 比较**——同算子的 int32/ARA 向量化路线在 arch35 上连续踩 256B 对齐、padding 垃圾、未声明变量 UB 等问题，hi/lo 拆分把这些对齐约束叠加两份；标量原生比较完全绕开向量化 API 的对齐/dtype 约束，且 int64 原生 `>` 的"严格更大才更新"语义与 torch.argmax 的"多最大值取首位（first-occurrence）"天然一致，索引不经 float 中转（全程 int64，消除 >2^53 精度损失）。

**1. AR 路径（归约轴最内轴）**：逐 chunk `DataCopyPad` 搬入（blockLen=chunkLen×8 字节）→ `GetValue` 取 int64 → C++ 原生 `>` 更新 globalMax/globalIdx → 尾块按 `rSize_ - chunkStart` 钳制；输出 8 字节 DataCopyPad 单元素写 GM：

```cpp
// op_kernel/arch35/bench_no_harness_arg_max.h:797-846
template <typename D_T_X, uint32_t MODE>
__aicore__ inline void BenchNoHarnessArgMax<D_T_X, MODE>::ProcessInt64AR()
{
    LocalTensor<int64_t> outLocal = outBuf.Get<int64_t>();

    for (int64_t row = rowStart_; row < rowEnd_; row++) {
        int64_t globalMax = INT64_MIN;
        uint64_t globalIdx = 0;
        uint64_t rowOffset = static_cast<uint64_t>(row) * rSize_;

        uint32_t numChunks = (static_cast<uint32_t>(rSize_) + ubFactorR_ - 1) / ubFactorR_;
        for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            uint64_t chunkStart = static_cast<uint64_t>(chunkIdx) * ubFactorR_;
            uint32_t chunkLen = static_cast<uint32_t>(rSize_ - chunkStart);
            if (chunkLen > ubFactorR_) { chunkLen = ubFactorR_; }

            LocalTensor<int64_t> inLocal = inQueueX.template AllocTensor<int64_t>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = chunkLen * sizeof(int64_t);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
            DataCopyPad(inLocal, inputGM[rowOffset + chunkStart], copyParams, padParams);
            inQueueX.EnQue(inLocal);

            LocalTensor<int64_t> inData = inQueueX.template DeQue<int64_t>();

            for (uint32_t i = 0; i < chunkLen; i++) {
                int64_t val = inData.GetValue(i);
                if (val > globalMax) {
                    globalMax = val;
                    globalIdx = chunkStart + static_cast<uint64_t>(i);
                }
            }
            inQueueX.FreeTensor(inData);
        }

        outLocal.SetValue(0, static_cast<int64_t>(globalIdx));
        PipeBarrier<PIPE_V>();

        DataCopyExtParams outCopyParams;
        outCopyParams.blockCount = 1;
        outCopyParams.blockLen = sizeof(int64_t);
        outCopyParams.srcStride = 0;
        outCopyParams.dstStride = 0;
        outCopyParams.rsv = 0;
        DataCopyPad(indicesGM[static_cast<uint64_t>(row)], outLocal, outCopyParams);
    }
}
```

**2. ARA 路径（归约轴非最内轴）**：逐行迭代，max/idx 缓冲直接用 int64 LocalTensor（不经 float 中转）；`curA0Len` 三重钳制（`nextASize_ - a0Base`、`ubFactorA0_`、`rowEnd_ - row`）处理尾块/核边界；每趟 r 内 `val > curMax` 严格大于更新保证多最大值取首位：

```cpp
// op_kernel/arch35/bench_no_harness_arg_max.h:852-925（核心循环）
        for (uint64_t r = 1; r < rSize_; r++) {
            LocalTensor<int64_t> inLocal = inQueueX.template AllocTensor<int64_t>();
            DataCopyPad(inLocal, inputGM[gmBase + r * nextASize_], copyParams, padParams);
            inQueueX.EnQue(inLocal);
            LocalTensor<int64_t> inData = inQueueX.template DeQue<int64_t>();

            int64_t rowIdx = static_cast<int64_t>(r);
            for (uint32_t i = 0; i < curA0Len; i++) {
                int64_t val = inData.GetValue(i);
                int64_t curMax = maxLocal.GetValue(i);
                if (val > curMax) {
                    maxLocal.SetValue(i, val);
                    idxLocal.SetValue(i, rowIdx);
                }
            }
            PipeBarrier<PIPE_V>();
            inQueueX.FreeTensor(inData);
        }
```

**3. 缓冲分配（Init 内 int64 分支，无需 tree/mask/float 等向量化辅助缓冲）**：

```cpp
// :261-263（AR，位于 Init 的 MODE==ARG_MAX_MODE_AR 分支内）
        } else if constexpr (std::is_same_v<D_T_X, int64_t>) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactorR_ * sizeof(int64_t));
            pipe.InitBuffer(outBuf, 32);
        }
// :292-297（ARA）
        } else if constexpr (std::is_same_v<D_T_X, int64_t>) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM,
                            ubFactorR_ * a0Align_ * sizeof(int64_t));
            pipe.InitBuffer(maxBuf, a0Align_ * sizeof(int64_t));
            pipe.InitBuffer(idxBuf, a0Align_ * sizeof(int64_t));
            pipe.InitBuffer(outBufAra, a0Align_ * sizeof(int64_t));
        }
```

**4. tiling 配套**：int64 分支 UB 预算 `perElementBytes = bufNum*8`，仅需 inQueueX + max/idx/outAra。交付检视确认该偏离功能正确（"实际 Int64-AR 用原生 int64 标量比较，反而比设计的 hi/lo 更直接，无 float 转换"）。

**5. 与 hi/lo 设计的关系**：hi/lo（复现增量第 3 条）仍是性能优先的候选路线，但仅在性能验收明确不达标时再引入；标量原生比较是正确性优先的稳妥落地（safety_fallback——大 R 性能显著低于向量化方案，本轨迹跳过性能验收无量化数据）。

### 复现增量验证信号

- int64 用例 44 条（AR/ARA 各类 shape）在 Ascend950PR 真机执行，42 条 bitwise-equal 通过（mismatch-rate=0，判定 np.array_equal）；2 条失败（[8,8] dim=1、[8,8,8] dim=2 的 AR 小 shape）经独立闭环确认是框架 int64Mode 限制跳过自定义 tiling 所致，与 kernel 实现无关（机制见 [int64+AR 动态 shape aclnn 直调跳过自定义 tiling](../precision/int64_ar_dynamic_shape_aclnn_skip_tiling.md)）（来源平台：950；证据等级：tool_observed）。
- 交付检视逐行核对确认 int64 两路径为标量原生比较实现、无 float 中转、`val > curMax` 严格大于语义保证多最大值取首位（与 torch.argmax 一致，multi_max_first 用例全通过）（来源平台：950；证据等级：tool_observed）。
- 大 shape int64 用例（如 1D 1048576）通过，chunk 循环 + curA0Len 三重钳制的尾块/核边界处理无误（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 性能边界：标量 GetValue/SetValue 循环不使用 Vector 计算单元，大 R（归约轴很长）场景时延显著高于向量化方案；本轨迹跳过性能验收，无量化数据（未验证，不推断具体差距）。
- 框架边界：int64 + AR 模式 + R≤8 + 多输出的极小 shape 会命中框架 int64Mode 限制（跳过自定义 tiling/kernel），任何 kernel 实现都救不了，需图模式或 CANN 升级。
- 平台边界：结论绑定 DAV_3510/arch35 的 API 支持缺口；若目标平台 Max/ReduceMax 支持 int64，应优先向量化路线（本卡标量方案退化为安全兜底）。
- DESIGN.md §3.7.2 仍描述 hi/lo 拆分方案，与实现不一致，文档待同步。

### 复现增量坏实践增补

- 把 int64 拆 hi/lo 用 int32 向量化比较虽可行，但在已知向量化对齐/padding 陷阱的平台上叠加实现复杂度——本例证明标量原生比较是更稳的首选（hi/lo 仅在性能验收明确不达标时再引入）。
- 索引经 float 中转存储（int64 索引先存 float 再 Cast 回 int64）在 R 超出 float 精确整数范围时引入误差；int64 路径应全程 int64。