---
type: "Runbook"
title: "BF16 标量 cast 后端失败—改用位序键"
description: "dav-3510 的 aicore 标量路径不接受 bfloat16_t 与 float 的 static_cast，可按原始 uint16 位模式比较并回写。"
tags: ["task.debugging", "topic.compute.elementwise.type_conversion", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# BF16 标量 cast 后端失败—改用位序键

## 现象

在 `operators/cummin/cummin.asc` 中实现 16 位浮点前缀扫描时，`remote_runner.sh ... build` 已完成 CMake 配置并进入 ASC 编译，但 bisheng 在实例化 `bfloat16_t` 分支时终止，原始错误为：

```text
fatal error: error in backend: not support bf16 type cast
bisheng: error: clang frontend command failed with exit code 70
```

当时代码把 `half` 与 `bfloat16_t` 都先 `static_cast<float>` 做标量比较，再 `static_cast<T>` 写回；错误发生在后端而非 C++ 语法解析阶段。

## 适用场景

- 工程形态：Ascend C kernel 直调工程；
- API 路径：`__global__ __aicore__ __vector__` kernel 内的 aicore scalar load/compare/store；
- 验证过的平台与版本：Ascend 950（dav-3510）/ CANN 9.0.1，与 frontmatter `platforms` 一致；
- 适用粒度：`operator_family`，适用于“从 BF16/FP16 输入中选择并原样输出某个元素”的比较、扫描或选择类 kernel；
- 路线优先级：`recommended_fastpath`，这是本轨迹中唯一通过目标编译并完成全套精度验证的 16 位标量比较路线；
- 明确不适用：需要真正执行 BF16 算术、舍入或生成新 BF16 数值的场景；位模式排序只适用于选择已有元素，不能替代数值转换。

## 根因

dav-3510 的 bisheng aicore 后端不支持在该标量 kernel 路径中实例化 `bfloat16_t` 与 `float` 的双向 `static_cast`。

## 解决配方

把 16 位浮点分支改为读取原始 `uint16_t` 位模式：先单独识别 NaN，再把 IEEE sign-magnitude 位模式映射为单调整数 key，比较 key 后原样写回被选中输入的 16 位模式。非负数映射为 `bits | 0x8000`，负数映射为 `(~bits) & 0xFFFF`，因此所有负数 key 小于所有非负数 key，且各自内部保持数值次序。

### before

修复前写入快照；path=`operators/cummin/cummin.asc`，symbol=`cummin_kernel<T>`，lines=71-150。该完整函数支撑“BF16 分支确实实例化双向 scalar cast”的结论。

```cpp
template <typename T>
__global__ __aicore__ __vector__ void cummin_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx,
    int64_t M, int64_t R, int64_t N, int64_t lanesPerCore)
{
    __gm__ T *xPtr = (__gm__ T *)x;
    __gm__ T *yPtr = (__gm__ T *)y;
    __gm__ int64_t *idxPtr = (__gm__ int64_t *)idx;

    int64_t totalLanes = M * N;
    int64_t blockId = AscendC::GetBlockIdx();
    int64_t laneStart = blockId * lanesPerCore;
    if (laneStart >= totalLanes) {
        return;
    }
    int64_t laneEnd = laneStart + lanesPerCore;
    if (laneEnd > totalLanes) {
        laneEnd = totalLanes;
    }

    // Walk lanes as (m, n): lanes are ordered m*N + n (n innermost, contiguous).
    int64_t m = laneStart / N;
    int64_t n = laneStart - m * N; // laneStart % N without a second division
    int64_t lane = laneStart;
    while (lane < laneEnd) {
        int64_t laneEndThisM = (m + 1) * N;
        if (laneEndThisM > laneEnd) {
            laneEndThisM = laneEnd;
        }
        int64_t nCur = n;
        for (int64_t l = lane; l < laneEndThisM; ++l, ++nCur) {
            int64_t base = m * R * N + nCur;
            // k = 0: init
            T x0 = xPtr[base];
            int64_t minIdx = 0;

            if constexpr (std::is_same<T, int32_t>::value) {
                int32_t minVal = static_cast<int32_t>(x0);
                yPtr[base] = static_cast<T>(minVal);
                idxPtr[base] = minIdx;
                for (int64_t k = 1; k < R; ++k) {
                    int64_t off = base + k * N;
                    int32_t cur = static_cast<int32_t>(xPtr[off]);
                    if (cur <= minVal) {
                        minVal = cur;
                        minIdx = k;
                    }
                    yPtr[off] = static_cast<T>(minVal);
                    idxPtr[off] = minIdx;
                }
            } else {
                // float / half / bfloat16_t: compare in f32 to avoid aicore
                // half/bf16 scalar relational restrictions; value written back
                // as original dtype. Selecting an existing input element means
                // half->f32->half is identity (no precision drift).
                float minVal = static_cast<float>(x0);
                yPtr[base] = static_cast<T>(minVal);
                idxPtr[base] = minIdx;
                for (int64_t k = 1; k < R; ++k) {
                    int64_t off = base + k * N;
                    float cur = static_cast<float>(xPtr[off]);
                    if (cummin_isnan_f32(cur)) {
                        minVal = cummin_make_nan_f32();
                        minIdx = k;
                    } else if (cummin_isnan_f32(minVal)) {
                        // NaN persists, idx unchanged
                    } else if (cur <= minVal) {
                        minVal = cur;
                        minIdx = k;
                    }
                    yPtr[off] = static_cast<T>(minVal);
                    idxPtr[off] = minIdx;
                }
            }
        }
        lane = laneEndThisM;
        m += 1;
        n = 0;
    }
}
```

### after

修复后写入快照；path=`operators/cummin/cummin.asc`，symbols=`cummin_key16`/`cummin_isnan16`/`cummin_scan_lane_u16`/`cummin_kernel<T>`，lines=75-84,135-166,171-242。以下均为事件流中的完整函数，支撑“用位序键比较并让 half/BF16 分支走 uint16 路径”的结论。

```cpp
__aicore__ inline int32_t cummin_key16(uint16_t bits)
{
    return (bits & 0x8000u) ? (int32_t)((~bits) & 0xFFFFu)
                           : (int32_t)(bits | 0x8000u);
}

__aicore__ inline bool cummin_isnan16(uint16_t bits, uint16_t expMask, uint16_t mantMask)
{
    return ((bits & expMask) == expMask) && ((bits & mantMask) != 0u);
}

__aicore__ inline void cummin_scan_lane_u16(
    __gm__ uint16_t *xP, __gm__ uint16_t *yP, __gm__ int64_t *iP,
    int64_t base, int64_t R, int64_t N, uint16_t expMask, uint16_t mantMask)
{
    uint16_t minBits = xP[base]; // k = 0
    int64_t minIdx = 0;
    yP[base] = minBits;
    iP[base] = minIdx;
    int32_t minKey = cummin_isnan16(minBits, expMask, mantMask) ? 0 : cummin_key16(minBits);
    bool minIsNan = cummin_isnan16(minBits, expMask, mantMask);
    for (int64_t k = 1; k < R; ++k) {
        int64_t off = base + k * N;
        uint16_t curBits = xP[off];
        if (cummin_isnan16(curBits, expMask, mantMask)) {
            minBits = curBits; // preserve cur's NaN pattern
            minIdx = k;
            minIsNan = true;
            minKey = 0;
        } else if (minIsNan) {
            // NaN persists, idx unchanged
        } else {
            int32_t curKey = cummin_key16(curBits);
            if (curKey <= minKey) { // <= last-occurrence
                minBits = curBits;
                minKey = curKey;
                minIdx = k;
            }
        }
        yP[off] = minBits;
        iP[off] = minIdx;
    }
}

template <typename T>
__global__ __aicore__ __vector__ void cummin_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx,
    int64_t M, int64_t R, int64_t N, int64_t lanesPerCore)
{
    __gm__ int64_t *idxPtr = (__gm__ int64_t *)idx;
    int64_t totalLanes = M * N;
    int64_t blockId = AscendC::GetBlockIdx();
    int64_t laneStart = blockId * lanesPerCore;
    if (laneStart >= totalLanes) {
        return;
    }
    int64_t laneEnd = laneStart + lanesPerCore;
    if (laneEnd > totalLanes) {
        laneEnd = totalLanes;
    }

    if constexpr (std::is_same<T, float>::value) {
        __gm__ float *xP = (__gm__ float *)x;
        __gm__ float *yP = (__gm__ float *)y;
        int64_t m = laneStart / N;
        int64_t n = laneStart - m * N;
        int64_t lane = laneStart;
        while (lane < laneEnd) {
            int64_t endM = (m + 1) * N;
            if (endM > laneEnd) { endM = laneEnd; }
            for (int64_t l = lane; l < endM; ++l, ++n) {
                cummin_scan_lane_f32(xP, yP, idxPtr, m * R * N + n, R, N);
            }
            lane = endM;
            m += 1;
            n = 0;
        }
    } else if constexpr (std::is_same<T, int32_t>::value) {
        __gm__ int32_t *xP = (__gm__ int32_t *)x;
        __gm__ int32_t *yP = (__gm__ int32_t *)y;
        int64_t m = laneStart / N;
        int64_t n = laneStart - m * N;
        int64_t lane = laneStart;
        while (lane < laneEnd) {
            int64_t endM = (m + 1) * N;
            if (endM > laneEnd) { endM = laneEnd; }
            for (int64_t l = lane; l < endM; ++l, ++n) {
                cummin_scan_lane_i32(xP, yP, idxPtr, m * R * N + n, R, N);
            }
            lane = endM;
            m += 1;
            n = 0;
        }
    } else {
        // half / bfloat16_t via raw uint16 patterns
        constexpr uint16_t expMask =
            std::is_same<T, half>::value ? 0x7C00u : 0x7F80u;
        constexpr uint16_t mantMask =
            std::is_same<T, half>::value ? 0x03FFu : 0x007Fu;
        __gm__ uint16_t *xP = (__gm__ uint16_t *)x;
        __gm__ uint16_t *yP = (__gm__ uint16_t *)y;
        int64_t m = laneStart / N;
        int64_t n = laneStart - m * N;
        int64_t lane = laneStart;
        while (lane < laneEnd) {
            int64_t endM = (m + 1) * N;
            if (endM > laneEnd) { endM = laneEnd; }
            for (int64_t l = lane; l < endM; ++l, ++n) {
                cummin_scan_lane_u16(xP, yP, idxPtr, m * R * N + n, R, N, expMask, mantMask);
            }
            lane = endM;
            m += 1;
            n = 0;
        }
    }
}
```

## 设计决策

- 选择位模式比较，是因为该算子只选择并回写已有输入元素，不需要把 BF16 数值转换为 FP32 后再生成新的 BF16；
- 放弃双向 scalar cast，因为它在目标后端直接拒绝编译；
- 放弃用 canonical NaN 覆盖输入，因为保留实际 NaN 位模式更接近“选择已有元素”的语义；
- 当前方案在本轨迹中属于 `recommended_fastpath`；需要执行 BF16 算术时应改走已验证的向量 Cast/Reg 路径，不能复用本配方。

## 系统信息

- toolchain：CANN 9.0.1，bisheng clang 15.0.5；
- SoC/NPU arch：Ascend950PR_957b / dav-3510；
- 代码文件：`operators/cummin/cummin.asc`；
- 受影响 dtype：`bfloat16_t` 明确触发，`half` 与 BF16 共用修复后的 16 位位模式路径。

## 预防

- 在 dav-3510 aicore 标量路径中看到 `static_cast<float>(bfloat16_t)` 或反向 cast 时，先做最小 build probe；
- 若算法只在输入元素之间做选择，优先保留原始位模式，避免无意义的 BF16 往返转换；
- 位序比较必须把 NaN 从普通 key 比较中剥离，且 half/BF16 的指数、尾数 mask 不同。

## 验证信号

这条知识来自同一 kernel 从首版编译失败到位模式路径替换后的远端目标构建与最终回归。

- 首版目标编译稳定报 `not support bf16 type cast`（Ascend 950 / CANN 9.0.1，证据等级: tool_observed）；
- 替换为 `uint16_t` 位序键路径后，ASC 对象成功编译并链接为目标库（Ascend 950 / CANN 9.0.1，证据等级: tool_observed）；
- 后续含 half/BF16 的完整 pytest 回归全部通过（Ascend 950 / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

- 位序键不能用于 BF16 加减乘除、舍入、饱和或需要产生新数值的 kernel；
- 负零、NaN payload 与 tie 规则必须由上层算子语义单独定义；本配方只给出可编译的比较/选择机制；
- A2/A3 未验证，推断可迁移但必须重新编译和上板确认（未验证，推断可迁移）。

## 坏实践

- 看到“选中已有 BF16 元素”就先转 FP32 再转回 BF16：既引入后端不支持的 scalar cast，也没有产生任何必要的新数值；
- 直接按无符号 `uint16_t` 比较：负数区间的顺序会反向；
- 把 NaN 当普通位序 key 比较：会破坏算子要求的 NaN 传播与索引更新规则。

## 误诊

首轮设计担心的是 `__vector__` kernel 中的 scalar GM 访问不可编译，但编译器已经越过 GM 访问并精确停在 BF16 cast 后端，故该次构建失败的直接根因不是 GM 指针能力。