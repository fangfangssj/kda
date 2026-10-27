---
type: "Runbook"
title: DataCopyPad 无 UB→UB 重载，改 GM 跨步直取
description: AscendC kernel 在 UB 内用带 DataCopyExtParams 的 DataCopyPad 做 strided 重排编译报 no matching function；CANN 9.1.0 只有 GM→UB/UB→GM/UB→L1 重载，LocalTensor→LocalTensor 方向仅 UB→L1，需改为 GM 跨步直取偶/奇元素再跨步写回。
tags: [task.debugging, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-16T10:23:52Z'
platforms: ["950"]
---

# DataCopyPad 无 UB→UB 重载，改 GM 跨步直取

## 现象

在实现 RoPE 类算子（apply_rotary_pos_emb）的 interleaved（偶奇交错）旋转模式时，最初的设计是：先把一整行 D 个元素用 DataCopyPad 从 GM 加载到 UB 的一个 fp32 buffer，然后在 UB 内部用带 `DataCopyExtParams` 的 `DataCopyPad` 以 stride 拆出偶数元素（`xFp32[0]`, `xFp32[1]`, ...）和奇数元素两个 buffer，计算完成后再用同样的方式在 UB 内交错写回（`yFp32[0]`, `yFp32[1]`）。

对这份代码执行 `bash build.sh --soc=ascend950 -j8` 编译时，op_host 与 op_api 均编译通过，但 kernel 编译阶段报出成片的重复错误（每个 TilingKey 模板实例化各报一次）：

```text
op_kernel/arch35/bench_no_harness_apply_rotary_pos_emb.h:338:5: error: no matching function for call to 'DataCopyPad'
op_kernel/arch35/bench_no_harness_apply_rotary_pos_emb.h:339:5: error: no matching function for call to 'DataCopyPad'
op_kernel/arch35/bench_no_harness_apply_rotary_pos_emb.h:366:5: error: no matching function for call to 'DataCopyPad'
op_kernel/arch35/bench_no_harness_apply_rotary_pos_emb.h:367:5: error: no matching function for call to 'DataCopyPad'
make[4]: *** [.../bench_no_harness_apply_rotary_pos_emb.make:3: ...] Error 1
```

这四个调用点正是 kernel 头文件里 UB→UB 的 strided DataCopyPad（两个加载 + 两个写回）。

## 适用场景

- 工程形态：自定义算子仓开发（op_host/op_kernel/op_api + build.sh），手写 AscendC kernel（TQue/TBuf/MemBase 路线）。
- API 路径：`AscendC::DataCopyPad`（`ascendc/include/basic_api/impl/kernel_operator_data_copy_intf_impl.h`），参数结构 `DataCopyExtParams` / `DataCopyPadExtParams<T>`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35，构建名 ascend950）+ CANN 9.1.0（编译闭环验证；未上板）。
- 适用粒度：operator_family（一切需要在 kernel 内做元素交错/拆分的 vector 类算子）。
- 路线优先级：recommended_fastpath（GM 跨步直取是编译期唯一可行路线之一，另一路线是 Gather/向量掩码，未验证）。
- 明确不适用 / 未验证场景：GM→UB、UB→GM、UB→L1 方向的 DataCopyPad 本身不受影响；aarch64 交叉编译环境与其他 CANN 版本未验证。

## 根因

CANN 9.1.0 的 DataCopyPad 重载集合里不存在 UB→UB 的 strided 版本，LocalTensor→LocalTensor 方向只有 UB→L1（要求 dst 在 TSCM/L1）。

机制链（头文件逐行核对）：`kernel_operator_data_copy_intf_impl.h` 中

- 行 1289/1363：`DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams&, const DataCopyPadParams&)` —— GM→UB（MTE2），dst 硬件位置断言为 `Hardware::UB` 或 `Hardware::L1`；
- 行 1485/1517：GM→UB 的 `DataCopyExtParams` + `DataCopyPadExtParams<T>` 覆盖版本；
- 行 1605/1631：`DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams&)` —— UB→GM（MTE3），没有 pad 参数；
- 行 1676 附近的 LocalTensor dst 版本仅用于 UB→L1（Nd2Nz 参数），不是 UB→UB。

因此 `DataCopyPad(ubA, ubB, stridedParams, padParams)` 无法解析到任何重载，直接编译失败。

## 解决配方

放弃"整行加载后在 UB 内拆分/交错"的结构，改为：

1. 用 strided GM→UB DataCopyPad 直接从 GM 取偶数元素（起点 `gmIn[rowOffset]`）和奇数元素（起点 `gmIn[rowOffset+1]`）；
2. Cast 升 fp32 计算（如需要）；
3. 用 strided UB→GM DataCopyPad 分别把偶/奇结果写回 `gmOut[rowOffset]` 与 `gmOut[rowOffset+1]`。

跨步参数推导（以加载为例）：`blockCount = halfD`，`blockLen = sizeof(INPUT_T)`（每块 1 个元素），`srcStride = sizeof(INPUT_T)`（GM 中相邻块起点间隔 = `blockLen + srcStride = 2*sizeof(INPUT_T)`，即隔 1 取 1），`dstStride = 0`（UB 内连续打包）。写回方向对调：`srcStride = 0`，`dstStride = sizeof(INPUT_T)`。

### 修复前（编译失败的 UB→UB 写法，函数级完整；编译失败点是下方"2. 提取偶数/奇数元素"与"4. 交错写回 yFp32"两组 DataCopyPad 调用）

```cpp
template <typename INPUT_T, int ROTARY_MODE>
__aicore__ inline void BenchNoHarnessApplyRotaryPosEmb<INPUT_T, ROTARY_MODE>::RotateInterleavedRow(
    GlobalTensor<INPUT_T>& gmIn, uint64_t rowOffset,
    GlobalTensor<INPUT_T>& gmOut)
{
    uint32_t halfD = halfD_;
    auto copyParams = MakeCopyParams(D_);

    // 1. 加载完整行 D 个元素
    auto xInLocal = xInQueue_.template AllocTensor<INPUT_T>();
    DataCopyPad(xInLocal, gmIn[rowOffset], copyParams, {false, 0, 0, 0});
    xInQueue_.EnQue(xInLocal);
    auto xInDeq = xInQueue_.template DeQue<INPUT_T>();

    auto xFp32 = xFp32Buf_.Get<COMPUTE_T>();
    if constexpr (NEED_CAST) {
        Cast(xFp32, xInDeq, RoundMode::CAST_NONE, D_);
    } else {
        // FP32: 需要从 xInDeq 复制到 xFp32
        // 使用 Cast<float, float> 作为 copy
        Cast(xFp32, xInDeq, RoundMode::CAST_NONE, D_);
    }
    xInQueue_.FreeTensor(xInDeq);

    // 2. 提取偶数/奇数元素（strided DataCopyPad UB→UB）
    auto xEvenFp32 = xEvenFp32Buf_.Get<COMPUTE_T>();
    auto xOddFp32 = xOddFp32Buf_.Get<COMPUTE_T>();

    DataCopyExtParams stridedParams;
    stridedParams.blockCount = static_cast<uint16_t>(halfD);
    stridedParams.blockLen = sizeof(COMPUTE_T);
    stridedParams.srcStride = sizeof(COMPUTE_T);
    stridedParams.dstStride = 0;
    stridedParams.rsv = 0;
    DataCopyPadExtParams<COMPUTE_T> padParams{false, 0, 0, 0};

    DataCopyPad(xEvenFp32, xFp32[0], stridedParams, padParams);
    DataCopyPad(xOddFp32, xFp32[1], stridedParams, padParams);

    // 3. 计算旋转
    auto cosFp32 = cosFp32Buf_.Get<COMPUTE_T>();
    auto sinFp32 = sinFp32Buf_.Get<COMPUTE_T>();
    auto temp = tempBuf_.Get<COMPUTE_T>();
    auto yEvenFp32 = yEvenFp32Buf_.Get<COMPUTE_T>();
    auto yOddFp32 = yOddFp32Buf_.Get<COMPUTE_T>();

    // yEven = xEven * cos - xOdd * sin
    Mul(yEvenFp32, xEvenFp32, cosFp32, halfD);
    Mul(temp, xOddFp32, sinFp32, halfD);
    Sub(yEvenFp32, yEvenFp32, temp, halfD);

    // yOdd = xOdd * cos + xEven * sin
    Mul(yOddFp32, xOddFp32, cosFp32, halfD);
    Mul(temp, xEvenFp32, sinFp32, halfD);
    Add(yOddFp32, yOddFp32, temp, halfD);

    // 4. 交错写回 yFp32
    auto yFp32 = yFp32Buf_.Get<COMPUTE_T>();
    DataCopyExtParams interleaveParams;
    interleaveParams.blockCount = static_cast<uint16_t>(halfD);
    interleaveParams.blockLen = sizeof(COMPUTE_T);
    interleaveParams.srcStride = 0;
    interleaveParams.dstStride = sizeof(COMPUTE_T);
    interleaveParams.rsv = 0;
    DataCopyPad(yFp32[0], yEvenFp32, interleaveParams, padParams);
    DataCopyPad(yFp32[1], yOddFp32, interleaveParams, padParams);

    // 5. 写回 GM
    auto yOutLocal = yOutQueue_.template AllocTensor<INPUT_T>();
    if constexpr (NEED_CAST) {
        Cast(yOutLocal, yFp32, RoundMode::CAST_NONE, D_);
    } else {
        Cast(yOutLocal, yFp32, RoundMode::CAST_NONE, D_);
    }
    yOutQueue_.EnQue(yOutLocal);
    auto yOutDeq = yOutQueue_.template DeQue<INPUT_T>();
    DataCopyPad(gmOut[rowOffset], yOutDeq, copyParams);
    yOutQueue_.FreeTensor(yOutDeq);
}
```

### 修复后（GM 跨步直取 + 跨步写回，函数级完整）

配套的 buffer 声明与初始化（interleaved 模式新增原始 dtype 偶/奇输入与输出 buffer）：

```cpp
    // Interleaved 模式额外 buffer
    TBuf<TPosition::VECCALC> xEvenInBuf_;    // 偶数元素（原始 dtype，从 GM strided 加载）
    TBuf<TPosition::VECCALC> xOddInBuf_;     // 奇数元素（原始 dtype）
    TBuf<TPosition::VECCALC> xEvenFp32Buf_;  // 偶数元素 FP32
    TBuf<TPosition::VECCALC> xOddFp32Buf_;   // 奇数元素 FP32
    TBuf<TPosition::VECCALC> yEvenFp32Buf_;  // 偶数结果 FP32
    TBuf<TPosition::VECCALC> yOddFp32Buf_;   // 奇数结果 FP32
    TBuf<TPosition::VECCALC> yEvenOutBuf_;   // 偶数输出（原始 dtype）
    TBuf<TPosition::VECCALC> yOddOutBuf_;    // 奇数输出（原始 dtype）
```

```cpp
    if constexpr (IS_INTERLEAVED_MODE) {
        pipe.InitBuffer(xEvenInBuf_, halfD * sizeof(INPUT_T));
        pipe.InitBuffer(xOddInBuf_, halfD * sizeof(INPUT_T));
        pipe.InitBuffer(xEvenFp32Buf_, halfD * sizeof(COMPUTE_T));
        pipe.InitBuffer(xOddFp32Buf_, halfD * sizeof(COMPUTE_T));
        pipe.InitBuffer(yEvenFp32Buf_, halfD * sizeof(COMPUTE_T));
        pipe.InitBuffer(yOddFp32Buf_, halfD * sizeof(COMPUTE_T));
        pipe.InitBuffer(yEvenOutBuf_, halfD * sizeof(INPUT_T));
        pipe.InitBuffer(yOddOutBuf_, halfD * sizeof(INPUT_T));
    }
```

```cpp
template <typename INPUT_T, int ROTARY_MODE>
__aicore__ inline void BenchNoHarnessApplyRotaryPosEmb<INPUT_T, ROTARY_MODE>::RotateInterleavedRow(
    GlobalTensor<INPUT_T>& gmIn, uint64_t rowOffset,
    GlobalTensor<INPUT_T>& gmOut)
{
    uint32_t halfD = halfD_;

    // strided DataCopyPad 参数：从 GM 加载每隔 1 个元素（偶数/奇数）
    DataCopyExtParams stridedParams;
    stridedParams.blockCount = static_cast<uint16_t>(halfD);
    stridedParams.blockLen = sizeof(INPUT_T);
    stridedParams.srcStride = sizeof(INPUT_T);   // GM 间隔 1 个元素
    stridedParams.dstStride = 0;                  // UB 连续打包
    stridedParams.rsv = 0;
    DataCopyPadExtParams<INPUT_T> padParams{false, 0, 0, 0};

    // 1. 从 GM strided 加载偶数元素 [0, 2, 4, ...] 和奇数元素 [1, 3, 5, ...]
    auto xEvenIn = xEvenInBuf_.Get<INPUT_T>();
    auto xOddIn = xOddInBuf_.Get<INPUT_T>();
    DataCopyPad(xEvenIn, gmIn[rowOffset], stridedParams, padParams);
    DataCopyPad(xOddIn, gmIn[rowOffset + 1], stridedParams, padParams);

    // 2. Cast 到 FP32（如果需要）并计算旋转
    auto cosFp32 = cosFp32Buf_.Get<COMPUTE_T>();
    auto sinFp32 = sinFp32Buf_.Get<COMPUTE_T>();
    auto temp = tempBuf_.Get<COMPUTE_T>();
    auto yEvenFp32 = yEvenFp32Buf_.Get<COMPUTE_T>();
    auto yOddFp32 = yOddFp32Buf_.Get<COMPUTE_T>();

    if constexpr (NEED_CAST) {
        auto xEvenFp32 = xEvenFp32Buf_.Get<COMPUTE_T>();
        auto xOddFp32 = xOddFp32Buf_.Get<COMPUTE_T>();
        Cast(xEvenFp32, xEvenIn, RoundMode::CAST_NONE, halfD);
        Cast(xOddFp32, xOddIn, RoundMode::CAST_NONE, halfD);

        // yEven = xEven * cos - xOdd * sin
        Mul(yEvenFp32, xEvenFp32, cosFp32, halfD);
        Mul(temp, xOddFp32, sinFp32, halfD);
        Sub(yEvenFp32, yEvenFp32, temp, halfD);

        // yOdd = xOdd * cos + xEven * sin
        Mul(yOddFp32, xOddFp32, cosFp32, halfD);
        Mul(temp, xEvenFp32, sinFp32, halfD);
        Add(yOddFp32, yOddFp32, temp, halfD);
    } else {
        // FP32: INPUT_T == COMPUTE_T, 直接使用输入 buffer
        // yEven = xEven * cos - xOdd * sin
        Mul(yEvenFp32, xEvenIn, cosFp32, halfD);
        Mul(temp, xOddIn, sinFp32, halfD);
        Sub(yEvenFp32, yEvenFp32, temp, halfD);

        // yOdd = xOdd * cos + xEven * sin
        Mul(yOddFp32, xOddIn, cosFp32, halfD);
        Mul(temp, xEvenIn, sinFp32, halfD);
        Add(yOddFp32, yOddFp32, temp, halfD);
    }

    // 3. Cast 回原 dtype（如果需要）并 strided 写回 GM
    DataCopyExtParams storeParams;
    storeParams.blockCount = static_cast<uint16_t>(halfD);
    storeParams.blockLen = sizeof(INPUT_T);
    storeParams.srcStride = 0;                    // UB 连续
    storeParams.dstStride = sizeof(INPUT_T);      // GM 间隔 1 个元素
    storeParams.rsv = 0;

    if constexpr (NEED_CAST) {
        auto yEvenOut = yEvenOutBuf_.Get<INPUT_T>();
        auto yOddOut = yOddOutBuf_.Get<INPUT_T>();
        Cast(yEvenOut, yEvenFp32, RoundMode::CAST_NONE, halfD);
        Cast(yOddOut, yOddFp32, RoundMode::CAST_NONE, halfD);
        DataCopyPad(gmOut[rowOffset], yEvenOut, storeParams);
        DataCopyPad(gmOut[rowOffset + 1], yOddOut, storeParams);
    } else {
        // FP32: 直接从 FP32 结果 buffer strided 写回
        DataCopyPad(gmOut[rowOffset], yEvenFp32, storeParams);
        DataCopyPad(gmOut[rowOffset + 1], yOddFp32, storeParams);
    }
}
```

## 设计决策

- 为何选 GM 跨步直取：编译期核查 DataCopyPad 重载后发现 UB→UB 路线根本不存在，GM→UB 与 UB→GM 的 `DataCopyExtParams` 覆盖版本是现成可用的对称入口；跨步语义（blockLen+srcStride 推导块间隔）恰好表达"隔 1 取 1"的偶/奇拆分。
- 放弃的候选：用 Gather API 配偏移 tensor 做拆分（需要额外构造索引 tensor，复杂度高）；用向量 API 构造偶/奇掩码提取（未验证可行性）；保留 UB 内拆分（不可行，无重载）。
- 代价权衡：GM 跨步逐元素搬运（blockLen=单元素）牺牲搬运效率（本阶段为骨架迭代，正确性优先），性能代价未测量，见遗留问题。
- 当前方案属 recommended_fastpath（相对"编译不过"这一硬约束）。

## 预防

- 在 kernel 内做元素重排/交错前，先核对 `kernel_operator_data_copy_intf_impl.h` 的重载方向：DataCopyPad 的 strided 能力只在 GM↔UB 方向成立。
- 设计文档评审时把"UB 内 strided 重排"列为需 API 核查项，避免直接照搬伪代码。

## 验证信号

问题与验证来源：在一次 RoPE 算子 interleaved 模式的骨架开发编译循环中发现并修复，修复后同一环境重新构建验证。

- 修复前 `bash build.sh --soc=ascend950 -j8`：`error: no matching function for call to 'DataCopyPad'` ×4 调用点（kernel 头文件 338/339/366/367 行），kernel 编译失败（来源平台：950；证据等级：tool_observed）。
- 修复后同一命令输出 `[INFO] Build completed successfully!`，生成 kernel 二进制 `build/op_kernel/ascendc_kernels/binary/ascend950/.../BenchNoHarnessApplyRotaryPosEmb_*.o`（3 个，构建系统对相同模板实例化去重）与算子包 `build/custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该结论基于 CANN 9.1.0 头文件（ascendc/include/basic_api/impl/kernel_operator_data_copy_intf_impl.h）的重载集合；其他版本需重新核对（未验证，推断可迁移）。
- 仅编译闭环验证，未上板运行；kernel 数值正确性由后续 Real ST 验证（本轨迹此阶段未执行）。
- 跨步搬运假设 `blockLen + srcStride = 2*sizeof(INPUT_T)` 的块间隔语义，对 GM→UB（dstStride 单位=字节）与 UB→GM（dstStride 单位=字节）分别成立；其他方向/参数单位的组合未验证。
- D 为奇数等非法 shape 依赖上层 tiling/校验拦截，本函数不处理。

**未决项**：

- 上板 Real ST 精度验证（本轨迹后续阶段执行；复现并验证数值正确性列为优先项）。
- GM 跨步逐元素搬运的性能代价未测量（无 profiling 信号）；若成为瓶颈，可评估 Gather/向量掩码拆分或整行加载+向量重排的替代实现。
## 坏实践

- 把 `DataCopyPad(ubA, ubB, stridedParams, padParams)` 当成通用内存重排原语使用——UB→UB 方向不存在该重载，编译期即失败。
- 设计文档先写"UB 内 strided 拆分"伪代码再实现，不先核对 API 重载方向，导致一轮必然失败的编译。

## 误诊

- 初步猜测是"strided DataCopyPad 参数类型不匹配"。实际核对头文件后确认不是参数问题，而是方向问题：任何参数组合下都不存在 UB→UB 的 strided DataCopyPad。
