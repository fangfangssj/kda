---
type: "Runbook"
title: "Gather L2 整寄存器读 offset 越界致 507035"
description: "dav_3510 的 Gather Level 2 实现用 Reg::LoadAlign 一次读取整条 256B 向量寄存器的 offset（64 个 uint32）；offset buffer 分配小于 256B 时越界读 UB，垃圾索引导致源越界 → aivec 异常 340 → ST 报 507035；小 D 用重排需求时改用 GetValue/SetValue 逐元素重排。"
tags: [task.debugging, topic.data_movement.memory_access]
status: stable
sources: []
created_at: '2026-09-17T01:42:07Z'
updated_at: '2026-09-17T01:42:07Z'
platforms: ["950"]
---

# Gather L2 整寄存器读 offset 越界致 507035

## 现象

自定义 RoPE 类算子（arch35）的 interleaved 模式用 Gather Level 2 做 UB 内偶/奇元素重排：连续加载整行 D 个元素到 UB，构建 uint32 offset 索引数组（buffer 按 `AlignToBlock(D * sizeof(uint32_t))` 分配），然后

```cpp
Gather(xReorg, xFp32, inputIdx, 0, D);
```

真机 ST 中，**interleaved 小 D 用例全部 507035（流同步失败/向量核异常）**，而 offset buffer ≥256B 的大 D 用例（D=64/128/1024）不崩（当时因其他独立缺陷错值）：

```
L0_002 (fp16, interleaved, D=4)  → [FAIL] 流同步失败: 507035
L0_007 (fp32, interleaved, D=8)  → [FAIL] 流同步失败: 507035
L0_009 (fp16, interleaved, D=8)  → [FAIL] 流同步失败: 507035
L0_014 (fp32, interleaved, D=28) → [FAIL] 流同步失败: 507035
L0_006 (fp32, interleaved, D=64) / L0_003 (bf16, D=128) / L0_012 (fp16, D=1024) → 不崩（错值/通过）
```

plog 中 507035 对应的底层报错是 aivec 异常：`error code = 340`，`vec error info: 0x410028180031090e`。

## 适用场景

- 工程形态：AscendC kernel 在 UB 内做非连续取数/重排（interleaved↔half 布局转换、deinterleave、按索引抽取）考虑使用 Gather Level 2 API 的场景。
- API 路径：`Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& srcOffset, uint32_t srcBaseOffset, uint32_t count)`（Level 2 count 模式）；dav_3510 实现 `kernel_operator_vec_gather_impl.h`（`VfGatherApi2B32` 等）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/dav_3510），CANN 9.1.0；与 frontmatter `platforms` 一致。
- 适用粒度：platform_wide（dav_3510 的 Gather 实现行为；256B 向量寄存器宽度来自 GetVecLen()）。
- 路线优先级：safety_fallback——Gather 语义本身可用，但小 buffer 场景有隐蔽越界；本会话验证的路线是 SetValue 逐元素重排（标量回退，正确性与稳健性优先，性能代价未测量）。若坚持用 Gather，offset buffer 至少按 256B 分配并全量初始化 padding 区索引，该约束未在本会话独立验证。
- 明确不适用 / 未验证场景：offset buffer ≥256B 且索引全量初始化时 Gather 的正确性未在本会话单独验证；B16/B64 类型的 Gather 实现分支未逐一核对（机制同源，推断同样受 256B 寄存器宽度约束，未验证）。

## 根因

dav_3510 的 Gather Level 2 count 模式实现按**整条向量寄存器**（`GetVecLen()` = 256B，B32 类型即 64 个 uint32）为单位加载 offset：

```cpp
uint32_t dstRepeatCount = static_cast<uint32_t>(GetVecLen() / sizeof(T));       // float: 64
uint32_t u32OffsetRepeatCount = static_cast<uint32_t>(GetVecLen() / sizeof(uint32_t));  // 64
uint16_t repeatTime = CeilDivision(count, dstRepeatCount);
for (uint16_t i = 0; i < repeatTime; ++i) {
    dstMask = Reg::UpdateMask<uint32_t>(sreg);
    Reg::LoadAlign(offsetReg, srcOffsetLocal + i * u32OffsetRepeatCount);       // ← 每次读满 256B
    ShiftRights(indexReg, offsetReg, (int16_t)2, indexMask);                    // 字节偏移→元素索引（÷sizeof(T)）
    Reg::Gather(dstReg, (__ubuf__ uint32_t *)src + srcBaseIndex, indexReg, dstMask);
    Reg::StoreAlign((__ubuf__ uint32_t *)dst + i * dstRepeatCount, dstReg, dstMask);
}
```

`Reg::LoadAlign` 无条件读取**整条 256B 向量寄存器**的 offset 值。当 offset buffer 按实际 count 分配（如 D=4 → `AlignToBlock(16)` = 32B，D=28 → 128B）时，LoadAlign 会读出 buffer 之外最多 224B 的 UB 内容（越界读，未必立刻 fault），得到垃圾索引；`Reg::Gather` 用垃圾索引访问 src，触发源越界 → aivec 异常 340 → 对外表现为 `aclrtSynchronizeStream` 返回 507035。这与失败集完全吻合：D=4/8/28（buffer 32/32/128B < 256B）崩，D=64 及以上（buffer ≥256B）不崩。

另注：该实现同时确认了 `srcOffset` 语义是**字节偏移**（内部 `ShiftRights` 除以 sizeof(T) 转元素索引，B16 除 2、B64 除 8），`srcBaseOffset` 也是字节——索引数组要填 `i * sizeof(T)` 倍数。

## 解决配方

放弃 Gather，用 `GetValue/SetValue` 逐元素重排（最终版 kernel verbatim，interleaved 与 half 共用一套"重排到对齐 half 布局"的通道）：

```cpp
template <typename INPUT_T, int ROTARY_MODE>
__aicore__ inline void BenchNoHarnessApplyRotaryPosEmb<INPUT_T, ROTARY_MODE>::ReorganizeInputToHalf(
    const LocalTensor<COMPUTE_T>& src, LocalTensor<COMPUTE_T>& dst)
{
    uint32_t halfD = halfD_;
    uint32_t alignedHalfD = AlignToElems(halfD);

    if constexpr (IS_HALF_MODE) {
        for (uint32_t i = 0; i < halfD; i++) {
            COMPUTE_T v0 = src.GetValue(i);
            COMPUTE_T v1 = src.GetValue(halfD + i);
            dst.SetValue(i, v0);
            dst.SetValue(alignedHalfD + i, v1);
        }
    } else {
        for (uint32_t i = 0; i < halfD; i++) {
            COMPUTE_T v0 = src.GetValue(2 * i);
            COMPUTE_T v1 = src.GetValue(2 * i + 1);
            dst.SetValue(i, v0);
            dst.SetValue(alignedHalfD + i, v1);
        }
    }
}
```

```cpp
template <typename INPUT_T, int ROTARY_MODE>
__aicore__ inline void BenchNoHarnessApplyRotaryPosEmb<INPUT_T, ROTARY_MODE>::ReorganizeOutputFromHalf(
    const LocalTensor<COMPUTE_T>& src, LocalTensor<COMPUTE_T>& dst)
{
    uint32_t halfD = halfD_;
    uint32_t alignedHalfD = AlignToElems(halfD);

    if constexpr (IS_HALF_MODE) {
        for (uint32_t i = 0; i < halfD; i++) {
            COMPUTE_T v0 = src.GetValue(i);
            COMPUTE_T v1 = src.GetValue(alignedHalfD + i);
            dst.SetValue(i, v0);
            dst.SetValue(halfD + i, v1);
        }
    } else {
        for (uint32_t i = 0; i < halfD; i++) {
            COMPUTE_T v0 = src.GetValue(i);
            COMPUTE_T v1 = src.GetValue(alignedHalfD + i);
            dst.SetValue(2 * i, v0);
            dst.SetValue(2 * i + 1, v1);
        }
    }
}
```

调用侧（ProcessRow 内，替换原 Gather 重排）：

```cpp
    // 2. 转为 FP32 并重排到 half 布局
    if constexpr (NEED_CAST) {
        auto xFp32 = xFp32Buf_.Get<COMPUTE_T>();
        Cast(xFp32, xInDeq, RoundMode::CAST_NONE, alignedDCast);
        xInQueue_.FreeTensor(xInDeq);
        ReorganizeInputToHalf(xFp32, xWork);
    } else {
        ReorganizeInputToHalf(xInDeq, xWork);
        xInQueue_.FreeTensor(xInDeq);
    }

    // 3. 旋转计算
    RotateHalfRow(xWork, yWork);

    // 4. 重排回原始布局并转为输出 dtype
    auto yOutLocal = yOutQueue_.template AllocTensor<INPUT_T>();
    if constexpr (NEED_CAST) {
        auto yFp32 = yFp32Buf_.Get<COMPUTE_T>();
        ReorganizeOutputFromHalf(yWork, yFp32);
        Cast(yOutLocal, yFp32, RoundMode::CAST_NONE, alignedDCast);
    } else {
        ReorganizeOutputFromHalf(yWork, yOutLocal);
    }
    yOutQueue_.EnQue(yOutLocal);
```

若必须保留 Gather：offset buffer 至少分配 256B 并把 padding 区索引显式置 0（本会话未对该约束做独立验证，仅机制推导）。

## 设计决策

- 为什么放弃 Gather 而不是"把 offset buffer 撑到 256B"：撑 buffer 只堵住越界读这一条路，256B 常量与向量寄存器宽度耦合（跨架构可能变化）；SetValue 逐元素重排对小 D（D ≤ 数百）的行级重排开销可接受（本算子行長 D ≤ 1024，实测精度全对），换来对缓冲布局约束的免疫。
- 被放弃的路线还有：strided DataCopyPad（blockLen=sizeof(T) 的偶/奇分离）——在 arch35 上不可靠（设计文档即标注为风险项），更早被弃。
- 当前方案路线档位：safety_fallback——SetValue 重排相对向量化 Gather 是标量回退路线；本算子未做两条路线的成对性能对比（无 profiling 信号），采用理由是正确性与稳健性，性能代价未测量。

## 系统信息

- 环境：Ascend950PR（arch35/dav_3510），CANN 9.1.0。
- 相关文件：`op_kernel/arch35/<op>.h`（Gather 调用与重写对象）；`$ASCEND_HOME_PATH/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_impl.h`（VfGatherApi2B32 实现，根因证据）；`$ASCEND_HOME_PATH/x86_64-linux/asc/include/basic_api/kernel_operator_vec_gather_intf.h`（Level 2 签名）。
- 复现命令形态：真机 ST（Real 模式）跑含 interleaved 小 D 用例的套件，观察 507035 与用例 D 的相关性。

## 预防

- 在 dav_3510 上用 Gather Level 2 前，先按 `offset buffer 字节数 ≥ GetVecLen()`（256B）检查分配；索引数组 padding 区显式初始化。
- 行级小规模重排（D 在几百以内）默认先写 SetValue 版本保正确，再按 profiling 决定是否向量化。

## 验证信号

来源说明：以下信号来自该算子 ST 真机联调（Ascend950PR、CANN 9.1.0）中 Gather 版本与 SetValue 重写版本的对比，以及 CANN 头文件实读。

- Gather 版本：interleaved 小 D（D=4/8/28，offset buffer 32-128B）用例全部 `流同步失败: 507035`；offset buffer ≥256B 的 D=64/128/1024 用例不崩（来源平台：950；证据等级：tool_observed）。
- plog 底层信号：`there is an aivec error exception, core id is 0, error code = 340, ... vec error info: 0x410028180031090e`（来源平台：950；证据等级：tool_observed）。
- 头文件证据：`VfGatherApi2B32` 中 `Reg::LoadAlign(offsetReg, srcOffsetLocal + i * u32OffsetRepeatCount)` 按整条 256B 向量寄存器加载 offset，`u32OffsetRepeatCount = GetVecLen()/4 = 64`（来源平台：950；证据等级：tool_observed）。
- SetValue 重写版本：原 507035 的 interleaved 用例全部 PASS（MERE=0.00e+00，最大 262144 elems）；重写后 ST 12/15（全部 FP16/FP32 用例零误差，仅 3 个 BF16 用例因独立的 CAST_NONE 问题失败，另有专卡）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 256B 来自 `GetVecLen()`（dav_3510）；其他架构向量寄存器宽度不同，阈值需按目标架构重取（未验证，推断可迁移）。
- "offset buffer ≥256B 时 Gather 正确"未在本会话独立验证（本会话直接走了替代路线）；坚持用 Gather 的读者需自行验证该约束。
- SetValue/GetValue 为标量访存，大 D 高频重排的性能代价未测量（本算子场景正确性优先）；性能敏感场景需另行对比（无成对数据，不宣称相对代价）。

**未决项**：

- aivec error 340 的 `vec error info` 寄存器逐位语义未解码，"垃圾索引→源越界"的中间环节是机制推导（与失败集分区完全吻合），未做单变量隔离实验；如需机制级闭环，可在受控 kernel 中仅变化 offset buffer 大小复测。
- "offset buffer 按 256B 分配 + padding 索引清零"的 Gather 保活方案未验证，留给后续按需复测。
## 坏实践

- 按实际 count 精确分配 offset buffer（`AlignToBlock(count * 4)`）——在 LoadAlign 语义下必然越界读。
- 早期版本把 srcOffset 当**元素索引**填（`inputIdx.SetValue(i, 2 * i)`），与实现的字节偏移语义不符——正确写法是 `2 * i * sizeof(COMPUTE_T)`（该语义修正后 D=1024 用例曾转正确，佐证字节偏移语义）。

## 误诊

- 507035 先后被怀疑为：DataCopyParams/ExtParams 单位错位（修后无效）、UB buffer 大小未对齐（修后无效）、GM buffer 越界读（测试侧加 32B GM padding，无效）、blockLen 对齐方式与 isPad 组合（无效）、向量操作数非对齐（独立真因之一，另有专卡）——以上均为同失败集上的并行根因排查，Gather 越界读是最后由头文件实现级分析确认的主因。
