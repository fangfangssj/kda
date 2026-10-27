---
type: "Runbook"
title: 门梯度整体错位 — CopyOut srcStride 误用行距
description: DataCopyPad 多段统一 burst 搬出时 srcStride 应为段距而误按逻辑行距传参，每 burst 多跳 2×aUnit 致 UB 读源漂移、门梯度两路全错位而 grad_hx 全过。
tags: [task.debugging, topic.data_movement.data_copy, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 门梯度整体错位 — CopyOut srcStride 误用行距


## 现象

实现一个 AscendC 算子（GRU Cell 反向，base 模板分支）的 kernel：输出共五路，其中 `grad_hx` 是 (B,H) 单段行张量，两路门梯度 `grad_input_gates`/`grad_hidden_gates` 是 (B,3H) 张量（行内 3 个 H 宽段），UB 侧门梯度 tile 布局为 [rChunk 行 × 3 段]（每段 aUnit 宽，段内 valid 长 aLen）。CopyOut 用 `AscendC::DataCopyPad` + `DataCopyExtParams` 做 UB→GM 搬运：`grad_hx` 一次调用 blockCount=rChunk（burst=行），门梯度一次调用 blockCount=3×rChunk（burst=行×段统一枚举）。

编译通过后首跑 20 条 ST 精度用例仅 1 条通过，失败模式高度结构化：

```
[ST] B0-a fp32 B=64 H=2176 bias=1 ...
     grad_input_gates   FAIL (mere=2.066e+02 mare=7.327e+06 th=1.0e-05 nan_inf_mismatch=0)
     grad_hidden_gates  FAIL (mere=2.831e+02 mare=6.010e+06 th=1.0e-05 nan_inf_mismatch=0)
     determinism OK(bitwise,2runs) | verdict FAIL
[ST] E03 bf16 B=4 H=8 ... grad_hidden_gates FAIL (mere=6.580e+25 mare=6.316e+27 ...)
[ST] S1 fp32（grad_hy 全零） 五路输出 equals 0.0 精确断言 PASS
[ST] N1 / B0-c ... determinism MISMATCH | verdict FAIL
```

判别性观察：**`grad_hx` 在全部 20 条用例中零失败**；两路门梯度在 19/20 条用例大幅错位（mere≈1e2 量级，不是精度噪声）；S1（全零输入）五路全过；N1、B0-c 两条 bitwise 连跑两次结果不一致；bias 两路输出数值与后续修复版完全一致（本 bug 不污染 bias tile）。

## 适用场景

- 工程形态：AscendC 算子仓开发（aclnn ST 精度联调阶段），op_kernel 设备侧 UB→GM 搬出代码。
- API 路径：`AscendC::DataCopyPad` + `AscendC::DataCopyExtParams`（blockCount/blockLen/srcStride/dstStride），多 burst 二维切片式搬运。
- 验证过的平台与 CANN 版本：Ascend 950（AIV/arch35）+ CANN 9.0.0-beta.2。与 frontmatter `platforms` 严格一致。
- 适用粒度：`operator_family`（任何用 DataCopyPad 做行内多段（行,段）统一 burst 搬出的算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：blockCount=行数、blockLen=整行宽的"每 burst 一整行"搬运中 srcStride 语义即行距（本卡陷阱不存在）；其它芯片代次未验证。

## 判别方法

失败信号的结构本身就是判别器：

1. **同批输出中单段张量全过、多段张量全错** → 逐元素计算链与单段 CopyOut 公式正确，错误集中在多段 CopyOut 的寻址。
2. **mere≈1e2 而非 1e-5 量级** → 数值整体错位（寻址错误），非精度误差。
3. **determinism MISMATCH（bitwise 连跑不一致）** → 读到了未初始化/漂移的 UB 数据（确定性计算不会产生这个）。
4. **全零输入用例通过** → 漂移读源落在全零 tile 内仍读出零，与"UB 读地址错、GM 写地址对"的模型吻合。

## 根因

DataCopyPad 的 `srcStride` 是**相邻 burst 之间**的 UB 间距；当 blockCount 枚举的是行内子段（3×rChunk 个 burst，每个 burst=一段的 aLen valid 元素）时，该间距应等于段距 aUnit，而实现按逻辑二维视图的**行距 3·aUnit** 传参，导致每个 burst 多跳 2·aUnit、UB 读源持续漂移。

机制推导（不跳步）：

- burst (r, s)（第 r 行第 s 段）的 UB 源偏移 = `(r·3·aUnit + s·aUnit)·elemSz`，故**相邻 burst 前进恒为 `aUnit·elemSz`**（段内 s→s+1 与跨行 (r,2)→(r+1,0) 同值——跨行差 `3aUnit−2aUnit=aUnit`）。
- 实现公式：下一 burst 起点 = 当前起点 + `CeilAlign(aLen·elemSz, 32)` + `srcStride×32`。误传 pitch=3·aUnit 后：`CeilAlign(aLen·elemSz,32) + (3·aUnit·elemSz − CeilAlign(aLen·elemSz,32)) = 3·aUnit·elemSz` —— 每 burst 多跳 `3aUnit−aUnit = 2·aUnit` 个元素，随 burst 序号线性漂移。
- GM 侧 `dstStride=(H−aLen)·elemSz`：段内下一 burst 起点 `aOff+aLen+(H−aLen)=aOff+H`（段距 H）；跨行从 (r,2) 尾到 (r+1,0) 头 = `3H−2H−aOff−aLen+aOff = H−aLen`，段内跨行**同值** → GM 落点全部正确。
- 因此只有 UB 源错、GM 落点对：门梯度两路整体错位；漂移读越过 tile 边界进入未初始化区域解释 determinism MISMATCH；全零 tile（S1）漂移读仍零解释 S1 通过；`grad_hx` 的 burst=行、行距=aUnit，公式恰好正确故全过。

## 解决配方

**before**（base.h 的 CopyOutElewise，节选自构建时版本；调用点以 pitch 区分三路）：

```cpp
// ---------------------------------------------------------------------------
// CopyOutElewise（§5.4 (a)）：单路逐元素 tile UB→GM（MTE3）
//   grad_hx：rowW=H / pitch=aUnit（blockCount=rChunk）；
//   门梯度：rowW=3H / pitch=3·aUnit（blockCount=3×rChunk，行×段统一 burst——
//   段内与跨行的 UB gap、GM gap 恰好各自相等：段间 UB gap =
//   pitch×elemSz − CeilAlign(aLen×elemSz,32)、GM gap = (H−aLen)×elemSz，跨行同值）。
//   blockCount 上界 3×rUbFactor ≤ 3×rIMax ≈ 1212 < 4095；A 方向 garbage lane 不拷出
//   （一律 valid aLen 口径）。
// ---------------------------------------------------------------------------
template <typename D_T>
__aicore__ inline void ThnnFusedGruCellGradBaseKernel<D_T>::CopyOutElewise(
    int32_t share, AscendC::GlobalTensor<D_T>& gm, int64_t rIdx0, uint32_t rChunk,
    int64_t aOff, uint32_t aLen, int64_t rowW, int64_t pitch)
{
    const int64_t elemSz = static_cast<int64_t>(sizeof(D_T));
    const int64_t aLenBytes = static_cast<int64_t>(aLen) * elemSz;
    const int64_t copyPadBytes = (aLenBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;

    AscendC::DataCopyExtParams cp{};
    cp.blockCount = (rowW == H_) ? static_cast<uint16_t>(rChunk)
                                 : static_cast<uint16_t>(3 * rChunk);   // 门梯度：行×段
    cp.blockLen = static_cast<uint32_t>(aLenBytes);                      // valid 字节
    cp.srcStride = (pitch * elemSz - copyPadBytes) / BLOCK_BYTES;        // UB 32B gap
    cp.dstStride = (H_ - static_cast<int64_t>(aLen)) * elemSz;           // GM byte gap = (H−aLen)×elemSz
    cp.rsv = 0;
    AscendC::DataCopyPad(gm[rIdx0 * rowW + aOff], SlotTensorD_T(share), cp);
}
```

调用点（before）：

```cpp
    CopyOutElewise(SLOT_BT, gmHx_, rIdx0, rChunk, aOff, aLen, H_, aUnit_);
    CopyOutElewise(SLOT_GATES_IN, gmInGates_, rIdx0, rChunk, aOff, aLen, 3 * H_, 3 * aUnit_);
    CopyOutElewise(SLOT_GATES_HID, gmHidGates_, rIdx0, rChunk, aOff, aLen, 3 * H_, 3 * aUnit_);
```

**after**（删除 pitch 参数，三路 burst 间距统一为段距 aUnit；注释同步修正语义）：

```cpp
// ---------------------------------------------------------------------------
// CopyOutElewise（§5.4 (a)）：单路逐元素 tile UB→GM（MTE3）
//   grad_hx：rowW=H（blockCount=rChunk，burst=行）；
//   门梯度：rowW=3H（blockCount=3×rChunk，burst=行×段统一）。
//   burst→burst 间距三路统一为段距 aUnit（门梯度段内与跨行的 UB gap、GM gap 恰好
//   各自相等：段间 UB gap = aUnit×elemSz − CeilAlign(aLen×elemSz,32)、
//   GM gap = (H−aLen)×elemSz，跨行同值——§5.4(a) 原文）。
//   blockCount 上界 3×rUbFactor ≤ 3×rIMax ≈ 1212 < 4095；A 方向 garbage lane 不拷出
//   （一律 valid aLen 口径）。
// ---------------------------------------------------------------------------
template <typename D_T>
__aicore__ inline void ThnnFusedGruCellGradBaseKernel<D_T>::CopyOutElewise(
    int32_t share, AscendC::GlobalTensor<D_T>& gm, int64_t rIdx0, uint32_t rChunk,
    int64_t aOff, uint32_t aLen, int64_t rowW)
{
    const int64_t elemSz = static_cast<int64_t>(sizeof(D_T));
    const int64_t aLenBytes = static_cast<int64_t>(aLen) * elemSz;
    const int64_t copyPadBytes = (aLenBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;

    AscendC::DataCopyExtParams cp{};
    cp.blockCount = (rowW == H_) ? static_cast<uint16_t>(rChunk)
                                 : static_cast<uint16_t>(3 * rChunk);   // 门梯度：行×段
    cp.blockLen = static_cast<uint32_t>(aLenBytes);                      // valid 字节
    cp.srcStride = (static_cast<int64_t>(aUnit_) * elemSz - copyPadBytes) / BLOCK_BYTES;  // UB 段距 32B gap
    cp.dstStride = (H_ - static_cast<int64_t>(aLen)) * elemSz;           // GM byte gap = (H−aLen)×elemSz
    cp.rsv = 0;
    AscendC::DataCopyPad(gm[rIdx0 * rowW + aOff], SlotTensorD_T(share), cp);
}
```

调用点（after）：

```cpp
    CopyOutElewise(SLOT_BT, gmHx_, rIdx0, rChunk, aOff, aLen, H_);
    CopyOutElewise(SLOT_GATES_IN, gmInGates_, rIdx0, rChunk, aOff, aLen, 3 * H_);
    CopyOutElewise(SLOT_GATES_HID, gmHidGates_, rIdx0, rChunk, aOff, aLen, 3 * H_);
```

## 设计决策

- 修复选择**删掉 pitch 参数**而非"给门梯度调用传 aUnit"：`grad_hx` 的 burst=行、间距=aUnit；门梯度的 burst=段、间距=aUnit——两路本就同值，保留参数只会留下"行距/段距二选一"的歧义陷阱；参数化一个恒定值即隐患。
- 顺带发现 burst 间距在段内与跨行**恰好同值**（UB 侧 aUnit、GM 侧 H−aLen）是"行×段统一 burst"这个设计成立的必要条件——若 tile 行距不是 3 段等宽（如异构段宽），统一 burst 方案不成立，需退回逐段调用。
- `dstStride` 不动：推导证明 GM 侧段内跨行同值，落点本来就对。

## 系统信息

- 工具链：CANN cann-9.0.0-beta.2；目标 ascend950 / arch35 / AIV；构建产物 `build/<Op>_ascend950/`。
- 相关文件：`op_kernel/arch35/<op>_base.h`（CopyOutElewise 函数及其调用点、门梯度 tile 布局常量）；ST 工程 `tests/aclnn/`（run.sh + 20 条用例清单 + 期望文件）。
- UB tile 布局：门梯度 tile [rChunk × 3·aUnit]（行内 3 段 s-major，段宽 aUnit，valid 长 aLen ≤ aUnit，A 方向 garbage lane 不拷出）。

## 验证信号

- 修复前：ST 20 条仅 1 条过；门梯度两路 19/20 大幅失败（B0-a mere=2.066e+02 / mare=7.327e+06；E03 bf16 mere=6.580e+25）、grad_hx 零失败、N1/B0-c determinism MISMATCH（来源平台：950；证据等级：tool_observed）。
- 修复后：ST 16/20——两路门梯度全部通过、grad_hx 保持通过、特殊值用例（全零/NaN/Inf 传播/饱和）全部通过、determinism 全部 OK(bitwise,2runs)（来源平台：950；证据等级：tool_observed）。
- bias 两路数值修复前后一致（如 P7 mare=8.085e-04 两轮相同），证明本 bug 只错 CopyOut 的 UB 读地址、bias tile 内容本就正确（来源平台：950；证据等级：tool_observed）。
- 回归：TilingUT 60/60 全绿不回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡针对"blockCount 枚举行内子段"的多 burst 搬运；若搬运按整行 burst（blockCount=行数、blockLen=行宽），srcStride 语义就是行距，不存在此陷阱。
- 段内/跨行间距同值依赖"行内 3 段等宽 aUnit"的布局假设；异构段宽或行尾 padding 不为 0 的布局未验证。
- 验证于 ascend950 arch35 AIV + CANN 9.0.0-beta.2；其它平台（未验证，推断可迁移——DataCopyExtParams 语义是 CANN 通用契约）。

## 坏实践

- 用"哪个 tile"的指针比较选槽位：同文件早版自查发现 CopyOutElewise 里写过 `reinterpret_cast<int64_t>(tile)` 比较 `__ubuf__` 指针选 share 槽位的 hack，编码前即重构为显式传 share 索引（本例无失败信号，作为负例记录：地址空间限定指针转整数比较既不可靠也难读）。
- 看到"门梯度全错、grad_hx 全对"就去复查门梯度计算链的数学——本例五链算术与 golden 位级一致（numpy 复算 bitwise 相等），错误纯在搬出寻址；先用"哪路输出对/哪路错"切分计算域与搬运域再定位。

## 误诊

- 曾把 bias 大 B 用例的 mare 超阈与门梯度 bug 归为同因——分析确认 CopyOut 只错 UB 读地址、bias 归约路径不经该 CopyOut，两问题独立（bias 问题另见同域诊断卡）。
- 曾以"garbage lane 污染归约 / 跨核 race / cacheBuf 跨迭代残留"假设解释 bias 小误差，逐一被布局与同步推导排除。

## 预防

- 写 DataCopyPad 前先**手写 burst 枚举序列**：列出每个 burst 的 (源首地址, 目的首地址, 长度)，逐对算出相邻 burst 的首地址差，再填 `srcStride`/`dstStride`；"行距"只在每 burst=整行时才等于"burst 间距"。
- 代码评审 checklist：`srcStride` 的注释语义与传参值是否一致（本例注释写的是段距公式、传参却是行距——公式对、参数错）。
- 联调出现 determinism MISMATCH 时优先怀疑"读到未初始化/越界 UB"，而不是计算顺序。

## 未决事项

- 无：根因推导（burst 间距算式）与修复后实测（门梯度/确定性全转绿）互为印证，闭环完整。
