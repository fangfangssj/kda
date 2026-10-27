---
type: "Runbook"
title: int8 量化尾段 Cast 走固定 half 两步中转
description: Cast 支持表按产品线分表：Atlas A3（Ascend950）上 float→int8_t 不在表内却编译通过、运行期静默错乱（y 全错 max_diff=255 而 scale 全对）；两步 Cast 的中间类型还必须固定为 half——用输入 dtype（bfloat16_t）作中间时 bf16 用例仍大量错值。判别=scale 全对 y 全错的分叉 + 查 Cast 支持表；修复=fp32→half(CAST_ROUND)→int8(CAST_NONE)，并复用 absBuf/reduceTmpBuf 控 UB 预算。
tags: [task.debugging, topic.precision, topic.compute.elementwise.type_conversion, topic.compute.quantization]
status: stable
sources: []
created_at: '2026-09-17T01:58:00Z'
updated_at: '2026-09-17T01:58:00Z'
platforms: ["950"]
---

# int8 量化尾段 Cast 走固定 half 两步中转

## 现象

动态量化算子（per-token 对称量化：`scale = max(|x|)/127`，`y = clamp(round(x/scale), -128, 127)`，x fp16/bf16 → y int8 + scale fp32）在 Ascend950PR（arch35）上完成开发：host/kernel 全部编译通过、Kernel 二进制生成、UT（Tiling+InferShape）12/12 通过、ST Mock（CPU golden 自测）4/4 通过。部署算子包后跑 ST 真机（NPU 实测对比 CPU golden，标准 y |actual-golden|≤1.0、scale MERE<2^-13 / MARE<10×2^-13），结果**所有 5 条真机用例 y(int8) 全部失败、scale(fp32) 全部通过**：

```
[Real] shape [128, 256]:
  mismatch [0]: expected=-32, actual=116, diff=148
  mismatch [1]: expected=75,  actual=0,   diff=75
  [FAIL] y(int8) 32370 mismatches (max_diff=255, threshold=1)
  [PASS] scale(fp32) MERE=0.00e+00, MARE=0.00e+00, 128 elems
[Real] shape [1024, 1024]  → y 1036244 mismatches (max_diff=255)，scale PASS
... 5/5 用例同型失败
```

当时代码状态：kernel 的量化段（Cast fp16→fp32 → Abs → ReduceMax → 标量 scale → 重 Cast → Muls×invScale → Round → Mins/Maxs clamp）全部基于 fp32，最后一步直接把 fp32 转 int8：

```cpp
    // Step 9: Cast fp32 → int8
    Cast(yLocal, xFp32, RoundMode::CAST_NONE, lastDim_);
```

`yLocal` 是 `LocalTensor<int8_t>`，`xFp32` 是 `LocalTensor<float>`。这一行**编译完全通过**，但真机输出错乱：不是全 0 也不是溢出翻转，而是有规律的错误值（大量 0、部分位置错位样值）。

改为两步 Cast（fp32→half→int8）后 fp16 用例全过，但**仅 bf16 输入 + 非 32 对齐 lastDim 的 2 个用例** y(int8) 仍大量错值（shape [2^8] 74 处失配、[11,13,67] 3115 失配，max_diff=254，整字节翻转量级），scale 全对、同形 fp16 用例通过——因为该版两步 Cast 用输入缓冲（T_IN）作中间类型，bf16 输入时中间就是 bfloat16_t。

## 适用场景

- 工程形态：AscendC kernel（Vector 侧）+ aclnn ST 真机精度验证；fp32 计算链路末端需要落 int8 输出的量化/低比特算子。
- API 路径：`Cast(dst, src, roundMode, count)`（asc-devkit docs/api/context/Cast.md 按产品线分表的支持矩阵）；`Round`、`Mins`/`Maxs` clamp 链。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35，Atlas A3 系产品线）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该产品线 Cast API 类型支持表上所有 fp32/bf16→int8 企图，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：src=half → dst=int8_t 是支持的（表内明列），本卡针对 float→int8_t 与 bfloat16_t→int8_t 两个组合；其他产品线（A2 等）的 Cast 支持表未逐一核对，迁移前查对应表。

## 根因

两个叠加的 Cast 组合缺口，均在"编译通过、运行期静默错值"形态暴露：

1. **float→int8_t 不在 Atlas A3 支持表内**：Ascend C `Cast` API 的类型支持表按产品线分表；在 Atlas A3 表中，`src=float` 的合法 dst 只有 float / int32_t / half / int64_t / int16_t / bfloat16_t，**不含 int8_t**。`Cast(yLocal, xFp32, CAST_NONE, ...)` 的组合不在支持表内——模板能实例化、编译通过，但运行期不产生正确转换。scale 路径（ReduceMax + 标量 + DataCopyPad fp32）不经过该 Cast 全对（MERE=0/MARE=0）→ 说明数据搬运、多核切分、reduce 都没问题；y 路径唯一的类型转换出口就是这个未支持的 Cast → y 全错且 max_diff 可达 255（int8 全幅）。
2. **bfloat16_t→int8_t 组合输出错误值**：两步 Cast 用输入 dtype（T_IN=bfloat16_t）作中间类型时，`Cast(bfloat16_t, fp32, CAST_ROUND)` + `Cast(int8_t, bfloat16_t, CAST_NONE)` 中后一组合在该平台输出错误值，而同一代码路径在 T_IN=half 时正确——中间类型固定换成 half 后问题消失。

## 解决配方

**修复一（两步 Cast，中间跟随输入缓冲——对 fp16 输入正确）**：

```cpp
    // Step 9: Cast fp32 → int8 (两步：float→half→int8, 因为 Cast 不支持 float→int8_t)
    //   9a: fp32 → half (CAST_ROUND, 值已在[-128,127]整数范围, half可精确表示)
    //   9b: half → int8_t (CAST_NONE, 值已是整数无需再舍入)
    Cast(xLocal, xFp32, RoundMode::CAST_ROUND, lastDim_);
    Cast(yLocal, xLocal, RoundMode::CAST_NONE, lastDim_);
```

**修复二（中间类型固定为 half，不再跟随 T_IN——消除 bf16 错值）**：

```cpp
    // Step 7: Cast fp32 → int8 (两步：float→half→int8, 因为 Cast 不支持 float→int8_t)
    //   统一使用 half 作为中间类型（不使用 T_IN），避免 bfloat16_t→int8_t Cast 兼容性问题
    //   7a: fp32 → half (CAST_ROUND, 值已在[-128,127]范围, half可精确表示整数, round-half-to-even)
    //   7b: half → int8_t (CAST_NONE, 值已是整数无需再舍入)
    LocalTensor<half> xHalf = halfBuf.Get<half>();
    Cast(xHalf, xFp32, RoundMode::CAST_ROUND, lastDim_);
    Cast(yLocal, xHalf, RoundMode::CAST_NONE, lastDim_);
```

**修复三（消除独立 halfBuf，复用 absBuf/reduceTmpBuf 控 UB；SINGLE_TOKEN 复用 absBuf——ReduceMax 之后 absBuf 已无用）**：

```cpp
    //   absBuf 在 ReduceMax 后不再需要，复用为 half 中间缓冲区
    LocalTensor<half> xHalf = this->absBuf.template Get<half>();
    Cast(xHalf, xFp32, RoundMode::CAST_ROUND, lastDim_);
    Cast(yLocal, xHalf, RoundMode::CAST_NONE, lastDim_);
```

MULTI_TOKEN 同理复用 reduceTmpBuf（循环内该行 ReduceMax 完成后 scratch 即可复用），模板类内注意依赖名需 `template Get<half>()`：

```cpp
        // reduceTmpBuf 在 ReduceMax 后不再需要，复用为 half 中间缓冲区
        LocalTensor<half> xHalf = this->reduceTmpBuf.template Get<half>();
        Cast(xHalf, xFp32[rowOffset], RoundMode::CAST_ROUND, lastDim_);
        Cast(yLocal[rowOffset], xHalf, RoundMode::CAST_NONE, lastDim_);
```

配套（op_host tiling）：MULTI_TOKEN 的 reduceTmpSize 需同时容得下归约 scratch 与 half 中间缓冲：

```cpp
    uint32_t reduceTmpSizeSingle = ComputeReduceBufSize(lastDimAlign, sizeof(float));
    uint32_t halfBufNeeded = lastDimAlign * 2;  // half 中间缓冲区大小（2 bytes per half element）
    uint32_t reduceTmpSizeMulti = std::max({ComputeReduceBufSize(lastDimAlign, sizeof(float)),
                                             halfBufNeeded, MIN_REDUCE_TMP_SIZE});
```

改后必须清干净重编：`rm -rf <算子>/build` 并清 atc kernel cache（`rm -rf $HOME/atc_data/kernel_cache*`），再 `bash build.sh --soc=ascend950 -j8`、重装 .run 包后复跑 ST。

机制链（为何两步转换在本场景无损）：

1. Round + clamp 之后 `xFp32 ∈ [-128, 127]` 且均为整数。
2. half（fp16）可在 `[-2048, 2048]` 整数范围内精确表示每个整数（fp16 尾数 10 位，`2^11` 以内整数全部可精确编码），因此 fp32→half（CAST_ROUND）对已 clamp 的整数值是**无损**的（round 到的仍是同一整数）。
3. half→int8_t 走表内支持组合（CAST_NONE，值已是整数无需再舍入，溢出按饱和处理——但值域已在 int8 内，不触发）。
4. bf16 尾数仅 8 位，即使转换语义正确也不是理想的整型中转——中间类型必须显式写 `half`。

## 设计决策

- 为什么选 fp32→half→int8 而不是 fp32→int16/int32 再截位：half 中转只需要一次额外的支持内转换，且 [-128,127] 整数在 half 上无损（见机制链）；int32/int16 中转需要再自行做 `& 0xFF` 或饱和截位，多一次元素级运算且易引入符号截断 bug，放弃。
- 为什么固定 half 而不是 T_IN：half 可精确表示 [-128,127] 内全部整数，CAST_ROUND 的 round-half-to-even 与 golden（torch.round）一致；用输入 dtype 兼作中转的类型正确性依赖于"恰好 T_IN 是个合适的中间类型"这一巧合（bf16 输入即翻车）。
- 为什么消除独立 halfBuf：SINGLE_TOKEN 的 UB 预算 = lastDimAlign×14 + reduceTmp + 32；若额外加 lastDimAlign×2 的 halfBuf，lastDim=16384 时约 260KB，超过 248KB UB 上限（该算术为预算推导，非运行时实测溢出）；复用 absBuf/reduceTmpBuf 后约 228KB，且无需新增任何缓冲。复用前提是归约与尾段 Cast 的生命周期不重叠（absBuf 只在 Abs~ReduceMax 间被读、reduceTmp 只在 ReduceMax 调用期间作 scratch）。
- 当前方案属 recommended_fastpath。

## 验证信号

来源：该算子在 Ascend950PR 上的三轮 NPU 联调（修复前全错 → 两步 Cast → 固定 half 中间与缓冲复用），CANN 9.1.0。

- 修复前：5/5 真机用例 y(int8) 全失败（mismatch 最高 1036244/1048576、max_diff=255/254/228）；同批 scale(fp32) 全过（MERE=0、MARE=0）；CPU golden 自测 4/4 过（golden 逻辑本身正确）（来源平台：950；证据等级：tool_observed）。
- 查证 API 支持表：asc-devkit Cast.md Atlas A3 产品表 src=float 无 int8_t dst、src=half 有（来源平台：950；证据等级：tool_observed）。
- 修复一后（清缓存重编译）：ST 5/5 通过，NPU 实测 y(int8) max_diff=0~1（标准 ≤1.0），scale MERE=0、MARE=0；UT 12/12（来源平台：950；证据等级：tool_observed）。
- 扩展 bf16 后：仅 2 个 bf16 用例 y(int8) 大量错值（74/3115 mismatches，max_diff=254），scale 全 PASS、同形 fp16 用例 PASS——两步 Cast 以 T_IN（bfloat16_t）作中间（来源平台：950；证据等级：tool_observed）。
- 修复二（固定 half 中间 halfBuf）：ST 31/33 → 33/33，y max_diff≤1、scale MERE=0（来源平台：950；证据等级：tool_observed）。
- 修复三（absBuf/reduceTmpBuf 复用替代独立 halfBuf）后回归：UT 21/21 + ST 33/33 NPU 通过，含 bf16 全部用例（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "fp32→half 无损"依赖值域先 clamp 到 `[-128,127]` 整数；若把未 clamp/未取整的任意 fp32 值走两步 Cast，CAST_ROUND 舍入与 half 精度（11 位有效尾数）会引入误差，结论不外推。
- 支持表按产品线分档：本卡结论绑定 Atlas A3（Ascend950）表；A2 等其他产品线 Cast 支持组合不同，迁移前必须查对应表（未验证）。
- bfloat16_t→int8_t Cast 的确切失败机制（不支持的类型组合产生错误指令 vs 数值语义差异）未定位；本卡结论是"该组合输出错误值，换固定 half 中间即正确"，不是对该 API 的完备性判定。
- 复用 absBuf/reduceTmpBuf 的前提是归约与尾段 Cast 的生命周期确实不重叠；改变计算顺序（如把 Cast 提到 ReduceMax 前）会破坏该复用。
- lastDim=16384 + 独立 halfBuf 的 UB 溢出为预算算术推导（未运行时触发）；复用方案本身经 UT/ST 实测。
- 仅 CANN 9.1.0 + ascend950 实测（其他版本未验证，推断可迁移）。

**未决项**：

- 【优先】复现 bfloat16_t→int8_t Cast 单 API 最小用例（固定输入值对照输出），确认是不支持组合还是数值错误。
- float→int8_t 未支持组合**为何能通过编译**（模板实例化成功）而在运行期输出错乱的指令级机制未追查（trace 内无反汇编证据）；复现并追到微架构层可作为后续优先项。
- CAST_ROUND 与 golden `nearbyint`（half-to-even）在第一步的等价性由结果反推（修后 max_diff≤1 含 ±1 容差），未做逐位对拍。
## 坏实践

- 见"编译通过"就认定类型转换组合被支持——Cast 的支持表是产品线相关的运行时契约，模板能实例化不等于硬件转换路径存在；对低比特（int8/int4）落点必须先查表。
- 用输入 dtype 的 LocalTensor 兼作转换中间缓冲（`Cast(xLocal, xFp32, ...)`）——类型正确性依赖于"恰好 T_IN 是个合适的中间类型"这一巧合。
- 为修精度问题盲目加独立 UB 缓冲而不核 UB 预算——大 lastDim 场景会从精度问题换成 UB 超限。
- y 全错时先怀疑数据搬运/DataCopyPad 对齐——本例 scale 同链路搬运全对，可直接排除搬运问题，应从两输出的分叉点（最后的类型转换）入手。

## 误诊

- bf16 错值问题最初混在 20 例失败里被当作 scale 搬运问题的同批表现（同 trace 另行修复的 507035 崩溃消除后）；通过"仅 bf16 失败、fp16 同形通过、y 错而 scale 对"的组合定位到 dtype 相关的 Cast 路径。
