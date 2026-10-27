---
type: "Runbook"
title: "双边广播输出错乱：合轴 stride 误用输出维度累乘"
description: "广播合轴后用单一 stride 变量按合轴输出维度统一累乘再分给两个输入，非广播轴 stride 被相邻广播轴放大（如 [1,512,1] 的 axis1 stride 算成 4 而非 1），kernel 读越界输出垃圾（MARE 3.12e+04）；须按各输入自身非广播轴分别累乘。"
tags: [task.debugging, topic.implementation.tiling, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# 双边广播输出错乱：合轴 stride 误用输出维度累乘

## 现象

自定义算子（y = max(x1, x2)，最大值类逐元素）刚扩展完全 6 dtype + 广播路径（24 个 TilingKey 实例，arch22/arch35 编译通过、55 个 UT 全绿），在 Ascend950PR 上跑 C++ ST 联调验证（NPU 执行 + CPU golden 比对），13 个用例中 11 个通过，唯独双边广播用例全面崩溃：

```
[Real] 双边广播用例: float32 [512,512,4]x[1,512,1]
  dtype=float32, x1=[512, 512, 4], x2=[1, 512, 1], out=[512, 512, 4]
  [FAIL] float32 MERE=1.15e+00 (thr=1.22e-04), MARE=3.12e+04 (thr=1.22e-03)
    不匹配 [4]: golden=4.222465, actual=0.032613
    不匹配 [5]: golden=4.222465, actual=0.032613
    不匹配 [7]: golden=4.222465, actual=3.201633
    不匹配 [8]: golden=-0.339886, actual=1.217715
```

特征：输出不是小偏差而是完全错乱（MARE 3.12e+04）；无广播、标量广播（x2=[1]）、单边广播用例全部 PASS；UT（host 侧 tiling 断言）当时也是全绿的——但后来发现 UT 的期望值本身就是按错误公式写死的。

## 适用场景

- 工程形态：aclnn 自定义算子的广播 tiling（op_host TilingFunc 做广播合轴/CollapseBroadcast，把 outDims/x1Strides/x2Strides 写入 TilingData 供 kernel 按 stride 计算 GM 偏移）。
- API 路径：TilingFunc 内广播合轴（对齐 rank → 标记广播轴 → 合并相邻同 flag 轴 → 计算 stride）；kernel 侧 `ComputeGmOffset`（按 axesIndices × strides 累加偏移）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/ascend950，CANN 9.1.0）实测复现并修复验证（与 frontmatter `platforms` 一致）；arch22 同源 tiling 同步修复，未上板。
- 适用粒度：operator_family——一切"合轴 + stride 表"式广播实现（逐算子通用模式）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：使用 AscendC `Broadcast<T, dim, axis>` 向量 API 的实现（stride 语义由 API 承担）；单输入无广播算子。

## 根因

广播合轴的 Step 3 用**一个** `stride` 变量按**合轴后的输出维度**统一累乘，再把同一结果同时填给 `x1Strides[j]` 和 `x2Strides[j]`（仅广播轴置 0）：

```cpp
    for (int64_t k = j + 1; k < result.dimNum; k++) {
        stride *= result.dims[k];   // dims 是合轴后的输出维度
    }
    result.x1Strides[j] = (result.flags[j] & 1) ? 0 : stride;
    result.x2Strides[j] = (result.flags[j] & 2) ? 0 : stride;
```

正确语义是**每个输入各用自身的维度**累乘：非广播轴上输入维度 == 输出维度可借用合轴 dims，但**广播轴（对某个输入 dim=1）不得参与该输入的 stride 累乘**。错在两处叠加：(a) 只广播轴被置 0，其维度却仍留在内层累乘里；(b) 两个输入共用同一累乘值。以 x2=[1,512,1]（out=[512,512,4]）为例：axis1 的正确 stride 是 1（x2 自身 axis1 内层只有 axis2，而 axis2 对 x2 是广播轴、dim=1 不参与），错误实现累乘了输出 axis2 的维度 4，得到 4——kernel 按 stride=4 读 x2 直接越界读到别的行，输出垃圾。

为什么"标量广播 [1] / 单边广播"此前能通过：这些场景合轴后只剩一个非平凡维或广播轴恰在端部，统一累乘恰好与正确值一致；双边广播且广播轴夹在中间（[1,512,1]）才把错误暴露出来。

## 解决配方

修复前（arch35 与 arch22 的 `op_host/arch*/<op>_tiling.cpp`，CollapseBroadcast Step 3；单 stride 按输出维度统一累乘，verbatim）：

```cpp
    // Step 3: 计算 stride（行优先，广播轴 stride=0）
    for (int64_t j = 0; j < result.dimNum; j++) {
        int64_t stride = 1;
        for (int64_t k = j + 1; k < result.dimNum; k++) {
            stride *= result.dims[k];
        }
        result.x1Strides[j] = (result.flags[j] & 1) ? 0 : stride;
        result.x2Strides[j] = (result.flags[j] & 2) ? 0 : stride;
    }
```

修复后（x1Stride/x2Stride 各自仅累乘"该输入非广播轴"的维度，verbatim）：

```cpp
    // Step 3: 计算 stride（行优先，广播轴 stride=0）
    // stride 基于各输入张量自身的维度累乘，而非合轴后的输出维度：
    //   - 非广播轴：输入维度 == 输出维度，使用合轴 dims 累乘；
    //   - 广播轴：不参与该输入的 stride 累乘（该输入 stride=0）。
    // 修复广播 stride 缺陷：原实现用合轴输出维度统一累乘，导致非广播轴
    // stride 被相邻广播轴维度错误放大（如 x2=[1,512,1] 的 axis1
    // stride 被算成 4 而非 1），kernel 读取越界产生垃圾数据。
    for (int64_t j = 0; j < result.dimNum; j++) {
        int64_t x1Stride = 1;
        int64_t x2Stride = 1;
        for (int64_t k = j + 1; k < result.dimNum; k++) {
            if ((result.flags[k] & 1) == 0) {
                x1Stride *= result.dims[k];  // x1 非广播轴：累乘自身维度（==输出维度）
            }
            if ((result.flags[k] & 2) == 0) {
                x2Stride *= result.dims[k];  // x2 非广播轴：累乘自身维度（==输出维度）
            }
        }
        result.x1Strides[j] = (result.flags[j] & 1) ? 0 : x1Stride;
        result.x2Strides[j] = (result.flags[j] & 2) ? 0 : x2Stride;
    }
```

配套修正（UT 期望值同步纠错，`tests/ut/op_host/test_<op>_tiling.cpp`，4 处按错误公式写死的断言）：

```cpp
    // 单边广播 [4,1]×[4,4] → [4,4], fp16, 单缓冲
    //   合轴: dims=[4,4], flags=[0,1(x1广播轴1)]
    //   x1S=[1,0](axis0 stride=x1自身内层维度=1), x2S=[4,1]
    //   ubSplitAxis=1(广播轴), innerProduct=1, Duplicate: ubFormer=4
    {"bcast_single_side_4x1_4x4_fp16",
     {4,1}, {4,4}, {4,4}, ge::DT_FLOAT16,
     KEY_FP16_BCAST_SINGLE, 16, 1, 4,
     2, {4,4}, {1,0}, {4,1},
     1, 1, 4, 1, 4},
```

（双边 [4,1]×[1,4] 的 x1S `{4,0}`→`{1,0}`；中维广播 [512,512,4]×[1,512,1] 的 x2S `{0,4,0}`→`{0,1,0}`——即本卡核心场景；异 rank [256,256]×[1,256,1] 的 x2S `{256,0}`→`{1,0}`。）

判别方法：双边广播输出整体错乱（MARE 数量级 1e+2 以上、不匹配值与 golden 完全无关）而单边/标量广播正常时，优先 dump TilingData 的 x1Strides/x2Strides，用"各输入自身维度累乘、广播轴不参与"手工核算每个非广播轴 stride。

## 设计决策

- 为什么在 tiling 侧修而不是 kernel 侧兼容：stride 表语义（行优先、广播轴 0）是 kernel `ComputeGmOffset` 的既有契约，kernel 无从分辨"stride 被放大"与"输入本来就稀疏"；错在生成侧就修生成侧。
- 为什么合轴后仍可用 `result.dims[k]` 累乘：合轴规则保证相邻同 flag 轴才合并，非广播轴上两输入维度==输出维度，`dims[k]` 即自身维度；只需用 flags 跳过"对该输入是广播轴"的项。
- 为什么同步修 UT 期望值而不是保留"历史行为"：原期望值是从错误实现反推的（`{4,0}`、`{0,4,0}`），保留会把 bug 固化成断言。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写广播 tiling 的 stride 表时固定心法："一个输入一条 stride 链，广播轴既不贡献维度也不贡献 stride"。
- UT 断言 stride 期望值必须从输入自身 shape 手工推导，不从实现输出反推；review 时把"x1S/x2S 是否相同"当作 smells（两输入 shape 不同却共享同一 stride 元组几乎必错）。
- ST 广播用例至少覆盖：双边广播 + 广播轴位于中间维度（[1,512,1] 型）——端部广播用例覆盖不到该缺陷。

## 验证信号

以下信号来自该算子联调验证会话：首轮 ST 双边广播用例崩溃，修复 tiling stride 计算并同步修正 UT 期望值后，重建重装算子包全量回归（Ascend950PR + CANN 9.1.0）：

- 修复前：ST 双边广播 float32 [512,512,4]×[1,512,1] FAIL，MERE=1.15e+00、MARE=3.12e+04，失配元素 golden=4.222465 vs actual=0.032613（输出垃圾）；同轮无广播/标量广播/单边广播及全 6 dtype 用例 PASS（来源平台：950；证据等级：tool_observed）。
- 修复前：UT 中 4 个广播用例的期望 stride 为 `{4,0}`/`{0,4,0}`/`{256,0}`（按错误公式断言且通过，未拦截该缺陷）（来源平台：950；证据等级：tool_observed）。
- 修复后：重建安装（ascend950），ST 13/13 通过，双边广播用例 MERE=0.00e+00、MARE=0.00e+00（1048576 元素全对）（来源平台：950；证据等级：tool_observed）。
- 修复后：UT 全量 55/55 无回归（op_host 32 含修正后的广播用例，op_api 23）（来源平台：950；证据等级：tool_observed）。
- 交付前 `grep` 确认 arch22/arch35 两份 tiling 中 buggy 模式 `stride *= result.dims[k]` 已完全移除、替换为 `x1Stride *=` / `x2Stride *=`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 修的是"合轴 + stride 表"自定义广播路径；若换用 AscendC Broadcast 向量 API 或框架自动广播，本卡不适用。
- stride 语义按行优先（C contiguous）输入推导；转置/非连续输入（stride ≠ 行优先）在本卡场景外（当时代码假定 `AutoContiguous` 已连续化）。
- arch22 侧 tiling 同步修复但仅编译验证、未上板（未验证，推断一致——两架构 tiling 逻辑同源）。

**未决项**：

- 无（根因已追到 tiling 公式层并双端验证）。
## 坏实践

- 合轴后用单一 stride 变量按输出维度累乘，再同时赋给两个输入的 stride 数组。
- UT 期望值从当前实现的实际输出抄录（`{0,4,0}` 即由此而来）——等于用被测代码给自己背书。
- 只用端部广播（[1,n]/[n,1]）做广播回归——覆盖不到中间广播轴。

## 误诊

- 首轮失败分析时一度把另一用例（MARE 3.29e-02、golden≈actual 的小偏差形态）与本用例（输出完全垃圾）混为"广播精度问题"同类；实际两者根因不同：本卡是 tiling stride 功能错误（输出错乱），另一用例是测试 golden 数据源不一致，须分开修。
