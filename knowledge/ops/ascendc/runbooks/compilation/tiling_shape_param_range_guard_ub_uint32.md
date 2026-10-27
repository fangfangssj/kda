---
type: "Runbook"
title: "tiling 派生参数须补范围守卫防 UB 溢出回绕"
description: Tiling 从输入 shape 派生的 hidden/numRows/numExperts/expertCapacity 等参数只做结构校验（dim/整除/枚举）无上下限时，超范围输入直达 kernel 的 UB 分配（约 hidden×20 字节）与 uint32 字节回绕；修复为与需求对齐的 constexpr 上限常量 + 派生点逐参数 OP_CHECK_IF 区间守卫。
tags: [task.implementation, topic.memory, topic.mixture_of_experts]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# tiling 派生参数须补范围守卫防 UB 溢出回绕

## 现象

一个 MoE finalize routing 自定义算子交付前代码检视（C++ 安全检视，SEC-4.1"外部输入合法性校验"条款）发现：`op_host/arch35/` 的 tiling 源文件 `GetInputMetaInfo()` 从输入 shape 派生四个维度参数——`hidden`（`expanded_permuted_rows` 最后一维）、`numRows`（`nk / topK`）、`numExperts`（`bias.shape[0]`）、`expertCapacity`（3D 输入的 `shape[1]`）。当时的代码只校验了结构合法性（dim 数 ≥ 2、`topK > 0`、`nk % topK == 0`、dropPadMode ∈ [0,3]、可选输入依赖关系），四个维度参数**均无上下限校验**：

```cpp
// 修复前（op_host/arch35/<op>_tiling.cpp，GetInputMetaInfo()）
    info.hidden = expandedPermutedShape.GetDim(static_cast<size_t>(epDimNum - 1));
    ...
    info.numRows = nk / info.topK;
    ...
    // E = bias.shape[0] if bias else 0
    if (info.hasBias) {
        auto biasShapePtr2 = context->GetInputShape(4);
        info.numExperts = biasShapePtr2->GetStorageShape().GetDim(0);
    }

    // C: drop_pad 模式下 expanded_permuted_rows 为 (E, C, H)，C = shape[1]
    // drop_pad_mode 1/3 时 epDimNum=3
    if (epDimNum == 3) {
        info.expertCapacity = expandedPermutedShape.GetDim(1);
    }
```

检视给出了量化的风险链（需求约束 H∈[1,8192]、NUM_ROWS∈[1,524288]、E∈[1,256]、C∈[1,1024]）：

1. **UB 溢出**：kernel 为低精度全可选路径分配 11 个 TBuf（out/gather/bias/skip 各"原 dtype + fp32"双份 + 索引/缩放标量缓冲），估算总占用约 `hidden × 20` 字节——H=8192 时约 160KB ≤ 248KB（arch35 UB 容量，设计校核通过）；若 hidden=16384 则 320KB > 248KB，`InitBuffer` 溢出。
2. **uint32 回绕**：kernel 侧 `uint32_t hBytes = static_cast<uint32_t>(hidden_) * sizeof(D_T_MAIN)`，hidden 超过 `UINT32_MAX/4`（约 10 亿）时回绕。
3. numRows/numExperts/expertCapacity 超限时多核切分与 `E*C` 行数计算负载失控。

## 适用场景

- 工程形态：AscendC 自定义算子 op_host tiling（gert::TilingContext），shape 派生参数进入 TilingData 并驱动 kernel UB 分配。
- API 路径：`GetInputShape()->GetStorageShape().GetDim()` 派生参数 + `OP_CHECK_IF`/`OP_LOGE` 守卫；kernel 侧 `TPipe::InitBuffer` UB 预算。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510，build `--soc=ascend950`），CANN 9.1.0，ST Real NPU 模式实测。
- 适用粒度：platform_wide（UB 容量预算与 uint32 回绕是平台级约束；具体上限值按各算子需求推导，不是通用常量）。
- 路线优先级：recommended_fastpath（tiling 侧拒绝非法输入，越早越好）。
- 明确不适用 / 未验证场景：拒绝路径（GRAPH_FAILED + 错误日志）无专项负向用例实测；其他平台 UB 容量不同需重推。

## 根因

Tiling 从输入 shape 派生参数后只做了结构合法性校验（dim/整除/模式枚举），没有把需求规定的参数范围变成守卫，超范围输入会一路传到 kernel 的 UB 分配与 uint32 字节计算——离错误源头远且难定位。

## 解决配方

先新增与需求对齐的上限常量，再对每个参数在派生点立即校验：

```cpp
// 修改后（新增范围常量，与需求 §4.6 约束一致，防止 UB 溢出和 uint32_t 回绕）
constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t UB_BLOCK_BYTES = 32;  // DAV_3510 UB 最小对齐 32B

// 参数上限常量（与需求 §4.6 约束一致，防止 UB 溢出和 uint32_t 回绕）
constexpr int64_t MAX_NUM_ROWS = 524288;        // NUM_ROWS ∈ [1, 524288]
constexpr int64_t MAX_TOP_K = 32;               // K ∈ [1, 32]
constexpr int64_t MAX_HIDDEN = 8192;            // H ∈ [1, 8192]（UB 容量约束）
constexpr int64_t MAX_NUM_EXPERTS = 256;        // E ∈ [1, 256]
constexpr int64_t MAX_EXPERT_CAPACITY = 1024;   // C ∈ [1, 1024]
```

```cpp
// 修改后（GetInputMetaInfo() 内逐参数守卫）
    info.hidden = expandedPermutedShape.GetDim(static_cast<size_t>(epDimNum - 1));
    OP_CHECK_IF(info.hidden <= 0 || info.hidden > MAX_HIDDEN,
        OP_LOGE(context, "hidden must be in [1, %ld], got %ld", MAX_HIDDEN, info.hidden),
        return ge::GRAPH_FAILED);
    ...
    info.numRows = nk / info.topK;
    OP_CHECK_IF(info.numRows <= 0 || info.numRows > MAX_NUM_ROWS,
        OP_LOGE(context, "numRows must be in [1, %ld], got %ld", MAX_NUM_ROWS, info.numRows),
        return ge::GRAPH_FAILED);
    ...
    // E = bias.shape[0] if bias else 0
    if (info.hasBias) {
        auto biasShapePtr2 = context->GetInputShape(4);
        info.numExperts = biasShapePtr2->GetStorageShape().GetDim(0);
        OP_CHECK_IF(info.numExperts <= 0 || info.numExperts > MAX_NUM_EXPERTS,
            OP_LOGE(context, "numExperts must be in [1, %ld], got %ld", MAX_NUM_EXPERTS, info.numExperts),
            return ge::GRAPH_FAILED);
    }

    // C: drop_pad 模式下 expanded_permuted_rows 为 (E, C, H)，C = shape[1]
    // drop_pad_mode 1/3 时 epDimNum=3
    if (epDimNum == 3) {
        info.expertCapacity = expandedPermutedShape.GetDim(1);
        OP_CHECK_IF(info.expertCapacity <= 0 || info.expertCapacity > MAX_EXPERT_CAPACITY,
            OP_LOGE(context, "expertCapacity must be in [1, %ld], got %ld",
                    MAX_EXPERT_CAPACITY, info.expertCapacity),
            return ge::GRAPH_FAILED);
    }
```

topK 原有的 `> 0` 校验同步扩为区间：

```cpp
// 修改前
    OP_CHECK_IF(info.topK <= 0,
        OP_LOGE(context, "K must be positive, got %ld", info.topK),
        return ge::GRAPH_FAILED);

// 修改后
    OP_CHECK_IF(info.topK <= 0 || info.topK > MAX_TOP_K,
        OP_LOGE(context, "K must be in [1, %ld], got %ld", MAX_TOP_K, info.topK),
        return ge::GRAPH_FAILED);
```

## 设计决策

- **校验放 tiling 而不是 kernel**：这些参数全部可由输入 shape 推导，tiling 是第一入口，越早拒绝越省（GRAPH_FAILED 直接打回，不会走到 kernel 的 UB 分配）；kernel 内只兜底数据内容级校验，两层分工。
- **上限值从需求推导，不是通用常量**：MAX_HIDDEN=8192 的依据是 UB 预算（约 `hidden × 20` 字节/低精度全可选路径 ≤ 248KB），迁移到 buffer 布局不同的算子时要按自己的 TBuf 清单重推。
- **日志带 got 值**：`OP_LOGE(... "must be in [1, %ld], got %ld", ...)` 让调用方一眼定位是哪个参数超限。
- **当前方案路线**：recommended_fastpath。

## 预防

- tiling 里每个从 shape 派生的维度参数，派生点紧跟 `[下限, 上限]` 校验；上限用 UB 预算/uint32 安全域推导并写成 constexpr、注释保留推导账。
- 涉及 `static_cast<uint32_t>(...) * sizeof(T)` 的字节计算，检查乘积是否可能超过 UINT32_MAX。

## 验证信号

来源说明：该算子交付前代码检视修复闭环（Ascend950PR，CANN 9.1.0）：按上述配方补齐 tiling 范围守卫后全量重新构建与回归。

- 修复后重新编译通过：3 个 dtype 的 kernel 二进制生成、CPack 打包成功（来源平台：950（Ascend950PR，交叉编译宿主 x86_64）；证据等级：tool_observed）。
- ST 回归 57/57 PASS（Real NPU 模式），其中 L1-052（fp16 H=8192 UB 容量边界）/L1-053（bf16 H=8192）在新增上限边界内通过（来源平台：950；证据等级：tool_observed）。
- UT 回归 40/40 PASS（tiling 单测）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "hidden × 20 字节 ≈ 160KB@H=8192、248KB 上限"的 UB 预算数字为检视对 TBuf 清单的估算（会话内推导），与设计文档 UB 容量校核一致，非实测 profiling。

- 守卫只拒绝 shape 派生参数的超范围输入，不覆盖数据内容级非法（索引值越界由 kernel 侧守卫负责）。
- 拒绝路径（超限输入返回 GRAPH_FAILED + 日志）**无专项用例实测**——现有 UT/ST 用例参数均在范围内，拒绝行为未触发过。
- 上限常量绑定本算子需求与 buffer 布局；UB 容量 248KB 绑定 arch35，其他 SoC 需重推（未验证，可迁移的是"推导方法"而非数值）。
- 仅在 Ascend950PR (arch35) 实测。

**未决项**：

- **优先项**：补超限输入（如 hidden=8193）的 UT/ST 负向用例，实测 OP_CHECK_IF 拒绝路径与错误日志。
- UB 预算的精确字节账（11 个 TBuf 逐项）可做成 tiling 侧断言或构建期检查，避免布局变更后上限失配——未尝试。
## 坏实践

- 只校验"结构合法"（dim 数、整除、枚举值）就放行维度参数——超限输入会在 kernel 侧以 UB 溢出/回绕的形式炸出，离错误源头远且难定位。
- 把上限硬编码成"拍脑袋"值而不绑定容量推导——换平台（UB 容量不同）或换 buffer 布局后守卫失效。

## 误诊

- 无误诊轮次。注意：UB 溢出/回绕后果是量化推断（检视推导），未在运行期复现（无超限输入用例）。
