---
type: "Runbook"
title: "移植tiling误拒rank=0标量输入"
description: "注册工程转 kernel 直调时 host tiling 的 shape 读取把 rank<1 当非法，拒绝 rank=0（标量输入 shape ()）致 4 条用例 frnsm_run failed: 1；原注册 tiling 只拒 r<0，标量走广播路径合法，须对齐边界谓词。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.compute.indexing"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# 移植tiling误拒rank=0标量输入

## 现象

在 Ascend950PR（CANN 9.2.0）上验收一个由注册工程忠实转换而来的 FusedRmsNormSigmoidMul `<<<>>>` kernel 直调工程（host tiling 被移植为 .so 内的 `frnsm_compute_tiling` C ABI）。用统一功能验收 runner（TTK golden + stat_rel_err 语义）跑 18 条 L0 用例：14 条 PASS，4 条失败——`aclnnFusedRMSNormSigmoidMul_L0_007`、`L0_011`、`L0_bound_024`、`L0_028`，报错原文：

```
[aclnnFusedRMSNormSigmoidMul_L0_007] ERROR RuntimeError: frnsm_run failed: 1
```

逐一核对 18 条用例的输入 ranks（x/gamma/other）发现失败的 4 条全部是 `other` 为标量（shape `()`，`ranks=[7, 7, 0]`、`[2, 1, 0]`、`[5, 1, 0]`、`[3, 3, 0]`），即 `otherRank=0`；其余 14 条（other rank ≥1）全部正常。错误发生在 tiling/launch 阶段（返回码 1），不是数值比对失败。

## 适用场景

- 工程形态：注册工程 → `<<<>>>` kernel 直调的忠实转换（host tiling 手工移植为独立函数）；也适用于任何手工移植 gert::StorageShape 读取逻辑的场景。
- API 路径：host 侧 tiling 的 shape 槽读取（原 `gert::TilingContext` → `GetInputShape(n)` → `GetStorageShape()` → `GetDimNum()`；移植版为裸 `(rank, shape[])` 参数 + 左补 1 的 8 槽数组）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"按 8 槽 shape + stride 广播语义"处理可选标量输入的算子转换（RMSNorm/LayerNorm 类带标量 gate/bias 输入尤其常见）。
- 路线优先级：recommended_fastpath（对齐原实现边界谓词的最小修复）。
- 明确不适用 / 未验证场景：原注册实现本身拒绝 rank=0 的算子（此时 `rank < 1` 才是忠实行为——修复前必须先核对原始谓词）；rank>8 在两版中一致拒绝，不在本卡范围。

## 根因

移植版 `ReadShapeSlots` 把合法下界写错了一格：`if (rank < 1 || rank > 8) return false;` 拒绝了 rank=0；而原始注册工程 tiling 的 `rd()` 只拒绝负 rank（`if(r<0||r>8)return false;`），rank=0（标量输入）是合法输入——8 槽 shape 全部保持左补的 1，随后广播判定把 `otherShape[i]==1 && xShape[i]!=1` 的维 stride 置 0，标量 other 因此走广播路径。移植时把"维度数为 0 合法"这一边界条件收紧成了"至少 1 维"，属于转换保真度缺陷：标量用例在 tiling 阶段被拒，launch 返回 1。

原始注册工程 tiling（节选，`rd()` 与其调用、广播 stride 处理，verbatim）：

```cpp
auto rd=[&](int n,int64_t* o,int32_t& r){auto*s=c->GetInputShape(n);if(!s)return false;auto q=s->GetStorageShape();r=q.GetDimNum();if(r<0||r>8)return false;std::fill(o,o+8,1);for(int i=0;i<r;++i){o[8-r+i]=q.GetDim(i);if(o[8-r+i]<=0)return false;}return true;};
if(!rd(0,t->xShape,t->xRank)||!rd(1,t->gammaShape,t->gammaRank)||!rd(2,t->otherShape,t->otherRank)||t->gammaRank<1||t->gammaRank>t->xRank)return ge::GRAPH_FAILED;
...
for(int i=0;i<8;++i){if(t->otherShape[i]!=1&&t->otherShape[i]!=t->xShape[i])return ge::GRAPH_FAILED;if(t->otherShape[i]==1&&t->xShape[i]!=1)t->otherStride[i]=0;}
```

## 解决配方

最小修复：把移植版 `ReadShapeSlots` 的下界谓词从 `rank < 1` 改回 `rank < 0`，与原实现 `r<0||r>8` 对齐。

修复前（移植版，verbatim）：

```cpp
// 原实现：rd() —— 读输入 shape（gert::StorageShape → 左补 1 的 8 槽数组）
bool ReadShapeSlots(int32_t rank, const int64_t* shape, int64_t* out8, int32_t& rankOut)
{
    rankOut = rank;
    if (rank < 1 || rank > 8) return false;
    std::fill(out8, out8 + 8, 1);
    for (int i = 0; i < rank; ++i) {
        out8[8 - rank + i] = shape[i];
        if (out8[8 - rank + i] <= 0) return false;
    }
    return true;
}
```

修复后（verbatim，唯一改动行已注释）：

```cpp
// 原实现：rd() —— 读输入 shape（gert::StorageShape → 左补 1 的 8 槽数组）
bool ReadShapeSlots(int32_t rank, const int64_t* shape, int64_t* out8, int32_t& rankOut)
{
    rankOut = rank;
    if (rank < 0 || rank > 8) return false;  // 原实现 r<0||r>8：允许 rank=0（标量）
    std::fill(out8, out8 + 8, 1);
    for (int i = 0; i < rank; ++i) {
        out8[8 - rank + i] = shape[i];
        if (out8[8 - rank + i] <= 0) return false;
    }
    return true;
}
```

配套核查（修复后执行）：rank=0 时循环零次、`shape` 指针不被解引用，C ABI 传 0 长数组指针安全（ctypes 0 长度 `(c_int64 * 0)()` 数组地址合法、传参无断言）。

## 设计决策

- 选择"对齐原谓词"而不是"在 runner 侧把标量用例跳过/改写"：转换契约要求保真——原始注册 tiling 允许的行为，直调版必须同样允许；改测试侧属于掩盖缺陷。
- 修复后必须做双向回归：既验证 4 条失败用例转好，也逐一对比此前 14 条 PASS 用例的 `blockDim/tilingKey/PASS` 状态零变化（防止谓词放宽改变其它 case 的 tiling 决策）。
- 当前方案属 recommended_fastpath（单行 diff、语义完全对齐）。

## 系统信息

- 平台：Ascend950PR，CANN 9.2.0；直调 .so（分裂 TU 构建 + AIVEC magic 补丁链路）。
- 相关文件：直调工程 `csrc/op_kernel/fused_rms_norm_sigmoid_mul_kernel.cpp`（host TU，含移植版 tiling，`ReadShapeSlots` 在 L107 起）；原始注册工程 `op_host/arch35/fused_rms_norm_sigmoid_mul_tiling_arch35.cpp`（`rd()` 在 TilingFuncPublic 内，L918 起）；统一验收 runner `run_acceptance.py`（TTK golden + stat_rel_err，18 条 L0 用例集）。

## 验证信号

以下信号来自同一次转换工程的验收会话（修复前一轮 + 修复后重建重跑）：

- 修复前：`ACCEPTANCE FAIL (14/18)`，4 条标量 other 用例 `RuntimeError: frnsm_run failed: 1`（来源平台：950；证据等级：tool_observed）。
- 修复后（重建 .so 重跑全量）：`ACCEPTANCE PASS (18/18)`，4 条标量用例 mere=0.0 mare=0.0（如 L0_007 blockDim=56 tilingKey=1、L0_bound_024 blockDim=1 tilingKey=0）（来源平台：950；证据等级：tool_observed）。
- 回归对照（逐 case 比较修复前后 run-manifest）：`previous: 14/18 PASS → new: 18/18 PASS`，此前 14 条 PASS 用例 blockDim/tilingKey/PASS 全部不变，`regressions: 0`（来源平台：950；证据等级：tool_observed）。
- 性能/路径不回归：perf case `fused_128x6x128` 仍 blockDim=24 tilingKey=0、`max_abs_err=0.000000 bad=0/98304`；group 路径 smoke 仍 `max_abs_err=0.001953 bad=0/262144`；tiling 交叉验证（vs 框架捕获 bin）仍 MATCH（来源平台：950；证据等级：tool_observed）。
- 正式验收落盘 gate=strict_pass 18/18（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本修复只恢复与原实现的语义等价（rank 0..8 合法）；不改变 rank>8 拒绝与 `dim<=0` 拒绝行为。
- 适用于"原始实现允许 rank=0"的算子；若某算子原始 tiling 确实拒绝标量，照抄本修复反而引入保真度缺陷——修复前必须核对原始谓词。
- 标量 other 的数值正确性依赖后续广播路径（stride=0 的 8 槽语义）；本卡只闭合 tiling 接受域，不覆盖广播计算本身的正确性（该部分由 18/18 验收覆盖）。
- 未验证：rank=0 出现在 x/gamma（而非 other）上的行为（本算子 x/gamma 恒有实际维度）。

**未决项**：

- 无未追根因项（根因与修复已闭环）。后续做"搬运合并快路"类优化时，快路门控若按"other 无广播维"判定，需显式让 rank=0（标量广播）回落一般路径，避免同类接受域问题在优化分支复发——属于后续优化轮次的注意点。
## 预防

- 移植 tiling/shape 读取逻辑时，把原始实现的**边界谓词逐字对照**（`r<0` vs `rank<1` 这类差一格的下界是高频转换缺陷）；对照方式可用 tiling 交叉验证（直调 tiling 输出 vs 框架捕获 tiling bin 逐字段比对，见 [direct_tiling_crosscheck_captured_tiling_bin](../validations/direct_tiling_crosscheck_captured_tiling_bin.md)）。
- 验收用例集必须包含标量输入（rank=0）用例——本会话 18 条 L0 中恰好有 4 条标量 other 用例，才使该缺陷在验收期而非性能期暴露。

## 坏实践

- 见到 `frnsm_run failed: 1`（tiling/launch 返回 1）先怀疑 kernel 数值逻辑——本缺陷在 tiling 阶段就拒绝了输入，根本没到 kernel。
- 转换验收只跑"正常 shape"用例——标量/广播边界用例是转换保真度的高频暴露点。

## 误诊

- 无（首轮即通过 ranks 清单定位到标量 other 子集，未走弯路）。
