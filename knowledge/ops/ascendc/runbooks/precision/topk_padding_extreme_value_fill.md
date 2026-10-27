---
type: "Runbook"
title: "TopK padding 区须 SetValue 填极值"
description: "inner 非 32 倍数时 [inner, innerPad) padding 区垃圾值被 TopK 选入 top-k，idx 超过 inner 且 y 错；DataCopyPad rightPadding 与非对齐 Duplicate 均不生效，须在 Cast/搬运完成后用 SetValue 逐元素填 SortT 极值再加屏障。"
tags: [task.debugging, topic.compute.ordering]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# TopK padding 区须 SetValue 填极值

## 现象

TopK 类自定义算子（Ascend950PR/arch35，CANN 9.1.0，TopK 高层 API 要求内轴长度为 32 的整数倍，故 tiling 把 inner 向上取整到 innerPad）在 ST Real 模式下，5D fp16、末维 inner=1013（非 32 倍数，innerPad=1024，rightPad=11）的用例失败，y 与 idx 同时出错，测试输出原文：

```
[Real] L0-014 5D fp16 末维 topk(k=10) 最大值
  dtype=float16, shape=[2, 4, 8, 16, 1013], k=10, dim=-1, largest=true
  [FAIL] idx 越界: outer=141, k_idx=9, after=0, idx=1013, inner=1013
  [FAIL] idx 越界: outer=318, k_idx=6, after=0, idx=1017, inner=1013
```

idx 出现 1013~1021 的值，恰好落在 padding 区 [1013, 1024)——即 TopK 把 padding 里的垃圾值当成了 top-k 元素选出来。同 shape 的 3D/4D（inner 为 32 倍数）用例全部通过，2D inner=1024 也通过；只有 inner 非 32 倍数的 5D 用例失败。

## 适用场景

- 工程形态：算子仓开发，kernel 用 TopK 高层 API（要求 inner 32 对齐）处理任意 inner 的输入。
- API 路径：`AscendC::TopK`（约束"内轴长度为 32 的整数倍"）、`DataCopyPad`（isPad/rightPadding 填充）、`Duplicate`（32 字节对齐限制）、`LocalTensor::SetValue`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510，arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"TopK/排序类 API + inner 对齐 padding"的算子。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：inner 恒为 32 倍数的调用方；其他 arch 未验证（推断可迁移）。

## 根因

TopK 输入缓冲的 padding 区 [inner, innerPad) 未被填充成"不被选入 top-k"的极值，垃圾值参与排序后被选为 top-k，产生 idx ≥ inner 的越界索引与错误 y。

机制链：TopK 要求 inner 为 32 倍数 → innerPad = CeilAlign(inner, 32) > inner 时，[inner, innerPad) 共 rightPad<32 个元素必须填入按 largest 方向不可选的极值（largest=true 填 SortT 最小值，largest=false 填最大值）；三种"自动填充"手段在本场景都不可靠：① `DataCopyPad` 的 `{isPad=true, leftPadding=0, rightPadding, paddingValue}` 在 blockLen 非 32 字节对齐时未按预期写入 padding（实测仍失败）；② 在搬运前预填原 dtype buffer 无效——Cast 会覆盖整个 [0, innerPad)；③ `Duplicate<SortT>(srcValueSort[inner_], padVal, rightPad)` 从非 32 字节对齐偏移（1013×2B=2026B）起写不生效（实测仍失败）。

## 解决配方

CopyIn 的搬运/Cast 全部完成后，对 padding 区用 `SetValue` 逐元素写入 SortT 极值，再 `PipeBarrier<PIPE_V>()`（代码 verbatim 取自修复提交）：

```cpp
    // 填充 padding 区域 [inner, innerPad) 为 SortT 极值
    // DataCopyPad 当 blockLen 非 32 对齐时 dummy 填"第一个元素值"，Cast 会覆盖 padding，
    // 因此必须在此显式填充极值，确保 padding 不被 TopK 选入。
    // 使用 SetValue 逐元素写入（padding 元素数 < 32，性能可接受），避免 Duplicate 的 32 字节对齐限制。
    if (innerPad_ > inner_) {
        const SortT sortPadVal = GetPadValue<SortT>(isLargest_);
        for (int64_t i = inner_; i < innerPad_; i++) {
            srcValueSort.SetValue(i, sortPadVal);
        }
        PipeBarrier<PIPE_V>();
    }
```

其中按排序 dtype 取极值的辅助（half/float 走位模式，int8/uint8 走边界值）：

```cpp
// 通用极值获取（按排序 dtype）
template <typename SortT>
inline __aicore__ SortT GetPadValue(bool isLargest)
{
    if constexpr (std::is_same_v<SortT, half>) {
        return GetPadValueHalf(isLargest);
    } else {
        return GetPadValueFloat(isLargest);
    }
}
```

修复前的失败尝试（保留作对照，均未生效）：

```cpp
    // 尝试 1（无效）：DataCopyPad isPad=true + rightPadding 主动填充原 dtype 极值
    const uint32_t rightPad = static_cast<uint32_t>(innerPad_ - inner_);
    const T origPadVal = GetPadValueOrig<T>(isLargest_);
    DataCopyPad<T>(srcValueOrig, gmX[rowIdx * srcRowStride_], copyParams,
                   {true, 0, rightPad, origPadVal});   // 仍 FAIL
    // 尝试 2（无效）：非 32 对齐偏移上 Duplicate 填充
    Duplicate<SortT>(srcValueSort[inner_], padVal, static_cast<int32_t>(innerPad_ - inner_));  // 仍 FAIL
```

## 设计决策

- 选 SetValue 标量循环而非 Duplicate：padding 元素数恒 < 32（innerPad-inner < 32），标量写开销可忽略，且不受 32 字节对齐限制；Duplicate 起始地址 inner*sizeof(SortT) 非 32 对齐时不生效。
- 填充时机放在 Cast/搬运完成之后：Cast 会重写整个 [0, innerPad)，任何在此之前对 padding 的填充都会被覆盖。
- 极值方向与 largest 反向：largest=true 取最小（不被选入最大 k），largest=false 取最大。
- 当前方案属 recommended_fastpath。

## 判别方法

- 信号特征：仅 inner 非 32 倍数的用例失败；idx 值落在 [inner, innerPad) 区间；y 与 idx 同时错。
- 探针：dump TopK 输出的 idx，检查是否 ≥ inner。

## 预防

- 使用要求对齐的排序/规约 API 时，把"padding 显式填充不可选极值 + 同步屏障"写进 kernel 骨架模板，不要依赖 DataCopyPad 的 pad 参数。
- CR 清单：inner 对齐 → padding 区显式填极值 → 屏障 → TopK。

## 验证信号

问题与验证来源：迭代二联调验证的多轮 ST Real 实测（同一用例跨 4 轮对照）。

- 修复前：L0-014 FAIL，`idx 越界: outer=141, k_idx=9, idx=1013, inner=1013`（run1/run2/run3 持续失败；run3 为 DataCopyPad rightPadding 方案，仍 FAIL）（来源平台：950；证据等级：tool_observed）。
- 尝试期间回归观测：run2 曾因调整填充顺序使一条 bf16 largest=false、inner=1024 无 padding 的用例由 PASS 变 FAIL，说明填充逻辑改动会影响无 padding 用例，须整轮回归（来源平台：950；证据等级：tool_observed）。
- 修复后：run4 起 L0-014 PASS（`y 比对: MERE=0.00e+00, MARE=0.00e+00 (10240 elems)` + `idx gather 校验: x.gather(dim, idx) == y`），终态轮（排除范围外跳过后）仍 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `DataCopyPad isPad=true+rightPadding` 与非对齐 `Duplicate` 未生效的确切机制未在指令层验证（文档称 rightPadding 单位为元素个数且不超 32 字节；实测未按预期填充），本卡结论以"显式 SetValue 填充"这一可靠路径为准。
- padding 元素数 ≥ 32 不会出现（innerPad-inner < 32 恒成立）；若切分层改为大块 padding，SetValue 循环需重新评估。
- bf16/int8/uint8 等 Cast 路径在填充正确后仍可能有独立的 Cast 精度问题，不要把剩余失败归到 padding。
- 其他 arch 未验证（推断可迁移）。

**未决项**：

- DataCopyPad rightPadding 不生效的机制层原因未追（对下一迭代非必须，SetValue 路径已稳定）。
## 坏实践

- 依赖 `DataCopyPad` 的 `{true, 0, rightPad, padVal}` 完成排序类 padding（本场景实测不生效）。
- 在 Cast 之前预填 padding（会被 Cast 覆盖整个 [0, innerPad)）。
- 在非 32 字节对齐偏移上用 `Duplicate` 填充。

## 误诊

- 无（本闭环根因与修复均被四轮对照锁定）。
