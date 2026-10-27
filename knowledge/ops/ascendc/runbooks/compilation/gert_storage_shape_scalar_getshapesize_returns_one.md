---
type: "Runbook"
title: "StorageShape scalar GetShapeSize 返回 1"
description: "空 axes 表示为 rank-0 scalar tensor 时 GetShapeSize() 返回 1（非 0），TilingFunc 误判 axesNum=1 并读 nullptr 数据；须用 GetDimNum()==0 检测 empty axes。"
tags: ["task.debugging", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["scalar_tensor", "empty_axes"]
---

# StorageShape scalar GetShapeSize 返回 1

## 现象

在编写 EuclideanNorm 算子公共 `TilingFunc`（`op_host/euclidean_norm_host.cpp`）的 `ParseAxes` 阶段时，TilingUT 的 `pad_r_pure_a_big` case（shape `{6,7}`, `axes={}` 即空 axes → 全 reduce）失败，24/25 PASS。其余 24 case 全过，仅此 1 case mismatch 且所有 TilingData 字段返回 0（像 TilingFunc 早退 / 未填）。

排查发现：测试用 `gert::StorageShape axesShape;`（默认构造，`dim_num_=0`，即 scalar）表示空 axes。`ParseAxes` 用 `axesStorageShape->GetStorageShape().GetShapeSize()` 取 axes 元素数，对 scalar tensor 得到 **1**（不是 0），于是 `axesNum=1`，进而尝试 `axesTensor->GetAddr()` 读 axes 数据，但空 axes 的 data 为 nullptr → `isReduceAxis` 全 false → 当作 pure-A（无 reduce 轴）处理 → TilingData 字段全 0 / 走错分支。

## 适用场景

- **工程形态**：op_host TilingFunc（算子仓 op_host 层）。
- **API 路径**：`gert::StorageShape::GetShapeSize()` / `GetDimNum()`；`gert::Tensor::GetAddr()`；`gert::TilingContext::GetInputShape()`。
- **验证平台与版本**：Ascend950，CANN 9.1.0（x86_64-linux）；host-only TilingUT。
- **适用粒度**：`platform_wide`（CANN gert::StorageShape API 语义，跨算子通用）。
- **路线优先级**：`recommended_fastpath`（用 GetDimNum 检测 scalar 是 API 正确用法）。
- **不适用或未验证场景**：a3/a2 未验证，推断可迁移（gert API 语义跨平台一致）。

## 判别方法

**触发条件**：算子有"axes"类 Const 输入（如 reduce 类算子的 `axes`），且测试 / 真机可能用 scalar tensor（rank 0）或 0 元素 1-D tensor 表示"空 axes = 全 reduce"。`ParseAxes` 用 `GetShapeSize()` 判空会误判。

**可执行探针**（在 TilingFunc 里加一次性 fprintf / 断言，host UT 可见）：

```cpp
const gert::StorageShape* axesStorageShape = context->GetInputShape(AXES_INPUT_INDEX);
const int64_t axesNumByShapeSize = axesStorageShape->GetStorageShape().GetShapeSize();
const int64_t axesDimNum = axesStorageShape->GetStorageShape().GetDimNum();
fprintf(stderr, "[ParseAxes] axesShapeSize=%ld dimNum=%ld addr=%p\n",
        (long)axesNumByShapeSize, (long)axesDimNum, axesTensor->GetAddr());
```

**原始观测**（trace seq 3998 reasoning，据 CANN `shape.h` 源码）：

> `GetShapeSize()` for a scalar (`dim_num_=0`) returns 1 (the initial value, the loop doesn't execute). So for the test's empty axes (default `gert::StorageShape axesShape;` which is scalar), `axesStorageShape->GetStorageShape().GetShapeSize()` returns 1.

即：`GetShapeSize()` 实现 = `size_t ret = 1; for (i in dims) ret *= dim[i]; return ret;`。对 rank-0 tensor，循环不执行，`ret` 保持初值 1（数学上 ∏ 空集 = 1，但工程上"空 axes"语义不是"1 个 axis"）。

**解释**：`GetShapeSize()` 表达的是"张量元素总数"（按乘积），对 scalar 返回 1 是数学正确的（标量就是 1 个值）。但"axes 是不是空"的判据应是"轴的个数"即 `GetDimNum()`：1-D shape `[N]` → `GetDimNum()=1, GetShapeSize()=N`；scalar → `GetDimNum()=0, GetShapeSize()=1`。用 `GetShapeSize()==0` 检测空 axes 永远不成立（最小也是 1），用 `GetDimNum()==0` 才能正确识别 scalar/空 axes。

**修复判据**：`ParseAxes` 改为先查 `GetDimNum()`：`if (axesDimNum == 0) → 空 axes → 全 reduce（所有轴都标 isReduceAxis=true）`；否则按 `GetShapeSize()`（此时 == N）读 N 个 axis 值。

## 验证信号

来源：EuclideanNorm 公共 TilingUT（host-only gtest），在 `ParseAxes` 改用 `GetDimNum()` 检测 scalar 后重建并运行。

- 修复前 `pad_r_pure_a_big`（{6,7}, axes={}）MISMATCH 全字段 act=0，24/25 PASS（来源平台：950；证据等级: tool_observed）
- 修复后 25/25 PASS，`pad_r_pure_a_big` 走全-reduce 路径正确（来源平台：950；证据等级: tool_observed）

## 预防

- 任何用 `GetShapeSize()` 判"输入是否为空"的逻辑都要改用 `GetDimNum()`（rank）或显式查首维 `GetDim(0)`；`GetShapeSize()` 对 scalar 永远返回 1，不能当"空"判据。
- reduce 类算子若支持"空 axes = 全 reduce"语义，`ParseAxes` 必须显式处理 scalar axes tensor。

## 失效边界

- 平台：Ascend950 / CANN 9.1.0 host-only UT 实测；a3/a2 未验证，推断可迁移（gert::StorageShape API 语义跨平台一致）。
- 边界：本卡仅覆盖"scalar tensor 表示空 axes"这一语义陷阱；若测试用真正 0 元素 1-D tensor（shape `[0]`，`GetDimNum()=1, GetShapeSize()=0`）表示空 axes，则 `GetShapeSize()==0` 判据成立，本卡不适用——但 CANN 测试框架默认构造 `gert::StorageShape` 是 scalar 而非 `[0]`，故 scalar 路径更常见。

## 误诊

无。
