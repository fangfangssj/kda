---
type: "Runbook"
title: StorageShape 需解包为 Shape 再用维度 API
description: "gert 把形状分两层：GetInputShape/GetDynamicInputShape 返回的是 StorageShape 包装，GetDimNum/GetDim/GetShapeSize/SetDim/AppendDim 全是内层 Shape 的成员；直接在 StorageShape 上调、或把 Shape 引用传给 StorageShape 形参，编译报 no member/引用初始化失败，先 GetStorageShape() 解包再用。增量（CANN 9.2.0）：TilingContext 与 InferShapeContext 的同名 GetInputShape 返回类型不同（StorageShape* vs Shape*），从 InferShape 复制取维写法进 TilingFunc 是高发入口；InferShape 侧建议显式 const gert::Shape* 声明暴露差异。"
tags: [task.debugging, topic.compilation, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-17T03:16:34Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# StorageShape 需解包为 Shape 再用维度 API

## 现象

两个独立算子的 host 侧 tiling 首次构建撞到同一类 gert shape 层级误用：

**形态一（工具函数形参误用 StorageShape）**：argmax 类算子（Ascend950PR/arch35）的 tiling 实现里，为计算归约抽象 (A1, R, A0)（归约轴前/中/后维度之积），把工具函数形参声明成 `const gert::StorageShape&` 并直接在其上调用 `GetDimNum()/GetDim()`。首次 `bash build.sh --soc=ascend950` 时宿主侧编译报：

```
error: 'const struct gert::StorageShape' has no member named 'GetDimNum'
error: 'const struct gert::StorageShape' has no member named 'GetDim'
error: invalid initialization of reference of type 'const gert::StorageShape&' from expression of type 'gert::Shape'
```

第三条来自调用侧：`context->GetInputShape(0)->GetStorageShape()` 返回 `const Shape&`，把它传给 `StorageShape&` 形参构成类型不匹配。

**形态二（动态输入上直接调 GetShapeSize）**：foreach 类 p-范数算子（TensorList 输入）的 tiling 里，为统计 TensorList 中每个张量的元素数，对 `context->GetDynamicInputShape(0, k)` 返回的 `const gert::StorageShape*` 直接调用了 `GetShapeSize()`。首次 `bash build.sh --soc=ascend950 -j8` 报错原文：

```text
op_host/arch35/<op_name>_tiling.cpp:178:36:
error: 'const struct gert::StorageShape' has no member named 'GetShapeSize'; did you mean 'GetShape'?
  178 |         int64_t elemCount = shape->GetShapeSize();
      |                                    ^~~~~~~~~~~~
      |                                    GetShape
```

## 适用场景

- 工程形态：registry-invoke 自定义算子 op_host tiling / infershape 代码（含动态输入 TensorList 场景）。
- API 路径：`gert::TilingContext::GetInputShape(index)` / `GetDynamicInputShape(irIndex, relativeIndex)`（均返回 `const gert::StorageShape*`）、`gert::StorageShape::GetStorageShape()`（返回 `const gert::Shape&`）、`gert::Shape::GetDimNum()/GetDim()/GetShapeSize()/SetDim()/AppendDim()`。
- 验证过的平台与 CANN 版本：Ascend950PR（--soc=ascend950 编译通过 / ascend950 交叉编译），CANN 9.1.0。
- 适用粒度：platform_wide（gert 类型层级是 CANN 全平台一致的运行时接口事实）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：InferShape 上下文里 `GetInputShape` 直接返回 `const Shape*`（无 StorageShape 包装），不需要解包；图模式/其他上下文 API 未逐一验证。

## 根因

gert 运行时把"形状"分成两层：`TilingContext::GetInputShape(0)` 返回的是 `const StorageShape*`（origin/storage 双 shape 包装），而 `GetDimNum()/GetDim()/GetShapeSize()/SetDim()/AppendDim()` 全部是内层 `Shape` 类的成员；`StorageShape::GetStorageShape()` 才返回 `const Shape&`。把 StorageShape 当 Shape 用（或把 Shape 引用传给 StorageShape 形参）都会在编译期报"无成员/引用初始化失败"。

## 解决配方

**形态一**：工具函数形参从 `StorageShape` 改为 `Shape`，调用侧保持解包后传入。

修复前（before）：

```cpp
// Compute A1, R, A0 from input shape and normalized dim
static void ComputeA1RA0(const gert::StorageShape& inputShape, int64_t dim,
                         uint64_t* a1, uint64_t* r, uint64_t* a0)
{
    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    *a1 = 1;
    *r = static_cast<uint64_t>(inputShape.GetDim(static_cast<size_t>(dim)));
    ...
}
```

调用侧：

```cpp
    auto inputShapePtr = context->GetInputShape(0);
    auto inputShape = inputShapePtr->GetStorageShape();   // 返回 const Shape&
    ...
    ComputeA1RA0(inputShape, dim, &a1, &r, &a0);           // Shape → StorageShape& 类型不匹配
```

修复后（after，完整函数）：

```cpp
// Compute A1, R, A0 from input shape and normalized dim
static void ComputeA1RA0(const gert::Shape& inputShape, int64_t dim,
                         uint64_t* a1, uint64_t* r, uint64_t* a0)
{
    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    *a1 = 1;
    *r = static_cast<uint64_t>(inputShape.GetDim(static_cast<size_t>(dim)));
    *a0 = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (i < dim) {
            *a1 *= static_cast<uint64_t>(inputShape.GetDim(static_cast<size_t>(i)));
        } else if (i > dim) {
            *a0 *= static_cast<uint64_t>(inputShape.GetDim(static_cast<size_t>(i)));
        }
    }
}
```

调用侧保持 `inputShapePtr->GetStorageShape()` 解包后传入（`inputShape` 即 `const Shape&`），tiling 主函数内直接 `inputShape.GetDimNum()` 也随之合法。

**形态二**：`shape->GetShapeSize()` → `shape->GetStorageShape().GetShapeSize()`。

修复前（tiling 主函数中遍历 TensorList 统计元素数的完整代码段）：

```cpp
    for (size_t k = 0; k < tensorNum; k++) {
        auto shape = context->GetDynamicInputShape(0, k);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t elemCount = shape->GetShapeSize();
        tiling->elemCounts[k] = elemCount;
        totalElem += elemCount;
    }
```

修复后（同一代码段）：

```cpp
    for (size_t k = 0; k < tensorNum; k++) {
        auto shape = context->GetDynamicInputShape(0, k);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t elemCount = shape->GetStorageShape().GetShapeSize();
        tiling->elemCounts[k] = elemCount;
        totalElem += elemCount;
    }
```

## 设计决策

- 为什么改形参而不是把 StorageShape 的访问包装进工具函数：tiling 里同时需要 rank（GetDimNum）与逐维 GetDim，统一收 `const Shape&` 让函数只依赖最小接口；这也是参考代码的模式（`EnsureNotScalar(inputX->GetStorageShape())` 收 `const gert::Shape&`）。
- 备选：用 `GetOriginShape()` 取原始 shape——对无转置/无 format 变换的 ND 输入两者等价，选 `GetStorageShape()` 与参考模板一致。
- 形态二不要按编译器提示 `did you mean 'GetShape'` 改成 `shape->GetShape()` 后继续链 `->GetShapeSize()`——`GetShape()` 返回的也不是带该方法的类型，会再次报错，应一次改对为 `GetStorageShape().GetShapeSize()`。
- 本方案属 recommended_fastpath。

## 预防

- 在 tiling 代码中需要维度信息时，先解包：`auto shape = context->GetInputShape(0)->GetStorageShape();` 再对 `shape` 调 GetDimNum/GetDim/GetShapeSize。
- 写维度工具函数时形参类型用 `const gert::Shape&`，不要用 StorageShape。
- 遇到 `'StorageShape' has no member` 类错误，优先怀疑层级用错（Shape 方法落在了包装类上），而不是怀疑 API 缺失；不确定时直接 grep 头文件：

```bash
grep -n "GetShapeSize\|GetStorageShape" $ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/storage_shape.h
```

## 验证信号

来源说明：两个独立算子（argmax 类 / foreach 范数类）的 host 侧 tiling 首次构建失败与修复后重构建实测。

- 形态一修复前：`error: 'const struct gert::StorageShape' has no member named 'GetDimNum'`（4 处）、`error: invalid initialization of reference of type 'const gert::StorageShape&' from expression of type 'gert::Shape'`（来源平台：950；证据等级：tool_observed）
- 形态一修复后：宿主侧 cust_optiling/cust_opapi/cust_op_proto 全部编译通过，tiling 目标无 Shape 相关报错；同一 tiling 随后在 op_host UT 中被加载执行（宿主侧 gtest 环境编译通过），InferShape 15/15 通过（来源平台：950 宿主侧；证据等级：tool_observed）
- 形态二修复前：首轮构建在 `op_host/arch35/<op_name>_tiling.cpp:178` 报 `'const struct gert::StorageShape' has no member named 'GetShapeSize'; did you mean 'GetShape'?`（来源平台：950 交叉编译；证据等级：tool_observed）
- 形态二修复后：改为 `GetStorageShape().GetShapeSize()` 后重构建，该错误串不再出现，host 侧 tiling 编译单元通过，构建推进到 op_api 阶段（来源平台：950 交叉编译；证据等级：tool_observed）

## 失效边界

- 仅覆盖 gert Shape/StorageShape 层级误用这一根因。
- `GetStorageShape()` 取的是存储布局 shape；若算子对 view shape 与 storage shape 不一致（非连续/带偏移）的输入有要求，两者可能不同——形态二算子声明了 `AutoContiguous()` 且未涉及该场景，未验证非连续输入下的差异。
- CANN 9.1.0 头文件为准；`Shape` 若未来直接提供同义成员不影响本卡结论（未验证，推断可迁移到其他平台）。

**未决项**：

- 无（该根因已在两个独立 trace 闭环到构建通过）。
- 附带记录（独立小坑，不构成独立根因卡）：`Ops::Base::CeilAlign(T x, T align)` 模板要求两实参同型，`CeilAlign(tmpBufSize, 32)`（uint32_t vs int）报 `no matching function ... deduced conflicting types for parameter 'T'`，改成 `CeilAlign(tmpBufSize, 32U)` 即过；报错自解释，属通用 C++ 模板推导问题。
## 坏实践

- 把 `context->GetInputShape(0)` / `GetDynamicInputShape(...)` 的返回值当 `Shape*` 直接 `->GetDimNum()` / `->GetShapeSize()`。
- 用 `reinterpret_cast` 或拷贝成员强转两层 shape——正确做法就是一次 `GetStorageShape()`。
- 按编译器 "did you mean" 提示改成 `GetShape()` 再链调维度 API——返回类型同样没有该成员，应一次改对。

## 误诊

- 无（两个 trace 的编译器诊断均直接指向层级误用，一步定位）。

## 增量（CANN 9.2.0 三条新 trace：Tiling/InferShape 同名 API 返回类型不同）

另一独立轨迹并入三个独立 trace（GRU 类 / TopK 类 / DepthwiseConv2d 类算子的 host 侧 TilingFunc 首轮构建，均报 `'const struct gert::StorageShape' has no member named 'GetDim'`（部分含 `GetDimNum`），CANN 9.1.0/9.2.0），补齐本卡的高发入口与判别锚点：

**1. 高发入口：从 InferShape 代码复制取维写法进 TilingFunc**。`gert::TilingContext::GetInputShape(index)` 返回 `const StorageShape*`，而 `gert::InferShapeContext::GetInputShape(index)` 返回 `const Shape*`——两处 API 同名、长得一样，`xShape->GetDim(0)` 只在 InferShape 侧合法，错误只出现在 Tiling 侧。头文件实测锚点：

```text
// storage_shape.h
struct StorageShape {
  const Shape &GetOriginShape() const { return origin_shape_; }
  const Shape &GetShape() const { return origin_shape_; }
  const Shape &GetStorageShape() const { return storage_shape_; }
  ...
};
// shape.h（Shape 才有 GetDim）
158:  int64_t GetDim(const size_t idx) const {
// infer_shape_context.h
37:  const Shape *GetInputShape(const size_t index) const {
115:  Shape *GetOutputShape(const size_t index) {
```

**2. 修复形态（GRU trace，before/after 函数级）**：

修复前（Tiling 侧照 InferShape 写法直取 GetDim）：

```cpp
// op_host/arch35/<op>_tiling.cpp（GetShapeAttrsInfo）
auto xShape = context->GetInputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

if (*batchFirst) {
    *batchSize = xShape->GetDim(0);
    *seqLen = xShape->GetDim(1);
} else {
    *seqLen = xShape->GetDim(0);
    *batchSize = xShape->GetDim(1);
}
```

修复后（Tiling 侧经 GetStorageShape() 取维）：

```cpp
// op_host/arch35/<op>_tiling.cpp（GetShapeAttrsInfo）
auto xShape = context->GetInputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

if (*batchFirst) {
    *batchSize = xShape->GetStorageShape().GetDim(0);
    *seqLen = xShape->GetStorageShape().GetDim(1);
} else {
    *seqLen = xShape->GetStorageShape().GetDim(0);
    *batchSize = xShape->GetStorageShape().GetDim(1);
}
```

配套（InferShape 侧改为显式类型，避免 auto 掩盖两侧差异）：

```cpp
// op_host/<op>_infershape.cpp（显式类型，避免 auto 掩盖两侧差异）
const gert::Shape* xShape = context->GetInputShape(0);  // x
OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
...
int64_t S, B;
if (batchFirst) {
    B = xShape->GetDim(0);
    S = xShape->GetDim(1);
} else {
    S = xShape->GetDim(0);
    B = xShape->GetDim(1);
}
```

DepthwiseConv2d trace 的等价形态：函数入口一次性取 `const gert::Shape& xStorageShape = xShape->GetStorageShape();`，后续全部经该引用调 GetDimNum/GetDim（含多输入 x/weight/bias 三处解包与 dimNum 校验）；TopK trace 覆盖 `GetDimNum` 同型错误（5 处）并按主卡同法修复。取 `GetStorageShape()`（运行时布局）而非 `GetOriginShape()`：tiling 切分关心实际搬运/计算存储 shape；ND 连续布局下两者等价，与参考实现 `EnsureNotScalar(inputX->GetStorageShape())` 一致。

**3. 判别补充**：TilingFunc 取 shape 维度的固定三步 `GetInputShape(i)` → `GetStorageShape()` → `GetDim(j)`；用显式类型声明代替 `auto` 接 GetInputShape 返回值，让两个 context 的类型差异在代码里显式可见，防互相复制。

### 复现增量验证信号

来源说明：三个独立算子（GRU 类 / TopK 类 / DepthwiseConv2d 类）的 host 侧 TilingFunc 首轮编译联调，修复前稳定复现，修复后该类错误消失。

- GRU trace：修复前一轮构建 4 处 `'const struct gert::StorageShape' has no member named 'GetDim'`（同一算子 InferShape 侧 `xShape->GetDim(0)` 合法，错误只出现在 Tiling 侧）；Tiling 侧改 `GetStorageShape().GetDim(idx)`、InferShape 侧显式 `const gert::Shape*` 后下一轮构建该类错误全部消失（来源平台：950；证据等级：tool_observed）。
- TopK trace：修复前 5 处 `no member named 'GetDimNum'/'GetDim'`（`op_host/arch35/<op>_tiling.cpp:75/103/113/114/124`）；入口取 `const gert::Shape& shape = inputShape->GetStorageShape();` 后全部消失，构建推进到下一独立错误，最终构建成功（来源平台：950 编译链；证据等级：tool_observed）。
- DepthwiseConv2d trace（CANN 9.2.0）：修复前 cmake configure 阶段（npu_op_code_gen → compile_ascendc_all_ops_so）报 `error: 'const struct gert::StorageShape' has no member named 'GetDim'`（tiling.cpp:100，`compile ascend all ops failed`）；修复后同一 configure 不再报此错，全链路修复后全量编译产出 .o 与 custom_opp 安装包（来源平台：950；证据等级：tool_observed）。
- 头文件事实锚点：`infer_shape_context.h:37` InferShapeContext::GetInputShape 返回 `const Shape*`；`shape.h:158` GetDim 在 Shape 上（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 结论绑定 CANN 9.1.0/9.2.0 的 exe_graph 头；其他版本未验证，推断可迁移。
- 本卡只验证了 ND 布局下按运行时 shape 取维；`GetOriginShape()` 与 `GetStorageShape()` 不一致的场景（带转置/排布变换的图）未验证。
- `StorageShape::GetShape()` 与 `GetOriginShape()` 都返回 origin_shape_ 的设计意图（为何 GetShape 不是 storage）未深究。

### 复现增量坏实践

- 把 InferShape 的 `xShape->GetDim(i)` 写法原样复制进 TilingFunc（同名 API 返回类型不同）。
- `auto` 接 GetInputShape 返回值，掩盖 Tiling/InferShape 两侧类型差异。