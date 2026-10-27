---
type: "Runbook"
title: "gert 属性读取返回 const 指针须解引用"
description: "gert::RuntimeAttrs::GetAttrPointer<T>(idx) 与 Attrs::GetFloat(idx) 返回 const T* 而非值，按值接收报 invalid conversion / cannot convert；须接收指针、判空后解引用。复现增量：RuntimeAttrs 也没有 GetPointer<T>()、TypedContinuousVector 也没有 GetPtr<T>()（did you mean GetAttrPointer）——名字陷阱形态与 GetListInt + GetData/GetSize 的 ListInt 访问式；同族伴随：CeilDiv/CeilAlign 须显式 include op_common/op_host/util/math_util.h。"
tags: ["task.implementation", "topic.compilation", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# gert 属性读取返回 const 指针须解引用

## 现象

两个独立算子工程的 op_host 代码在同一 API 契约上以两种报错形态撞到同一根因。

形态一（infershape，int64_t 属性）：MlaProlog 算子的 `op_host/mla_prolog_infershape.cpp` 读取属性 `n_heads` 时写成直接赋值；执行 `bash build.sh --soc=ascend950 -j8` 编译，报错（原文）：

```
op_host/mla_prolog_infershape.cpp:53:52:
error: invalid conversion from 'const long int*' to 'int64_t' {aka 'long int'} [-fpermissive]
     53 |     int64_t nHeads = attrs->GetAttrPointer<int64_t>(0);
        |                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~
```

形态二（tiling，float 属性）：融合算子 `op_host/arch35/<op>_tiling.cpp` 的 GetShapeAttrsInfo 辅助函数按"`GetFloat` 返回 float"的直觉写 `*epsilon = attrs->GetFloat(0);`，同轮编译报：

```
<op>_tiling.cpp:85:31: error: cannot convert 'const float*' to 'float' in assignment
     85 |     *epsilon = attrs->GetFloat(0);  // 第 0 个属性
```

同一次输出里还伴随 `error: 'CeilDiv' has not been declared in 'Ops::Base'` / `'CeilAlign' has not been declared`（伴随问题，见解决配方末尾）。

## 适用场景

- 工程形态：自定义算子 op_host 侧 infershape/tiling 中读取算子属性（`gert::RuntimeAttrs` / `gert::Attrs`）。
- API 路径：`context->GetAttrs()->GetAttrPointer<T>(idx)`（T ∈ int64_t、float 等）与 `Attrs::GetFloat(index)`（两者返回类型同为 `const T*`）；伴随的 `Ops::Base::CeilDiv/CeilAlign`（`op_common/op_host/util/math_util.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR 目标（`--soc=ascend950` 构建链），CANN 9.1.0 与 9.2.0 目录（GetAttrPointer 形态为 cann-9.2.0 目录、9.1.0 基线；GetFloat 形态为 cann-9.1.0）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本所有 op_host 属性读取。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：返回值语义不同名的旧属性 API；整数属性 `GetInt` 未实测；本卡只覆盖 GetAttrPointer/GetFloat 的指针返回契约。

## 根因

`gert::RuntimeAttrs::GetAttrPointer<T>(idx)` 与 `gert::Attrs::GetFloat(index)` 的返回类型是 `const T*`（属性在运行时以指针形式暴露，可能为空），不是值；按值接收即编译失败，且裸指针若不判空直接解引用有崩溃风险。

同文件的 `CeilDiv/CeilAlign` 报错是同轮的伴随问题：这两个工具函数声明在 `op_common/op_host/util/math_util.h` 的 `Ops::Base` 命名空间里，tiling.cpp 只 include 了 `platform_util.h`，`using Ops::Base::CeilDiv;` 找不到声明。

## 解决配方

接收指针 → `OP_CHECK_NULL_WITH_CONTEXT` 判空 → 解引用取值。infershape（int64_t 属性）与 tiling（int64_t + float 属性）同式。

修复前（infershape，verbatim）：

```cpp
    // 获取属性 n_heads
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t nHeads = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(nHeads <= 0,
             OP_LOGE(context, "n_heads must be positive, got %ld", nHeads),
             return ge::GRAPH_FAILED);
```

修复后（infershape，verbatim）：

```cpp
    // 获取属性 n_heads
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* nHeadsPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, nHeadsPtr);
    int64_t nHeads = *nHeadsPtr;
    OP_CHECK_IF(nHeads <= 0,
             OP_LOGE(context, "n_heads must be positive, got %ld", nHeads),
             return ge::GRAPH_FAILED);
```

tiling 侧多属性读取（修复后，verbatim，GetFloat 形态）：

```cpp
    // epsilon 属性
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* epsPtr = attrs->GetFloat(0);  // 第 0 个属性，返回 const float*
    OP_CHECK_NULL_WITH_CONTEXT(context, epsPtr);
    *epsilon = *epsPtr;
    return ge::GRAPH_SUCCESS;
```

tiling 侧 int64_t + 双 float 属性（修复后，verbatim，GetAttrPointer 形态）：

```cpp
    // 属性: n_heads, epsilon_cq, epsilon_ckv
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* nHeadsPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, nHeadsPtr);
    *N = *nHeadsPtr;
    const float* epsCqPtr = attrs->GetAttrPointer<float>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsCqPtr);
    *epsCq = *epsCqPtr;
    const float* epsCkvPtr = attrs->GetAttrPointer<float>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsCkvPtr);
    *epsCkv = *epsCkvPtr;
```

伴随修复（CeilDiv/CeilAlign）：include 块补一行 `#include "op_common/op_host/util/math_util.h"`（只 include `platform_util.h` 不够）：

```cpp
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
```

## 设计决策

- 选"`GetFloat` + 判空 + 解引用"而不是 `GetAttrPointer<float>(0)`：两者返回类型相同（`const float*`），前者语义直白且和 CANN 头文件内 `GetFloat` 的注释一致；`OP_CHECK_NULL_WITH_CONTEXT` 保证属性缺失时带上下文报错而不是解引用空指针。
- 备选"不依赖工具函数、手写 `(M + blockDim - 1) / blockDim`"被放弃：`math_util.h` 的 `CeilDiv/CeilAlign` 是官方 host 侧实现，补一个 include 即可，避免重复造轮子。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写属性读取时按"指针 + 判空 + 解引用"三步模板写；`GetAttrPointer` 名字里的 Pointer 即返回类型，不要按旧 API 的值语义 muscle memory 直赋。
- 在 TilingFunc 里读任何 `Attrs::GetXxx` 前先确认返回类型是指针（`const T*`）。
- 用 `Ops::Base::CeilDiv/CeilAlign/FloorDiv/FloorAlign` 时记得 `#include "op_common/op_host/util/math_util.h"`。

## 验证信号

来源说明：两个独立算子工程的 op_host 编译修复闭环（Ascend950PR，CANN 9.1.0/9.2.0 构建链）。

- 修复前形态一：`error: invalid conversion from 'const long int*' to 'int64_t'`（infershape.cpp:53）（来源平台：950 构建链；证据等级：tool_observed）。
- 修复前形态二：`error: cannot convert 'const float*' to 'float' in assignment` 与 `CeilDiv/CeilAlign` 未声明（tiling.cpp 31/32/85/112/121 行）（来源平台：950 构建链；证据等级：tool_observed）。
- 修复后（形态一工程）：op_host 侧（cust_optiling/cust_op_proto）编译通过，无该错误残留；infershape 属性读取的运行验证 UT `MlaPrologInfershape` 6/6 通过（含基于 n_heads 属性推导 4 个输出 shape 的正常路径与非法路径）（来源平台：950 构建链与 x86 host UT；证据等级：tool_observed）。
- 修复后（形态二工程）：两处修复（include + 指针解引用）写入后重建，tiling.cpp 的全部编译错误消失，构建推进到 opbuild dtype 映射阶段；最终全工程编译成功并产出 kernel 二进制与 custom_opp 包（来源平台：950 构建链；证据等级：tool_observed）。

## 失效边界

- 针对 CANN 9.1.0/9.2.0 目录的 `gert` 属性 API（`exe_graph/runtime/runtime_attrs.h`）；版本升级后接口若改为值返回需重新核对（未验证，推断同机制）。
- `GetAttrPointer<float>` 与 `GetFloat` 返回类型一致性在两条轨迹互相印证；`GetBool` 形态见相邻卡（指针隐式转 bool 的静默错误，非编译错）。
- 其余平台未验证（推断可迁移）。

**未决项**：

- 无（该编译问题已闭环；`GetFloat` 空指针分支的运行期行为未实测，但代码路径已加防御）。
## 坏实践

- `*N = *attrs->GetAttrPointer<int64_t>(0);` 裸解引用不判空——属性缺失时直接崩溃。
- 把 `attrs->GetFloat(0)` 当 `float` 直接赋值（编译错是好运；若接口换成 void* 转换则可能是运行期错）。
- 只 include `platform_util.h` 就 `using Ops::Base::CeilDiv`。

## 误诊

- 无（两种报错形态的 candidate/convert 信息都直接指向指针-值不匹配；伴随的 CeilDiv 未声明在补 include 后消失，确认独立）。

## 增量（GetPointer/GetPtr 名字陷阱形态与 GetListInt/TypedContinuousVector 访问式）

另一独立轨迹并入第三条独立 trace（Conv3D filter 梯度反向算子，ascend950 + CANN 9.1.0，x86_64 交叉编译），补齐同族 API 的"方法名不存在"误写形态与 ListInt 属性的完整访问式：

**1. 触发形态（方法名不存在，与"返回指针须解引用"互补的另一面）**：InferShape/Tiling 里读属性按通用容器直觉写了 `attrs->GetPointer<int64_t>(4)` 与 `stridesVec->GetPtr<int64_t>()`，首次编译报错：

```text
op_host/bench_no_harness_conv_3d_backprop_filter_infershape.cpp:32:40:
error: 'const class gert::RuntimeAttrs' has no member named 'GetPointer'; did you mean 'GetAttrPointer'?
   32 |     const int64_t* filterSize = attrs->GetPointer<int64_t>(/* filter_size attr index */ 4);
      |                                        ^~~~~~~~~~
      |                                        GetAttrPointer
op_host/bench_no_harness_conv_3d_backprop_filter_infershape.cpp:32:58:
error: expected primary-expression before '>' token
```

**2. RuntimeAttrs 公开面（runtime_attrs.h 原文节选，事实源）**：

```cpp
namespace gert {
class RuntimeAttrs {
 public:
  template <typename T>
  const T *GetAttrPointer(size_t index) const {
    return reinterpret_cast<const T *>(GetPointerByIndex(index));
  }
  const int64_t *GetInt(const size_t index) const {
    return GetAttrPointer<int64_t>(index);
  }
  const TypedContinuousVector<int64_t> *GetListInt(const size_t index) const {
    return GetAttrPointer<TypedContinuousVector<int64_t>>(index);
  }
  const ContinuousVectorVector *GetListListInt(const size_t index) const;
  const char *GetStr(const size_t index) const;
  const Tensor *GetTensor(const size_t index) const;
  const float *GetFloat(const size_t index) const;
  const bool *GetBool(const size_t index) const;
  const TypedContinuousVector<float> *GetListFloat(const size_t index) const;
  size_t GetAttrNum() const;
  ...
};
}
```

`TypedContinuousVector<int64_t>` 的元素访问（continuous_vector.h）：`const T *GetData() const` 与 `size_t GetSize() const`，即 `[GetData(), GetData() + GetSize())` 为容器内容——没有 `GetPtr<T>()`。

**3. 修复（InferShape 的 ListInt 属性，verbatim before/after）**：

```cpp
// before：GetPointer<T> 不存在
const int64_t* filterSize = attrs->GetPointer<int64_t>(/* filter_size attr index */ 4);
OP_CHECK_NULL_WITH_CONTEXT(context, filterSize);
gert::Shape* outputShape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
outputShape->SetDimNum(5);
for (int64_t i = 0; i < 5; i++) {
    outputShape->SetDim(i, filterSize[i]);
}
```

```cpp
// after：ListInt 属性走 GetListInt + GetData
auto filterSizeVec = attrs->GetListInt(4);
OP_CHECK_NULL_WITH_CONTEXT(context, filterSizeVec);
const int64_t* filterSize = filterSizeVec->GetData();
OP_CHECK_NULL_WITH_CONTEXT(context, filterSize);

gert::Shape* outputShape = context->GetOutputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

outputShape->SetDimNum(5);
for (int64_t i = 0; i < 5; i++) {
    outputShape->SetDim(i, filterSize[i]);
}
```

**4. 修复（Tiling 的向量元素访问与 GetInt 解引用，verbatim before/after）**：

```cpp
// before：TypedContinuousVector 无 GetPtr<T>()
const int64_t* stridesPtr = stridesVec->GetPtr<int64_t>();
const int64_t* padsPtr = padsVec->GetPtr<int64_t>();
const int64_t* dilationsPtr = dilationsVec->GetPtr<int64_t>();
```

```cpp
// after：元素指针用 GetData()
const int64_t* stridesPtr = stridesVec->GetData();
const int64_t* padsPtr = padsVec->GetData();
const int64_t* dilationsPtr = dilationsVec->GetData();
```

```cpp
// before：GetInt 返回的是指针，直接当值赋给 int64_t 是类型错误
int64_t groups = attrs->GetInt(3);
```

```cpp
// after：GetInt 返回 const int64_t*，需解引用
int64_t groups = 1;
const int64_t* groupsPtr = attrs->GetInt(3);
OP_CHECK_NULL_WITH_CONTEXT(context, groupsPtr);
groups = *groupsPtr;
OP_CHECK_IF(groups <= 0, OP_LOGE(context, "groups(%ld) must > 0", groups), return ge::GRAPH_FAILED);
```

**5. 设计要点**：标量 Int 属性与 ListInt 属性的读取路径不同——`GetInt(index)` 返回指针（属性缺失时可为 null，需判空后解引用），`GetListInt(index)` 返回向量指针（需先判向量再 `GetData()`）；属性 index 按 `_def.cpp` 中 `Attr(...)` 的注册顺序计数，改动注册顺序必须同步 index。

### 复现增量验证信号

- 修复前：`error: 'const class gert::RuntimeAttrs' has no member named 'GetPointer'; did you mean 'GetAttrPointer'?` 与级联的 `expected primary-expression before '>' token`（来源平台：950 交叉编译；证据等级：tool_observed）。
- 修复后：三处修正（GetPointer→GetListInt/GetData、GetPtr→GetData、GetInt 解引用）后 `cmake -DASCEND_COMPUTE_UNIT=ascend950 ..` configure 通过，`GetPointer`/`GetPtr` 报错消失；最终 build.sh 完整编译通过，产出含 cust_optiling（tiling so）在内的算子包（来源平台：950 交叉编译；证据等级：tool_observed）。

### 复现增量坏实践

- 按通用容器 API 直觉（`GetPtr<T>`/`At`/`operator[]`）猜 RuntimeAttrs 与 TypedContinuousVector 的方法；`GetPointer` 只存在于 `gert::KernelContext` 系（如 tiling context 的 `GetPointerByIndex` 私有实现），不在 RuntimeAttrs 公开面。
- `int64_t groups = attrs->GetInt(3);` 直接赋值——这是编译错误而非静默 bug，但改法必须是解引用加判空，不能强转绕过。
- 读属性前不看 runtime_attrs.h：方法名与返回类型一目了然。

### 复现增量失效边界

- 绑定 CANN 9.1.0 的 gert 头文件（`exe_graph/runtime/runtime_attrs.h`、`continuous_vector.h`）；其他版本需重新核对（未验证，推断可迁移）。
- GetListFloat/GetListListInt/GetTensor/GetStr 等同族方法推断同理，未逐一验证。
- 属性 index 依赖注册顺序，本增量示例中的 0-4 编号仅适用于该算子的注册顺序（strides/pads/dilations/groups/filter_size）。