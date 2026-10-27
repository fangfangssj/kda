---
type: "Runbook"
title: Conv2d list 属性经 OP_ATTR→TilingContext 传递
description: 自定义算子 strides/pads/dilations 三个 list 属性的完整传递配方：OpDef Attr().ListInt() 声明、L2 用 aclGetIntArraySize 校验长度、L0 用 OP_ATTR 打包进 launcher、tiling 用 GetAttrs()->GetAttrPointer<TypedContinuousVector<int64_t>> 按声明序读值；aclIntArray 为不透明类型，L0 侧手解内部布局的 {size;values[]} 内联假设已被同算子后续 ST 证伪（实际 {vtable;data指针;size}），该读法 banned。
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T04:50:06Z'
updated_at: '2026-09-17T04:50:06Z'
platforms: ["950"]
---

# Conv2d list 属性经 OP_ATTR→TilingContext 传递

## 现象

为带三个 list 属性（strides[2]、pads[4]、dilations[2]）的自定义 Conv2d 基准算子（bench_no_harness_conv_2d）搭五层工程（op_graph proto、op_host def+tiling、op_api L0+L2）时，属性如何从 aclnn 入参一路传到 host tiling 并读出数值，文档里没有一条贯通的示例（参考样例 add_example 无 list 属性）。排查 CANN 9.1.0 头文件得到两个非显然事实：

1. `gert::TilingContext`（exe_graph/runtime/tiling_context.h）自身没有属性访问方法，它继承自 `ExtendedKernelContext`，属性经 `context->GetAttrs()`（返回 `const RuntimeAttrs*`）→ `GetAttrPointer<T>(index)` 按声明序取值；list 属性落为 `ContinuousVector`（`GetData()/GetSize()` 读内容），int 标量属性用 `GetInt(index)`。
2. `aclIntArray` 是不透明类型（`typedef struct aclIntArray aclIntArray`）：公开 C 接口只有 `aclGetIntArraySize(const aclIntArray*, uint64_t*)` 取长度，没有取元素值的公开函数。创建侧 `aclCreateIntArray(const int64_t* value, uint64_t size)` 的签名曾让人推断内部布局为 `{uint64_t size; int64_t values[]}` 内联数组——**该推断是错的**：同算子后续 ST 实测其布局为 `{vtable 指针; const int64_t* data; uint64_t size}`（data 为指向值数组的指针，非内联），按内联假设读值会读到 vtable/指针垃圾（详见同域 561103 aclIntArray 布局卡）。

按此链路实现后 host 侧三个目标（cust_optiling/cust_opapi/cust_op_proto）编译通过。

## 适用场景

- 工程形态：算子仓自定义算子（ops-registry-invoke）五层工程中 list 类型属性（strides/pads/dilations 类）的声明、校验、打包与 tiling 读取。
- API 路径：`ops::OpDef::Attr().ListInt()`；`aclGetIntArraySize`；`ADD_TO_LAUNCHER_LIST_AICORE(..., OP_ATTR(...))`；`gert::TilingContext → ExtendedKernelContext::GetAttrs() → RuntimeAttrs::GetAttrPointer<gert::TypedContinuousVector<int64_t>>(idx) → GetData()`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950，arch35/DAV_3510）+ CANN 9.1.0（x86_64-linux 安装），编译级验证（三个 host 目标编译通过）；同算子后续 ST 上板验证整链（见 561103 卡的修复后信号）。
- 适用粒度：platform_wide（绑定 CANN 9.1.0 的 opdev/exe_graph 接口形态，与算子类别无关）。
- 路线优先级：属性传递接线（OpDef ListInt → L2 长度校验 → L0 OP_ATTR 打包 → tiling GetAttrPointer 读取）为 recommended_fastpath；L0 侧按 `{size; values[]}` 内联布局 reinterpret 读 aclIntArray 值为 **banned**（已被实测证伪），确需在 L0 读元素值时按实测布局经 data 指针读（safety_fallback，依赖非公开 ABI，能用公开接口避开时应避开）。
- 明确不适用/未验证场景：运行时取值正确性未在本阶段上板验证（由同算子后续 ST 覆盖）；attr 作为 int 标量（GetInt）的路径未在本算子使用；按属性名访问不存在（RuntimeAttrs 全系只收 size_t 索引）。

## 解决配方（完整链路，自底向上四段）

**第 1 段：op_host 算子定义——声明 list 属性（OpDef）**

```cpp
        // 属性: strides [stride_h, stride_w]
        this->Attr("strides").ListInt();
        // 属性: pads [pad_top, pad_bottom, pad_left, pad_right]
        this->Attr("pads").ListInt();
        // 属性: dilations [dilation_h, dilation_w]
        this->Attr("dilations").ListInt();
```

（声明顺序决定后续 GetAttrPointer 的 index：strides=0、pads=1、dilations=2。）

**第 2 段：op_host tiling——按声明序读 ContinuousVector**

```cpp
// 获取输入 shape 和属性
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context,
    int64_t* N, int64_t* CIn, int64_t* H, int64_t* W,
    int64_t* COut, int64_t* KH, int64_t* KW,
    int64_t* strideH, int64_t* strideW,
    int64_t* padTop, int64_t* padBottom, int64_t* padLeft, int64_t* padRight,
    int64_t* dilationH, int64_t* dilationW,
    int64_t* hOut, int64_t* wOut, ge::DataType* dataType)
{
    // 获取 x shape: [N, C_in, H, W]
    auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() != 4,
                OP_LOGE(context, "x shape dim must be 4"), return ge::GRAPH_FAILED);
    *N = xShape->GetStorageShape().GetDim(0);
    *CIn = xShape->GetStorageShape().GetDim(1);
    *H = xShape->GetStorageShape().GetDim(2);
    *W = xShape->GetStorageShape().GetDim(3);

    // 获取 filter shape: [C_out, C_in, K_h, K_w]
    auto filterShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, filterShape);
    OP_CHECK_IF(filterShape->GetStorageShape().GetDimNum() != 4,
                OP_LOGE(context, "filter shape dim must be 4"), return ge::GRAPH_FAILED);
    *COut = filterShape->GetStorageShape().GetDim(0);
    *KH = filterShape->GetStorageShape().GetDim(2);
    *KW = filterShape->GetStorageShape().GetDim(3);

    // 获取属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto stridesVec = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesVec);
    const int64_t* strides = stridesVec->GetData();
    *strideH = strides[0];
    *strideW = strides[1];

    auto padsVec = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsVec);
    const int64_t* pads = padsVec->GetData();
    *padTop = pads[0];
    *padBottom = pads[1];
    *padLeft = pads[2];
    *padRight = pads[3];

    auto dilationsVec = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilationsVec);
    const int64_t* dilations = dilationsVec->GetData();
    *dilationH = dilations[0];
    *dilationW = dilations[1];

    // 计算输出空间维度
    *hOut = (*H + *padTop + *padBottom - *dilationH * (*KH - 1) - 1) / *strideH + 1;
    *wOut = (*W + *padLeft + *padRight - *dilationW * (*KW - 1) - 1) / *strideW + 1;

    // 获取数据类型
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();

    return ge::GRAPH_SUCCESS;
}
```

**第 3 段：op_api L0——OP_ATTR 打包进 launcher（值读取见段后警示）**

```cpp
static const aclTensor* BenchNoHarnessConv2dAiCore(
    const aclTensor* x, const aclTensor* filter, const aclTensor* bias,
    const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
    const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessConv2dAiCore, x, filter, bias, y);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessConv2d,
        OP_INPUT(x, filter, bias), OP_OUTPUT(y),
        OP_ATTR(strides, pads, dilations));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessConv2dAiCore failed."),
             return nullptr);
    return y;
}
```

L0 的 InferShape 需要属性数值，但 aclIntArray 无公开取值接口。本阶段写入的辅助读法如下（**banned，勿照抄**——按 `{uint64_t size; int64_t values[]}` 内联布局 reinterpret，已被同算子后续 ST 证伪：读出的是 vtable/指针位组合的垃圾值，触发 561103）：

```cpp
// aclIntArray 内部布局辅助访问（已证伪的形态，仅作历史记录）
// aclCreateIntArray(const int64_t* value, uint64_t size) 创建的内部结构
struct AclIntArrayRepr {
    uint64_t size;
    int64_t values[];
};

static inline int64_t GetIntArrayValue(const aclIntArray* arr, uint64_t index)
{
    const auto* repr = reinterpret_cast<const AclIntArrayRepr*>(arr);
    return repr->values[index];
}
```

实测正确布局（x86_64，CANN 9.1.0，最小探针程序 dump 确认，完整探针与修复过程见同域 561103 aclIntArray 布局卡）：

```cpp
// 实际内存布局（x86_64, CANN 9.1.0）：
//   offset 0: vtable 指针 (op::Object 虚函数表)
//   offset 8: const int64_t* data (指向值数组，非内联)
//   offset 16: uint64_t size (元素数量)
struct AclIntArrayRepr {
    void* vtable;            // offset 0: vtable pointer
    const int64_t* data;     // offset 8: pointer to values array
    uint64_t size;           // offset 16: number of elements
};

static inline int64_t GetIntArrayValue(const aclIntArray* arr, uint64_t index)
{
    const auto* repr = reinterpret_cast<const AclIntArrayRepr*>(arr);
    return repr->data[index];
}
```

若编译单元里 opdev 头可见，元素访问亦可直接 `(*arr)[i]` + `Size()`（见同域 aclIntArray 下标访问卡），无须手解布局。`repr->data[index]` 无内建边界检查，用前先以公共 `aclGetIntArraySize` 校验长度。

**第 4 段：op_api L2——公开接口校验长度（只用公开 API）**

```cpp
static bool CheckAttrs(const aclIntArray* strides, const aclIntArray* pads,
                       const aclIntArray* dilations)
{
    uint64_t stridesSize, padsSize, dilationsSize;
    aclGetIntArraySize(strides, &stridesSize);
    aclGetIntArraySize(pads, &padsSize);
    aclGetIntArraySize(dilations, &dilationsSize);

    OP_CHECK(stridesSize == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides length must be 2, got %lu", stridesSize),
             return false);
    OP_CHECK(padsSize == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pads length must be 4, got %lu", padsSize),
             return false);
    OP_CHECK(dilationsSize == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilations length must be 2, got %lu", dilationsSize),
             return false);
    return true;
}
```

L2 的 `aclnnBenchNoHarnessConv2dGetWorkspaceSize` 在 CheckParams 后把三个 `aclIntArray*` 原样传给 L0（Contiguous 只作用于 tensor，属性指针直传），L0 经 `OP_ATTR(strides, pads, dilations)` 打包进 launcher，host tiling 侧即按第 2 段读出——整条链不需要任何序列化代码。

## 设计决策

- **tiling 侧选 `GetAttrPointer<TypedContinuousVector<int64_t>>(index)` 按声明序取**：TilingContext 没有按属性名取值的接口（头文件确认 GetAttrs 在基类 ExtendedKernelContext 上），index 与 OpDef 中 `Attr()` 声明顺序一一对应；用 `OP_CHECK_NULL` 防属性缺失。
- **L0 侧 InferShape 需要属性数值，但 aclIntArray 无公开取值接口**：骨架阶段按创建函数签名推断内联布局并 reinterpret——这是一个明确的风险决策（依赖内部 ABI 而非公开契约），且后来在同算子的 ST 阶段实际爆雷（读出 -1e17 量级垃圾值、GetWorkspaceSize 561103）。教训：对不透明类型按直觉猜字段顺序前，先用最小探针程序实测布局（构造已知值数组 → dump 内存 → 指针解引用验证）。
- **L2 侧只用公开 API（aclGetIntArraySize）做长度校验**：能不用内部布局的地方就不用，把 reinterpret 读取限制在 L0 InferShape 这一处最小范围（并优先考虑把值级计算推迟到 tiling 侧，走 GetAttrPointer 的框架路径）。
- **属性值校验收敛到单一辅助函数**：repr 读值若用于 OP_CHECK 值级校验（strides>0/pads>=0/dilations>0），先以 aclGetIntArraySize 校验长度再按下标读。

## 系统信息

- 工具链/版本：CANN 9.1.0（安装根 cann-9.1.0），x86_64-linux include 布局；构建 `build.sh --soc=ascend950`。
- 目标平台：Ascend950PR（arch35/DAV_3510）。
- 关键头文件：`include/exe_graph/runtime/tiling_context.h`（TilingContext，无 GetAttrs）、`include/exe_graph/runtime/extended_kernel_context.h`（GetAttrs 在此类，`const RuntimeAttrs* GetAttrs()`）、`include/exe_graph/runtime/runtime_attrs.h`（`GetAttrPointer<T>(index)`/`GetInt(index)`/`GetListInt(index)` 等，全系 size_t 索引）、`include/exe_graph/runtime/continuous_vector.h`（`GetData()/GetSize()`）、`include/aclnn/acl_meta.h`（`aclCreateIntArray`/`aclGetIntArraySize`，aclIntArray 不透明）、`include/aclnn/opdev/op_arg_def.h`（`OP_ATTR(x...)` 宏）。
- 目录布局与相关文件：`op_host/bench_no_harness_conv_2d_def.cpp`（Attr 声明）、`op_host/arch35/bench_no_harness_conv_2d_tiling.cpp`（属性读取）、`op_api/bench_no_harness_conv_2d.cpp`（L0：AclIntArrayRepr/InferShape/AiCore）、`op_api/aclnn_bench_no_harness_conv_2d.cpp`（L2：CheckAttrs/GetWorkspaceSize）、`op_graph/bench_no_harness_conv_2d_proto.h`。

## 验证信号

本配方来自该算子五层工程的实际搭建：按上述链路实现后全量构建。

- host 侧 cust_optiling、cust_opapi、cust_op_proto 三个目标全部编译通过（构建输出显示三个 target Built target；错误推进到 kernel 侧），证明整条属性传递链的接口签名与头文件引用正确（来源平台：950；证据等级：tool_observed）。
- 头文件事实（TilingContext 无 GetAttrs、GetAttrs 在 ExtendedKernelContext、RuntimeAttrs 提供 GetAttrPointer/GetInt、aclIntArray 仅有 aclGetIntArraySize 公开取长度接口）均由 grep/读 CANN 9.1.0 头文件确认（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `AclIntArrayRepr` 内联布局假设已在同算子后续 ST 被证伪（运行期读到 vtable/指针垃圾 → 561103）；实测布局 `{vtable; const int64_t* data; uint64_t size}` 绑定 x86_64 + CANN 9.1.0，aarch64 与后续版本未验证——内部布局不是公开契约，凡有公开替代接口时应立即替换（safety_fallback 声明：手解布局仅用于 L0 InferShape 这类"公开 API 无取值手段"的场合，且必须带版本锚点注释）。
- 运行时数值正确性（tiling 读到的 strides/pads/dilations 与用户传入一致、L0 InferShape 算出的 H_out/W_out 正确）在本阶段未上板验证；同算子后续 ST 在修正布局后 50/50 全过（见 561103 卡），整链含 tiling 侧读取经真实调用验证。
- `GetAttrPointer` 的 index 绑定 OpDef 属性声明顺序：调整属性声明顺序或插入新属性时 index 需同步，否则读到错位数据（编译期不会报错）。

**未决项**：

- 框架侧正式读值路径（替代手解布局）未在本轨迹实施；若后续 CANN 提供公开按下标读 aclIntArray 的 API，应整体替换 repr 方案（优先项：调研并迁移）。
- `TypedContinuousVector` 与 `GetListInt(size_t)` 两条 tiling 侧读法的适用差异未做对比验证（本轨迹实测前者）。
## 坏实践

- 对 aclIntArray 按创建函数签名推断"内联数组"布局直接 reinterpret 读值——同算子已实测读到垃圾（本卡第 3 段的历史形态）。
- 在 L2/多处散布 reinterpret 读值：把内部布局依赖扩大到不必要的范围；应把读取收敛到单一辅助函数。
- 按属性名在 tiling 里找接口：TilingContext 没有按名访问，先查基类 ExtendedKernelContext 与 RuntimeAttrs 再动手，不要凭空猜 API。

## 误诊

- 排查初期容易以为 TilingContext 缺 GetAttrs 是"版本太老/头文件没包全"，四处找 `GetAttrs` 的其他头；实际方法在基类上，继承链即可用（该弯路为真实排查过程）。
- 同算子后续 ST 曾把 561103（属性读坏后 InferShape 报错的表象）沿"哪个 l0 接口返回 nullptr"方向逐个排查——真实失败点是布局读错，见 561103 卡的误诊记录。
