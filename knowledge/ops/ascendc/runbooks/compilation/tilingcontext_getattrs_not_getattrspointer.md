---
type: "Runbook"
title: Tiling 属性用 GetAttrs 而非 GetAttrsPointer
description: "CANN 9.1.0 gert::TilingContext 没有 GetAttrsPointer()，也没有 GetAttr<T>(idx) 模板成员；属性须用 GetAttrs() 拿 RuntimeAttrs 再 GetInt/GetBool/GetFloat/GetAttrPointer<T> 读取（返回 const 指针须判空解引用），否则 tiling 在 npu_op_code_gen 阶段编译报 \"has no member named 'GetAttrsPointer'\" 或 \"expected primary-expression before 'bool'\"。复现增量：GQA 算子独立 trace 复现 GetAttr<float>/<bool> 模板误用形态（did you mean 'GetInputPointer'）。复现增量：RuntimeAttrs 全系只收 size_t 索引、无按名访问——GetAttrPointer<T>(\"name\") 报 invalid conversion from 'const char*' to 'size_t'，猜名 GetAttrInt64/GetAttrBool 不存在；索引即 OpDef Attr 声明顺序，多属性须建索引注释。"
tags: [task.debugging, topic.tiling, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# Tiling 属性用 GetAttrs 而非 GetAttrsPointer

## 现象

为一个 3D 网格采样类自定义算子（双输入 x:(N,C,D,H,W)、grid:(N,Dout,Hout,Wout,3)，单输出 y，带 interpolationMode/paddingMode/alignCorners 三个属性）编写 arch35 平台的 host 侧 Tiling 实现 `op_host/arch35/*_tiling.cpp`。在算子工程根目录首次执行 CMake 配置：

```bash
mkdir -p build && cd build && cmake -DASCEND_COMPUTE_UNIT=ascend950 \
    -DCMAKE_PREFIX_PATH=<ASCEND_HOME>/x86_64-linux/lib64/cmake ..
```

配置阶段（`npu_op_code_gen` 调 `compile_ascendc_all_ops_so` 编译 tiling 源文件）直接失败，报错原文：

```text
op_host/arch35/bench_no_harness_grid_sampler_3d_tiling.cpp:88:48:
error: expected primary-expression before 'bool'

     88 |     bool alignCorners = *attrs->GetAttrPointer<bool>(2);
        |                                                ^~~~

  RuntimeError: compile ascend all ops failed
  op_host/CMakeLists.txt:44 (npu_op_code_gen)
  -- Configuring incomplete, errors occurred!
```

当时代码凭记忆写成 `auto attrs = context->GetAttrsPointer();`，再用 `*attrs->GetAttrPointer<int64_t>(0)` / `*attrs->GetAttrPointer<bool>(2)` 逐个取属性。

## 适用场景

- 工程形态：算子仓式自定义算子开发（op_host TilingFunc，`IMPL_OP_OPTILING` 注册，CANN 9.1.0 exe_graph 新 tiling 框架）。
- API 路径：`gert::TilingContext` 属性读取——`context->GetAttrs()` 返回 `const RuntimeAttrs*`，其上用 `GetInt(index)` / `GetBool(index)` / `GetAttrPointer<T>(index)`；相关头文件 `exe_graph/runtime/extended_kernel_context.h`、`exe_graph/runtime/runtime_attrs.h`、`exe_graph/runtime/tiling_context.h`。
- 验证过的平台与 CANN 版本：ascend950 目标编译（Ascend950PR 开发机，CANN 9.1.0，x86_64 主机）——与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准。
- 适用粒度：platform_wide（该 API 契约随 CANN 版本走，与具体算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：旧版（非 exe_graph）tiling 框架的 context API；其他 CANN 版本未逐一验证；`GetAttrPointer<T>` 模板路径本身存在但本卡未实测（修复直接改用 GetInt/GetBool）。

## 根因

CANN 9.1.0 的 `gert::TilingContext` 没有 `GetAttrsPointer()` 成员，属性读取入口是继承自 `ExtendedKernelContext` 的 `GetAttrs()`，返回的 `RuntimeAttrs` 才提供类型化访问器；凭记忆写 `GetAttrsPointer()` 会让 `attrs` 推导失败，其后的模板调用 `GetAttrPointer<bool>` 被编译器当作非模板名解析，报 "expected primary-expression before 'bool'"。

头文件证据（本轨迹内实际读取）：

- `<ASCEND_HOME>/x86_64-linux/include/exe_graph/runtime/extended_kernel_context.h:133: const RuntimeAttrs *GetAttrs() const`
- `<ASCEND_HOME>/x86_64-linux/include/exe_graph/runtime/runtime_attrs.h:29: const T *GetAttrPointer(size_t index) const`（模板成员）
- 对 `tiling_context.h` grep `GetAttr|GetAttrs|Attrs` 无命中——TilingContext 自身不直接暴露属性方法。

## 解决配方

修复前（tiling 函数内属性读取段，op_host/arch35/*_tiling.cpp，verbatim 取自首次写入的文件）：

```cpp
    // 3. 获取属性
    auto attrs = context->GetAttrsPointer();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t interpMode = *attrs->GetAttrPointer<int64_t>(0);  // 0=bilinear, 1=nearest
    int64_t padMode = *attrs->GetAttrPointer<int64_t>(1);     // 0=zeros, 1=border, 2=reflection
    bool alignCorners = *attrs->GetAttrPointer<bool>(2);
```

修复后（同一位置，改用 GetAttrs() + GetInt/GetBool）：

```cpp
    // 3. 获取属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t interpMode = *attrs->GetInt(0);   // 0=bilinear, 1=nearest
    int64_t padMode = *attrs->GetInt(1);      // 0=zeros, 1=border, 2=reflection
    bool alignCorners = *attrs->GetBool(2);
```

属性读取值随后照常填入 TilingData（`tiling->paddingMode = padMode; tiling->alignCorners = alignCorners ? 1 : 0; tiling->interpolationMode = interpMode;`），不因读取方式改变。

## 设计决策

- `GetInt(idx)`/`GetBool(idx)` 与 `GetAttrPointer<T>(idx)` 都在 `RuntimeAttrs` 上可用；本卡选前者——按索引取标量属性的意图更直接，且避免模板参数写错类型（如把 bool 写成 int64_t 解引用错位）。两者性能差异未比较。
- 判定 API 是否存在的第一手段是 grep CANN 安装目录头文件（`include/exe_graph/runtime/`），而非参考旧示例或记忆。

## 预防

- 写 TilingFunc 前先确认所用 CANN 版本的 `TilingContext` 头文件签名；不信任从旧示例/记忆迁移来的 context 方法名。
- 遇到 "expected primary-expression before '<类型>'" 出现在模板调用处时，优先怀疑被调用对象本身没有解析成功（上游方法名/类型错误），而不是模板语法。

## 验证信号

问题与验证来源：在一次算子骨架开发的 tiling 首轮 CMake 配置中发现并修复，随后同一命令重新配置验证。

- 修复前：`cmake` 配置阶段 npu_op_code_gen 报 `error: expected primary-expression before 'bool'`，`RuntimeError: compile ascend all ops failed`，`-- Configuring incomplete, errors occurred!`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一配置命令输出 `-- Opbuild generating sources - done`、`-- Configuring done / -- Generating done / -- Build files have been written to: .../build`，tiling 编译单元通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅验证 CANN 9.1.0（x86_64-linux 安装）；其他版本 API 可能增删，迁移时按头文件重新确认（未验证，推断可迁移）。
- `RuntimeAttrs::GetAttrPointer<T>` 模板路径在头文件中存在，但本卡未实测其编译行为；若使用它仍需保证 `attrs` 来自 `GetAttrs()`。

**未决项**：

- 无——根因已定位到 API 契约层并有头文件证据。
## 坏实践

- 凭记忆/类比旧框架把不存在的方法名（`GetAttrsPointer`）写进 tiling 代码，报错位置还容易误导到模板语法本身。
- 报错后不查头文件直接试改写法（加 template 关键字、换类型参数等），会在错误层面打转。

## 误诊

无。

## 后续增量（2026-09-17，并行策展批：GetAttrPointer<T> 直调变体与编译器误导告警）

另一 trace（BenchNoHarnessCummin，Ascend950PR + CANN 9.1.0）以第二种写法踩中同一 API 契约：按旧样例直觉直接在 context 上调用模板方法 `context->GetAttrPointer<int64_t>(0)`，`bash build.sh --soc=ascend950 -j8` 的 `compile ascend_all_ops` 阶段报错：

```text
op_host/arch35/*_tiling.cpp:77:31:
error: 'class gert::TilingContext' has no member named 'GetAttrPointer'; did you mean 'GetInputPointer'?
     77 |     auto dimScalar = context->GetAttrPointer<int64_t>(0);
        |                               ^~~~~~~~~~~~~~
        |                               GetInputPointer
error: expected primary-expression before '>' token
```

**编译器建议的 `GetInputPointer` 是输入张量访问接口，与属性无关——采纳它会拿错数据**（本增量新增的坏实践警示）。修复同本卡入口约定，且本增量实测了原卡未验证的 `GetAttrPointer<T>` 模板路径：

```cpp
// 修复前（TilingFunc 属性获取段，verbatim）
    auto dimScalar = context->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimScalar);
    int64_t dim = *dimScalar;

// 修复后（先取 RuntimeAttrs，再取属性指针；空指针用 OP_CHECK_NULL_WITH_CONTEXT 归一化处理）
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dimScalar = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimScalar);
    int64_t dim = *dimScalar;
```

定位探针（可复用）：

```bash
grep -rn "GetAttrs\|GetAttrPointer" "$ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/tiling_context.h"
# 命中：const RuntimeAttrs *GetAttrs() const { ... compute_node_info->GetAttrs(); }
grep -rn "class TilingContext" "$ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/"
# 命中：tiling_context.h:34: class TilingContext（: public ExtendedKernelContext）
```

配套事实：属性在 `*_def.cpp` 侧用 `.Int()` 声明、读取宽度由 `GetAttrPointer<int64_t>` 决定；多属性算子的索引按 `Attr(...)` 声明顺序数（本例单属性，多属性顺序未验证）。

### 复现增量验证信号

来源说明：该算子迭代一骨架搭建会话的编译修复循环。

- 修复前：`error: 'class gert::TilingContext' has no member named 'GetAttrPointer'; did you mean 'GetInputPointer'?` + `expected primary-expression before '>' token`（来源平台：950；证据等级：tool_observed）。
- 头文件核查：`tiling_context.h:133: const RuntimeAttrs *GetAttrs() const`（经 ComputeNodeInfo 转发）；`TilingContext : public ExtendedKernelContext`（来源平台：950；证据等级：tool_observed）。
- 修复后：`cust_optiling` 目标编译链接通过（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- `GetAttrPointer<T>` 与 `GetInt/GetBool` 均在 `RuntimeAttrs` 上可用（原卡实测后者、本增量实测前者）；按索引取标量属性时两者等价，模板路径须保证模板类型与属性声明宽度一致。
- 仅验证 CANN 9.1.0 头文件布局；其他版本接口归属可能调整（未验证，推断可迁移）。

## 增量（GQA 算子独立复现：GetAttr<T>(idx) 模板形态与 GetFloat/GetBool 指针解引用）

另一独立轨迹并入第二条独立 trace（GQA 分组查询注意力算子，Ascend950PR + CANN 9.1.0），补齐同根因的另一误写形态与 GetFloat/GetBool 取值式：

**1. 触发形态（模板成员不存在 + 候选提示不同）**：TilingFunc 里按"直觉 API"写成 `context->GetAttrsPointer()` 加 `attrs->GetAttr<float>(0)` / `attrs->GetAttr<bool>(1)`，`bash build.sh --soc=ascend950 -j8` 在 cmake 配置阶段（autogen 编译 ascendc_all_ops）报错：

```
op_host/arch35/bench_no_harness_gqa_tiling.cpp:136:27:
error: 'class gert::TilingContext' has no member named 'GetAttrsPointer';
did you mean 'GetInputPointer'?
    136 |     auto attrs = context->GetAttrsPointer();

op_host/arch35/bench_no_harness_gqa_tiling.cpp:138:43:
error: expected primary-expression before 'float'
    138 |     float scaleValueAttr = attrs->GetAttr<float>(0);   // Float attr

op_host/arch35/bench_no_harness_gqa_tiling.cpp:139:36:
error: expected primary-expression before 'bool'
    139 |     bool isCausal = attrs->GetAttr<bool>(1);           // Bool attr
```

随后 make 报 `Error 1`，python 封装 `compile_ascendc_all_ops_so.py` 抛 `RuntimeError: compile ascend all ops failed`，`op_host/CMakeLists.txt:39 (npu_op_code_gen)` 处 Configuring incomplete。

**2. 修复（GetAttrs + GetFloat/GetBool 指针解引用 + 判空，verbatim before/after）**：

```cpp
// before（编译失败）
    auto attrs = context->GetAttrsPointer();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    float scaleValueAttr = attrs->GetAttr<float>(0);   // Float attr
    bool isCausal = attrs->GetAttr<bool>(1);           // Bool attr
```

```cpp
// after（GetAttrs + GetFloat/GetBool 指针解引用 + 判空）
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scaleValuePtr = attrs->GetFloat(0);
    const bool* isCausalPtr = attrs->GetBool(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, isCausalPtr);
    float scaleValueAttr = *scaleValuePtr;
    bool isCausal = *isCausalPtr;
```

**3. 定位链路（可复用）**：报错文本的候选提示（`did you mean 'GetInputPointer'?`）说明该上下文对象上确无 GetAttrsPointer 成员；再在 CANN 安装头文件反查正确入口：`grep -rn "GetAttrs" $ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/` → 命中 `extended_kernel_context.h:132 const RuntimeAttrs *GetAttrs() const` 与 `runtime_attrs.h class RuntimeAttrs`，据此确认返回类型与取值方式；属性在 IR 中的声明顺序即索引（`GetFloat(0)` 对应 op_def 中第一个 `.Attr`）。

**4. 边界增补**：`RuntimeAttrs` 的取值方法名以目标 CANN 版本头文件为准（9.1.0 为 `GetFloat/GetBool` 返回指针）；List/String 类型属性取值方法不同（本增量只验证了标量 float/bool，未验证）。

### 复现增量验证信号

- 修复前：`build.sh --soc=ascend950` 在 autogen 编译 `bench_no_harness_gqa_tiling.o` 时报上述三条错误（来源平台：950；证据等级：tool_observed）。
- 修复后：同目录重新 configure+build，后续两轮构建的错误输出中不再出现任何 GetAttrsPointer/GetAttr 相关错误，且 `[66%] Built target cust_optiling` 表明 tiling 目标编译通过（来源平台：950；证据等级：tool_observed）。

### 复现增量坏实践

- 凭记忆写 `GetAttr<float>(0)` 这类"看起来自然"的模板接口：gert 体系没有该成员，报错还叠加 `expected primary-expression before 'float'` 语法噪声，容易误判为宏或头文件问题。
- 只改报错行不查返回类型：`GetFloat(i)` 返回指针，若不解引用直接当 float 用（如 `tiling->scaleValue = attrs->GetFloat(0);`）会把指针隐式转整数/报新错。

## 增量（RuntimeAttrs 无按名访问：GetAttrPointer<T>("name") 与猜名 GetAttrInt64/GetAttrBool 全被拒）

另一独立轨迹并入第三条独立 trace（GRU 类基准算子，Ascend950 + CANN 9.2.0），补齐 RuntimeAttrs 取值签名的另一面：**只提供索引访问，没有按名访问**。InferShape（`gert::InferShapeContext::GetAttrs()`）与 TilingFunc（`gert::TilingContext::GetAttrs()`）两侧同契约。

两种真实踩坑形态（同一会话先后出现）：

```text
// 形态一：按名传 const char*
int64_t inputSize = *attrs->GetAttrPointer<int64_t>("inputSize");
→ error: invalid conversion from 'const char*' to 'size_t' {aka 'long unsigned int'} [-fpermissive]
  note: initializing argument 1 of 'const T* gert::RuntimeAttrs::GetAttrPointer(size_t) const'

// 形态二：猜方法名 GetAttrInt64/GetAttrBool（不存在）
→ error: 'gert::RuntimeAttrs' has no member named 'GetAttrInt64'; did you mean 'GetAttrPointer'?
→ error: 'GetAttrBool'; did you mean 'GetBool'?
```

头文件实测签名（`$ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/runtime_attrs.h`，全系 `size_t index`）：`GetAttrPointer<T>(size_t)` / `GetInt(size_t)` / `GetListInt(size_t)` / `GetListListInt(size_t)` / `GetStr(size_t)` / `GetTensor(size_t)` / `GetFloat(size_t)` / `GetBool(size_t)`。索引即 OpDef 中 `this->Attr(...)` 的声明顺序（0 起）。

修复形态（多属性算子，7 个属性；**索引-属性对应注释是防静默错位的最低成本手段**——插入/删除属性会改变索引且不报编译错，运行期才取错值）：

```cpp
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // 属性索引与 OpDef 定义顺序一致:
    // 0:inputSize 1:hiddenSize 2:numLayers 3:bias 4:batchFirst 5:dropout 6:bidirectional
    int64_t inputSize = *attrs->GetInt(0);
    bool batchFirst = *attrs->GetBool(4);
    bool bidirectional = *attrs->GetBool(6);
```

### 复现增量验证信号

- 按名 `GetAttrPointer<T>("name")` 报 `invalid conversion from 'const char*' to 'size_t'`，编译器 note 直接给出正确签名（来源平台：950；证据等级：tool_observed）。
- 猜测的 `GetAttrInt64`/`GetAttrBool` 报 no member + `did you mean` 提示（来源平台：950；证据等级：tool_observed）。
- 改为 `*attrs->GetInt(idx)`/`*attrs->GetBool(idx)` 后该类错误消失，host 侧多轮后全部编译通过（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 结论绑定 CANN 9.2.0 `runtime_attrs.h`（9.1.0 侧由原卡/并行轨迹 实测同族契约）；其他版本未验证，推断可迁移。
- 坏实践：按 Python/其他框架习惯猜 `GetAttrByName`/`GetAttrInt64` 一类名字（猜名浪费一轮分钟级构建）；无注释地裸写魔法索引（跨算子复制代码时错位不报编译错）。
