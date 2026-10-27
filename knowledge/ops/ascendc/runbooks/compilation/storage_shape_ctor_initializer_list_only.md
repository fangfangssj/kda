---
type: "Runbook"
title: UT 构造 StorageShape 须用 initializer_list
description: "gert::StorageShape 只有 (initializer_list<int64_t>&, initializer_list<int64_t>&) 构造；op_host UT 用 vector/迭代器对花括号构造报 no matching function，须把测试参数的 shape 字段改为 std::initializer_list<int64_t>。复现增量：两条新路线——helper 形参改 initializer_list + 直接构造语法，以及\"默认构造 + MutableOriginShape/MutableStorageShape + SetDimNum/SetDim 逐维填充\"的运行期构造辅助函数（后者覆盖纯运行期生成形状的场景，此前未验证）。复现增量：第三条路线——默认构造 + AppendDim 逐维追加（与 SetDimNum/SetDim 等价且更简），及 infershape 用例输出留空 shape 由被测函数推导的组织模式。复现增量（12 个独立 trace 复现，扩 CANN 9.2.0）：错误形态补全——单组 dims 单层花括号须改双层 {{dims},{dims}} 二元组、SetDimNum/SetDim 误在 StorageShape 上直调报 no member、TilingContextPara 构造参数不匹配为级联噪声、单变量表达式花括号重载决议不稳定。"
tags: [task.testing, topic.compilation, topic.terminology]
status: stable
sources: []
created_at: '2026-09-16T11:15:37Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# UT 构造 StorageShape 须用 initializer_list

## 现象

为 BenchNoHarnessConv2d 搭建 op_host UT（tiling/infershape 单测，gtest + tiling_context_faker 框架）时，测试参数结构体里 shape 存的是 `std::vector<int64_t>`，构造输入形状写成：

```cpp
gert::StorageShape xShape({param.xShape.begin(), param.xShape.end()},
                          {param.xShape.begin(), param.xShape.end()});
```

编译报（多个构造点同报）：

```
.../tests/ut/op_host/test_bench_no_harness_conv_2d_tiling.cpp:175:88: error: no matching function for call to
'gert::StorageShape::StorageShape(<brace-enclosed initializer list>, <brace-enclosed initializer list>)'
  175 |                                    {param.filterShape.begin(), param.filterShape.end()});
      |                                                                                        ^
.../include/exe_graph/runtime/storage_shape.h:27:3: note: candidate: 'gert::StorageShape::StorageShape(
    const std::initializer_list<long int>&, const std::initializer_list<long int>&)'
   27 |   StorageShape(const std::initializer_list<int64_t> &origin_shape, const std::initializer_list<int64_t> &storage_shape)
      |                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
...: note: no known conversion for argument 1 from '<brace-enclosed initializer list>' to
    'const std::initializer_list<long int>&'
（default/copy/move 构造均 "candidate expects N arguments"）
```

## 适用场景

- 工程形态：自定义算子 op_host UT（tiling/infershape faker 框架），需要手工构造 `gert::StorageShape`。
- API 路径：`gert::StorageShape`（`include/exe_graph/runtime/storage_shape.h:27`）；测试参数结构体的 shape 字段类型。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（x86_64 host 侧 UT，被测为 ascend950 包 tiling 源）。
- 适用粒度：platform_wide（CANN 头文件构造函数集合，与算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：CANN 提供 SetDim/Build 之类运行时逐维构造 API 的版本（本版本头文件未见，未验证）。

## 根因

`gert::StorageShape` 只有四个构造：默认、拷贝、移动、以及 `(const std::initializer_list<int64_t>&, const std::initializer_list<int64_t>&)`——没有接收 vector/迭代器区间的构造。`{param.xShape.begin(), param.xShape.end()}` 是"两个迭代器"的花括号列表，被推导为 `std::initializer_list<iterator>` 而非 int64_t 列表，无法匹配任何候选。`std::initializer_list` 不能在运行时从 vector 构造，只能由静态数组/字面量支撑——所以修复必须改测试参数的字段类型，而不是改构造写法。

## 解决配方

修复前（vector 字段 + 迭代器对花括号）：

```cpp
struct Conv2dTilingParam {
    std::string caseName;
    std::vector<int64_t> xShape;        // <- 问题字段
    ge::DataType xDtype;
    std::vector<int64_t> filterShape;
    std::vector<int64_t> biasShape;
    ...
};
...
gert::StorageShape xShape({param.xShape.begin(), param.xShape.end()},
                          {param.xShape.begin(), param.xShape.end()});   // no matching function
```

修复后（字段改 `std::initializer_list<int64_t>`，静态测试用例数组支撑；构造直接双列表）：

```cpp
struct Conv2dTilingParam {
    std::string caseName;
    // 输入 x: [N, Cin, H, W]
    std::initializer_list<int64_t> xShape;      // <- initializer_list 由静态数组支撑
    ge::DataType xDtype;
    // filter: [Cout, Cin, Kh, Kw]
    std::initializer_list<int64_t> filterShape;
    // bias: [Cout]
    std::initializer_list<int64_t> biasShape;
    // 属性
    ...
};
...
static Conv2dTilingParam testCases[] = {
    {"basic_conv_3x3", {1, 64, 56, 56}, ge::DT_FLOAT16, {64, 64, 3, 3}, {64}, ...},
    ...
};
...
gert::StorageShape xShape = {param.xShape, param.xShape};   // origin == storage
```

按下标取维度的代码用 `.begin()[index]`（`param.xShape.begin()[0]`）；yShape 这类运行期拼出的形状用已读出的标量值组字面量列表再构造。

## 设计决策

- 为什么改字段类型而不是写 vector→initializer_list 适配函数：`std::initializer_list` 不拥有数据、无法在运行时从 vector 合成，任何"适配"都退化为 UB 或拷贝静态数组；官方 add_example 的 UT 参数结构体正是 `std::initializer_list<int64_t>` 字段 + 静态 testCases 数组的模式，直接对齐。
- 静态数组支撑的合法性：initializer_list 绑定到 testCases 静态存储期数组元素，生命周期覆盖测试执行。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 环境：x86_64 主机，CANN 9.1.0（ASCEND_HOME_PATH 指向安装根）；UT 工程位于 `operators/<op>/tests/ut/{common,cmake,op_host}`。
- 头文件路径修正（同轮 UT 搭建实测）：`op_impl_space_registry_v2.h` 在 `$ASCEND_HOME_PATH/x86_64-linux/pkg_inc/base/registry/`（不在 `include/`）；`op_impl_registry.h` 在 `include/register/`。
- 依赖治理：tiling_case_executor 去除了参考模板对 nlohmann/json.hpp 的依赖（直接构造平台信息），避免引入 CANN 外部头文件路径（如 miniconda 的 json.hpp）。

## 预防

- 搭 op_host UT 时直接抄官方 add_example 的参数结构体模式（`std::initializer_list<int64_t>` 字段 + 静态用例数组），不要先用 vector 再改。

## 验证信号

以下信号来自该算子 UT 开发会话（x86_64 host 侧编译运行，被测为 ascend950 包 tiling 源）：

- 修复前：tiling 测试多个构造点报 `no matching function for call to 'gert::StorageShape::StorageShape(...)'`，候选列表显示仅 initializer_list 构造可用（来源平台：950；证据等级：tool_observed）。
- 头文件锚点：`include/exe_graph/runtime/storage_shape.h:27` 为唯一的形状构造重载（来源平台：950；证据等级：artifact_verified）。
- 修复后：UT 编译通过；`./test_op_host` 运行 16 个用例全部通过（Conv2dInfershapeTest 7 + TilingKey 判定 2 + Tiling 7），含 dynamic_shape 等边界用例（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 构造函数集合按 CANN 9.1.0 头文件；后续版本若新增 vector 构造或逐维 Set 接口，本卡写法仍兼容但不再是唯一解（未验证）。
- initializer_list 字段模式要求测试用例数据为静态存储期；纯运行期生成的形状不适用（须另找 API，未验证）。

**未决项**：

- 无。
## 坏实践

- `{vec.begin(), vec.end()}` 当作"转 initializer_list"的写法（编译失败）。
- 为了绕开构造限制去 memcpy/pun StorageShape 内部（本轨迹未犯，列为禁项）。

## 误诊

- 无。

## 增量（两条并行修复路线与运行期构造 API）

两个独立 trace（scatter 算子与交叉熵损失算子的 op_host UT，均为 CANN 9.1.0 头文件、x86_64 UT 主机 / ascend950 编译目标）以不同写法撞到同一构造约束（`{vector, vector}` 花括号初始化报 `could not convert '<brace-enclosed initializer list>' to 'gert::StorageShape'`），并各自落地一条修复路线：

**路线一：helper 形参改 initializer_list + 直接构造语法**（交叉熵实例；与主卡"字段改 initializer_list"同族，但不动用例结构体，只改 helper 签名）：

```cpp
// 修复前：helper 形参是 const vector<int64_t>&，构造用拷贝列表初始化（必报错）
static TilingInfo RunTiling(const vector<int64_t>& inputShape, ...)
{
    StorageShape xShape = {inputShape, inputShape};

// 修复后：形参改 std::initializer_list<int64_t>，构造改直接初始化
static TilingInfo RunTiling(std::initializer_list<int64_t> inputShape, ...)
{
    StorageShape xShape(inputShape, inputShape);
```

调用侧 `RunTiling({4, 10}, {4}, "none", -100)` 花括号列表直接绑定到 initializer_list 形参，无需改动。

**路线二：默认构造 + Mutable 访问器 + SetDim 逐维填充**（scatter 实例；覆盖主卡失效边界中"纯运行期生成的形状不适用"的缺口）：

```cpp
// 从 std::vector 构造 StorageShape（origin == storage 的常用形态）
// 注：StorageShape 仅提供 initializer_list 构造，不接受 vector，需逐维设置
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    auto& origin = s.MutableOriginShape();
    auto& storage = s.MutableStorageShape();
    origin.SetDimNum(dims.size());
    storage.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        origin.SetDim(i, dims[i]);
        storage.SetDim(i, dims[i]);
    }
    return s;
}
```

即：`gert::Shape` 提供 `SetDimNum(const size_t)`/`SetDim`、`StorageShape` 提供 `MutableOriginShape()/MutableStorageShape()` 访问器——不开洞 initializer_list 约束的运行期逐维构造路径在 CANN 9.1.0 头文件中实际可用（主卡曾标注"未见，未验证"，本增量以编译+UT 运行证实）。参数化用例把 shape 存 `std::vector<int64_t>`（运行期数据）时优先该路线；shape 为静态字面量时 initializer_list 路线更短。

验证信号（复现增量，两个独立 trace）：

- scatter 实例：修复前 4 处 `error: could not convert '{tc...dataShape, ...}' from '<brace-enclosed initializer list>' to 'gert::StorageShape'`（make Error 2）；头文件锚点 `storage_shape.h:27` 仅 initializer_list 双参构造、`shape.h:40` 无 vector 构造但提供 SetDimNum/SetDim；替换 MakeStorageShape 后编译通过、同套件修完 tilingKey 期望后 8 tests PASSED（来源平台：950 编译目标/x86_64 UT 主机；证据等级：tool_observed）。
- 交叉熵实例：修复前三处同型报错；helper 形参改 initializer_list + 直接构造后 `make -j` 成功、UT 18/18（9 InferShape + 9 Tiling）通过、run.sh 路径复跑同样 18/18（来源平台：950 编译目标/x86_64 UT 主机（SOC_VERSION=ascend950 配置）；证据等级：tool_observed）。

失效边界（复现增量）：

- 路线一依赖 `std::initializer_list` 引用调用处临时数组、只能立即用于构造，不能存起来跨函数/跨语句使用；需要持久化 shape 的场景用路线二。
- 路线二按 origin == storage（连续布局）构造；需要 origin ≠ storage 的场景（非连续 format）需在辅助函数中扩展两份不同维度。
- 两条路线均按 CANN 9.1.0 头文件验证；其他版本签名未验证（未验证，推断可迁移：以实际头文件为准）。

## 增量（第三条 AppendDim 路线与 infershape 输出留空模式）

另一独立轨迹并入三个独立 trace（指数族 Elementwise / TensorList 多张量 / SwiGlu 类算子的 op_host UT，均 CANN 9.1.0 头文件、ascend950 编译目标）的同型复现与两条新内容：

**路线三：默认构造 + AppendDim 逐维追加**（与路线二等价但更简——不需要先 SetDimNum；`gert::Shape` 提供 `AppendDim`）：

```cpp
// 辅助函数：从 vector 构造 StorageShape
static StorageShape MakeStorageShape(const vector<int64_t>& dims)
{
    StorageShape shape;
    for (auto d : dims) {
        shape.MutableOriginShape().AppendDim(d);
        shape.MutableStorageShape().AppendDim(d);
    }
    return shape;
}
```

配套事实：`gert::Shape` 虽有 `Shape(const std::initializer_list<int64_t>&)`，同样没有迭代器区间构造——`{{vec.begin(), vec.end()}, {vec.begin(), vec.end()}}` 花括号里放迭代器对会被当成一个 initializer_list 元素推导失败，编译器报"无法从 brace-enclosed initializer list 转换为 gert::StorageShape"；报错不直接指出"缺少该构造重载"，容易误以为是嵌套层级写错（另有同文件 `initializer_list` 漏写 `std::` 前缀报 `'initializer_list' was not declared in this scope` 的伴生小错，补 `using namespace std;` 或写全限定名即解）。

**路线一扩展（infershape 用例组织）**：输出 shape 不在测试里手工推导（如 dim 折半），而是留给 infershape 一个默认构造的空 `StorageShape` 让被测函数填写——既绕开 initializer_list 不能从 vector 构造的限制，又让用例只声明"输入是什么、期望输出是什么"，不在测试侧复刻被测逻辑（复刻逻辑会引入第二个可能出错的实现）：

```cpp
static void RunInfershape(
    initializer_list<int64_t> inputShape,
    DataType dtype,
    int64_t dim,
    ge::graphStatus expectedResult,
    const std::vector<std::vector<int64_t>>& expectOutputShape = {})
{
    // 构造 input StorageShape (origin == storage, 连续布局)
    StorageShape inShape(inputShape, inputShape);
    // output shape 空白, 由 infershape 函数推导
    StorageShape outShape;

    vector<InfershapeContextPara::TensorDescription> inputDesc = {
        {inShape, dtype, FORMAT_ND}
    };
    vector<InfershapeContextPara::TensorDescription> outputDesc = {
        {outShape, dtype, FORMAT_ND}
    };
    vector<InfershapeContextPara::OpAttr> attrs = {
        {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim)}
    };

    InfershapeContextPara para(OP_NAME, inputDesc, outputDesc, attrs);
    ExecuteTestCase(para, expectedResult, expectOutputShape);
}
```

（tiling 用例同理可复用输入 shape 作输出描述：tiling 函数不消费 output shape，占位合法。）

### 复现增量验证信号

- TensorList 多张量 trace：修复前 `make -j8` 报 `could not convert '<brace-enclosed initializer list>' to 'gert::StorageShape'`（多处）；改 MakeStorageShape（SetDimNum/SetDim 逐维填充）后 UT 编译通过并运行 21/21（含 listLen=2/4/8 多张量用例）（来源平台：950；证据等级：tool_observed）
- 指数族 trace：修复前 `make` 在测试目标处失败，报 "could not convert '{{...begin(), ...end()}, ...}' from '<brace-enclosed initializer list>' to 'gert::StorageShape'"；头文件锚点确认 StorageShape 仅有默认构造与双 initializer_list 构造、Shape 提供 AppendDim/SetDim；改 AppendDim 辅助构造后 `[100%] Built target <op>_op_host_ut`，后续 tiling/infershape 用例可执行（来源平台：950 环境主机侧编译；证据等级：tool_observed）
- SwiGlu 类 trace：修复前 tiling/infershape 两个测试文件编译失败（迭代器对构造 + `initializer_list` 漏写 `std::` 前缀）；helper 形参改 initializer_list 直传 + infershape 输出留空后 `[100%] Built target <op>_op_host_ut`，InferShape 套件 11/11 全部 OK（覆盖 dim=-1/0/-2/1、fp32/bf16、4D，以及奇数 dim、rank=1、dim 越界 4 个预期失败用例）（来源平台：950；证据等级：tool_observed）

### 复现增量失效边界

- AppendDim 与 SetDimNum/SetDim 两条运行期路线等价，均需双填 origin/storage（只填一侧时 tiling 读到另一套 shape，行为与真实框架不一致）。
- 编译期字面量 shape 直接双 initializer_list 构造最短（`StorageShape xShape({1024, 1024}, {1024, 1024});`），不需要辅助函数。
- dims 数超过 `kMaxDimNum` 时 `Shape(initializer_list)` 直接返回空 shape（头文件行为），本卡未验证该边界。
- 运行时动态 rank（无法静态写 initializer_list）场景的替代途径未尝试、无证据。

## 增量（12 个独立 trace 复现：错误形态补全与级联噪声判别）

另一独立轨迹并入 12 个独立 trace（GRU/TopK/Unique/MoE 重排/Transpose/RMSNorm 动态量化/conv3d 反传 filter/resize 双线性/gather/DepthwiseConv2d 等算子的 op_host UT 首编，CANN 9.1.0 与 9.2.0）的同根因复现，补齐错误形态分类与两个新判别点：

**1. 错误形态分类（四种失败写法 → 报错签名）**：

| 失败写法 | 报错签名 |
|---|---|
| `StorageShape xShp(xShape, xShape)`（vector 圆括号/花括号） | `no matching function for call to 'gert::StorageShape::StorageShape(const std::vector<long int>&, const std::vector<long int>&)'`（note: candidate 仅 initializer_list 版本） |
| `{{vec.begin(), vec.end()}, ...}`（迭代器对，主卡/上一独立开发会话已载） | `could not convert '<brace-enclosed initializer list>' to 'gert::StorageShape'` |
| `StorageShape x1Shape = {M, N}`（**单组 dims 单层花括号**） | `could not convert '{...}' from '<brace-enclosed initializer list>' to 'gert::StorageShape'`（多行同报） |
| `xStorageShape.SetDimNum(...)` / `xStorageShape.SetDim(i, ...)`（维度 API 误在 StorageShape 上直调） | `no member named 'SetDimNum'` / `no member named 'SetDim'`（错误混在大量 CANN 头文件 deprecation pragma 输出里，须 grep `error:` 而非肉眼看尾部） |

第三种形态的正确写法是**双层嵌套**（StorageShape 是 storageDims+originDims 二元组，外层给两个 Shape 成员、内层各给一组 dims）：

```cpp
    // 修复前（单组 dims 初始化，编译失败）：
    gert::StorageShape x1Shape = {static_cast<int64_t>(param.M), static_cast<int64_t>(param.N)};
    // 修复后（storageDims 与 originDims 双 Shape 嵌套初始化）：
    gert::StorageShape x1Shape = {{static_cast<int64_t>(param.M), static_cast<int64_t>(param.N)},
                                  {static_cast<int64_t>(param.M), static_cast<int64_t>(param.N)}};
```

第四种形态说明维度写入 API 在 `gert::Shape` 上、StorageShape 的可写入口只有 `MutableShape()`（origin）与 `MutableStorageShape()`（storage）——与库内 [StorageShape 需解包为 Shape 再用维度 API](gert_storageshape_unwrap_dim_api.md) 是同一接口分布在读/写两个方向的呈现。

**2. 级联噪声判别**：StorageShape 构造失败会连带出 `TilingContextPara` 构造函数"argument 4 from 'int'"不匹配一类报错，看起来像 faker 构造参数个数错了，实际是前者失败的级联错误——先修第一个 StorageShape 错误再看剩余报错（后者大概率是级联噪声）。

**3. 单变量表达式花括号的重载决议陷阱**：`{cfg.numShape.GetDim(1)}` 这种"变量表达式花括号"构造单元素 initializer_list 在重载决议里不稳定（同样报 no matching function）；最稳妥是把维度先取出到局部 `int64_t` 变量再用字面量列表构造，或直接拷贝一个已构造好的 StorageShape（拷贝构造可用）。取维度本身须经 `shape.GetStorageShape().GetDim(i)`（见 unwrap 卡）。

**4. 空 shape 的两种等价写法**：infershape 用例的输出留空可用默认构造 `gert::StorageShape outShape;`（独立 trace 路线）或空双列表 `gert::StorageShape yEmpty = {{}, {}};`（本增量 TopK trace 形态），两者等价；不要写 `StorageShape({}, {})` 传空 vector（同样不匹配）。

### 复现增量验证信号

来源说明：12 个独立 trace 的 op_host UT 首编修复（修复前均稳定复现上表报错签名之一，修复后编译通过并进入测试执行）；以下为代表性实测。

- GRU trace（CANN 9.2.0）：修复前 `no matching function for call to 'gert::StorageShape::StorageShape(const std::vector<long int>&, ...)'`（note 列出唯一 initializer_list 候选）；辅助函数形参改 `std::initializer_list<int64_t>` 后 op_host UT 编译通过、14/14 PASSED（tiling 9 + infershape 5）（来源平台：950；证据等级：tool_observed）。
- conv3d 反传 filter trace：修复前多条 `could not convert '{param.xShape, param.xShape}' from '<brace-enclosed initializer list>' to 'gert::StorageShape'` + 级联的 TilingContextPara 构造不匹配；新增 MakeStorageShape（MutableOriginShape/MutableStorageShape + SetDimNum/SetDim 双填）后 UT 目标编译通过（`[100%] Built target ..._op_host_ut`）、全量 10/10（来源平台：x86 UT 主机；证据等级：tool_observed）。
- Transpose trace：修复前 `could not convert '{{xShape.begin(), xShape.end()}, ...}'`（tiling:58/61、infershape:37 三处）；MakeStorageShape 替换后 `make -j` 编译链接通过（仅剩 CANN 头文件弃用 pragma 警告），UT 23/23 PASSED（InferShape 8 + Tiling 15），run.sh 汇总 PASS（来源平台：950（Ascend950 UT 配置，x86_64 宿主执行）；证据等级：tool_observed）。
- 单组 dims 形态 trace：双层嵌套 + CMake 补 context_builder 目录后 UT 10/10 通过（5 Tiling + 5 InferShape），清理重建复跑仍 10/10；后续迭代同框架扩到 20/20（来源平台：host 侧 UT 编译（x86 g++，未上板）；证据等级：tool_observed）。
- DepthwiseConv2d trace（CANN 9.2.0）：修复前 `no member named 'SetDimNum'`/构造不匹配（错误混在 deprecation pragma 输出里）；MakeStorageShape 双填后编译通过，最终 UT 27/27（10 InferShape + 6 TilingSuccess + 11 TilingFailure），InferShape 用例 yShape 推导值全部符合预期（来源平台：950；证据等级：tool_observed）。
- 参数化/错误路径扩展 trace：MakeStorageShape（AppendDim 双填）落地后 UT 58/58 PASS（13 InferShape + 33 参数化 Tiling + 8 错误路径 TilingErrorTest + 独立测试），其中 8 条错误路径用例（1D/3D/M=0/N=0/fp32/int8/double/缺 epsilon）全部按 `ExecuteTestCase(..., ge::GRAPH_FAILED)` 正确拒绝；`bash run.sh` 清理重建从零复跑同样 58/58（来源平台：950；证据等级：tool_observed）。
- 复用性：同一 MakeStorageShape 写法被后续 TilingKey 探针（临时 gtest 用例）与用例扩展（47/47）直接复用，无再编译问题（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 构造签名与接口分布验证于 CANN 9.1.0/9.2.0 的 exe_graph 头；其他版本若增加 vector/迭代器构造或把 SetDimNum/SetDim 提到 StorageShape 上，本卡过时（未验证，推断可迁移）。
- MakeStorageShape 各 trace 版本均按 origin == storage（连续布局）双填；需要故意构造 storage≠origin（带 padding 的物理布局）的 UT 需扩展 helper（未验证）。
- 空 shape 默认构造与 `{{}, {}}` 两种写法的等价性以编译+运行通过为准，未做布局级核对。

### 复现增量坏实践

- 按普通 shape 容器的直觉用单组 dims（单层花括号）初始化 StorageShape——它是 storageDims+originDims 二元组，必须两层。
- 从 `cfg.tokensShape.GetDim(0)` 这类"变量表达式花括号"构造单元素 initializer_list——重载决议不稳定，先取局部变量。
- 编译报错里出现 faker 构造函数参数不匹配时先去数参数个数——先修第一个 StorageShape 错误，剩余大概率是级联噪声。
- faker 头文件找不到时去改 faker 的 include 路径——应在自己的 UT CMake 里补 CANN 头目录（context_builder 子目录缺失见库内 [UT 编译缺 context_builder include 路径](ut_context_builder_include_path.md)）。
- 在 DO_TILING 等 faker 框架宏内部用 `//` 行注释——会吃掉行尾 `\` 续行符导致宏断裂，报出一串 `tilingRet/tilingContext 未声明` 的错位信息（某 trace 实际踩到，删除注释解决）。

### 复现增量遗留问题

- 无（12 个 trace 均闭环到编译通过 + UT 运行；个别 trace 的运行期失败为独立根因，已各自成卡）。