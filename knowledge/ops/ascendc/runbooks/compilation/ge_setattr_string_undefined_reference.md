---
type: "Runbook"
title: "GE SetAttr 字符串属性链接失败，改用 AttrValue"
description: "ge::Operator::SetAttr(const string&, const string&) 在 CANN 9.2.0 头文件有声明但发布链接库未导出定义，编译通过链接失败 undefined reference；字符串属性须用 ge::AttrValue + AscendString 装箱，误以为缺库补链 metadef 无效。"
tags: [task.debugging, topic.compilation, topic.implementation]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# GE SetAttr 字符串属性链接失败，改用 AttrValue

## 现象

为自定义算子 BenchHarnessScatter（scatter 类，3 输入 data/indices/updates → 1 输出 y，属性 dim 为 int、reduce 为字符串枚举 update/add/multiply/amax/amin）开发 GEIR ST（图模式系统测试）时，把 tests/geir 的图适配器从 2 输入 add 通用模板重写为 3 输入 scatter 适配器后，执行构建：

```bash
cd operators/BenchHarnessScatter_package/tests/geir
source ../../../../docs/develop/env.sh          # CANN 9.2.0 环境
export GEIR_VENDOR_DIR="$(./run.sh stage)"      # 安装 staging vendor
./run.sh build                                  # cmake 构建 geir_route/geir_suite/geir_infer
```

编译全部通过，但链接 geir_route 时失败（BUILD_EXIT=2），报错原文：

```text
/usr/bin/ld: CMakeFiles/geir_route.dir/route_main.cpp.o: in function `geir_test::adapter::BuildGraph(ge::Graph&, geir_test::adapter::Case const&, bool, ge::DataType)':
route_main.cpp:(.text._ZN9geir_test7adapter10BuildGraphERN2ge5GraphERKNS0_4CaseEbNS1_8DataTypeE[_ZN9geir_test7adapter10BuildGraphERN2ge5GraphERKNS0_4CaseEbNS1_8DataTypeE]+0x2c1): undefined reference to `ge::Operator::SetAttr(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
collect2: error: ld returned 1 exit status
gmake[3]: *** [CMakeFiles/geir_route.dir/build.make:97: geir_route] Error 1
gmake[2]: *** [Makefile2:87: CMakeFiles/geir_route.dir/all] Error 2
gmake[1]: *** [Makefile2:94: CMakeFiles/geir_route.dir/rule] Error 2
gmake: *** [Makefile:124: geir_route] Error 2
```

报错调用点是适配器 `BuildGraph` 里给算子节点设字符串属性的一行 `node.SetAttr("reduce", test_case.reduce);`（`test_case.reduce` 为 `std::string`）；同一函数里 `node.SetAttr("dim", static_cast<int64_t>(test_case.dim));`（整型属性）不报错。当时 geir_route 只链接 CANN 的常规库集（graph/ge_runner/graph_base/opp_registry 等）。

## 适用场景

- 工程形态：算子仓开发流程中的 GEIR ST / 图模式测试适配器开发；也适用于任何用 GE C++ API 编程式构图（`ge::Operator` / `ge::OperatorFactory`）的 host 侧代码——ST 适配器、构图辅助工具、图构建脚本。
- API 路径：`ge::Operator::SetAttr`（`$ASCEND_HOME_PATH/x86_64-linux/include/graph/operator.h`）、`ge::AttrValue::SetAttrValue(const ge::AscendString&)`（`graph/attr_value.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR 开发环境，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——host 侧 GE API 行为，与具体算子和 kernel 实现无关，同一 CANN 版本内通用。
- 路线优先级：recommended_fastpath（AttrValue 装箱是该版本验证过可链接的字符串属性通路）。
- 明确不适用 / 未验证场景：
  - `SetAttr(const std::string&, const std::vector<std::string>&)`（同头文件声明的 vector 重载）是否同样链接失败未验证；需要时应先小样验证或走 `AttrValue::SetAttrValue(const std::vector<AscendString>&)`。
  - 带 `ATTRIBUTED_DEPRECATED` 标记的 `(const char_t*, const char_t*)` 旧重载可用性未验证。
  - 属性字符串值是否被 kernel 侧原样读到：本轨迹未验证（见失效边界）。

## 根因

CANN 9.2.0 的 `graph/operator.h` 声明了 `Operator &SetAttr(const std::string &name, const std::string &attr_value)`，但发布的链接库（graph/ge_runner/graph_base/metadef 等）均未导出该符号的定义，因此调用它的代码编译通过、链接失败。

机制链分三步：

1. **编译通过**：重载决议命中头文件里的 `(string, string)` 声明（同版本 cann-9.2.0 头文件可核对，声明位于 operator.h 的 SetAttr 重载区），无编译错误；
2. **链接失败**：链接器在被链接的全部 CANN 库里找不到该符号定义 → `undefined reference to ge::Operator::SetAttr(string const&, string const&)`；
3. **可用通路**：字符串属性改走 `ge::AttrValue` 装箱——`AttrValue::SetAttrValue(const AscendString&)`（`graph/attr_value.h` 声明）设置值，再经 `SetAttr(const std::string&, AttrValue &&)`（`graph/operator.h` 声明）提交，这两个符号可正常解析。

会话内的诊断命令（grep 头文件重载集，verbatim）：

```bash
grep -rn "SetAttr.*String\|SetAttr.*char\|SetAttr.*AscendString" \
  /workspace/…/Ascend/cann-9.2.0/x86_64-linux/include/graph/operator.h 2>/dev/null | head -10
echo "==="
grep -n "SetAttr" \
  /workspace/…/Ascend/cann-9.2.0/x86_64-linux/include/graph/operator.h 2>/dev/null | head -20
```

输出确认可用重载为 `int64_t/int32_t/uint32_t/std::vector<int64_t>/vector<int32_t>/vector<uint32_t>/initializer_list<int64_t>/float32_t/vector<float32_t>/AttrValue &&`，以及 `ge::AttrValue` 的 `SetAttrValue(const AscendString&)`。注意该 grep 模式用大写 `String`，匹配不到小写 `std::string` 的重载声明，容易得出"没有 string 重载"的半对结论——声明其实在，缺的是库里的定义；判定"符号是否可链接"应以链接器/`nm -D` 结果为准，不以 grep 命中为准。

## 解决配方

修复分两处（同一文件 `operators/<OP>_package/tests/geir/adapter/generated_plan.h`）。

**1）文件头部 include 区增加 `attr_value.h`**（verbatim）：

```cpp
#include "attr_value.h"
#include "operator_factory.h"
#include "support/geir_adapter.h"
#include "support/geir_common.h"
#include "support/geir_manifest.h"
```

**2）`BuildGraph` 里字符串属性改 AttrValue 装箱。**

修复前（编译通过、链接失败的原写法——`BuildGraph` 属性设置段；同函数整型属性 dim 不受影响）：

```cpp
    // Set attributes.
    node.SetAttr("dim", static_cast<int64_t>(test_case.dim));
    node.SetAttr("reduce", test_case.reduce);
```

修复后：

```cpp
    // Set attributes.
    node.SetAttr("dim", static_cast<int64_t>(test_case.dim));
    ge::AttrValue reduce_av;
    reduce_av.SetAttrValue(
        ge::AscendString(test_case.reduce.c_str()));
    node.SetAttr("reduce", std::move(reduce_av));
```

`BuildGraph` 其余部分（OperatorFactory 建节点、make_data 建 Data 源、SetInput/UpdateOutputDesc）不变，函数级完整 before/after 见下节。

## 完整代码

修复前的 `BuildGraph` 完整函数（verbatim 取自修复前最近一次完整写入的代码状态，仅属性设置段为 string 直传写法）：

```cpp
inline GraphBinding BuildGraph(ge::Graph &graph, const Case &test_case,
                                bool dynamic_inputs,
                                ge::DataType output_decl_dtype)
{
    const std::string node_name = SafeNodeName(test_case.id);
    ge::Operator node =
        ge::OperatorFactory::CreateOperator(node_name.c_str(), kOpType);

    // Helper to create a Data source op for each input.
    auto make_data = [&](const char *port_name, int64_t index,
                         const std::vector<int64_t> &shape,
                         DType dtype) -> ge::Operator {
        const std::string data_name = node_name + "_" + port_name;
        const ge::TensorDesc desc = dynamic_inputs
            ? MakeUnknownRankDesc(ToGeType(dtype))
            : MakeDesc(shape, ToGeType(dtype));
        ge::Operator data =
            ge::OperatorFactory::CreateOperator(data_name.c_str(), "Data");
        data.SetAttr("index", index);
        data.UpdateInputDesc("x", desc);
        data.UpdateOutputDesc("y", desc);
        graph.AddOp(data);
        return data;
    };

    ge::Operator data_op = make_data("data", 0, test_case.data_shape,
                                     test_case.data_dtype);
    ge::Operator indices_op = make_data("indices", 1, test_case.indices_shape,
                                        test_case.indices_dtype);
    // updates: shape == indices_shape, dtype == data_dtype
    ge::Operator updates_op = make_data("updates", 2, test_case.indices_shape,
                                        test_case.data_dtype);

    node.SetInput(kInputData, data_op);
    node.SetInput(kInputIndices, indices_op);
    node.SetInput(kInputUpdates, updates_op);

    // Set attributes.
    node.SetAttr("dim", static_cast<int64_t>(test_case.dim));
    node.SetAttr("reduce", test_case.reduce);

    node.UpdateOutputDesc(kOutputY, MakeUnknownRankDesc(output_decl_dtype));
    graph.SetInputs(std::vector<ge::Operator>{data_op, indices_op, updates_op})
        .SetOutputs(std::vector<ge::Operator>{node});
    return {node, node_name};
}
```

修复后的 `BuildGraph` 完整函数（verbatim 取自该会话终版代码状态，`// Set attributes.` 段为 AttrValue 装箱写法）：

```cpp
inline GraphBinding BuildGraph(ge::Graph &graph, const Case &test_case,
                                bool dynamic_inputs,
                                ge::DataType output_decl_dtype)
{
    const std::string node_name = SafeNodeName(test_case.id);
    ge::Operator node =
        ge::OperatorFactory::CreateOperator(node_name.c_str(), kOpType);

    // Helper to create a Data source op for each input.
    auto make_data = [&](const char *port_name, int64_t index,
                         const std::vector<int64_t> &shape,
                         DType dtype) -> ge::Operator {
        const std::string data_name = node_name + "_" + port_name;
        const ge::TensorDesc desc = dynamic_inputs
            ? MakeUnknownRankDesc(ToGeType(dtype))
            : MakeDesc(shape, ToGeType(dtype));
        ge::Operator data =
            ge::OperatorFactory::CreateOperator(data_name.c_str(), "Data");
        data.SetAttr("index", index);
        data.UpdateInputDesc("x", desc);
        data.UpdateOutputDesc("y", desc);
        graph.AddOp(data);
        return data;
    };

    ge::Operator data_op = make_data("data", 0, test_case.data_shape,
                                     test_case.data_dtype);
    ge::Operator indices_op = make_data("indices", 1, test_case.indices_shape,
                                        test_case.indices_dtype);
    // updates: shape == indices_shape, dtype == data_dtype
    ge::Operator updates_op = make_data("updates", 2, test_case.indices_shape,
                                        test_case.data_dtype);

    node.SetInput(kInputData, data_op);
    node.SetInput(kInputIndices, indices_op);
    node.SetInput(kInputUpdates, updates_op);

    // Set attributes.
    node.SetAttr("dim", static_cast<int64_t>(test_case.dim));
    ge::AttrValue reduce_av;
    reduce_av.SetAttrValue(
        ge::AscendString(test_case.reduce.c_str()));
    node.SetAttr("reduce", std::move(reduce_av));

    node.UpdateOutputDesc(kOutputY, MakeUnknownRankDesc(output_decl_dtype));
    graph.SetInputs(std::vector<ge::Operator>{data_op, indices_op, updates_op})
        .SetOutputs(std::vector<ge::Operator>{node});
    return {node, node_name};
}
```

## 设计决策

- **为何直接改 API 用法而不是继续排查库**：补链 metadef 后错误逐字重现，说明该符号不在已发布的库里；头文件分析给出 `AttrValue` 是声明的字符串属性通道，换用法是唯一有出路的修复。
- **备选与放弃原因**：`ATTRIBUTED_DEPRECATED` 的 `(const char_t*, const char_t*)` 旧重载未尝试（带弃用标记，且当时无法确认可链接性）；`vector<string>` 重载同样未验证。
- **路线档位**：recommended_fastpath——纯 host 侧 API 通路修复，不涉精度与性能取舍。

## 系统信息

- 硬件/软件：Ascend950PR 单卡开发环境，CANN 9.2.0（`$ASCEND_HOME_PATH/x86_64-linux/include/graph/*.h`）。
- 相关文件（与事件流实际读写一一对应）：
  - `operators/<OP>_package/tests/geir/adapter/generated_plan.h`——GEIR 图适配器（本次修复点）。
  - `operators/<OP>_package/tests/geir/CMakeLists.txt`——geir_route 链接配置（误诊尝试中追加过 metadef，修复后保留无害但非修复手段）。
  - `$ASCEND_HOME_PATH/x86_64-linux/include/graph/operator.h`、`graph/attr_value.h`——API 声明（诊断时 grep 核对）。

## 预防

- 用 GE C++ API 设字符串属性时，直接写 AttrValue 装箱形态，不要写 `SetAttr(name, std::string 值)`。
- 遇到 CANN 头文件类库的 undefined reference，先用 `nm -D <lib>` / `c++filt` 确认符号是否真的被导出，再决定补库还是换 API；grep 头文件时注意大小写（`String` 匹配不到 `std::string`）。

## 验证信号

以下信号来自该算子 GEIR ST 开发会话：链接失败 → 头文件诊断 → AttrValue 改写 → 重建，随后三支 ST 二进制（route/infer/reject）全链路运行。

- 修复前：`./run.sh build` 链接 geir_route 报 `undefined reference to ge::Operator::SetAttr(std::string const&, std::string const&)`，BUILD_EXIT=2（来源平台：950；证据等级：tool_observed）。
- 误诊尝试后：CMakeLists 给 geir_route 追加 `metadef`、`dl` 重新构建，同一 undefined reference 逐字重现（来源平台：950；证据等级：tool_observed）。
- 修复后：`./run.sh build` 三个目标全部 Built target（geir_route/geir_suite/geir_infer），BUILD_EXIT=0（来源平台：950；证据等级：tool_observed）。
- 修复后路由探测：geir_route 以 AttrValue 属性构图成功，日志出现 `GRAPH-ADD mode=route ...` 与 `CallInferFunc:Op n_route_target[BenchHarnessScatter] Call InferShapeFuncV2`（来源平台：950；证据等级：tool_observed）。
- 修复后 infer：`INFER-SUMMARY route=V2 total=10 pass=10 fail=0`、`RUN-EXIT status=0`（InferShapeV2/InferDataType 回调从 staging 的 libcust_opmaster_rt2.0.so 加载并全部通过）（来源平台：950；证据等级：tool_observed）。
- 修复后 reject：`GEIR-SUMMARY total=1 pass=1 fail=0 skip=0`、`RUN-EXIT status=0`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 CANN 9.2.0（x86_64-linux include 布局）验证；其他小版本是否补上该符号定义未验证（若补上，原写法也能链接，本卡修复写法仍兼容）。
- 本卡验证到"链接成功 + 图构建 / shape-dtype 推导 / 端口拒绝路径运行"层面；`reduce` 字符串属性值是否被 kernel 侧原样读到未验证——该会话中全图执行（RunGraph）对 float 用例失败（设备错误），未走到属性消费路径。
- `vector<string>` 与 `const char_t*` 旧重载未验证（见适用场景）。
- 头文件内该声明可在同版本 `graph/operator.h` 的 SetAttr 重载区核对（挖掘时已复核），其他版本行号可能不同。

**未决项**：

- `SetAttr(const std::string&, const std::vector<std::string>&)` 是否同样链接失败未验证。
- 属性值端到端到达 kernel 的验证缺失：本轨迹 RunGraph 对 float 用例失败（设备错误）挡住了这条验证路，待该失败根因解决后补验。
## 坏实践

- 见到 undefined reference 先补库链接（本例加 metadef）而不核对符号是否被任何库导出。
- 用大小写敏感的 grep（`SetAttr.*String`）扫头文件就下"没有 string 重载"的结论——声明用的是小写 `std::string`，会被该模式漏掉。

## 误诊

- "缺库"假设：把 undefined reference 当作链接库缺失，给 geir_route 补链 metadef 后错误逐字重现——该符号在发布的 CANN 库中根本没有定义，问题在 API 选择而非链接配置。
