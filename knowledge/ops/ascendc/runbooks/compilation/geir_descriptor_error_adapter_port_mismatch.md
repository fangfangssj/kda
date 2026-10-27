---
type: "Runbook"
title: "GEIR 路由 DESCRIPTOR_ERROR：适配器端口未对齐 OpDef"
description: "自定义算子 GEIR 路由探测报 DESCRIPTOR_ERROR/path=UNKNOWN 时，先比对测试适配器端口与 OpDef——UpdateOutputDesc 找不到端口名只打 WARNING，失败延迟到 SHAPE-INIT 才暴露；grep 证据日志的 Cannot find the desc named 行定位缺失端口。"
tags: [task.testing, topic.compilation, topic.implementation]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# GEIR 路由 DESCRIPTOR_ERROR：适配器端口未对齐 OpDef

## 现象

在一个 Ascend950PR（CANN 9.2.0）自定义算子的 GEIR ST 验收中：算子是 3 输入的 scatter 类算子（输入 `data`/`indices`/`updates`，必选属性 `dim`、可选属性 `reduce`，输出 `y`，已在 `op_host/*_def.cpp` 用 `OP_ADD` 注册并打包进 vendor staging）。验收流程从当前源码重建 staging（`tests/geir/run.sh stage`）、构建三个测试目标 `geir_route`/`geir_suite`/`geir_infer`（全部编译成功，且 link.txt 显示不链接生产源码、隔离通过），然后执行路由探测：

```bash
cd operators/<Op>_package/tests/geir
./run.sh route
```

输出为：

```
ROUTE op=BenchHarnessScatter path=UNKNOWN status=DESCRIPTOR_ERROR probe_exit=1
```

且此前的探测行显示失败发生在 SHAPE-INIT 阶段（`AddGraph` 本身成功）：

```
ROUTE-PROBE node=n_route_target gid=1 stage=SHAPE-INIT pipeline=FAIL value=FAIL
```

检查测试适配器 `tests/geir/adapter/generated_plan.h` 后发现：它是从通用样例模板复制来的 **2 输入 add 适配器**——端口常量是 `x`/`y`/`z`，期望值按 `x + y` 计算，完全没有按算子契约适配（真实契约是 3 输入 + 2 属性）。

## 适用场景

- 工程形态：自定义算子（aclnn 算子包）GEIR/graph 模式 ST 开发与验收——用 `ge::OperatorFactory::CreateOperator` + `SetInput`/`UpdateOutputDesc` 构图、`ge::Session::AddGraph` 加图、`run.sh route` 做 V1/V2 路由探测的测试工程。
- API 路径：`ge::OperatorFactory::CreateOperator`、`ge::Operator::SetInput`/`UpdateOutputDesc`、`ge::Session::AddGraph`、vendor staging（`ASCEND_CUSTOM_OPP_PATH`）下的 OpDef 注册（`OP_ADD`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——GE 按端口名解析 OpDef 契约的行为与具体算子无关，任何走 graph 模式测试/构图调用的自定义算子工程都适用。
- 路线优先级：recommended_fastpath（遇到 DESCRIPTOR_ERROR 时的第一判别步骤；静态比对 5 分钟内可完成）。
- 明确不适用 / 未验证场景：
  - aclnn 单算子直调路径（不经 GE 构图，不会出现该信号）。
  - `SetInput` 侧端口名错误的行为未在日志中观测到（只观测到 `UpdateOutputDesc` 侧 WARNING，见失效边界）。
  - 其他 CANN 版本上 WARNING 文案与失败阶段是否一致未验证（推断可迁移）。

## 判别方法

根因（单句）：**GEIR 测试适配器构图用的端口名与 OpDef 注册的输入/输出端口名不一致（且未设置必选属性），GE 对不存在的输出描述符只打 WARNING 不报错，失败延迟到 SHAPE-INIT 才以 DESCRIPTOR_ERROR 暴露。**

判别步骤：

1. **拿到原始 WARNING**：`run.sh route` 的 stdout 只有汇总行，必须 grep 证据日志（默认写到 `build/evidence/route.debug.log`）：

```bash
grep -iE "Cannot find the (input|output)" build/evidence/route.debug.log
grep -E "^(GRAPH-|SHAPE-|ROUTE)" build/evidence/route.debug.log
```

本例命中（原文）：

```
[WARNING] GE(...):... [op_desc.cc:777]... UpdateOutputDesc:[Update][OutputDesc] Cannot find the output desc named z
GRAPH-ADD mode=route case=route-target node=n_route_target gid=1 inputs={x:[2,3]/DT_FLOAT,y:[2,3]/DT_FLOAT}
SHAPE-INIT case=route-target outputs={z:{shape:[],dtype:DT_FLOAT}}
```

三条线互相印证：构图只喂了 `x`/`y` 两个输入、输出描述符 `z` 在 OpDef 中不存在、于是输出 shape 推导为空 `[]`，SHAPE-INIT 失败 → `status=DESCRIPTOR_ERROR`、`path=UNKNOWN`。

2. **静态比对适配器端口与 OpDef**。出错适配器 `tests/geir/adapter/generated_plan.h`（未适配的通用模板，关键内容 verbatim）：

```cpp
inline constexpr const char *kOpType = "BenchHarnessScatter";
inline constexpr const char *kInputX = "x";
inline constexpr const char *kInputY = "y";
inline constexpr const char *kOutputZ = "z";

inline const std::vector<std::string> &OrderedInputPorts()
{
    static const std::vector<std::string> ports = {kInputX, kInputY};
    return ports;
}

inline const std::vector<std::string> &OrderedOutputPorts()
{
    static const std::vector<std::string> ports = {kOutputZ};
    return ports;
}
```

构图函数只创建 2 个 `Data` 输入并绑定到 `x`/`y` 端口（`BuildGraph`，verbatim 节选）：

```cpp
inline GraphBinding BuildGraph(ge::Graph &graph, const Case &test_case,
                               bool dynamic_inputs,
                               ge::DataType output_decl_dtype)
{
    const std::string node_name = SafeNodeName(test_case.id);
    ge::Operator node =
        ge::OperatorFactory::CreateOperator(node_name.c_str(), kOpType);
    const ge::TensorDesc x_desc =
        dynamic_inputs
        ? MakeUnknownRankDesc(ToGeType(test_case.x_dtype))
        : MakeDesc(test_case.x_shape, ToGeType(test_case.x_dtype));
    const ge::TensorDesc y_desc =
        dynamic_inputs
        ? MakeUnknownRankDesc(ToGeType(test_case.y_dtype))
        : MakeDesc(test_case.y_shape, ToGeType(test_case.y_dtype));

    const std::string x_name = node_name + "_x";
    ge::Operator x =
        ge::OperatorFactory::CreateOperator(x_name.c_str(), "Data");
    x.SetAttr("index", static_cast<int64_t>(0));
    x.UpdateInputDesc("x", x_desc);
    x.UpdateOutputDesc("y", x_desc);
    graph.AddOp(x);
    node.SetInput(kInputX, x);

    const std::string y_name = node_name + "_y";
    ge::Operator y =
        ge::OperatorFactory::CreateOperator(y_name.c_str(), "Data");
    y.SetAttr("index", static_cast<int64_t>(1));
    y.UpdateInputDesc("x", y_desc);
    y.UpdateOutputDesc("y", y_desc);
    graph.AddOp(y);
    node.SetInput(kInputY, y);

    node.UpdateOutputDesc(
        kOutputZ, MakeUnknownRankDesc(output_decl_dtype));
    graph.SetInputs(std::vector<ge::Operator>{x, y})
        .SetOutputs(std::vector<ge::Operator>{node});
    return {node, node_name};
}
```

独立期望值也还是 add 语义（`MakeExpected`，verbatim）：

```cpp
inline ExpectedOutput MakeExpected(const Case &test_case)
{
    ExpectedOutput expected;
    expected.shape = test_case.expected_shape;
    expected.dtype = test_case.expected_dtype;

    const std::vector<double> x =
        InputValues(test_case.x_shape, 0);
    const std::vector<double> y =
        InputValues(test_case.y_shape, 1);
    expected.values.resize(
        static_cast<size_t>(std::max<int64_t>(0, Numel(expected.shape))));
    for (size_t i = 0; i < expected.values.size(); ++i) {
        const double x_value = RoundTo(x[i], test_case.x_dtype);
        const double y_value = RoundTo(y[i], test_case.y_dtype);
        expected.values[i] =
            RoundTo(x_value + y_value, test_case.expected_dtype);
    }
    return expected;
}
```

而权威 OpDef（`op_host/bench_harness_scatter_def.cpp` 构造函数，verbatim 节选）注册的是完全不同的契约：

```cpp
class BenchHarnessScatter : public OpDef {
public:
    explicit BenchHarnessScatter(const char* name) : OpDef(name)
    {
        // --- Input "data": 被写张量，scatter 的目标（10 种 dtype 组合按列对齐）---
        this->Input("data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT64, ge::DT_INT64})
            ...
            .AutoContiguous();

        // --- Input "indices": 索引张量，值 ∈ [0, data.size(dim))，dtype ∈ {int32, int64} ---
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ...})
            ...
            .AutoContiguous();

        // --- Input "updates": 更新值张量，dtype 同 data，shape 同 indices ---
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT64, ge::DT_INT64})
            ...
            .AutoContiguous();

        // --- Output "y": 结果张量，shape/dtype 同 data ---
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT64, ge::DT_INT64})
            ...
            .AutoContiguous();

        // --- Attr "dim": scatter 维度，必选（无默认值） ---
        this->Attr("dim").AttrType(REQUIRED).Int();

        // --- Attr "reduce": 聚合方式，可选，默认 "update"（覆盖模式）---
        this->Attr("reduce").AttrType(OPTIONAL).String("update");
        ...
    }
};
```

比对结论：适配器 2 输入（`x`/`y`）+ 1 输出（`z`）+ 0 属性，OpDef 3 输入（`data`/`indices`/`updates`）+ 1 输出（`y`）+ 2 属性（`dim`/`reduce`）——端口名、输入数、属性三处全部不一致。`node.UpdateOutputDesc("z", ...)` 在 OpDef 中找不到名为 `z` 的输出描述符，GE 只打 WARNING；输出描述符从未被更新，`LogShapeInit` 读到的输出 shape 为空，路由探测在 SHAPE-INIT 阶段失败并归类为 DESCRIPTOR_ERROR。

3. **顺带检查用例清单与脚本路径是否同源**。本例中 `cases.tsv` 也还是 2 输入样例表头（`input0_dtype`/`input1_dtype` 两列），TTK `static.csv`/`dynamic.csv` 的 `input_shapes` 为二元组——与 DESIGN 的 32 条矩阵不映射；`run.sh` 里 TTK_SPEC/RESULT_PARSER/GEIR_PLAN/VALIDATOR 四条路径全部指向不存在的位置。这些是同一个"样例模板未适配"根因的不同表现，可一并确认。

## 设计决策

- 为什么判别顺序是"先 grep WARNING 再静态比对"而不是直接重写适配器：DESCRIPTOR_ERROR 是汇总状态，可能由输入侧或输出侧描述符缺失引起；WARNING 行能直接点名是哪一个端口（本例 `output desc named z`），把静态比对范围缩到一行。
- 为什么不采信"构建成功"作为 ST 可用信号：本例三个测试目标全部编译通过、隔离检查通过，但路由第一步即失败——GEIR ST 的可用性必须以 `run.sh route` 输出 `status=UNIQUE`（V1/V2 路由固定）为准，编译绿不等于适配对。
- 当前判别路线属 recommended_fastpath：静态比对无环境依赖，可在 5 分钟内排除/确认；修复（按算子契约重写适配器）不属本卡范围。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.2.0（`ASCEND_HOME_PATH=/workspace/.../cann-9.2.0`），TTK 位于 `/tmp/ops-test-kit`（GEIR 用例执行需 source `docs/develop/env.sh`）。
- 相关文件（本场景实际读/写，与事件流一一对应）：
  - `operators/<Op>_package/tests/geir/adapter/generated_plan.h`——出问题的适配器（端口/期望值/构图）。
  - `operators/<Op>_package/tests/geir/adapter/op_adapter.h`、`adapter/infer_adapter.h`——只 include generated_plan.h 的薄壳。
  - `operators/<Op>_package/tests/geir/route_main.cpp` / `suite_main.cpp` / `infer_main.cpp`——三个测试入口（构图与探测逻辑本身正常）。
  - `operators/<Op>_package/tests/geir/cases.tsv`、`ttk/static.csv`、`ttk/dynamic.csv`——用例清单（同为未适配样例）。
  - `operators/<Op>_package/tests/geir/run.sh`——stage/build/route/test 入口与证据日志路径（`build/evidence/route.debug.log`）。
  - `operators/<Op>_package/op_host/bench_harness_scatter_def.cpp`——权威 OpDef（端口/属性/dtype 组合）。
- 复现命令（完整）：

```bash
source docs/develop/env.sh
cd operators/<Op>_package/tests/geir
GEIR_VENDOR_DIR=$(./run.sh stage)   # 从当前源码重建 staging
export GEIR_VENDOR_DIR
./run.sh build                       # 三个目标编译（会通过）
./run.sh route                       # 路由探测 → DESCRIPTOR_ERROR
grep -iE "Cannot find the (input|output)" build/evidence/route.debug.log
```

## 完整代码

本卡为诊断规则卡，无修复实现；判别所需的完整 verbatim artifact（适配器 `BuildGraph`/`MakeExpected`/端口常量、OpDef 构造函数、WARNING/GRAPH-ADD/SHAPE-INIT 原始输出、复现命令）已全部内联于《判别方法》《系统信息》两节。

## 预防

- 为新算子生成 GEIR ST 时，把"适配器端口名/输入数/属性 ↔ OpDef 注册契约"做成显式检查项：`generated_plan.h` 的 `kInput*`/`kOutput*` 常量、`OrderedInputPorts()`/`OrderedOutputPorts()`、`BuildGraph` 中的 `SetInput`/`UpdateOutputDesc`、以及必选属性的 `SetAttr`，逐项对照 `op_host/*_def.cpp` 的 `Input(...)`/`Output(...)`/`Attr(...)`。
- 样例模板复制后先跑 `run.sh route` 再写用例矩阵：路由 UNIQUE 是适配面正确的最小信号，比编译通过强得多。
- 验收/评审 GEIR ST 时，`build/evidence/route.debug.log` 里的 `[WARNING] ... Cannot find the ... desc named <name>` 是模板未适配的直接指纹，应当作 blocker 而非噪音。

## 验证信号

以下信号来自该算子 GEIR ST 验收会话：从当前源码重建 staging、构建测试目标、执行路由探测并提取证据日志。

- staging 重建成功：`.run` 包安装出 vendor 目录（含 `libcust_opmaster_rt2.0.so`/`libcust_opsproto_rt2.0.so`/ascend950 kernel `.o`）（来源平台：950；证据等级：tool_observed）。
- 三个测试目标构建成功且不链接生产源码：link.txt 仅含 `-lgraph -lge_runner -lgraph_base [-lopp_registry -lmetadef -ldl]`，compile_commands 无 `op_graph/`/`op_host/` 路径（来源平台：950；证据等级：tool_observed）。
- 路由探测失败：`ROUTE op=BenchHarnessScatter path=UNKNOWN status=DESCRIPTOR_ERROR probe_exit=1`，`ROUTE-PROBE ... stage=SHAPE-INIT pipeline=FAIL value=FAIL`（AddGraph 成功、失败在 SHAPE-INIT）（来源平台：950；证据等级：tool_observed）。
- 证据日志原文：`[WARNING] ... UpdateOutputDesc:[Update][OutputDesc] Cannot find the output desc named z`；`GRAPH-ADD ... inputs={x:[2,3]/DT_FLOAT,y:[2,3]/DT_FLOAT}`；`SHAPE-INIT ... outputs={z:{shape:[],dtype:DT_FLOAT}}`（来源平台：950；证据等级：tool_observed）。
- 适配器与 OpDef 静态比对：2 输入(x/y)+1 输出(z)+0 属性 vs 3 输入(data/indices/updates)+1 输出(y)+2 属性(dim/reduce)（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡判别的是"端口/属性契约不一致 → DESCRIPTOR_ERROR"这一类根因；路由失败还可能有其他根因（vendor 未加载、OpType 未注册等），若 WARNING 行不存在需另行排查。
- 只观测到 `UpdateOutputDesc` 侧的静默 WARNING；`SetInput` 用不存在端口名时 GE 是静默忽略还是打日志未在本场景验证（未验证）。
- 必选属性 `dim` 从未被适配器设置——它对失败的独立贡献没有单独分离验证（本例失败已在输出描述符阶段定性）（未验证）。
- WARNING 文案（`op_desc.cc` 的 `Cannot find the ... desc named`）是 CANN 9.2.0 的观测值，其他版本文案可能不同（未验证，推断可迁移：按 "Cannot find" + "desc" 模糊匹配）。
- 修复后的路由结果（应得 `status=UNIQUE` + V1/V2 路径）未验证——本场景为只读验收，未实施修复。

**未决项**：

- 按算子契约重写适配器（3 输入 + `dim`/`reduce` 属性 + scatter 期望值）并复验路由 `status=UNIQUE`：未实施（本会话为只读验收角色），是本卡结论的下一步接续点。
- `SetInput` 端口名不存在时 GE 的确切行为（静默/告警/错误）待单独构造实验验证。
- 缺失必选属性 `dim` 的独立失败信号待分离验证（构造"端口全对但漏属性"的用例）。
## 坏实践

- 把样例/模板 GEIR 工程直接当某算子的 ST 交付：编译能过、隔离能过，但端口/属性/期望值全是 add 样例的，路由一步即失败。
- 只看 `run.sh route` 的汇总行（`status=DESCRIPTOR_ERROR`）不看 `route.debug.log` 的 WARNING 原文——汇总行不含端口名，无法定位是哪个描述符缺失。
- 用"构建成功 + 用例文件存在"作为 ST 验收通过标准（正确标准：路由 UNIQUE + 用例矩阵映射 DESIGN + TTK GEIR 用例无 SKIP 执行）。

## 误诊

- 初看可能把 DESCRIPTOR_ERROR 归因为 vendor 包没装好或 OpType 未注册（`GetOpAllAICoreConfig:Cannot find aicore config for op[...]` 这类 DEBUG 行在日志里大量出现，容易误读为根因）——实际 `AddGraph` 成功、算子类型能解析，失败精确落在输出描述符 `z` 上，WARNING 行可排除此类误诊。
- 也可能以为是 TTK/脚本环境问题（本例 run.sh 确实另有四条断路径）——但路由失败在进入 TTK 阶段之前就已发生，两件事应分开归因。
