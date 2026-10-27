---
type: "Runbook"
title: proto.h ATTR 无默认值须改 REQUIRED_ATTR
description: 源码树 op_graph proto.h 用 .ATTR(name, ListInt) 不带默认值，operator_reg.h 宏展开出空表达式编译失败；无默认值的必选属性须写 .REQUIRED_ATTR。复现增量：Float 属性同型报错（lr/beta1/beta2/weight_decay 逐一命中），须保持 ATTR 形态时必须显式补默认值（如 .ATTR(lr, Float, 0)），且必填语义须由 aclnn/OP_CHECK 层校验兜底（默认值会进 IR）。
tags: [task.implementation, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-17T04:13:22Z'
platforms: ["950"]
---

# proto.h ATTR 无默认值须改 REQUIRED_ATTR

## 现象

在 Ascend950PR 开发机上为 BenchNoHarnessStridedSlice 算子编写 GE IR 调用示例（示例 cpp `#include` 源码树的 `op_graph/bench_no_harness_strided_slice_proto.h`），CMake 配置成功、编译时报宏展开语法错误：

```
/workspace/.../op_graph/bench_no_harness_strided_slice_proto.h: In member function
'ge::Operator::OpListInt ge::op::BenchNoHarnessStridedSlice::get_attr_begin() const':
$ASCEND_HOME_PATH/include/graph/operator_reg.h:317:31: error: expected primary-expression before ';' token
  317 |     Op##Type ret = __VA_ARGS__;                                          \
      |                               ^
```

报错点在安装版 CANN 头文件 `graph/operator_reg.h` 的 `ATTR` 宏内，看似 SDK 问题，实际是源码树 proto.h 的属性声明方式问题。对照发现：安装到 vendor 目录的 `op_proto.h`（由算子包构建链生成）用的是 `.REQUIRED_ATTR(begin, ListInt)`，而源码树 `op_graph/bench_no_harness_strided_slice_proto.h` 用的是 `.ATTR(begin, ListInt)`（不带默认值）。

## 适用场景

- 工程形态：算子仓开发流程的自定义算子 op_graph 声明（proto.h）+ GE IR（XIR）调用示例/图模式接入编译。
- API 路径：CANN `graph/operator_reg.h` 的 `REG_OP` 链式宏（`.ATTR` / `.REQUIRED_ATTR`）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 `operator_reg.h` 宏对所有无默认值 ATTR 声明的行为。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：带默认值的 `.ATTR(name, Int, 0)` 写法本身合法（有 `__VA_ARGS__` 可展开），不受本问题影响；其他 CANN 版本宏实现未验证。

## 根因/解决配方

根因：`.ATTR(begin, ListInt)` 不带默认值时，`operator_reg.h:317` 的宏展开 `Op##Type ret = __VA_ARGS__;` 中 `__VA_ARGS__` 为空，生成 `OpListInt ret = ;` 这样的空表达式语法错误；CANN 的 REG_OP 宏不支持"ATTR 无默认值"，无默认值的必选属性必须用 `.REQUIRED_ATTR(name, Type)` 声明（它不生成取默认值的 getter 展开路径）。

修复前（op_graph/bench_no_harness_strided_slice_proto.h，节选自修复前读取与 edit oldString）：

```cpp
REG_OP(BenchNoHarnessStridedSlice)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(begin, ListInt)
    .ATTR(end, ListInt)
    .ATTR(strides, ListInt)
    .ATTR(begin_mask, Int)
    .ATTR(end_mask, Int)
    .ATTR(ellipsis_mask, Int)
    .ATTR(shrink_axis_mask, Int)
    .ATTR(new_axis_mask, Int)
    .OP_END_FACTORY_REG(BenchNoHarnessStridedSlice);
```

修复后（edit verbatim：三个 ListInt 必选属性改为 REQUIRED_ATTR；五个 mask 属性 vendor 生成版同为 REQUIRED_ATTR，本算子保持一致）：

```cpp
REG_OP(BenchNoHarnessStridedSlice)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(begin, ListInt)
    .REQUIRED_ATTR(end, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(begin_mask, Int)
    .REQUIRED_ATTR(end_mask, Int)
    .REQUIRED_ATTR(ellipsis_mask, Int)
    .REQUIRED_ATTR(shrink_axis_mask, Int)
    .REQUIRED_ATTR(new_axis_mask, Int)
    .OP_END_FACTORY_REG(BenchNoHarnessStridedSlice);
```

修改 proto.h 后需重编算子包并重装（GE IR 示例 include 的是源码树文件，但图执行用的是安装的 op_proto 库），再编 GE IR 示例。

## 设计决策

- 对齐依据：vendor 包构建链生成的 `op_proto/inc/op_proto.h` 全部用 `REQUIRED_ATTR`，说明算子包构建侧早已把"无默认值必选属性"规范化为 REQUIRED_ATTR；源码树 proto.h 是手写遗留的 ATTR 写法，两处不一致正是踩坑来源。
- 修改范围取"全部 8 个属性改 REQUIRED_ATTR"而非只改三个 ListInt：与安装版声明保持逐字一致，避免源码树与安装包的 proto 声明分叉。

## 系统信息

- 环境：Ascend950PR，CANN 9.1.0，g++ 13.3.0。
- 相关文件：`op_graph/bench_no_harness_strided_slice_proto.h`（修复点）；`$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_strided_slice_custom/op_proto/inc/op_proto.h`（对照基准，REQUIRED_ATTR 写法）；`$ASCEND_HOME_PATH/include/graph/operator_reg.h:317`（宏展开报错点）；`examples/test_geir_bench_no_harness_strided_slice.cpp`（触发编译的示例）。

## 验证信号

段首来源说明：GE IR 示例工程在同一环境下的修复前后编译对比。

- 修复前：GE IR 示例编译失败于 `operator_reg.h:317 error: expected primary-expression before ';' token`（get_attr_begin/get_attr_end/get_attr_strides getter 展开）（来源平台：950；证据等级：tool_observed）。
- 修复后（仅改 proto.h 三处 ATTR→REQUIRED_ATTR）：同一示例工程重新 cmake + make 编译通过（`[100%] Built target test_geir_bench_no_harness_strided_slice`）（来源平台：950；证据等级：tool_observed）。
- 对照事实：安装版 op_proto.h 用 `REQUIRED_ATTR(begin, ListInt)` 等 8 个 REQUIRED_ATTR 声明（grep 输出逐行可见）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"ATTR 不带默认值"的声明形态；`.ATTR(name, Int, <默认值>)` 合法不受影响。
- 编译错误发生在**包含源码树 proto.h 的下游工程**（如 GE IR 示例）；算子包自身的构建链会规范化输出 REQUIRED_ATTR，所以只跑 build.sh 不写下游示例时不会暴露。
- 推断可迁移（未验证）：CANN 9.x 各版本 operator_reg.h 的 ATTR 宏实现预计同样不支持空 `__VA_ARGS__`；其他版本未实测。

**未决项**：

- 未在 CANN 官方文档中核实 ATTR/REQUIRED_ATTR 的规范表述（本卡归因依据是宏展开行为 + vendor 生成版对照）。
## 坏实践

- 源码树 proto.h 手写 `.ATTR(name, Type)` 不带默认值——与构建链产物不一致，且下游一 include 就编译炸；新写 proto.h 时无默认值必选属性一律 `REQUIRED_ATTR`。

## 误诊

- 初看报错在 CANN SDK 头文件 `operator_reg.h:317`，怀疑是 SDK 宏 bug 或版本问题；实际宏行为是既定约束，错误用法在算子侧 proto.h（对照安装版 op_proto.h 即可自证）。

## 增量（2026-09-17，并行策展批：Float 属性同型报错与显式默认值路线）

独立 trace（ApplyAdamW 类算子，Ascend950PR + CANN 9.1.0）复现同根因并补第二条修复路线：

**现象（Float 属性形态）**：GE IR 调用示例直接 include 算子仓的 `op_graph/<op>_proto.h`，算子包本身 build.sh 编译安装正常，示例工程 cmake+make 时在同一头文件上报一串编译错误：

```text
/.../op_graph/<op>_proto.h: In member function 'ge::Operator::OpFloat ge::op::<Op>::get_attr_lr() const':
${ASCEND_HOME_PATH}/aarch64-linux/include/graph/operator_reg.h:317:31: error: expected primary-expression before ';' token
  317 |     Op##Type ret = __VA_ARGS__;
      |                               ^
/.../op_graph/<op>_proto.h:48:6: note: in expansion of macro 'ATTR'
   48 |     .ATTR(lr, Float)
      |      ^~~~~~
（beta1/beta2/weight_decay 的 .ATTR(..., Float) 同样逐一报错）
```

报错点全部落在**无默认值的两参 ATTR** 上；同头文件里带默认值的 `.ATTR(epsilon, Float, 1e-8)`、`.ATTR(step, Int, 1)`、`.ATTR(maximize, Bool, false)` 不报错——与 ListInt 形态互证"该宏的第三参数（默认值）在 CANN 9.1.0 是必填"。

**第二条修复路线（保持 ATTR 形态时显式补默认值）**：属性语义上必填但需要保持 `ATTR` 写法（或属性实际可选）时，必须给默认值实参（verbatim）：

```cpp
REG_OP(<Op>)
    .INPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(m, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(lr, Float, 0)
    .ATTR(beta1, Float, 0)
    .ATTR(beta2, Float, 0)
    .ATTR(weight_decay, Float, 0)
    .ATTR(epsilon, Float, 1e-8)
    .ATTR(step, Int, 1)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(<Op>)
```

必填属性的"必填"语义交给 aclnn/OP_CHECK 层校验（该算子 op_api 的 `CheckAttrValid` 已做范围检查），proto 层默认值仅满足宏的语法要求。

**两路线取舍**：必选属性首选 `REQUIRED_ATTR`（语义正确，调用方漏传在 proto 层即被拦截，与本卡主修复一致）；显式默认值路线适用于需要保持 ATTR 形态的场景，其代价是默认值会进入 IR——调用方漏传必填属性时 proto 层取默认值（如 0）不再拦截，**必须**由 aclnn 层属性校验兜底；没做属性校验的算子慎用"默认 0"写法。

**验证信号（复现增量）**：

- 修复前：GE IR 示例编译失败于 `operator_reg.h:317:31: error: expected primary-expression before ';' token`，报错行逐一对应 `.ATTR(lr/beta1/beta2/weight_decay, Float)`（带默认值的 ATTR 不报错）（来源平台：950；证据等级：tool_observed）。
- 修复后（补默认值）：同一示例编译通过 `[100%] Built target test_geir_<op>`（仅剩 array_ops.h 弃用 warning）（来源平台：950；证据等级：tool_observed）。
- 修复后示例可执行并进入图执行阶段（图执行阶段的另一问题见对应 runbook）（来源平台：950；证据等级：tool_observed）。

**边界补充（复现增量）**：

- 默认值进入 IR 后，必填语义的拦截责任转移到 aclnn/OP_CHECK 层——采用该路线的算子必须确认 op_api 侧有对应属性校验，否则调用方漏传属性会静默取默认值。
- Float/ListInt/Int/Bool 属性均命中同一宏约束（两 trace 分别以 ListInt 与 Float 实测）；其他属性类型的宏实现未逐一验证（未验证，推断同约束）。