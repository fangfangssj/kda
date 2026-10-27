---
type: "Runbook"
title: E80003 属性 Python 类型须与 OpDef 逐例一致
description: TTK kernel 模式动态编译把 CSV 属性按 Python 字面量原样传给 para_check 的严格 isinstance 检查（无 int/float 互转、无联合类型），Int 收 float 或 Float 收 int 均报 E80003；须 CSV 字面量、OpDef、tiling 粘合层、vendor autogen 装饰器多处类型对齐。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# E80003 属性 Python 类型须与 OpDef 逐例一致

## 现象

在 Ascend950PR（CANN 9.2.0，TTK 以源码方式 `python3 -m ttk` 运行）上给自定义算子 BenchHarnessApplyAdamW 跑 Branch-0 白盒用例（TTK kernel 模式，34 条用例，命令形如 `python3 -m ttk kernel -i ttk_design_doc_cases.csv --op bench_harness_apply_adam_w --plat Ascend950PR_9579 --plugin golden_plugin.py -o result.csv --pc 1`），12/34 条用例 `dyn_precision=DYN_COMPILE_FAILURE`、`perf_status=FAIL`，且全部集中在显式携带 `step` 属性的用例（step=1.0/10.0/1000.0，均为 Python float）；其余 22 条（attributes={} 或只含个别键）编译正常。result.csv 解析结果：

```
dyn_precision categories: {'DYN_COMPILE_FAILURE': 12, 'GOLDEN_FAILURE': 22}
case00001  FAIL   DYN_COMPILE_FAILURE    FAIL   step=1.0
case00002  FAIL   DYN_COMPILE_FAILURE    FAIL   step=10.0
...（case00001-00012 全部 step 为 float）
```

full_debug.log 中的报错原文（以 step=1000.0 的用例为例，OpDef 声明 Int）：

```
RuntimeError: ({'errCode': 'E80003', 'op_name': 'bench_harness_apply_adam_w', 'param_name': 'step',
'param_type': "<class 'int'>", 'actual_type': 'float'},
"In op[bench_harness_apply_adam_w], the parameter[step]'s type should be [<class 'int'>], but actually is [float].")
```

当时代码状态：OpDef（`op_host/bench_harness_apply_adam_w_def.cpp`）声明 `this->Attr("step").AttrType(OPTIONAL).Int(1);`；而白盒用例生成阶段为避免 int 枚举采样自动引入非法 step=0（step=0 会使偏置校正分母 1-β^0=0 除零），把 step 按 float32 采样，CSV 里落在 Python 字面量 1.0/10.0/1000.0。同算子的 aclnn C++ ST（int64_t step）一直 21/21 通过——问题只出现在 TTK kernel 模式动态编译（dyn）路径。

把 OpDef 改成 `Float` 修复 float 采样用例后，同一 CSV 里 `{'step': 1}`（Python int 字面量）的用例又**反向**触发 E80003：

```
RuntimeError: ({'errCode': 'E80003', 'op_name': 'bench_harness_apply_adam_w', 'param_name': 'step',
'param_type': "<class 'float'>", 'actual_type': 'int'},
"In op[bench_harness_apply_adam_w], the parameter[step]'s type should be [<class 'float'>], but actually is [int].")
```

两分支对同一 attr 的字面量类型相反，是踩坑的直接诱因——检查是逐例严格 isinstance，两个方向都不宽容。

## 适用场景

- 工程形态：aclnn 自定义算子的 TTK kernel 模式白盒/精度测试（动态编译 dyn 路径），以及任何把属性按 Python 字面量传入算子动态编译入口的测试链路。
- API 路径：TTK `python3 -m ttk kernel` → `op_interface._compile_op`（`op_func(*tensor_list, **op_kwargs)`）→ CANN `asc_op_compile_base/common/utils/para_check.py` 的 `@para_check.check_op_params` 装饰器；算子侧 OpDef `this->Attr(...).Int/.Float(...)` 与 arch 粘合层 `attrs->GetInt/GetFloat`；类型检查的期望值来自 vendor 安装目录的 autogen Python 装饰器（`$ASCEND_OPP_PATH/vendors/<V>/<V>_impl/dynamic/<op_snake>.py`）。
- 验证过的平台与 CANN 版本：Ascend950PR（`--plat Ascend950PR_9579`），CANN 9.2.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：platform_wide——机制由 CANN para_check 与 TTK 透传逻辑决定，与具体算子无关。
- 路线优先级：recommended_fastpath（类型对齐是正解；改 CANN site-packages 放宽检查只是临时手段）。
- 明确不适用 / 未验证场景：TTK aclnn（e2e）路径——该路径的 `param_plan.build_positional_args` 有 `coerce_value` 做 int/float 归一（静态读码确认，未实测）；aclnn C++ 侧传参（走 aclnn apis 的类型转换链，不经该 Python 装饰器）；不经过 para_check 的静态编译/二进制（bin）路径；其他 CANN 版本若引入类型宽容需重新验证。

## 根因

CANN 动态编译入口的 para_check 对**显式传入**的属性做严格 `isinstance` 类型检查（无 int/float 互转、无联合类型），而 TTK kernel 模式把 CSV 属性按 Python 字面量原样透传，因此 CSV 中每个属性的 Python 类型必须与 OpDef 声明类型逐例精确一致。

机制链（逐层读码 + 实测确认）：

1. TTK 组参：`op_kwargs = param_transformation(testcase.spec_attrs, op_func_parameters)`（op_interface.py:254）。`param_transformation` 即 `adapt_pickup_by_names`（container_utils.py:254），只按算子函数参数名挑键重命名，**不做任何类型转换**；CSV 的 `{'step': 1000.0}` 经 `process_dict` 的 `eval` 解析后就是 Python float。
2. 调用：`op_func(*copy.deepcopy(tensor_list), **copy.deepcopy(op_kwargs))`（op_interface.py:351）——属性按 kwargs 显式传入编译入口。
3. para_check：`_in_wrapper` 只遍历**显式传入**的 args/kwargs 逐个检查；`_check_attr_type` 执行 `isinstance(op_param, py_type)`（para_check.py:471）。`OPTION_ATTR_INT` 对应 `isinstance(x, int)`，float 值直接抛 E80003；`OPTION_ATTR_FLOAT` 对应 `isinstance(x, float)`，Python 的 `isinstance(1, float)` 为 `False`（int 不是 float 的实例），int 值同样抛 E80003。
4. **没有联合类型**：para_check 的标量属性检查类型只有 INT/FLOAT/STR/BOOL/TYPE（para_check.py:48-59、190），不存在 INT_OR_FLOAT/NUMBER，int 与 float 互不兼容。
5. **默认值不检查**：用例不传 step 时走生成入口的函数默认值（生成器把 `.Float(1.0f)` 渲染成 `step=1` int 字面量），但默认参数不经 `_in_wrapper` 的显式检查，所以 attributes={} 的用例不受默认值字面量类型影响——只有 CSV 里显式写的值被逐例检查。
6. **类型检查的期望值来自 vendor 安装目录里的 autogen Python 文件装饰器**，不是 ops-info.json，也不是源码里的 OpDef——所以只改 OpDef 源码不重新部署，检查行为不变（配套的部署不刷新问题见同批"OpDef 改动不生效"卡）。

## 解决配方

以 step 从 Int 改为 Float（CSV 以 float 采样）为例，须**多处类型对齐**：

1. OpDef 声明（`op_host/bench_harness_apply_adam_w_def.cpp`）：

   修复前（Int 声明；CSV 以 float 采样 step → E80003）：

   ```cpp
   // --- Attributes (proto.md §1: all OPTIONAL with defaults) ---
   this->Attr("lr").AttrType(OPTIONAL).Float(1.0e-3f);
   this->Attr("beta1").AttrType(OPTIONAL).Float(0.9f);
   this->Attr("beta2").AttrType(OPTIONAL).Float(0.999f);
   this->Attr("weight_decay").AttrType(OPTIONAL).Float(0.01f);
   this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-8f);
   this->Attr("step").AttrType(OPTIONAL).Int(1);
   this->Attr("maximize").AttrType(OPTIONAL).Bool(false);
   ```

   修复后（step 声明为 float：白盒 CSV 按 design_facts 以 float32 采样 step（1.0/10.0/1000.0，避免枚举采样引入非法 step=0），para_check 对 Int attr 严格拒收 float（E80003）。数学上 step 仅用于偏置校正分母 β^t，float 与 int 等价（值为整数））：

   ```cpp
   this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-8f);
   this->Attr("step").AttrType(OPTIONAL).Float(1.0f);
   this->Attr("maximize").AttrType(OPTIONAL).Bool(false);
   ```

2. arch35 tiling 粘合层同步改读取方式（`op_host/arch35/bench_harness_apply_adam_w_tiling_arch35.cpp`）：

   修复前（按 Int 读 step）：

   ```cpp
   const int64_t* iPtr = attrs->GetInt(5);   if (iPtr != nullptr) step         = *iPtr;
   ```

   修复后（step declared Float in OpDef (CSV float32 sampling); read as float then cast to int64_t — values are whole numbers, cast is exact.）：

   ```cpp
   fPtr = attrs->GetFloat(5);                if (fPtr != nullptr) step         = static_cast<int64_t>(*fPtr);
   ```

   （反向把 Float 改回 Int 时同理：`GetInt` 声明于 `exe_graph/runtime/runtime_attrs.h`。）

3. 重新 `./build.sh BenchHarnessApplyAdamW_package` 构建并安装 vendor 包（ops-info 只在 cmake configure 阶段再生成，须强制重建，见同批部署卡）；安装后生成入口的 para_check 由 `OPTION_ATTR_INT` 变为 `OPTION_ATTR_FLOAT`，aclnn autogen 头文件中 step 形参从 `int64_t` 变为 `double`。若该 vendor autogen Python 文件未被构建刷新，须手工同步装饰器对应参数位（`grep -n "OPTION_ATTR" $ASCEND_OPP_PATH/vendors/<V>/op_impl/ai_core/tbe/<V>_impl/dynamic/<op>.py` 核对）：

   ```python
   # before（step 位为 OPTION_ATTR_FLOAT，第 11 参）
   @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
   # after（仅 step 一位 FLOAT→INT，以 Int 方向为例）
   @para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
   ```

4. golden 插件侧对 step 做归一后再喂 oracle（如 `int(step)`，oracle 数学上按整数 step 计算 β^t）。
5. 回归 aclnn C++ ST 确认 API 兼容（整数值 int64_t→double 隐式转换无损）。

判别方法（可复用）：

1. 从 result.csv 过滤 `dyn_precision=DYN_COMPILE_FAILURE` 的用例，比对它们的 attributes 与 PASS 用例的差异（失败用例是否全部显式携带同一属性）。
2. 在 full_debug.log 搜 `errCode`：E80003 文本直接给出 param_name/param_type/actual_type 三元组。
3. `grep -n "OPTION_ATTR" $ASCEND_OPP_PATH/vendors/<V>/op_impl/ai_core/tbe/<V>_impl/dynamic/<op_snake>.py` 确认生成入口里该属性的检查类型。
4. 核对 CSV 属性的 Python 字面量类型：`ast.literal_eval(attrs).get('step')` 的类型——**逐条用例**都必须与 OpDef 声明一致，混用会在部分用例上单独爆 E80003；换分支/换生成策略时先 diff 两份 CSV 的同一 attr 字面量类型。

## 设计决策

- 为什么改 OpDef 为 Float 而不是把 CSV 改回 int：白盒生成侧按 float32 采样是有意设计（int 枚举采样会自动加 range(5) 引入非法 step=0，step=0 使偏置校正分母 1-β^0=0 除零）；且数学上 step 只出现在偏置校正 β^t 中，整数值的 float 与 int 等价。改 OpDef 一次对齐全部用例，避免逐条修 CSV。反向场景（CSV 以 int 采样）则应把 OpDef 改回 Int——类型语义上 step 是整数计数时声明 Int 才是对的。
- 代价评估：OpDef Int→Float 是 API 语义变化（autogen aclnn 形参 int64_t→double）；aclnn ST 21/21 复跑确认无回归后接受。
- 备选（临时手段，跑完必须恢复）：改 CANN site-packages 的 `para_check.py` 放行 int（`py_type is float and isinstance(op_param, int)` 分支）——不可交付且会掩盖其他算子的真实类型错误（包括把 1.5 传 Int 这种真 bug）；golden_plugin 包装层转换属性——无效，`_wrap` 只作用于 golden 计算，不在编译路径上。
- 已知残留：同批 CSV 里 case00026 的 `{'step': 1}` 是 Python int，OpDef 改 Float 后反向触发 E80003（`isinstance(1, float)`=False）——同一严格检查的对称证据；正确做法是把该用例值归一为 1.0。
- 当前方案路线档位：recommended_fastpath（属性类型对齐是编译正确性必需）。

## 预防

- 用 TTK kernel 模式跑属性敏感算子前，先对 CSV 的 attributes 列做一次 Python 类型盘点（int/float/bool 分组），与 OpDef 声明逐一对照；任何一条用例类型不一致都会单独 E80003。
- OpDef 属性类型一旦变更，同一变更集内必须同步：arch 粘合层 GetInt/GetFloat、vendor autogen 装饰器、golden 插件归一、aclnn autogen 签名，然后重装 vendor 包并全量重跑 ST。
- 看到 E80003 先看报错里的 actual_type/param_type 字段——类型三元组直接定位，不要先怀疑 kernel 逻辑或重装环境。

## 验证信号

以下信号来自该算子 Branch-0 白盒失败修复会话（两个方向的 E80003 都实测）：先用 TTK 全量跑出失败分布，再实施类型对齐修复、重装 vendor 包后复跑验证。

- 修复前（Int←float 方向）：TTK kernel 模式 total=34 pass=0 fail=34，其中 12 条 DYN_COMPILE_FAILURE 全部为显式 float step 用例，报错原文 `E80003 ... the parameter[step]'s type should be [<class 'int'>], but actually is [float]`（来源平台：950；证据等级：tool_observed）。
- 修复重装后：vendor 生成入口 step 位置的检查由 `OPTION_ATTR_INT` 变为 `OPTION_ATTR_FLOAT`，aclnn autogen 头文件 step 形参变为 `double`（grep 生成产物确认）（来源平台：950；证据等级：tool_observed）。
- 修复后回归：aclnn C++ ST 21/21 PASSED（0 FAILED），含 step_1 边界用例（来源平台：950；证据等级：tool_observed）。
- 修复后 TTK 复跑：total=34 pass=29 fail=5；原 12 条 float-step 用例中 10 条 PASS（case00003/07/08/10/11/23/28/29/31/32），case00005/09 仅因无关的 DYN_OPERATOR_BUILD_LOST（编译产物/环境问题）失败（来源平台：950；证据等级：tool_observed）。
- 对称证据（float←int 方向）：case00026（CSV `{'step': 1}` Python int）在 Float OpDef 下反向 E80003（`the parameter[step]'s type should be [<class 'float'>], but actually is [int]`，DYN_COMPILE_FAILURE），确认检查是逐例严格 isinstance、无联合类型（来源平台：950；证据等级：tool_observed）。
- 三处改齐 + 重新部署后（Int 方向）：同一用例不再报 E80003，报错层级推进到 tiling（当时叠加的 vendor 符号冲突问题），证明类型检查已通过；再修复 load_priority 后 kernel 正常执行 `STATUS: PASS`（来源平台：950；证据等级：tool_observed）。
- para_check.py 临时补丁已恢复原样：`grep -c "py_type is float and isinstance"` 返回 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡机制基于 CANN 9.2.0 的 `asc_op_compile_base` para_check 实现；其他版本若引入类型宽容或 int/float 联合检查需重新验证（未验证）。
- TTK aclnn（e2e）路径存在 `param_plan.coerce_value` 类型归一（静态读码确认），本卡结论不适用该路径（未实测）。
- "默认值不参与检查"的结论来自 para_check `_in_wrapper` 只遍历显式 args/kwargs 的读码分析，并有 attributes={} 用例修复后编译通过的实证佐证；若未来 CANN 把默认值纳入检查，`.Float(1.0f)` 被生成器渲染成 int 字面量 `step=1` 的组合可能出问题（未验证）。
- 混合字面量类型（同属性部分用例 int、部分 float）没有任何单一 OpDef 类型能同时满足，必须归一 CSV（case00026 即此残留）。
- int 对 float 在数值上安全（widen），但 float 对 int（如 1.5 传 Int attr）是真错误，para_check 拒收是合理的——不要用"放行 int"补丁长期替代类型对齐。

**未决项**：

- case00026 的 `{'step': 1}`（int）在 Float OpDef 下仍会 E80003——需把该 CSV 值归一为 1.0（或由生成器统一属性字面量类型），当次会话未完成。
- 修复后 case00005/09 的 DYN_OPERATOR_BUILD_LOST 及清空 kernel_meta 后全量 BUILD_LOST（手动复现 TVMError EB9000 get platform info failed）未定位根因，属编译产物/环境问题，未闭环。
- 可选改进：在用例生成器导出 CSV 时按 opspec attr 类型强制字面量类型，从源头消灭该类不一致。
## 坏实践

- 只改 OpDef 类型不改 arch 粘合层读取（GetInt/GetFloat 与声明不匹配）——属性会读到错误类型或空指针。
- 同一属性在 CSV 里混用 Python 字面量类型，修一头漏一头。
- 看到 DYN_COMPILE_FAILURE 就去重装 CANN/清缓存，而不先读日志里的 E80003 类型三元组。
- 改 CANN site-packages 的 `para_check.py` 让 int 过 float 检查来"跑绿"并长期保留。
- 在 golden_plugin 里做属性类型转换期望修编译——包装层不在编译路径上，无效。

## 误诊

- 曾担心 OpDef 改 Float 后生成入口的 int 默认值 `step=1` 会让 attributes={} 用例反向 E80003：读码确认 para_check 只检查显式传入的 args/kwargs、不检查默认值，且修复后 attributes={} 用例全部编译通过，排除该假设。
- 排查中一度被"上一分支 CSV 里 `{'step': 1}`（int）曾 PASS"迷惑，怀疑 TTK 有隐藏的类型强转；实际时间线是：那次结果文件生成于 03:55，而带 Float step 的 vendor 在 04:49 才重建部署——旧部署态（step 为 Int 的 OpDef）接受了 int。教训：判断"以前能过"前先核对当时部署的 vendor 构建时间与源码提交时间是否一致（该时间线推演基于 mtime/git log，机制未直接复现，属推断性归因）。

## 增量（falsy 跳过规则与 autogen 手改的脆弱性）

同算子白盒修复会话的另一分片补齐两条机制边界与一条坏实践：

**falsy 跳过规则**：`para_check._check_attr` 在类型检查前有 `if not op_param: return`——falsy 值（0 / False / None）会**跳过类型检查**，因此 `step=0`、`weight_decay=0` 之类取值碰巧不触发 E80003。测试设计含义：用 0 值用例验证类型对齐会得到假阴性；类型对齐验证须用非零值。

**方向二的 autogen 手改路线（已证伪为持久修复）**：对"调用方送 int、OpDef 已是 Float"的反向案例（CSV `{'step': 1}` Python int / C++ 测试结构体存 int64_t 而 C API 形参是 double），本会话曾手改安装目录 autogen Python 包装层——把 step 位置的 `para_check.OPTION_ATTR_FLOAT` 换成 `None` 跳过类型检查，并在 `_build_args` 里 `float(step)` 归一：

```python
# 安装目录 op_impl/ai_core/tbe/<Op>_impl/dynamic/<op>.py（autogen，勿手改——见坏实践）
@para_check.check_op_params(..., para_check.OPTION_ATTR_FLOAT, ...,
    None,   # step —— 原 OPTION_ATTR_FLOAT，跳过 int/float 严格校验
    para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
```

该手改能让反向用例通过，但它是安装产物上的未入库改动：本会话内被 build 安装**静默覆盖 2 次**、每次需重打，不可作为持久修复；且验收阶段重建安装后 ST 直调 41 例全部报 ACLNN 561002（`aclnnXxxGetWorkspaceSize` 处），验收方已定位 C API `double step` / 测试 `int64_t` / OpDef Float 的同一类型线索——该 561002 与手改回退的因果闭环在记录截断处未完成（见增量遗留）。方向二更稳妥的源级做法是让调用方统一送 float 字面量（CSV 写 `1.0`、C++ 侧传 `double`），使 OpDef 单一 Float 类型与全部调用方对齐。

验证信号（复现增量，同算子白盒失败修复会话）：

- 修复前（TTK 白盒汇总）：total=34、pass=21、fail=13，其中 12 例 step 以 float 采样被 OpDef Int 声明拒收（DYN_COMPILE_FAILURE，E80003）（来源平台：950；证据等级：tool_observed）。
- 方向一修复后（OpDef step=Float + tiling GetFloat，重建安装）：34 例中 29 通过，12 例 float-step 编译失败全部消除；剩余失败为反向 int 用例、golden 形状用例与精度类用例（来源平台：950；证据等级：tool_observed）。
- 方向二处置后（autogen Python 跳过 step 类型检查 + float(step) 归一）：反向用例从失败清单消失，全量收敛至 33/34（剩 1 例 bf16 极值精度用例）（来源平台：950；证据等级：tool_observed）。
- 该 autogen 手改在本会话内被 build 安装覆盖 2 次、每次需重打；验收阶段重建安装后 ST 直调 41 例全部 561002，类型线索已定位、闭环未完成（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 手改安装目录下 autogen 生成的算子 Python 包装层来绕过 para_check——改动不在源码树，任何重建/重装都会静默回退，且削弱算子输入校验；本会话两次被覆盖，验收阶段疑似关联 561002 全量回归。
- 修复代码后不重跑评测、沿用旧 result.csv 交差——本会话第一轮修复即因未重跑被验收方凭 mtime 与失败分布逐位比对识破。

失效边界（复现增量）：

- falsy 跳过规则基于 CANN 9.2.0 安装目录 `asc_op_compile_base/common/utils/para_check.py` 的 `_check_attr` 源码（`if not op_param: return`）；其他版本若调整该行为需重核对。
- C++ aclnn 直调路径（`aclnnXxxGetWorkspaceSize`）下 int64_t→double→attr 的类型归一行为未单独验证。

遗留问题（复现增量）：

- 优先项：重建后复现验收阶段 41/41 ST ACLNN 561002，确认是否为 autogen 补丁被覆盖后 para_check 再拒收，并给出源级修复（当次记录在分片边界截断）。
