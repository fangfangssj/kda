---
type: "Runbook"
title: "DTYPE_X 未定义：dtype 宏名随输入名生成"
description: "从样例算子复制工程后 apt.cpp 沿用 DTYPE_X 编译报 use of undeclared identifier；codegen 按输入名注入 DTYPE_KEYS 等宏，宏名=输入名大写。R2（，CANN 9.1.0）：反向形态——宏存在而 kernel 类模板参数名撞宏：模板参数命名 DTYPE_X 被 -DDTYPE_X=float 宏展开，报晦涩的 expected a qualified name after 'typename'；类模板参数应改名（如 T），入口函数保持 D_T_X，依赖类型上的模板成员调用须加 .template 前缀。R3（CANN 9.2.0）：同形态独立复现，连锁 no template named 报错，sed 批量改名闭环。R4（，CANN 9.1.0）：输出侧参数同样注入 DTYPE_<名> 宏（独立名无 _REF 后缀），另有 ORIG_DTYPE_*/FORMAT_* 同族与 DTYPE_MAP（float32→float、float16→half、bfloat16→bfloat16_t）；kernel 单源码多 dtype 用 using T = DTYPE_<输入名> 消费（构建按 dtype 组合生成独立二进制与 kernel json，可探针核对），双 dtype 变体 ST 15/15 逐位一致。"
tags: ["task.implementation", "topic.compilation", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-09-14T08:44:28Z"
updated_at: "2026-09-17T03:16:41Z"
platforms: ["950"]
---

# DTYPE_X 未定义：dtype 宏名随输入名生成

## 现象

按 ACLNN 算子仓流程新建一个自定义算子工程：以 `add_custom` 样例包为底版复制改名（样例第一输入名为 `x`），spec 中本算子第一输入名为 `keys`（bfloat16）。在 tiling 结构体落地、`*_apt.cpp` 核函数入口仍沿用样例写法时构建：

```bash
cd operators && bash build.sh engram_gate_fusion_package
```

失败，核心报错（另伴随 ld.lld cannot open ..._0.o 的 FATBIN 阶段连坐错误）：

```text
op_kernel/bench_harness_engram_gate_fusion_apt.cpp:58:40: error: use of undeclared identifier 'DTYPE_X'
ld.lld: error: cannot open .../kernel_meta/..._0.o: No such file or directory
[ERROR] ... An error occurred during compile phases of CompileStage.FATBIN ...
```

同一条流水线构建 `add_custom_package` 样例本身完全正常——说明问题不在环境，而在本工程与样例的输入命名差异。

## 适用场景

- 工程形态：ACLNN 算子仓开发——从样例算子包（add_custom/scale_custom 等）复制创建新算子工程，或手写 `*_apt.cpp` 的 kernel 入口模板实例化。
- API 路径：Ascend C 构建系统 codegen 注入的 `DTYPE_<NAME>` 宏（编译期以 `-DDTYPE_<NAME>=<cpp类型>` 形式注入，值由 `build/.../bin_param/*_param.json` 的 inputs 定义决定）。
- 验证过的平台与 CANN 版本：Ascend 950（950PR）+ CANN 9.2.0。
- 适用粒度：operator_family（所有按 spec 输入名生成 DTYPE 宏的 ACLNN 工程）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：输入名恰为 `x` 的算子（DTYPE_X 天然存在）；直接用 `bfloat16_t`/`half` 字面类型而不引用 DTYPE 宏的写法。
- 反向形态扩展（独立复现 R2，独立 trace，CANN 9.1.0，ascend950）：宏存在而 **kernel 类模板参数名撞宏**——kernel 实现类的模板参数命名为 `DTYPE_X` 时，被编译系统按输入名注入的 `-DDTYPE_X=float` 宏整体替换，报与真实原因相距甚远的 `expected a qualified name after 'typename'`；同一轮常伴随依赖类型上 `ReinterpretCast` 缺 `.template` 前缀的报错。处置：类模板参数改名 `T` 等不撞宏的名字，入口函数模板参数保持 `D_T_X`（与 `ASCENDC_TPL_DATATYPE_DECL(D_T_X, ...)` 契约一致），见解决配方 R2 增量。

## 根因

`DTYPE_X` 不是固定宏名，而是 codegen 按**spec 输入参数名**生成的 `DTYPE_<输入名大写>` 宏；样例工程的输入叫 `x` 所以是 `DTYPE_X`，本算子第一输入叫 `keys`，注入的是 `DTYPE_KEYS=bfloat16_t`，源码里引用 `DTYPE_X` 自然未定义。

机制链：构建系统先由 OpDef/spec 枚举出 `build/<OpType>_ascend950/bin_param/<bin>_param.json`，其中 `op_list[].inputs[]` 逐项列出 `name/dtype/format`；随后按每个输入生成 `-DDTYPE_<NAME>=<cpp-type>` 注入 kernel 编译（同一 `.cpp` 按 dtype 组合编译多份）。宏名与输入名一一对应，工程间不可移植。

R2 补充（独立 trace，CANN 9.1.0）：生成脚本 `build/op_kernel/<OpType>_<soc>/dynamic/<OpType>.py` 中有 `options.append("-DDTYPE_{n}={t}".format(n=x_n_in_kernel, t=DTYPE_MAP.get(x_dtype)[1]))` 与 `-DORIG_DTYPE_{n}` 同族——输入名为 `x` 时对每个 dtype 变体传 `-DDTYPE_X=half/float/bfloat16_t`。kernel 类模板参数若命名为 `DTYPE_X`，在对应变体的编译单元里模板参数名被宏替换成类型名，`template <typename float, ...>` 语法崩掉。

相邻卡片与区分条件：

- [TPL 双 bool 模板入口卡](tpl_template_entry_subkernel_instantiation_verify.md)——机制层近邻：模板入口与 sub-kernel 实例化的验证方法；本卡是 codegen DTYPE 宏名随输入名生成的现象级 runbook，查宏名先查 `_param.json` 而不是猜。
- [bisheng 动态 shape 段错误卡](bisheng_extern_c_dtype_dynamic_shape_segfault.md)——同为 extern C + DTYPE_* 宏入口家族但现象不同：那卡是动态 shape TBE 重编译触发 bisheng 段错误，本卡是静态编译期 use of undeclared identifier。
- [ld.lld cannot open 判别卡](ld_lld_cannot_open_mix_aiv.md)——本卡报错伴随的 FATBIN ld.lld 连坐噪声判别：`.o` 缺失几乎总是上游编译错误的连锁，先翻真正的 `error:` 行。

## 解决配方

**before（沿用样例宏名）：**

```c
    // Milestone 1 stub — kernel Init/Process are empty placeholders.
    BenchHarnessEngramGateFusionKernel<DTYPE_X, isEmpty> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();
```

**after（改用本算子输入名对应的宏）：**

```c
    // Milestone 1 stub — kernel Init/Process are empty placeholders.
    // DTYPE_KEYS is defined by codegen from keys.dtype (bfloat16_t, per §2.2).
    BenchHarnessEngramGateFusionKernel<DTYPE_KEYS, isEmpty> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();
```

排查入口（确认自己工程注入了哪些宏）：

```bash
find operators/<pkg>/build -name "*_param.json" -path "*bin_param*" \
  | xargs python3 -m json.tool | head -80   # 看 op_list[].inputs[].name 与 dtype
```

`inputs[].name` 的大写形式即宏名后缀（`keys` → `DTYPE_KEYS`，dtype bfloat16 → `bfloat16_t`）。

### R2 增量（独立 trace：模板参数名撞 -DDTYPE_X 宏）

触发形态（kernel 类头文件，多 dtype 变体并行编译，FP32 变体最先撞上）：

```text
op_kernel/arch35/<op>.h:55:20: error: expected a qualified name after 'typename'
   55 | template <typename DTYPE_X, int COMPUTE_BRANCH>
      |                    ^~~~~~~~~~
<command line>:1:17: note: expanded from here
    1 | #define DTYPE_X float
```

伴随报错（依赖类型上裸调模板成员）：

```text
op_kernel/arch35/<op>.h:184:49: error: use 'template' keyword to treat 'ReinterpretCast' as a dependent template name
```

修复：类模板参数统一改名 `T`（入口函数保持 `D_T_X`，与 TPL 机制契约一致，不撞宏）；依赖类型上的 `ReinterpretCast` 加 `.template` 前缀：

```cpp
// 修复前：模板参数名撞 -DDTYPE_X 宏 + 依赖类型裸调 ReinterpretCast
template <typename DTYPE_X, int COMPUTE_BRANCH>
class Kernel<Op> { ... GlobalTensor<DTYPE_X> xGm_; ... };

LocalTensor<float> yLocalFloat = yLocal.ReinterpretCast<float>();

// 修复后：类模板参数改名 T；入口函数保持 D_T_X
template <typename T, int COMPUTE_BRANCH>
class Kernel<Op> { ... GlobalTensor<T> xGm_; ... };

template <typename D_T_X, int COMPUTE_BRANCH>
__global__ __aicore__ void <op>(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(<Op>TilingData);
    GET_TILING_DATA_WITH_STRUCT(<Op>TilingData, tilingData, tiling);
    <KernelClass><D_T_X, COMPUTE_BRANCH> op;   // 实现类模板参数已改名 T，入口保持 D_T_X
    op.Init(x, y, &tilingData);
    op.Process();
}

// 依赖类型上的模板成员一律 .template / ::template 前缀
LocalTensor<float> xLocalFloat = xLocal.template ReinterpretCast<float>();
```

确认宏来源的探针（任意 CANN 环境可复跑）：

```bash
grep -n "DTYPE_{n}" <算子构建目录>/op_kernel/<OpType>_<soc>/dynamic/<OpType>.py
# 期望输出包含：
#   options.append("-DDTYPE_{n}={t}".format(n=x_n_in_kernel, t=DTYPE_MAP.get(x_dtype)[1]))
#   options.append("-DORIG_DTYPE_{n}={ot}".format(n=x_n_in_kernel, ot=DTYPE_MAP.get(x_dtype)[0]))
```

不选 `#undef DTYPE_X` 的对抗写法：宏由编译系统按输入名批量生成，对抗宏不如规避保留名；输入名换一个字母宏名就变，规避命名才是稳定解。

R3 增量（另一独立 trace，CANN 9.2.0，regbase 自定义算子 softmax kernel）：同根因的另一组下游报错形态——模板参数名撞宏后，除 `expected a qualified name after 'typename'` 外还连锁报 `expected ',' or '>' in template-parameter-list`、`expected unqualified-id`，以及入口文件 `no template named '<KernelClass>'`（模板声明已坏，实例化点跟着报"类不存在"）；宏展开污染模板参数列表的报错与 DTYPE_X 无字面关联，极易被当作模板语法错误。批量改名配方（实测）：

```bash
cd operators/<Pkg>_package/op_kernel/arch35 && \
sed -i 's/template <typename DTYPE_X, bool isTailR>/template <typename T, bool isTailR>/' <kernel>.h && \
sed -i 's/DTYPE_X/T/g' <kernel>.h
```

入口点保持 `KernelClass<DTYPE_X, true> kernel;` 不变——`DTYPE_X` 作为**模板实参**是合法且必需的用法（宏在实参位置展开为具体类型），出问题的只是把它用作**模板形参名**；命名选择 `T` 与 SDK 参考实现（euclidean_norm、scale_custom 均用 `T`）一致。

## 设计决策

- 选择"改宏引用"而不是把类型写死为 `bfloat16_t`：DTYPE 宏让同一源码随 dtype 枚举多份编译（构建系统按 `_param.json` 组合实例化），写死会破坏多 dtype 扩展且与范式不符。
- 当前方案属 recommended_fastpath。

## 预防

- 从样例包复制工程后，第一时间 grep `DTYPE_` 并对照本算子 spec 的输入名逐一替换；样例里的 `DTYPE_X` 只是样例输入名 `x` 的产物。
- 构建报 `use of undeclared identifier 'DTYPE_*'` 时，先查 `bin_param/*_param.json` 的 inputs 名单再改代码，不要猜宏名。
- （R2 补充）kernel 类模板参数名永远不要用 `DTYPE_<输入名>` 形态（`DTYPE_X`/`DTYPE_Y`…），统一用 `T`/`T_X` 等不与 `-DDTYPE_*`、`-DORIG_DTYPE_*`、`-DFORMAT_*` 宏族冲突的名字。
- （R2 补充）看到 `expected a qualified name after 'typename'` 且行号指向模板参数列表时，先检查 `<command line>` 展开是否被 `-D` 宏替换，而不是怀疑模板语法。

## 验证信号

以下信号均来自同一条开发会话内的构建实测：

- 修复前：干净重建（`rm -rf build` 后 `build.sh`）稳定报 `op_kernel/..._apt.cpp:58:40: error: use of undeclared identifier 'DTYPE_X'`，并连坐 `ld.lld: cannot open ..._0.o`（来源平台：950；证据等级：tool_observed）。
- 修复后：同命令重建输出 `SUCCESS`，再次执行退出码 `EXIT_CODE=0`（来源平台：950；证据等级：tool_observed）。
- 产物核查：构建检查器确认 1 个 `.o` × 2 个子 kernel（Normal `_0` / Empty `_1`）、dtype 枚举 bfloat16/ND 等与 spec 一致（来源平台：950；证据等级：tool_observed）。

R2 增量信号（独立 trace，Elementwise 指数算子骨架开发会话，Ascend950PR + CANN 9.1.0）：

- 修复前：kernel 变体编译报 `expected a qualified name after 'typename'`（`<command line>` 展开显示 `#define DTYPE_X float`）与 `use 'template' keyword to treat 'ReinterpretCast' as a dependent template name`（来源平台：950；证据等级：tool_observed）。
- 机制证据：生成的 `dynamic/<OpType>.py` 中确认 `-DDTYPE_{n}={t}` 编译选项生成逻辑（输入名大写 → `DTYPE_X`）（来源平台：950；证据等级：tool_observed）。
- 修复后：两类编译错误消失，构建推进到链接阶段（后续失败为 TPL_SEL 未注册组合，属另一根因）；最终 3 个 dtype 变体全部编译 Done（来源平台：950；证据等级：tool_observed）。

R3 增量信号（softmax 类 regbase 算子 kernel 首版编译会话，Ascend950 + CANN 9.2.0）：

- 修复前：`op_kernel/arch35/<kernel>.h:77:20: error: expected a qualified name after 'typename'`、`expected ',' or '>' in template-parameter-list`、`expected unqualified-id`，连锁 `op_kernel/<op>_apt.cpp:34:9: error: no template named '<KernelClass>'`，随后整个 opc 流程 `RuntimeError: Kernel Compilation Error`（来源平台：950；证据等级：tool_observed）。
- 修复后：模板形参批量改名（sed）后同批编译中该组错误全部消失，仅剩后续独立的 LoadAlign/Cast 匹配错误（另一问题），最终 build.sh 退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 宏名生成规则以 CANN 9.2.0 的 codegen 实测为准（输入名大写）；含混合大小写/下划线的输入名如何映射（如 `conv_weight` → `DTYPE_CONV_WEIGHT`）未逐一验证，改前先查 `_param.json`。
- 可选输入（ParamType Optional）是否也注入 DTYPE 宏未验证。
- 该修复只解决宏名问题；tiling 结构体/多子 kernel 的编译另属各自问题。
- （R2 补充）`-DDTYPE_{输入名}` 宏生成行为验证于 CANN 9.1.0 的构建链；其他版本若改变宏命名策略（如加前缀），冲突面会变，但"模板参数避开编译系统注入的宏名"这一原则不变（未验证其他版本，推断可迁移）。`.template ReinterpretCast` 仅在模板参数依赖路径上必需；非模板代码直接 `ReinterpretCast<float>()` 即可。
- （R3 补充）`DTYPE_Y`/`DTYPE_Z` 等同族宏同理；host 侧代码无此宏不受影响；若报错行实为其他自定义宏撞名，同一排查思路适用但根因不同。

**未决项**：

- 无（本卡根因已闭合）。

### 增量（输出侧 DTYPE 宏与单源码多 dtype 消费法）

softplus_v2_grad（ops-info.ini `input0.dtype=float16,float32`，Ascend950 + CANN 9.1.0）补上输出侧宏注入与"一份源码服务全部 dtype 组合"的消费法（均为 build 产物可复现核对）：

**1) 注入侧全景**：输入 `grad_output`/`self` 得 `-DDTYPE_GRAD_OUTPUT=<type>`、`-DDTYPE_SELF=<type>`；**输出** `grad_input` 同样得 `-DDTYPE_GRAD_INPUT=<type>`（输出名不在输入集合内故无 `_REF` 后缀；`_REF` 后缀规则——同名参数在输入输出都出现时输出侧加后缀——本例未见，组合情况未实测）；另有 `-DORIG_DTYPE_*=DT_FLOAT` 与 `-DFORMAT_*=FORMAT_ND` 同族。类型映射 `DTYPE_MAP`（`build/ascendc_kernels/binary/dynamic/<op>.py`）：`float32→["DT_FLOAT", "float"]`、`float16→["DT_FLOAT16", "half"]`、`bfloat16→["DT_BF16", "bfloat16_t"]`；`get_dtype_fmt_options` 按 input/output 参数名生成宏名。

**2) kernel 侧消费法（多 dtype 单源码）**：构建已按 dtype 组合生成独立二进制（本例 2 个 .o + 2 个 kernel json，json 的 `inputs[].dtype` 分别为 float32/float16），kernel 只需别名消费：

```cpp
// dtype is chosen by the Kernel template parameter (DESIGN §5.1(3)); the build
// injects -DDTYPE_GRAD_OUTPUT / -DDTYPE_SELF / -DDTYPE_GRAD_INPUT (=float for
// the fp32 variant, =half for the fp16 variant). All three tensors share one
// dtype (DESIGN §2.2), so a single T alias is sufficient.
using T = DTYPE_GRAD_OUTPUT;
```

三 tensor 同 dtype 时单别名足够；异 dtype 组合（如 fp16 计算 fp32 输出）应分别用 `DTYPE_GRAD_OUTPUT/DTYPE_SELF/DTYPE_GRAD_INPUT`（未实测）。内置 matmul 即 `DTYPE_X1/DTYPE_X2` 同款用法。

**3) 探针**：`cat <pkg>/build/ascendc_kernels/binary/dynamic/<op>.py`（DTYPE_MAP 与 get_dtype_fmt_options）；`cat $ASCEND_OPP_PATH/.../kernel/ascend950/<Op>_*.json`（每个 dtype 组合一条，`inputs[].dtype` 字段）；`ls` 安装 kernel 目录数 `.o` 数。

坏实践补充：kernel 源码写死 `float`（fp16 组合被编译成错误 dtype 的访存/计算）；在 kernel 里写运行时 dtype 判断（tiling 传 dtype 标志做分支）——构建已按组合实例化，机制上多余且引入双倍代码路径；靠猜宏名写 kernel 而不先读 build 产物核对。

验证信号（复现增量，kernel 实现前机制调查 + 实现后构建/上板，均 tool_observed/950）：

- 机制判据：构建脚本 `DTYPE_MAP` 含 float32→float、float16→half；`get_dtype_fmt_options` 按 input/output 名生成 DTYPE_GRAD_OUTPUT/DTYPE_SELF/DTYPE_GRAD_INPUT。
- 实例化证据：安装目录下 2 个 kernel json + 2 个 .o（fp32/fp16 组合各一）。
- 实现验证：kernel 用 `using T = DTYPE_GRAD_OUTPUT` 后 `./build.sh` 退出 0、两个 dtype 变体均生成；ST 15/15 PASSED（fp32 与 fp16 用例都过）且两次运行输出逐位一致。

失效边界（复现增量）：

- 宏名由参数名大写化而来：参数名含非字母数字字符或与保留宏冲突时命名会变；`_REF` 后缀规则未实测。
- `DTYPE_MAP` 里标 `"unknown"` 的 dtype（string、resource 等）不能这样用。
- 本机制为编译期分发：dtype 组合数决定二进制数（组合爆炸时注意产物体积）；运行时选型由 aclnn 按 json 元数据路由。
- 其他 CANN 版本的注入宏名/映射表需重跑探针核对（未验证，推断机制稳定）。
## 误诊

- 本流曾在定位真错前先怀疑 TBE 编译器头文件/bisheng 内建类型问题（追查 `__BISHENG_BUILTIN_TYPE_PROXY_Float8_E6M2x2`、`ASCENDC_TPL_KERNEL`、`max_align_t` typedef 重定义等线索多轮）：FATBIN 阶段的 `ld.lld cannot open .o` 是 kernel 编译失败的**下游连坐**，真正原因藏在更早的 `error:` 行里。做法应该是干净重建并 grep 首个 `error:`，而不是顺着 python 栈追 TBE 内部。
- （R2 补充）报错后只盯模板语法行反复改声明顺序，不展开 `<command line>` 看宏——本例真实原因在命令行 `-DDTYPE_X=float`。
- （R3 补充）按报错行号去改 `typename` 关键字或模板语法——报错位置是宏展开后的残留，不改名永远修不好；在 kernel 头文件里 `#undef DTYPE_X` 会破坏入口点的模板实参传递。
