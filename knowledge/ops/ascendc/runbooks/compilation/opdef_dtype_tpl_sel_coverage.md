---
type: "Runbook"
title: DEF dtype 集须被 ASCENDC_TPL_SEL 全覆盖
description: "op_def 的 DataType 列表声明多 dtype 而 ASCENDC_TPL_SEL 只注册部分组合时，asc_op 编译系统仍按 DEF dtype 生成 tilingkey 包装调用，报 no matching function for call to '<kernel>_0_tilingkey' 并连锁 ld.lld cannot open .o；SEL 的 dtype 集合必须覆盖 DEF 全部 dtype，未实现分支用空 else 占位。复现增量：对无法/无需 SEL 注册的 key-0 包装单元，另一条实测修复路线是给 kernel 入口全部模板参数补默认值（默认值取主推分支）——TEMPLATE_PARAMS_LEN==0 时 compile_op.py 以无模板实参形式调用入口，带默认值的模板入口可同时承接模板/非模板两种调用形态；附 ccec -E -DASCENDC_TPL_PRE 宏声明探针。"
tags: [task.implementation, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# DEF dtype 集须被 ASCENDC_TPL_SEL 全覆盖

## 现象

在 x86_64 交叉编译环境（CANN 9.2.0）为自定义算子 BenchNoHarnessDepthwiseConv2d（目标 Ascend950PR/ascend950）做首版骨架编译。算子定义文件 `op_host/bench_no_harness_depthwise_conv_2d_def.cpp` 里给三个输入和输出都声明了 `DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})`（fp32/bf16 留待后续迭代实现），而模板参数文件 `op_kernel/arch35/bench_no_harness_depthwise_conv_2d_tiling_key.h` 里 `ASCENDC_TPL_SEL` 按当期计划只注册了 fp16 一个 `ARGS_SEL` 组合。host 侧（cust_optiling/cust_op_proto/cust_opapi）全部编译通过后，`cmake --build . --target binary` 在 kernel 阶段失败，报错原文：

```
.../kernel_meta_.../kernel_meta/..._kernel.cpp:27:5:
error: no matching function for call to 'bench_no_harness_depthwise_conv2d_0_tilingkey'
[ERROR] ... ld.lld: error: cannot open .../BenchNoHarnessDepthwiseConv2d_..._0.o: No such file or directory
[ERROR] ... Op[BenchNoHarnessDepthwiseConv2d] of index[0] compile failed
```

makefile 显示编译系统生成了 3 个 kernel target（`bench_no_harness_depthwise_conv2d0/1/2`，对应 3 个 bin_param json）。直接用 `asc_opc` 复现（`--main_func=bench_no_harness_depthwise_conv2d --soc_version=Ascend950PR_9599`）同样 `EXIT=1` 且 stdout 无错误详情，只能靠读编译系统 Python 源码定位。

## 适用场景

- 工程形态：registry-invoke 自定义算子仓开发（AscendC 模板编程，`ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL` + `op_def` 的 `DataType` 列表），按迭代分批实现多 dtype 的算子。
- API 路径：`ascendc/host_api/tiling/template_argument.h` 的 `ASCENDC_TPL_*` 宏族；`asc_op_compile_base`（compile_op.py / kernel_info_infer.py）驱动的 `asc_opc` kernel 编译流水线。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510），CANN 9.2.0，x86_64 交叉编译（验证等级为编译级，未上板）。
- 适用粒度：platform_wide——约束来自 asc_op 编译流水线本身，与具体算子无关。
- 路线优先级：recommended_fastpath（DEF/SEL 一致性是编译通过的必要条件）。
- 明确不适用 / 未验证场景：不使用 ASCENDC_TPL 模板编程（单 TilingKey 或 TILING_KEY_IS 旧宏）的算子；其他 CANN 版本的 key 分配算法未验证。

## 根因

算子定义（op_def 的 `DataType` 列表）声明的 dtype 集合必须被 `ASCENDC_TPL_SEL` 注册的模板实例化组合完全覆盖；编译系统按 DEF 侧 dtype 生成 tilingkey 包装调用，SEL 未注册的 dtype 组合没有对应实例化，包装调用解析失败。

机制链（本会话实测 + 编译系统源码读取）：

1. `asc_op_compile_base/asc_op_compiler/compile_op.py:2142/2368`：编译命令带 `-D{origin_func_name}={origin_func_name}_{tiling_key}_tilingkey`，即核函数在编译单元内被宏重命名为 `<kernel>_<tilingkey>_tilingkey`。
2. 生成的 `*_kernel.cpp` 包装内 `gen_template_tiling_params` 产出 `#if {TILING_KEY_MACRO} == {key}UL / #define TEMPLATE_PARAMS {TypeFromId<dtype>::type, ...}` 分档，再调用重命名后的模板函数。
3. DEF 声明 3 dtype 时编译系统生成 3 个 binary target；SEL 只注册 fp16（`ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)` × `IS_STRIDED∈{0,1}`）时，包装侧引用的 `_0_tilingkey`（对应未注册 dtype 的 key）无匹配实例化 → `no matching function for call`，前置编译失败又导致 `.o` 缺失 → `ld.lld: cannot open ... .o`。

## 解决配方

修复前（tiling_key.h：SEL 仅注册 fp16 一个组合；kernel 入口只有 fp16 分支无 else）：

```cpp
// op_kernel/arch35/bench_no_harness_depthwise_conv_2d_tiling_key.h（修复前）
// SEL 列出实际实例化的模板组合
// 首版：仅 (fp16, IS_STRIDED=0)
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(IS_STRIDED, ASCENDC_TPL_UI_LIST, 0, 1)
    )
);
```

```cpp
// op_kernel/bench_no_harness_depthwise_conv_2d_arch35.cpp（修复前）
    // 按 dtype 分发到具体实现类（if constexpr 编译期分发）
    // 首版：仅 fp16 路径实例化
    if constexpr (std::is_same_v<D_T_X, half>) {
        NsBenchNoHarnessDepthwiseConv2d::BenchNoHarnessDepthwiseConv2dOp<IS_STRIDED> op;
        op.Init(x, weight, bias, y, &tilingData);
        op.Process();
    }
    // 后续扩展：fp32 / bf16 路径
```

修复后（SEL 覆盖 DEF 声明的全部 3 dtype × IS_STRIDED={0,1}；kernel 入口为未实现 dtype 补空 else 分支保证可实例化）：

```cpp
// op_kernel/arch35/bench_no_harness_depthwise_conv_2d_tiling_key.h（修复后）
// SEL 注册全 dtype（fp16/fp32/bf16）× IS_STRIDED={0,1}
// 当期仅实现 fp16+IS_STRIDED=0 路径，其余分支留空（后续迭代填充）
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(IS_STRIDED, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(IS_STRIDED, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(IS_STRIDED, ASCENDC_TPL_UI_LIST, 0, 1)
    )
);
```

```cpp
// op_kernel/bench_no_harness_depthwise_conv_2d_arch35.cpp（修复后）
    // 按 dtype 分发到具体实现类（if constexpr 编译期分发）
    // 当期仅 fp16 路径实现，fp32/bf16 路径留空（后续迭代填充）
    if constexpr (std::is_same_v<D_T_X, half>) {
        NsBenchNoHarnessDepthwiseConv2d::BenchNoHarnessDepthwiseConv2dOp<IS_STRIDED> op;
        op.Init(x, weight, bias, y, &tilingData);
        op.Process();
    } else {
        // 后续扩展：fp32 / bf16 路径（当前留空，避免实例化失败）
    }
```

判别方法（本会话走通的排障链路）：

1. 看到 `no matching function for call to '<kernel>_<N>_tilingkey'` 先核对该 N 是否落在你 SEL 注册的组合内：本例 `_0_tilingkey` 的 0 对应 DEF 里声明但 SEL 未注册的 dtype。
2. 编译系统报错点在被自动清理的 `kernel_meta/*/_kernel.cpp`（失败后目录被删，`find` 找不到）；读 `python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py` 的 `-D{origin_func_name}=..._{tiling_key}_tilingkey`（2142/2368 行）与 `gen_template_tiling_params`/`gen_kernel_fun` 确认包装如何实例化。
3. 对照 `build/op_kernel/<Op>_<soc>/bin_param/*.json`（含 attrs 与 bin_filename）和该目录下 makefile 的 target 列表，确认编译系统实际要生成几个 binary、对应哪些 dtype。

## 设计决策

- 为什么选择"SEL 补齐全 dtype + 空分支"而不是"DEF 只声明 fp16"：设计文档要求 DEF 声明完整支持范围（接口契约），迭代计划靠 SEL/实现侧渐进填充；把 DEF 收窄到 fp16 会让接口契约随迭代漂移。空 `else` 分支是显式的迭代占位，编译期无代码。
- 备选方案"SEL 只注册 fp16、DEF 也只写 fp16，后续再同步扩两者"被放弃：接口文档已按 3 dtype 发布，回改契约成本更高。
- 当前方案路线档位：recommended_fastpath（多 dtype 算子迭代开发的标准姿势）。

## 系统信息

- 硬件/软件：x86_64 主机，CANN 9.2.0，目标 ascend950（Ascend950PR）。
- 相关文件：`op_host/bench_no_harness_depthwise_conv_2d_def.cpp`（DataType 声明）、`op_kernel/arch35/bench_no_harness_depthwise_conv_2d_tiling_key.h`（SEL 注册）、`op_kernel/bench_no_harness_depthwise_conv_2d_arch35.cpp`（模板入口）、`build/op_kernel/BenchNoHarnessDepthwiseConv2d_ascend950/`（bin_param json + makefile）、`$ASCEND_HOME_PATH/python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py`。
- 复现命令：`cd build && rm -rf * && cmake -DASCEND_COMPUTE_UNIT=ascend950 .. && cmake --build . --target binary -- -j8`。

## 预防

- 写 `ASCENDC_TPL_SEL` 前先抄一遍 op_def 的 `DataType({..})` 列表：SEL 的 dtype 集合 ⊇ DEF 的 dtype 集合是硬约束，逐项核对。
- 分迭代实现多 dtype 时，一律"DEF 全量 + SEL 全量 + kernel `if constexpr` 空 else 占位"，不要"DEF 全量 + SEL 子集"。
- 编译失败时不要只看 ld.lld 的 `.o` 缺失（那是连锁错误），向上找第一条 `error:`——本例真正的锚点是 `no matching function for call to ..._tilingkey`。

## 验证信号

以下信号来自该算子首版骨架的编译联调：修复前 kernel 阶段稳定复现 `_0_tilingkey` 报错，补齐 SEL 与空分支后全量重编通过。

- 修复前：`cmake --build . --target binary` 报 `error: no matching function for call to 'bench_no_harness_depthwise_conv2d_0_tilingkey'`（两个 kernel_meta 均命中）+ `ld.lld: error: cannot open ..._0.o: No such file or directory` + `Op[BenchNoHarnessDepthwiseConv2d] of index[0] compile failed`；`asc_opc` 直接调用 EXIT=1（来源平台：950；证据等级：tool_observed）。
- 修复后：`rm -rf build && cmake -DASCEND_COMPUTE_UNIT=ascend950 .. && cmake --build . --target all binary package -- -j8` 全绿，输出 `Self-extractable archive "custom_opp_ubuntu_x86_64.run" successfully created`（来源平台：950；证据等级：tool_observed）。
- 修复后：`op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_depthwise_conv2d/` 下生成 3 个 `.o` 与 3 个 `.json`；`bash build.sh` 完整流程同样成功（来源平台：950；证据等级：tool_observed）。
- 机制锚点：`compile_op.py:2142/2368` 的 `-D{origin_func_name}={origin_func_name}_{tiling_key}_tilingkey` 与 `gen_template_tiling_params` 的 `#if TILING_KEY == key` / `TypeFromId<>::type` 模板参数注入（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论以 CANN 9.2.0 的 asc_op_compile_base 实现为准；其他版本 key 分配或 DEF/SEL 一致性检查逻辑可能变化（未验证）。
- 验证止于编译级：fp32/bf16 空分支算子被调用到时行为未定义（本会话 TilingFunc 侧对非 fp16 dtype 显式报错拦截），上板行为未验证。
- "6 个 SEL 组合生成 3 个 binary"的归并规则未逐条核实，不影响 DEF⊆SEL 结论但影响对 target 数量的精确预期（未验证）。

**未决项**：

- 优先项：把 asc_op 编译系统"DEF dtype 列表 → tilingkey 列表 → binary target 数"的精确映射逐行核实（本会话只验证了补齐 SEL 即通过，未追到 key 分配算法机制层）。
- fp32/bf16 空分支在上板路径的实际兜底行为（当前依赖 host 侧 TilingFunc 的 dtype 校验拦截，kernel 侧无防御）未验证。
## 坏实践

- DEF 声明多 dtype"先把接口立起来"、SEL 只注册当前迭代子集——编译系统不会替你跳过未注册 dtype 的包装生成。
- 看到 `ld.lld: cannot open *.o` 就去查磁盘/权限/清理脚本——它是前置编译失败的连锁产物，不是根因。
- `asc_opc` 手工复现时 stdout 无输出就放弃——错误细节在编译系统内部日志/源码里，需要读 compile_op.py。

## 误诊

- 曾怀疑 kernel 入口模板用了 `if constexpr` 导致实例化异常：把入口改回普通模板分发前先重命名了函数，错误随之变化，证明 `if constexpr` 不是因；真因是 SEL 覆盖不全。
- 曾试图从 `kernel_meta/*/_kernel.cpp` 读包装代码：失败后目录被自动清理（find 无结果），改从 compile_op.py 的生成逻辑还原——直接证据拿不到时要换证据源而不是猜。

## 增量（CANN 9.1.0 复现与失败替代路径）

以下增量来自另一算子（BenchNoHarnessConv2d，同样 op 定义 3 dtype / SEL 仅声明 fp16）在 CANN 9.1.0 + `bash build.sh --soc=ascend950 -j8` 构建流上的独立复现，与上文 9.2.0 结论一致（DEF/SEL 一致性是跨版本约束）。

**内置算子先例（"实现可滞后，声明必须齐全"的官方对照）**：对照 CANN 内置 arch35 算子 `acosh`（`opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/acosh/arch35/acosh_tiling_key.h`），其 tiling_key.h 同样把 3 个 dtype 全部声明进 `ASCENDC_TPL_DATATYPE_DECL`/`ASCENDC_TPL_SEL`，注释明确"仅 FP32 路径完整实现；FP16/BF16 路径后续完成。这里同时声明 3 个 dtype 仅为保持 TilingKey 接口稳定"——即多 dtype 渐进实现时的标准姿势是"DEF 全量 + SEL 全量 + 实现滞后"，与本卡"空 else 占位"方案互为印证。

**两条实测失败的替代路径（勿再走）**：

1. 手写显式实例化"绕过"SEL：

```cpp
template __global__ __aicore__ void bench_no_harness_conv2d<half, 0>(GM_ADDR, ...);
```

报 `explicit instantiation cannot be 'inline'`（`__aicore__` 隐含 inline，C++ 不允许显式实例化 inline 函数）——实例化由 ASCENDC_TPL 系统生成，不应手写。

2. 退回非模板 kernel + 手工 `SetTilingKey(0)` 并清空 tiling_key.h 宏：报 `bench_no_harness_conv2d_1_tilingkey does not name a template but is followed by template arguments`（构建系统仍按 DEF dtype 生成带后缀的包装目标，非模板入口无法承接），且丢掉多 dtype 能力。模板规范同时禁用旧式 `TILING_KEY_IS` 宏。

**dtype 编码锚点**：`C_DT_FLOAT=0、C_DT_FLOAT16=1、C_DT_BF16=27`（与 `ge::DT_*` 对应，CANN 9.1.0 头文件/内置算子佐证）——bin_param 侧确认 kernel_N 与 dtype 对应关系时可直接对照。

**边界补充**：

- `ASCENDC_TPL_SEL` 条目书写顺序决定 TilingKey 编号的分配顺序，DECL 与 SEL 的 dtype 顺序须保持一致。
- 补全声明后，未实现的 dtype 路径会参与编译——若该 dtype 的 kernel 代码本身不兼容（如 bf16 标量运算限制），会在该目标上报新的编译错误（属实现侧问题，不是本根因复发）。
- `-j8` 并行下先看到 `ld.lld: cannot open ..._0.o` 链接错误，须串行 `-j1` 重跑或向上翻第一条 `error:` 拿真实编译错误（与上文预防一致，9.1.0 复现再证）。

**验证信号（复现增量，CANN 9.1.0 构建流）**：

- 修复前：kernel_0（fp16）生成 .o 成功，kernel_1/kernel_2（fp32/bf16）报 `no matching function for call to 'bench_no_harness_conv2d_0_tilingkey'` + `candidate template ignored: couldn't infer template argument 'D_T_X'`（来源平台：950；证据等级：tool_observed）。
- 修复后：`bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 三个 kernel 二进制（fp16/fp32/bf16）全部生成，`Build completed successfully`，`custom_opp_ubuntu_x86_64.run` 打包成功；再次 clean 重建复跑通过，`binary/ascend950/<op>/` 下 3 个 .o 齐全，`--force` 安装成功（来源平台：950；证据等级：tool_observed）。
- bin_param 取证：读 `build/op_kernel/<Op>_<soc>/bin_param/*.json` 与 kernel_meta 目录，确认 3 个编译目标按 dtype（fp16/fp32/bf16）而非 Cube/Vector 通道划分（来源平台：950；证据等级：tool_observed）。

**坏实践补充**：

- op 定义声明 3 种 dtype，tiling_key.h 只写当前迭代实现的那一种（"先只声明做得到的"）。
- 手写 `template ... <half, 0>` 显式实例化，或退回非模板 + 手工 `SetTilingKey`——实例化与包装目标由构建系统按 DEF 驱动，绕不开。

## 增量（入口模板参数默认值吸收 key-0 无模板实参调用）

同一报错家族（`no matching function for call to '<op>_0_tilingkey'`）在"SEL/DEF/入口签名本身一致、但存在 key-0 无模板实参包装单元"的形态下的另一条实测修复路线（GQA 类算子，Ascend950，CANN 9.1.0，纯 Vector 模板 kernel，`ASCENDC_TPL_DATATYPE_DECL(D_T_Q, C_DT_FLOAT16, C_DT_BF16)` + 两个 8bit UINT，SEL 仅展开 fp16 × {0,1} × {0,1} = key 1/257/65537/65793）。

**机制（compile_op.py 源码直读）**：`gen_kernel_fun` 生成的 kernel 封装以 `#if defined(TEMPLATE_PARAMS_LEN) && TEMPLATE_PARAMS_LEN != 0` 切换两种调用形态——模板参数可提取时调 `算子名<TEMPLATE_PARAMS>(args...)`，否则调 `算子名(args...)`（无模板实参）。key-0 包装单元（无论其来源是 DEF 中 key 为 0 的 dtype 还是生成器的兜底目标）没有可提取的模板参数，`TEMPLATE_PARAMS_LEN` 保持 0，走非模板调用；模板入口不写默认值时无法解析该调用。反向验证：把 kernel 改成完全非模板后，SEL 列表单元反而报 `'...' does not name a template but is followed by template arguments`（与复现增量的同名失败形态一致）——两类包装单元并存，只有"模板入口 + 全默认参数"能同时承接两种调用形态。

**修复（仅改一行，before/after verbatim）**：

```cpp
// 修复前（kernel 入口，模板参数无默认值）
template <typename D_T_Q, int IS_CAUSAL, int SCALE_MODE>
__global__ __aicore__ void bench_no_harness_gqa(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessGqaTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGqaTilingData, tilingData, tiling);
    NsBenchNoHarnessGqa::BenchNoHarnessGqa<D_T_Q, IS_CAUSAL, SCALE_MODE> op;
    op.Init(query, key, value, y, workspace, &tilingData);
    op.Process();
}

// 修复后（全部模板参数补默认值，默认值取主推 TilingKey 分支：fp16 + 非因果 + 自动 scale）
template <typename D_T_Q = half, int IS_CAUSAL = 0, int SCALE_MODE = 0>
__global__ __aicore__ void bench_no_harness_gqa(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessGqaTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessGqaTilingData, tilingData, tiling);
    NsBenchNoHarnessGqa::BenchNoHarnessGqa<D_T_Q, IS_CAUSAL, SCALE_MODE> op;
    op.Init(query, key, value, y, workspace, &tilingData);
    op.Process();
}
```

默认参数只保证"key-0 单元可编译"，兜底单元实际执行语义等于默认分支；若 host 侧从不下发 TilingKey=0 则运行期不受影响（本轨迹未观测到运行期走兜底分支）。

**宏声明探针（排障可复用，一分钟排除"宏写错"假设）**：

```bash
ccec -E -DASCENDC_TPL_PRE -I op_kernel -I op_kernel/arch35 <入口.cpp> | grep '@@'
# 能打出 @@ASCENDC_TPL_ARGS_DECL_<OpName>@@ 完整标记说明 ASCENDC_TPL 宏声明与预处理无误，
# 不要再改 tiling_key.h；报错另有来源（包装调用形态）。
```

**验证信号（复现增量，GQA 骨架编译会话，Ascend950，CANN 9.1.0）**：

- 修复前：`cmake OK / build exit: 2`，kernel_1 报 `no matching function for call to 'bench_no_harness_gqa_0_tilingkey'`；改成非模板后 kernel_0 报 `'bench_no_harness_gqa_1_tilingkey' does not name a template but is followed by template arguments`（枚举出 TilingKey 1/257/65537/65793 四分支）（来源平台：950；证据等级：tool_observed）。
- 修复前：手动 `ccec -E -DASCENDC_TPL_PRE` 可见 `@@ASCENDC_TPL_ARGS_DECL_BenchNoHarnessGqa@@` 完整标记，证明宏声明与预处理无误（来源平台：950；证据等级：tool_observed）。
- 修复后（仅补默认模板参数）：`cmake OK / build exit: 0`，`custom_opp_ubuntu_x86_64.run` 成功打包；`build/op_kernel/ascendc_kernels/binary/ascend950/bench_no_harness_gqa/` 下生成 4 个 kernel 二进制（与 4 个 SEL TilingKey 对应）（来源平台：950；证据等级：tool_observed）。
- API 事实锚点：`compile_op.py` 生成代码以 `#if defined(TEMPLATE_PARAMS_LEN) && TEMPLATE_PARAMS_LEN != 0` 切换模板/非模板调用（来源平台：950（CANN 9.1.0 安装树源码读取）；证据等级：tool_observed）。

**失效边界补充**：默认参数路线的前提是"SEL/DEF/入口四处签名一致、失败单元确为无模板实参调用"——若失败单元对应 DEF 声明而 SEL 未注册的 dtype（本卡主形态），应走 SEL 补齐/DEF 收敛路线，默认参数只是让未注册单元可编译的兜底，不消除路由缺口。判别顺序：先核 bin_param 与 SEL 覆盖（本卡判别方法），确认无缺口后才考虑默认参数。CANN 9.1.0 的 `asc_op_compile_base` 生成逻辑为据；其他版本未验证（推断可迁移）。

## 后续增量（2026-09-17，并卡合并：三条独立 trace 的修复路线补全——收窄 def、五层全收敛与 TilingData 字段分流）

#### 并入自 def_dtype_list_tpl_sel_instantiation_mismatch（路线 B 收窄 def、wrapper 宏重命名细节与 DataTypeList 笛卡尔积形态）

- **wrapper 报错细节（判别锚点）**：生成的 `*_kernel.cpp` 内 `#define <op> <op>_0_tilingkey` 后以 GM_ADDR 实参直接调用核函数，报 `no matching function for call to '<op>_0_tilingkey'` + `note: candidate template ignored: couldn't infer template argument 'D_T_MAIN'`（typename 模板参数无法从 GM_ADDR 实参推断）；fp16 binary 先 `Generating ... Done`、失败的是未注册 dtype 的 kernel_1/kernel_2（bin_param 的 `*_param.json` 内 `inputs[0].dtype` 可证对应关系）。任一 dtype 组合实例化失败都会使整个 `binary` 目标失败——即使 fp16 那份已经编译成功。
- **路线 B（收窄 def，接口契约未发布时）**：def.cpp 与 op_graph proto.h 的 dtype 列表全部收窄为长度 1（与 SEL 已注册组合一致），后续迭代再同步扩展。取舍：def.cpp dtype 列表同时承担"算子声明支持范围（编译期生成 binary）"与"opbuild 等长约束"两个角色；收窄后 aclnn L2 的运行时 dtype 校验列表可保留多 dtype，运行期由 `CheckDtypeValid` 拒绝未支持项。后续扩 N 组 TilingKey 时必须三处同步——def.cpp 列表、SEL、kernel 模板分支。
- **DataTypeList 笛卡尔积形态（另一独立轨迹并入）**：def 用 `DataTypeList` 想表达"输出 dtype 跟随输入"，实际语义是多个输入/输出之间做**笛卡尔积**（x 3 × scalar 1 × y 3 = 9 组合，其中出现 x:float32→y:float16 这类非法组合），与 TPL_SEL 模板组失配；缺失 `.o` 编号恰是 BF16 组的 TilingKey 值（27/283/539/795/1051/1307 = `27 | n<<8`）。**INI 组合数探针**：读 `build/autogen/aic-<soc>-ops-info.ini` 的 `input0.dtype=/output0.dtype=` 行数组合数（9 ≠ 模板组数 3 即失配）。修复两处一起改：**1.** def 回到等长 `DataType`（zip 出恰好 N 个合法组合）；**2.** tiling_key.h 把单组多值 `ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16)` 拆成每个 dtype 一个 `ARGS_SEL` 组（每组含全部 UINT 值）——"等长 DataType + 每 dtype 独立组"而不是"保留 DataTypeList 并扩模板"（9 组合里含非法语义，扩模板等于为非法组合买单）。`ASCENDC_TPL_DATATYPE_SEL` 单组多值在 def 侧组合正确时是否可用未验证。
- 误诊补充（并入）：曾怀疑 kernel 入口 GM_ADDR 参数个数与 def 注册的 input/output 数不匹配（7 输入 + 1 输出 + workspace + tiling = 10）——逐一核对 wrapper 调用与核函数签名完全一致，排除；曾怀疑 `ASCENDC_TPL_INPUT(0)` 绑定写法与官方样例不一致——对比模板逐字一致，排除。

验证信号（并入，MoE FinalizeRouting + foreach 范数两个独立算子，Ascend950PR + CANN 9.1.0）：

- 修复前：3 份 `bin_param/*_param.json`（fp16/fp32/bf16 的 hash 可对照），kernel_1 wrapper 报 `no matching function ... 'bench_no_harness_moe_finalize_routing_0_tilingkey'` + `candidate template ignored: couldn't infer template argument 'D_T_MAIN'`，`ld.lld: cannot open ..._0.o`（来源平台：950；证据等级：tool_observed）。
- 修复后（路线 B 收窄）：清空重建全量成功，生成 kernel 二进制、`custom_opp_ubuntu_x86_64.run`（455055 字节）及 liboptiling/libcust_opapi/libcust_opsproto 等库；复跑 grep error/warning 为空、`EXIT: 0`（来源平台：950；证据等级：tool_observed）。
- 独立 trace：DataTypeList 状态 INI 生成 9 个 dtype 组合（含非法组合），kernel 编译报 `no matching function ... 'bench_no_harness_foreach_norm_0_tilingkey'`，ld.lld 连锁缺 `_mix_aiv_27.o / _283.o / _539.o / _795.o / _1051.o / _1307.o`；改等长 DataType + 每 dtype 独立 ARGS_SEL 组后同错误消失，与 BF16 Cast 修复共同完成后 `Build completed successfully!`、产出 3 个 kernel binary（每 dtype 一个、各含 6 NORM_TYPE）（来源平台：950；证据等级：tool_observed）。

#### 并入自 tpl_sel_dtype_variant_link_fail（五层全收敛路线与 TilingKey-only 收敛无效的对照实验）

- **路线 F（骨架期五层全收敛单 dtype）**：除"SEL 补齐 + 空 else 占位"外，另一条实测走通的路线是把 **OpDef/proto/tiling 白名单/op_api 白名单（L0/L2）/TilingKey** 五层的 dtype 集合全部收窄为已实现集合（适用于接口契约尚未发布、迭代一只交付单 dtype 的场景）。
- **对照实验（路线必要性证明）**：仅把 `tiling_key.h` 的 `ASCENDC_TPL_DATATYPE_DECL` 收敛为单 dtype（及 TilingKey/入口/类/SEL_PARAM 四处签名一致化）后，非 FP32 失败变体依旧存在；只有把 OpDef `.DataType({ge::DT_FLOAT})`、proto `TensorType({DT_FLOAT})`、tiling `supportedDtype = {ge::DT_FLOAT}`、L0/L2 白名单 `{DataType::DT_FLOAT}` 全链路收敛后，失败变体才消失——**变体生成集合由 OpDef dtype 列表驱动，TilingKey 声明不决定变体生成集合**。
- **RegBase 形态的定位困难（并入边界）**：失败变体的真实编译错误可能既不回显也找不到 kernel_meta 编译日志（失败变体的 `kernel_meta/` 目录不生成，`find` 只有 bin_param json），定位靠变体哈希差分 + `bin_param/*_param.json` 反查 dtype。
- 误诊补充（并入）："ASCENDC_TPL_SEL 在非 ASCENDC_DEBUG 模式是空宏，所以 SEL 不影响编译"——从 `template_argument.h` 源码看 `#define ASCENDC_TPL_SEL(...)` 在非 debug 下确实展开为空，据此曾推断 SEL 与变体生成无关；但成对观测（补注册前后）证明 SEL 注册集合事实上影响变体 wrapper 能否解析到函数实例，宏在 kernel 编译路径中被消费的机制层未完全追明。
- 两条路线的选择依据（并入）：接口契约（aclnn 头/测试设计）已按多 dtype 发布 → SEL 补齐 + 空 else 占位（改动最小）；契约未发布、当期只交付单 dtype → 五层全收敛（变体数最小化缩短报错→定位→修复循环）。
- 坏实践补充（并入）：想通过收窄 `ASCENDC_TPL_SEL` 来"少编译几个变体"——变体生成由 op_def 的 dtype 集合驱动，收窄 SEL 只会让未注册变体链接失败；并行构建（-j8）下看到 `ld.lld cannot open *.o` 就去查链接器/路径权限——先串行重跑拿编译期真实报错。

验证信号（并入，Exp/ApplyAdamW/ForeachAddcdivScalar 三个独立算子，Ascend950PR + CANN 9.1.0）：

- Exp 修复前（-j1 串行）：FP16 变体 `Done`；FP32 变体报 `error: no matching function for call to 'bench_no_harness_exp_0_tilingkey'`（wrapper 第 27 行）+ `ld.lld: cannot open ..._0.o`，FATBIN 阶段失败；`bin_param/` 下 3 个变体脚本，失败变体 param.json 为 `dtype: float32`；修复后三个变体全部 `Done`，run 包生成，3 个 kernel 二进制 `.o` 与 `.json` 全部落盘（来源平台：950；证据等级：tool_observed）。
- ApplyAdamW（OpDef 3 dtype、SEL 只选 FP32+单缓冲）：FP16/BF16 在 FATBIN 报 `ld.lld: cannot open ..._0.o`；`-j1` 全量日志 grep 出 `kernel.cpp:27:5: error: no matching function for call to 'bench_no_harness_apply_adam_w_0_tilingkey'`；SEL 补齐 3 dtype 单缓冲组合 + `if constexpr` 占位后 3 个 kernel 全部生成、构建退出码 0（来源平台：950；证据等级：tool_observed）。
- ForeachAddcdivScalar（五层收敛路线）：连续多轮构建失败变体只呈现 `ld.lld cannot open ..._27.o/_283.o` 与 `...json not generated`；仅收敛 TilingKey 声明后非 FP32 失败变体依旧；OpDef/proto/tiling/L0/L2 五层全收敛 FP32 后 `Build completed successfully!`，唯一变体 `.o`（10024 字节）与 run 包（446127 字节）生成（来源平台：950；证据等级：tool_observed）。

#### 并入自 opdef_dtype_exceeds_tpl_sel_wrapper（template_tiling.py 机制链与 TilingData 字段分流路线）

- **机制链补全（编译器源码直读）**：`TEMPLATE_PARAMS_LEN` 来自 template_tiling_info——由 `ASCENDC_TPL_ARGS_DECL`/`SEL` 的预处理 token 与 OpDef 的 dtype 声明**共同**解析（`template_tiling.py` 的 `extract_decl_param_options` 会读取 OpDef 输入的 dtype 列表）；OpDef 3 dtype vs SEL 1 dtype 时，SEL 内已覆盖变体 wrapper 带 `TypeFromId<...>::type` 模板实参、编译通过，未覆盖变体落入裸调用分支 → 模板推导失败 → `.o` 未生成 → FATBIN 链接失败；wrapper 同时用 `-D` 把入口函数重命名为 `<func>_<tilingKey>_tilingkey`。
- **路线 G（单 dtype 模板参数 + 分支标识走 TilingData 字段）**：分支标识（如 downsample/global_avg/upsample）不需要编译期特化时，去掉 UINT 模板参数（`TILING_KEY`），只保留 dtype 模板参数；tilingKey 值写入 `tiling->tilingKey` 字段、kernel 侧从 TilingData 读取分支；host 侧选择调用相应收敛为 `ASCENDC_TPL_SEL_PARAM(context, dTypeX)`。取舍：CANN 9.1.0 的 `template_tiling.py` 对 UI_LIST 值串的解析（`extract_num` 用 `\d+` 抽数字）让"UI 标志位是否计入值列表"的行为不透明，去掉 UINT 模板参数可消除这一变量。
- **判别中间步骤（并入）**：先去掉 TILING_KEY 模板参数、OpDef 仍声明 3 dtype 时，已覆盖变体（如 6b6f…）`Done` 而未覆盖变体（另一独立 trace 的实例）仍报同一 no matching function 错——据此定位 OpDef dtype 列表是剩余变量。
- 坏实践补充（并入）：把 ASCENDC_TPL 宏当成"kernel 侧也能拿到模板参数"的机制——`ASCENDC_TPL_KERNEL` 阶段这些宏展开为空，kernel 编译单元拿不到任何类型信息，类型只能来自 wrapper 的显式模板实参。
- 误诊补充（并入）：先按"ASCENDC_TPL_UI_LIST 的值解析把 UI 标志位计入值列表"的假设去掉 TILING_KEY 模板参数；修复后未覆盖变体仍失败，说明 UI_LIST 解析并非（唯一）根因——真正的剩余变量是 OpDef dtype 列表。

验证信号（并入，AdaptiveAvgPool3d，Ascend950PR + CANN 9.1.0）：

- 修复中途：`[ascend950] Generating BenchNoHarnessAdaptiveAvgPool3d_6b6f7d4d65f2573ee6043af0dfd3eba5 Done`，另一独立 trace 的变体仍报同一 no matching function 错——定位 OpDef dtype 列表是剩余变量（来源平台：950；证据等级：tool_observed）。
- OpDef 收敛为 `{ge::DT_FLOAT}` 后：只生成 6b6f… 一个 kernel 变体，`[INFO] Build completed successfully!`，产出 `binary/ascend950/bench_no_harness_adaptive_avg_pool3d/BenchNoHarnessAdaptiveAvgPool3d_6b6f7d4d65f2573ee6043af0dfd3eba5.o`（12368 字节）与 `custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。

并入后失效边界汇总：

- `ASCENDC_TPL_UINT_SEL`/UI_LIST 多值作模板参数的写法在并入轨迹未走通（与路线 G 的取舍互证）；`ASCENDC_TPL_SEL` 为 def dtype 组合超集时的行为（多余条目被忽略还是报错）未验证。
- "def zip"与"模板拆组"两个改动在同一次重建中联合应用，各自独立贡献未隔离验证。
- autogen 为"OpDef 有、SEL 无"的 dtype 生成无模板实参 wrapper 的确切枚举路径未直证；复现时可保留 `build/op_kernel` 下生成的 `*_kernel.cpp` 对照分析（优先项）。