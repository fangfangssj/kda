---
type: "Runbook"
title: "kernel entry not implement — 入口须随算子名"
description: "算子重命名后 TBE 编译在 infer compile info 阶段报 kernel entry not implement：kernel 入口 __global__ 函数名必须等于算子名的 snake_case（照设计文档写 PascalCase、跟随文件名写 _arch35 后缀、照目录名拆数字段写 conv_3d/conv_2d/_3d 同样触发）。复现增量：四条独立 trace 定形 CamelCase→snake_case 数字段转换规则（Conv3d→conv3d、Conv2d→conv2d、GridSampler3d→grid_sampler3d，数字不开启新分段），makefile --main_func 与 binary 输出目录名可直接取官方推导名。复现增量：Conv2d 与 AdaptiveAvgPool3d 两条独立 trace 再证数字段规则（含 3 kernel 变体并行生成全败与修复后 .o 落盘闭环）。"
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# kernel entry not implement — 入口须随算子名

## 现象

在 Ascend950PR + CANN 9.2.0 环境下，为解决自定义算子与内置算子的 aclnn 符号冲突，把算子从 `MlaProlog` 全链路重命名为 `BenchNoHarnessMlaProlog`（`op_kernel/CMakeLists.txt` 的 `OP_TYPE` 与 `op_kernel/arch35/mla_prolog_tiling_key.h` 的 `ASCENDC_TPL_ARGS_DECL` 均已同步改新名）。随后清理构建目录重新执行 `bash build.sh --soc=ascend950`，宿主侧目标（cust_optiling/cust_opapi/cust_op_proto）全部编译通过，但在 kernel 生成阶段失败：

```
[ 83%] Built target ascendc_kernels_ops_info_gen_ascend950
[100%] Built target mla_prolog_custom_ascend950_cust_optiling
[ascend950] Generating BenchNoHarnessMlaProlog_764544817f6b6950f47d1ff41032d6c7 ...
[ERROR] TBE(1583983,python3):2026-07-23-19:18:15.797.688 [log_utils.py:232][plog_print] [ascend950] kernel entry `bench_no_harness_mla_prolog' not implement in `mla_prolog_arch35.cpp', please check whether the function name is correct in the kernel file.
Opc tool start working now, please wait for a moment.
[2026-07-23 19:18:15] [ERROR] [ascend950] kernel entry `bench_no_harness_mla_prolog' not implement in `mla_prolog_arch35.cpp', please check whether the function name is correct in the kernel file.
[ERROR] TBE(1583983,python3):2026-07-23-19:18:15.797.871 [op_compilation.py:397][__single_op_compile] Exception occured, An error occurred during stage of infer compile info
```

报错明确指向 `op_kernel/mla_prolog_arch35.cpp`：构建系统在找名为 `bench_no_harness_mla_prolog` 的 kernel 入口函数，而该文件里的入口函数仍叫旧名 `mla_prolog`。

## 适用场景

- 工程形态：算子仓开发（Ascend C kernel + custom opp 包），`bash build.sh --soc=<soc>` 构建。
- API 路径：`ASCENDC_TPL_ARGS_DECL(<OpName>, ...)`（tiling key 声明）与 `op_kernel/CMakeLists.txt` 的 `OP_TYPE <OpName>`；TBE/opc 编译链按算子名派生 kernel 入口。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（编译目标 soc=ascend950）。
- 适用粒度：operator_family——任何 Ascend C 自定义算子的命名/重命名操作。
- 路线优先级：recommended_fastpath（重命名算子时入口函数必须同步改，无替代方案）。
- 明确不适用/未验证场景：不涉及算子改名的普通 kernel 修改；其他 CANN 版本的报错文案可能不同但派生规则一致（未验证，推断可迁移）。

## 根因

构建系统（opc/TBE 编译链）按 `OP_TYPE` 算子名的 **snake_case** 派生 kernel 入口函数名：`BenchNoHarnessMlaProlog` → `bench_no_harness_mla_prolog`（构建时生成的 `BenchNoHarnessMlaProlog.py` 包装器按该名字调用 C++ 入口）。`ASCENDC_TPL_ARGS_DECL(BenchNoHarnessMlaProlog, ...)` 声明的模板实参也与该入口绑定。而 `mla_prolog_arch35.cpp` 里的入口函数仍是旧算子名的 snake_case `mla_prolog`，构建系统在文件中找不到 `bench_no_harness_mla_prolog`，于是在 infer compile info 阶段直接报 `kernel entry ... not implement`。入口函数名跟随的是**算子名**，不是源文件名（文件名 `mla_prolog_arch35.cpp` 可以不改）。

## 解决配方

把 kernel 入口函数改名为新算子名的 snake_case（模板签名与函数体不动，仅改函数名；改动前后完整函数签名如下，取自当次修复的实际 edit）：

```cpp
// before（op_kernel/mla_prolog_arch35.cpp）
template <uint32_t SCH_MODE>
__global__ __aicore__ void mla_prolog(
    GM_ADDR token_x, GM_ADDR w_dq, GM_ADDR w_uq_qr, GM_ADDR w_uk, GM_ADDR w_dkv_kr,
    GM_ADDR rmsnorm_gamma_cq, GM_ADDR rmsnorm_gamma_ckv, GM_ADDR rope_sin, GM_ADDR rope_cos,
    GM_ADDR query, GM_ADDR query_rope, GM_ADDR c_kv, GM_ADDR k_rope,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MlaPrologTilingData);
    GET_TILING_DATA_WITH_STRUCT(MlaPrologTilingData, tilingData, tiling);
    NsMlaProlog::MlaPrologKernel<SCH_MODE> op;
    op.Init(token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr,
            rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos,
            query, query_rope, c_kv, k_rope, workspace, &tilingData);
    op.Process();
}

// after：仅函数名 mla_prolog → bench_no_harness_mla_prolog
template <uint32_t SCH_MODE>
__global__ __aicore__ void bench_no_harness_mla_prolog(
    GM_ADDR token_x, GM_ADDR w_dq, GM_ADDR w_uq_qr, GM_ADDR w_uk, GM_ADDR w_dkv_kr,
    GM_ADDR rmsnorm_gamma_cq, GM_ADDR rmsnorm_gamma_ckv, GM_ADDR rope_sin, GM_ADDR rope_cos,
    GM_ADDR query, GM_ADDR query_rope, GM_ADDR c_kv, GM_ADDR k_rope,
    GM_ADDR workspace, GM_ADDR tiling)
{
    ...（函数体不变）
}
```

内核类名/命名空间（如 `NsMlaProlog::MlaPrologKernel`）是内部实现，不参与构建系统入口查找，可不改。

## 设计决策

- 入口名跟随算子名（构建系统派生契约），文件名与内部类名保持原样：最小化重命名爆炸半径，同时满足构建系统查找契约。
- 重命名算子的完整 checklist（本例验证过的链路）：`REG_OP` → `OpDef` 类/`OP_ADD` → `IMPL_OP_INFERSHAPE`/`IMPL_OP_OPTILING` → `OP_TYPE` → `ASCENDC_TPL_ARGS_DECL` → `__global__` 入口函数名 → op_api 的 extern "C" aclnn 符号与手写头 → UT/ST 引用。

## 系统信息

- 芯片 Ascend950PR（arch35）；CANN 9.2.0；构建命令 `bash build.sh --soc=ascend950`（清理 `build/` 后全量重建）。
- 相关文件：`op_kernel/mla_prolog_arch35.cpp`（入口函数）、`op_kernel/arch35/mla_prolog_tiling_key.h`（`ASCENDC_TPL_ARGS_DECL`）、`op_kernel/CMakeLists.txt`（`OP_TYPE`）、构建生成物 `build/op_kernel/BenchNoHarnessMlaProlog_ascend950/dynamic/BenchNoHarnessMlaProlog.py`（按新名调用入口）。

## 预防

- 重命名算子时把"kernel 入口函数名 = snake_case(算子名)"写进重命名 checklist；也可全局搜索旧 snake_case 名（`mla_prolog(`）防遗漏。
- 新建算子时入口函数直接从算子名生成，不要按源文件名起入口名。
- （复现增量）设计文档给出 PascalCase 入口签名时，以工程模板实际约定为准，不要照抄文档签名。

## 验证信号

来源说明：以下信号来自该算子重命名后的重建过程（修复前后的构建输出对比）。

- 修复前：`[ERROR] [ascend950] kernel entry 'bench_no_harness_mla_prolog' not implement in 'mla_prolog_arch35.cpp'` + `An error occurred during stage of infer compile info`，构建失败（来源平台：950；证据等级：tool_observed）。
- 修复后：构建成功，产物清单包含 `BenchNoHarnessMlaProlog_764544817f6b6950f47d1ff41032d6c7.o`、同名 `.json`、`kernel/config/ascend950/bench_no_harness_mla_prolog.json` 与 `dynamic/bench_no_harness_mla_prolog.py`（来源平台：950；证据等级：tool_observed）。
- 入口改名后重装包，UT 12/12 通过（tiling/infershape 走新入口注册链正常）（来源平台：950；证据等级：tool_observed）。

增量信号（PascalCase 入口形态 + bin_param 机制证据，独立 trace，BenchNoHarnessApplyAdamW，Ascend950PR/arch35 + CANN 9.1.0）：

- 修复前：`bash build.sh --soc=ascend950 -j8` host 侧三目标（cust_optiling/cust_opapi/cust_op_proto）全部通过后，kernel 阶段报 `[ERROR] [ascend950] kernel entry 'bench_no_harness_apply_adam_w' not implement in 'bench_no_harness_apply_adam_w_arch35.cpp', please check whether the function name is correct in the kernel file.` + `An error occurred during stage of infer compile info`——当时入口函数按详细设计文档伪代码写成 PascalCase 的 `BenchNoHarnessApplyAdamW`（来源平台：950；证据等级：tool_observed）。
- 机制证据：编译产物 `build/op_kernel/BenchNoHarnessApplyAdamW_ascend950/bin_param/BenchNoHarnessApplyAdamW-bench_no_harness_apply_adam_w-1.sh` 中框架生成的编译命令为 `res=$(asc_opc $1 --main_func=bench_no_harness_apply_adam_w --input_param=.../BenchNoHarnessApplyAdamW_1bf467bd5719bf7a8c7ef719708886ca_param.json --soc_version=Ascend950PR_9599 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )`——`--main_func` 取的是 snake_case 名，asc_opc 按该名字在 kernel 源文件中查找入口函数，找不到即报 "kernel entry not implement"（来源平台：950；证据等级：tool_observed）。
- 修复后（仅改入口函数名为 `bench_no_harness_apply_adam_w`，函数体不变，类名/命名空间不改）：该报错消失，kernel 编译推进到下一阶段；同批次全部修复后 3 个 kernel 二进制（FP32/FP16/BF16 的 `.o`）全部生成，`bash build.sh --soc=ascend950 -j8` 退出码 0，算子包 `custom_opp_ubuntu_x86_64.run` 生成（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡针对"算子名与入口函数名不一致"的构建失败；若入口名已一致仍报 not implement，需查 `ASCENDC_TPL_ARGS_DECL`/`OP_TYPE` 是否漏改（同一契约的另一环）。
- 入口函数体内的 `REGISTER_TILING_DEFAULT`/`GET_TILING_DATA_WITH_STRUCT` 使用的 tiling 结构名与本卡无关，不需要随算子名改。

**未决项**：

- 无。
## 坏实践

- 重命名算子时只改 host 侧注册与 op_api，漏改 kernel 入口函数名。
- 把入口函数名写成源文件名或文件名衍生的名字（如 `mla_prolog_arch35`），构建系统同样找不到。
- （复现增量）按设计文档伪代码的 PascalCase 签名直接落码入口函数，未对照工程模板的入口命名。

## 误诊

- 无（报错文案已直接点名文件与期望的入口函数名，定位无歧义）。

## 增量（数字段转换规则定形 + _arch35/PascalCase 形态，四条独立 trace）

另一独立轨迹并入五条独立 trace（MlaProlog/TopK/Conv3dBackpropFilter/DepthwiseConv2d/GridSampler3d 类算子，Ascend950PR + CANN 9.1.0/9.2.0），把入口名推导的**数字段规则**定形，并补齐两种新误写形态：

**1. CamelCase→snake_case 转换规则（依报错与规则文档核对）**：首字符大写转小写；大写前为小写/数字时插入下划线；**数字前为小写时不插入下划线**（数字段跟随前缀字母段、不开启新分段）。即 `Conv3d → conv3d`、`Conv2d → conv2d`、`GridSampler3d → grid_sampler3d`，而不是目录 snake_case 习惯的 `conv_3d`/`conv_2d`/`grid_sampler_3d`。OP_TYPE 含数字段（2d/3d/V2 等）时，入口名与目录名天然不同构，必须从 OpType 推导而非从目录名复制。

三种新误写形态（同报错 `kernel entry '<期望名>' not implement in '<文件>'`）：

- 照目录名拆数字段：入口写 `bench_no_harness_conv_3d_backprop_filter`，期望 `bench_no_harness_conv3d_backprop_filter`；入口写 `bench_no_harness_depthwise_conv_2d`，期望 `bench_no_harness_depthwise_conv2d`；入口写 `bench_no_harness_grid_sampler_3d`，期望 `bench_no_harness_grid_sampler3d`。修复均只改函数名（conv_3d→conv3d 等），文件名/目录名/include/类名全部不动。
- 入口函数照抄文件名后缀（`_arch35` 写进函数名）：入口写 `mla_prolog_arch35`，期望 `mla_prolog`——`_arch35` 后缀只允许出现在文件名。
- 入口写成 PascalCase 类名风格：入口写 `BenchNoHarnessTopK`，期望 `bench_no_harness_top_k`；入口函数名（snake_case，工具链查找用）与 kernel 实现类名（PascalCase，C++ 类命名）应分离——类在命名空间内保持 PascalCase，仅入口函数服从 snake_case 约定。

**2. 官方推导名的两个零成本事实源**（不猜规则）：

- 构建产物 makefile：`build/op_kernel/<Op>_<soc>/<entry>.make` 内 `asc_opc $1 --main_func=<entry> ...` 即编译系统推导名（本批实测 `--main_func=bench_no_harness_depthwise_conv2d`）。
- 报错原文与 binary 输出目录名：报错里反引号括起的名字、`binary/ascend950/<entry_name>/` 目录名即期望入口名，逐字符比对（下划线差异肉眼易漏）。

**3. 报错可被上层噪声掩盖**：Conv3d 实例的失败先以 Python traceback 形式出现（`Exception: An error occurred during stage of infer compile info` + `RuntimeError: Kernel Compilation Error`），像工具链内部错误；单独 `cmake --build . --target binary -- -j1` 过滤 error 才露出 `kernel entry ... not implement` 根因行。ascendc kernel 编译失败时优先 grep `not implement`/`error:` 找第一因。

### 复现增量验证信号（五条独立 trace，均为 ascend950 编译目标）

- MlaProlog 实例（_arch35 后缀形态，CANN 9.2.0 目录/9.1.0 基线）：修复前 `[ERROR] [ascend950] kernel entry 'mla_prolog' not implement in 'mla_prolog_arch35.cpp'` + `dfx_for_func_name` 抛异常 + `An error occurred during stage of infer compile info`；函数名 mla_prolog_arch35→mla_prolog 后 kernel entry 检查通过、进入源码编译阶段，最终 `[100%] Built target MlaProlog_ascend950`、二进制 `MlaProlog_157c9d7f5324f6d1edefc24cb87040f5.o` 生成（来源平台：950 构建链；证据等级：tool_observed）。
- TopK 实例（PascalCase 形态，CANN 9.1.0）：修复前 `kernel entry 'bench_no_harness_top_k' not implement in 'bench_no_harness_top_k_arch35.cpp'`；入口改 snake_case 后错误消失、推进到下一独立错误，最终生成 `BenchNoHarnessTopK_9921bbcc9384f9f94ab1a08309c1028e.o` 与 custom_opp 包（来源平台：950；证据等级：tool_observed）。
- Conv3dBackpropFilter 实例（数字段形态，CANN 9.1.0）：函数名 conv_3d→conv3d 后 `cmake --build . --target binary -- -j1` 输出 `[ascend950] Generating BenchNoHarnessConv3dBackpropFilter_....o Done`（两个 tiling key 组合各一个二进制）；完整 build.sh 通过并产出算子包（来源平台：950；证据等级：tool_observed）。
- DepthwiseConv2d 实例（数字段形态 + makefile 锚点，CANN 9.2.0）：修复前 `[ascend950] kernel entry 'bench_no_harness_depthwise_conv2d' not implement in 'bench_no_harness_depthwise_conv_2d_arch35.cpp'`；修复后错误推进为 SEL 覆盖问题（独立根因），该问题修复后全量编译产出 3 个 `.o` 与 custom_opp 安装包；`bench_no_harness_depthwise_conv2d.make` 内 `asc_opc $1 --main_func=bench_no_harness_depthwise_conv2d ... --soc_version=Ascend950PR_9599` 佐证推导名（来源平台：950；证据等级：tool_observed）。
- GridSampler3d 实例（尾部数字段形态，CANN 9.1.0）：修复前 `kernel entry 'bench_no_harness_grid_sampler3d' not implement in 'bench_no_harness_grid_sampler_3d_arch35.cpp'`、`.json not generated!`；修复后 entry 检查通过，最终 `binary` target 100% 构建成功，`binary/ascend950/bench_no_harness_grid_sampler3d/` 下两个 kernel 二进制生成（来源平台：950；证据等级：tool_observed）。

### 复现增量坏实践

- 照抄目录名/snake_case 当核函数入口名，而不从 OpType 推导（数字段是高发区）。
- 把入口函数名写成源文件名或文件名衍生的名字（如 `mla_prolog_arch35`），构建系统同样找不到。
- 入口函数与算子类同名（PascalCase），以为"和 OpType 一致就行"。
- 看到 Python traceback（infer compile info 阶段的 Exception）就当工具链 bug 去查环境——先过滤 error 找 `not implement` 第一因。

### 复现增量失效边界

- 命名转换规则由报错行为与规则文档核对，数字段在名称中段（如 Foo2Bar）或连续多个数字段的行为未从构建系统源码验证；不确定时直接编译一次从报错里抄准确名字。
- 结论绑定 CANN 9.1.0/9.2.0 的 opc/asc_op 编译链（`kernel_info_infer.py` 的 `dfx_for_func_name` 检查）；其他版本以实际报错中的期望名为准（未验证，推断可迁移）。

## 增量（2026-09-17，先前增量 · Conv2d/AdaptiveAvgPool3d 两条独立 trace 再证数字段规则）

**Conv2d 实例**（BenchNoHarnessConv2d，目录名 bench_no_harness_conv_2d，Ascend950PR/arch35 + CANN 9.1.0）：host 侧三个目标全部编译通过后，kernel 侧编译由 asc_opc 工具接手，报：

```text
[ERROR] [ascend950] kernel entry `bench_no_harness_conv2d' not implement in `bench_no_harness_conv_2d_arch35.cpp', please check whether the function name is correct in the kernel file.
Exception: An error occurred during stage of infer compile info
.../binary/ascend950/bench_no_harness_conv2d/BenchNoHarnessConv2d_f9ff...json not generated!
```

手写入口按目录名习惯写成 `bench_no_harness_conv_2d`（Conv 和 2d 之间有下划线），工具期望 `bench_no_harness_conv2d`（数字跟随前词、不插下划线）。修复仅改函数名（模板签名与函数体不动；before/after verbatim）：

```cpp
// before
template <typename D_T_X, int TILING_MODE>
__global__ __aicore__ void bench_no_harness_conv_2d(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessConv2dTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv2dTilingData, tilingData, tiling);

    NsConv2d::Conv2d<D_T_X, TILING_MODE> op;
    op.Init(x, filter, bias, y, workspace, &tilingData);
    op.Process();
}
```

```cpp
// after
template <typename D_T_X, int TILING_MODE>
__global__ __aicore__ void bench_no_harness_conv2d(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessConv2dTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessConv2dTilingData, tilingData, tiling);

    NsConv2d::Conv2d<D_T_X, TILING_MODE> op;
    op.Init(x, filter, bias, y, workspace, &tilingData);
    op.Process();
}
```

目录名/文件名可保留自己的 snake_case 风格（目录 `bench_no_harness_conv_2d` 与入口函数 `bench_no_harness_conv2d` 共存没有问题），入口函数名单独服从 OP_TYPE 推导规则，不要从目录名复制。

**AdaptiveAvgPool3d 实例**（BenchNoHarnessAdaptiveAvgPool3d + CANN 9.1.0）：op_api 与 host 侧库全部编译通过后，构建进入 kernel 生成阶段，为算子并行生成 3 个 kernel 变体（哈希名 6b6f/b092/535a），全部失败，报错原文：

```text
[ascend950] Generating BenchNoHarnessAdaptiveAvgPool3d_6b6f7d4d65f2573ee6043af0dfd3eba5 ...
[ascend950] Generating BenchNoHarnessAdaptiveAvgPool3d_b0922e6ba03b0f41a4a4416de281c8d2 ...
[ascend950] Generating BenchNoHarnessAdaptiveAvgPool3d_535ab9f39abab55261bd87dfe1944fc2 ...
[ERROR] [ascend950] kernel entry `bench_no_harness_adaptive_avg_pool3d' not implement in `bench_no_harness_adaptive_avg_pool_3d_arch35.cpp', please check whether the function name is correct in the kernel file.
[ERROR] An error occurred during stage of infer compile info
```

入口按算子目录的 snake_case 写成 `bench_no_harness_adaptive_avg_pool_3d`（pool 与 3d 之间有下划线），期望 `bench_no_harness_adaptive_avg_pool3d`。判别：报错原文直接给出期望名（反引号内）与被检查文件名，逐字符对比——数字段前是否带下划线是最常见差异点。修复（函数改名；下示同一条构建链修完后的单模板参最终形态，verbatim——模板参数从 2 个收敛为 1 个是同一构建链上另一处独立修复，本增量的因果单元只有函数名 `..._pool_3d → ..._pool3d`）：

```cpp
// before
template <typename D_T_X, int TILING_KEY>
__global__ __aicore__ void bench_no_harness_adaptive_avg_pool_3d(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessAdaptiveAvgPool3dTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessAdaptiveAvgPool3dTilingData, tilingData, tiling);

    NsBenchNoHarnessAdaptiveAvgPool3d::KernelAdaptiveAvgPool3d<D_T_X, TILING_KEY> op;
    op.Init(x, y, workspace, &tilingData);
    op.Process();

    // TILING_KEY 分支说明：
    // - TILING_KEY == 0: 通用 downsample 路径（完整实现）
    // - TILING_KEY == 1: 全局平均池化（复用 downsample 逻辑，output=[1,1,1]）
    // - TILING_KEY == 2: upsample（复用 downsample 逻辑，窗口大小为 1）
    // 当前所有 TILING_KEY 共用同一计算逻辑，后续迭代可特化优化
}
```

```cpp
// after
template <typename D_T_X>
__global__ __aicore__ void bench_no_harness_adaptive_avg_pool3d(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessAdaptiveAvgPool3dTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessAdaptiveAvgPool3dTilingData, tilingData, tiling);

    NsBenchNoHarnessAdaptiveAvgPool3d::KernelAdaptiveAvgPool3d<D_T_X> op;
    op.Init(x, y, workspace, &tilingData);
    op.Process();
}
```

### 复现增量验证信号

- Conv2d 实例：改名后重新 `build.sh --make_clean` 全量编译，`kernel entry ... not implement` 报错消失，编译推进到 kernel 头文件内部的下一处独立编译错误（HardEvent::FIX_V 不存在），证明入口名已被工具接受（来源平台：950；证据等级：tool_observed）。
- AdaptiveAvgPool3d 实例：改名后重跑 `bash build.sh --soc=ascend950 --make_clean`，不再出现 "kernel entry not implement"，报错推进到 kernel wrapper 的模板实参推导阶段（`no matching function for call to '..._0_tilingkey'`），说明入口名已被框架找到（来源平台：950；证据等级：tool_observed）。
- AdaptiveAvgPool3d 实例最终整包构建成功，生成 kernel 二进制 `binary/ascend950/bench_no_harness_adaptive_avg_pool3d/BenchNoHarnessAdaptiveAvgPool3d_6b6f7d4d65f2573ee6043af0dfd3eba5.o`（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 两实例均为 CANN 9.1.0 opc/asc_op 编译链（与复现增量的版本边界一致）；其他版本以实际报错中的期望名为准（未验证，推断可迁移）。
- AdaptiveAvgPool3d 的 before/after 之间模板参数收敛属同链另一修复（ASCENDC_TPL 与 OpDef dtype 不一致，另卡），不并入本卡因果单元；改名只解决入口解析，入口函数签名（GM_ADDR 参数表）与 tiling 宏的匹配是另一层问题。

## 后续增量（2026-09-17，并卡合并：`_kernel` 后缀误写形态）

#### 并入自 kernel_entry_name_must_match_op_name（入口带 `_kernel` 后缀形态与 SwiGlu 实例）

第四种误写形态（同报错 `kernel entry '<期望名>' not implement in '<文件>'`）：入口模板函数写成 `<op>_kernel`（带 `_kernel` 后缀），期望名为算子 snake_case。SwiGlu 实例（BenchNoHarnessSwiGlu，Ascend950PR + CANN 9.1.0，ASCENDC_TPL 多 dtype/多 tiling path 模板实例化构建流）：

```cpp
// 修复前（入口带 _kernel 后缀）
template <typename D_T_X, int TILING_PATH>
__global__ __aicore__ void bench_no_harness_swi_glu_kernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessSwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessSwiGluTilingData, tilingData, tiling);

    NsBenchNoHarnessSwiGlu::KernelBenchNoHarnessSwiGlu<D_T_X, TILING_PATH> op;
    op.Init(input, output, workspace, &tilingData);
    op.Process();
}

// 修复后（入口名 = 算子 snake_case 名，函数体不变）
template <typename D_T_X, int TILING_PATH>
__global__ __aicore__ void bench_no_harness_swi_glu(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{ ...（同上） }
```

- **报错伴随形态（并入）**：该错误对每个模板实例化分支（每个 dtype）都出现一遍，且随后 `ascendc_compile_kernel.py` 抛出 `RuntimeError: Kernel Compilation Error: OpType BenchNoHarnessSwiGlu Kernel File .../op_kernel/bench_no_harness_swi_glu_arch35.cpp!`，日志里每个 dtype 都有 `BenchNoHarnessSwiGlu_*.json not generated!`，make 以 Error 1/2 终止。
- **机制补充（并入）**：ASCENDC_TPL 构建流按算子名（OpType 的 snake_case）查找/生成 kernel 入口符号：KERNEL_FILE 命名为 `<op>_arch35.cpp` 时，要求文件内存在名为 `<op>` 的 `__global__ __aicore__` 模板入口函数；写成 `<op>_kernel` 不被识别。KERNEL_FILE 按 `<op>_arch35.cpp` 命名时入口函数直接取算子 snake_case 名；类/命名空间可自由命名，入口不行。
- 误写形态谱系汇总（本卡四处 + 并入一处）：照目录名拆数字段（conv_3d→conv3d）、`_arch35` 后缀写进函数名、PascalCase 类名风格、`_kernel` 后缀——全部同修法：只改入口函数名，模板签名与函数体不动，类名/命名空间/文件名不改。

验证信号（并入，BenchNoHarnessSwiGlu，x86_64 主机 + ascend950 目标，CANN 9.1.0）：

- 修复前：每个 dtype 实例化分支报 `kernel entry 'bench_no_harness_swi_glu' not implement in 'bench_no_harness_swi_glu_arch35.cpp'` + `RuntimeError: Kernel Compilation Error`（来源平台：950；证据等级：tool_observed）。
- 修复后：重编后 3 个 dtype 实例全部生成：`[ascend950] Generating BenchNoHarnessSwiGlu_57f2fefc... / af1bcee8... / 9449b1db... Done`、`[100%] Built target BenchNoHarnessSwiGlu_ascend950`；产物含 3 个 kernel 二进制 `.o`（约 13–14KB）及 `custom_opp_ubuntu_x86_64.run` 安装包，build 退出码 0（来源平台：950；证据等级：tool_observed）。