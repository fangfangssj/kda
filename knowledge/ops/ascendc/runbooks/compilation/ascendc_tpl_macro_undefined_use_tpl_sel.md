---
type: "Runbook"
title: ASCENDC_TPL 宏未定义，须用 ASCENDC_TPL_SEL
description: 官方示例模板的 tiling_key.h 用 ASCENDC_TPL( 列出 TilingKey 选择表，但 CANN 9.1.0 全头文件树没有该宏，host 编译报 expected constructor before '('；正确宏名是 ASCENDC_TPL_SEL(。
tags: [task.debugging, topic.compilation, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# ASCENDC_TPL 宏未定义，须用 ASCENDC_TPL_SEL

## 现象

在搭建一个 Mish 激活算子（y = x·tanh(softplus(x))）的算子仓工程时（AscendC 自定义算子，目标 Ascend950PR、工程目录 op_kernel/arch35，编译机 x86_64、CANN 9.1.0），参照官方示例模板 add_example 编写模板参数声明头 `op_kernel/arch35/*_tiling_key.h`：用 `ASCENDC_TPL_ARGS_DECL(...)` 声明 dtype 模板参数 D_T_X，用 `ASCENDC_TPL(...)` 列出 3 个 TilingKey 的选择表。这是工程首版代码，从未编译过。首次运行 `bash build.sh --soc=ascend950` 时，host 侧 autogen 编译（npu_op_code_gen 阶段）报语法错误：

```
op_kernel/arch35/bench_no_harness_mish_tiling_key.h:38:12: error: expected constructor, destructor, or type conversion before '(' token
   38 | ASCENDC_TPL(
      |            ^
```

同一轮编译还伴随 `'platform_ascendc' has not been declared`（那是另一个独立根因，另有专卡记录）。

## 适用场景

- 工程形态：算子仓工程（ops-registry-invoke 类）op_kernel 的 tiling_key.h 模板参数声明 + host autogen 编译。
- API 路径：`asc/include/tiling/template_argument.h` 的 ASCENDC_TPL_* 宏族（ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_ARGS_SEL / ASCENDC_TPL_SEL / ASCENDC_TPL_SEL_PARAM）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，--soc=ascend950 交叉编译），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（所有走 ASCENDC_TPL 模板参数机制的算子工程）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：旧式 `TILING_KEY_IS` 宏路由（本机制明确禁止）；其他 CANN 版本的宏集合未验证。

## 根因

CANN 9.1.0 的 `template_argument.h` 在全部三个条件分支（`ASCENDC_TPL_PRE` autogen 预处理、`ASCENDC_TPL_KERNEL` kernel 编译、默认 host g++ 编译）中都只定义 `ASCENDC_TPL_SEL(...)`，从未定义无后缀的 `ASCENDC_TPL(...)`。对整个 CANN 安装树执行 `grep -rn "define ASCENDC_TPL(" $ASCEND_HOME_PATH/x86_64-linux/asc/include/` 零命中；而 `define ASCENDC_TPL_SEL` 有 4 处（第 95/214/275/277 行，分别对应 PRE 占位符、默认空定义、ParamStruct 版本、内核选择版本）。宏未定义时预处理器把 `ASCENDC_TPL` 展开为空，剩下的孤立 `(...)` 触发 "expected constructor, destructor, or type conversion before '(' token"。

关键背景：官方示例模板 add_example 的 tiling_key.h 同样写着 `ASCENDC_TPL(...)`——但该模板的 build/autogen 目录为空，即它在本环境从未成功编译过。示例模板的宏名不能当作事实源，必须以本机 CANN 头文件实际定义为准。

## 解决配方

修复前（tiling_key.h 首版，SEL 列表用了不存在的宏）：

```cpp
ASCENDC_TPL(
    // TilingKey 1: FP16 直接计算路径
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)
    ),
    // TilingKey 2: FP32 直接计算路径
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    ),
    // TilingKey 3: BF16 升精度计算路径
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16)
    ),
);
```

修复后（仅换宏名，`ASCENDC_TPL_ARGS_DECL` 与 tiling.cpp 侧的 `ASCENDC_TPL_SEL_PARAM(context, dTypeX)` 均不受影响）：

```cpp
ASCENDC_TPL_SEL(
    // TilingKey 1: FP16 直接计算路径
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)
    ),
    // TilingKey 2: FP32 直接计算路径
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    ),
    // TilingKey 3: BF16 升精度计算路径
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16)
    ),
);
```

配套的声明（无需改动，供对照）：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessMish,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);
```

## 设计决策

- 为什么确认宏名靠 grep 而不是继续读模板示例：示例模板自身从未编译通过（build/autogen 为空），照抄只会复现错误；`template_argument.h` 的条件分支结构（PRE/KERNEL/默认三分支 × ASCENDC_DEBUG 两分支）读全之后能确定所有可用宏。
- 当前方案路线档位：recommended_fastpath（单行宏名替换，无替代方案）。

## 系统信息

- 编译机：x86_64 Ubuntu（GCC 13.3），CANN 9.1.0，`aarch64-linux` 与 `x86_64-linux` 互为软链接。
- 关键头文件：`$ASCEND_HOME_PATH/x86_64-linux/asc/include/tiling/template_argument.h`（338 行；`include/ascendc/host_api/tiling` 是指向 `asc/include/tiling` 的软链接目录）。
- 相关文件：`op_kernel/arch35/*_tiling_key.h`（出错与修复文件）、`op_host/arch35/*_tiling.cpp`（include 该头）。

## 预防

- 抄示例模板的宏名前，先在本机验证宏存在：`grep -rn "define ASCENDC_TPL(" $ASCEND_HOME_PATH/x86_64-linux/asc/include/`；同理可验证 `ASCENDC_TPL_SEL`、`ASCENDC_TPL_ARGS_DECL`。
- 遇到 "expected constructor … before '(' token" 且落在宏调用行，优先怀疑宏未定义（被展开为空），而不是括号配对或 include 路径。

## 验证信号

以下信号来自该 Mish 算子工程的编译联调会话（Ascend950PR 交叉编译目标，CANN 9.1.0）：首编报错 → grep 确认宏不存在 → 单行宏名替换 → 重新编译。

- 修复前：autogen 编译在 tiling_key.h:38 报 `error: expected constructor, destructor, or type conversion before '(' token`（来源平台：950；证据等级：tool_observed）。
- 宏存在性探针：全 CANN `grep "define ASCENDC_TPL("` 零命中、`define ASCENDC_TPL_SEL` 4 处命中（来源平台：950；证据等级：tool_observed）。
- 修复后（同轮还剩独立的 platform_ascendc 报错待修）：ASCENDC_TPL 语法错误从编译输出中消失（来源平台：950；证据等级：tool_observed）。
- 全部修复完成后：`bash build.sh --soc=ascend950` exit 0，3 个 TilingKey 的 kernel .o 生成（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 CANN 9.1.0 验证；其他 CANN 版本的宏集合可能不同（未验证，换版本时应重新 grep）。
- 该结论只覆盖 "SEL 列表宏"；`ASCENDC_TPL_ARGS_DECL`、`ASCENDC_TPL_DATATYPE_DECL/SEL`、`ASCENDC_TPL_SEL_PARAM` 的可用性在本工程同轮编译中一并得到验证。

**未决项**：

- 无（宏名已经由本机头文件 grep 确证）。
## 坏实践

- 把官方示例模板当作宏名/API 事实源直接照抄，不核对本地头文件定义。
- 宏展开为空导致的语法错误容易被误判为"括号/语法笔误"而只调整排版，掩盖真实根因。

## 误诊

- 第一轮排查曾假设是 npu_op_code_gen 阶段未定义 `ASCENDC_TPL_PRE` 预处理宏、或 include 路径找不到 template_argument.h 导致宏未定义；逐层读 template_argument.h 条件分支并对全 CANN grep 后确认：宏名本身在任何分支都不存在，与预处理模式、include 路径无关（该假设无工具输出佐证，仅作排查路径记录）。
