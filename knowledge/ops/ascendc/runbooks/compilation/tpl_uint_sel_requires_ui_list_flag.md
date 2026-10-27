---
type: "Runbook"
title: ASCENDC_TPL_UINT_SEL漏UI_LIST标志致TBE报错
description: 模板 tiling 的 SEL 段中，DECL 用 ASCENDC_TPL_UI_LIST 声明的 UINT 参数，其 ASCENDC_TPL_UINT_SEL 必须同样带 ASCENDC_TPL_UI_LIST 标志加取值列表；只写裸值会触发 TBE python 报 "Length of ASCENDC_TPL_UINT_SEL xxx is too short"。
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# ASCENDC_TPL_UINT_SEL漏UI_LIST标志致TBE报错

## 现象

在 Ascend950PR（CANN 9.1.0）上开发 GQA 自定义算子，`op_kernel/arch35/<op>_tiling_key.h` 用模板 tiling 宏声明了三个模板参数，其中 SCALE_MODE 是 UINT 列表参数（8 bit，取值 {0,1}）：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessGqa,
    ASCENDC_TPL_DATATYPE_DECL(D_T_Q, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(IS_CAUSAL, 0, 1),
    ASCENDC_TPL_UINT_DECL(SCALE_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);
```

修其他编译错时把 SEL 实例化段"简化"成只给裸值 `ASCENDC_TPL_UINT_SEL(SCALE_MODE, 0)`（意图：迭代一只实例化 TK_0 一个分支）。重新构建（cmake 通过、进入 kernel 二进制生成阶段）时 TBE python 侧直接抛异常，报错原文：

```
[ERROR] TBE(...):[op_compilation.py:311][__single_op_compile] Exception occured,
Length of ASCENDC_TPL_UINT_SEL SCALE_MODE is too short.
[ERROR] TBE(...):[opc_common.py:227][_log_error] Traceback (most recent call last):
  File ".../asc_op_compile_base/asc_op_compiler/kernel_info_infer.py", line 602, in get_tiling_key_list_and_simple_infer_code_channel
  File ".../asc_op_compile_base/asc_op_compiler/template_tiling.py", line 401, in extract_template_tiling_info
```

该错误发生在 python 解析阶段（TBE 侧 extract_template_tiling_info），不伴随任何 C++ 编译错误行，容易让人以为工程配置坏了。

## 适用场景

- 工程形态：使用模板 tiling（ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL）声明多 TilingKey 分支的 aclnn 自定义算子工程，迭代式开发（先实例化一个分支）。
- API 路径：`ascendc/host_api/tiling/template_argument.h` 的 `ASCENDC_TPL_UINT_DECL/ASCENDC_TPL_UINT_SEL/ASCENDC_TPL_UI_LIST`；TBE 侧 `template_tiling.py::extract_template_tiling_info`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（模板 tiling 宏契约，与算子无关）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：UINT 参数用 RANGE 语义声明（vals 为 {start, count} 对）时 SEL 的写法未验证；非 ASCENDC_DEBUG 构建下 SEL 宏在 C++ 侧展开为空、只有 TBE python 侧解析它，这一点决定了错误形态。

## 根因

`ASCENDC_TPL_UINT_DECL(SCALE_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)` 把该 UINT 参数声明为**显式取值列表**（UI_LIST 标志 + 列表 {0,1}）。SEL 段的 `ASCENDC_TPL_UINT_SEL` 是 TBE python 逐 token 解析的，对 UI_LIST 声明的参数要求 SEL 里同样携带 `ASCENDC_TPL_UI_LIST` 标志再跟取值列表，二者按位置配对解析；只写 `ASCENDC_TPL_UINT_SEL(SCALE_MODE, 0)` 时 python 侧按列表语义取到的元素不足，抛 "Length of ASCENDC_TPL_UINT_SEL SCALE_MODE is too short"。头文件侧机制佐证：`template_argument.h` 的 `ParamStruct::ParseTplUintValue()` 里 `vals[0]==ASCENDC_TPL_UI_LIST(=1)` 时 erase 掉首元素按列表处理，否则按 {start, rangeNum} 展开范围——DECL 与 SEL 的语义必须一致。

## 解决配方

修复前（TBE 报 Length too short）：

```cpp
ASCENDC_TPL_SEL(
    // TK_0: fp16 + non-causal + auto-scale
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_Q, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(IS_CAUSAL, 0),
        ASCENDC_TPL_UINT_SEL(SCALE_MODE, 0)
    )
);
```

修复后（恢复 UI_LIST 标志 + 取值列表）：

```cpp
ASCENDC_TPL_SEL(
    // TK_0: fp16 + non-causal + auto-scale
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_Q, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(IS_CAUSAL, 0),
        ASCENDC_TPL_UINT_SEL(SCALE_MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    )
);
```

判别 / 定位方法：

1. TBE 报 `Length of ASCENDC_TPL_XXX_SEL <参数名> is too short` 且堆栈落在 `template_tiling.py::extract_template_tiling_info` → 直接对照该参数 DECL 行，检查 SEL 行的标志 token 是否与 DECL 一致（UI_LIST/RANGE）。
2. 注意 C++ 侧不会报这个错（非 ASCENDC_DEBUG 构建 SEL 宏展开为空），所以 grep C++ 编译输出找不到线索；要去 build 日志的 `[ERROR] TBE` 行看 python 堆栈。

## 设计决策

- SEL 只实例化一个分支（TK_0）时保留完整宏形态而不是"精简"参数：SEL 的 token 结构由 DECL 决定，实例化几个分支只影响 `ASCENDC_TPL_SEL` 里放几个 `ASCENDC_TPL_ARGS_SEL` 组，不影响组内每个参数的写法。
- 当前方案路线档位：recommended_fastpath（宏契约固定，无备选写法）。

## 验证信号

本结论来自该 GQA 算子迭代一编译循环：SEL 写成裸值后构建报 `Length of ASCENDC_TPL_UINT_SEL SCALE_MODE is too short`（来源平台：950；证据等级：tool_observed）；恢复 `ASCENDC_TPL_UI_LIST, 0, 1` 形态后同工程重建，TBE 解析阶段通过，错误推进到 kernel C++ 编译阶段，后续构建输出中该 TBE 异常不再出现（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡验证的是 UI_LIST 语义；RANGE 语义（DECL vals 为 {start, count}）下 SEL 的对应写法未验证，迁移前先读 `template_argument.h` 的 ParseTplUintValue 展开逻辑。
- CANN 版本升级若改变 TBE 解析脚本（template_tiling.py），报错形态可能变化，宏契约以版本头文件+构建实测为准。

**未决项**：

- 无（本根因已闭环）。
## 坏实践

- 把 SEL 段当 C++ 代码"重构精简"：SEL 是 TBE python 按位置解析的数据，不是可自由简化的表达式；本例的故障正是由一次"简化"引入。
- 只看 C++ 编译错误不看 `[ERROR] TBE` 行：模板 tiling 的 SEL 问题只从 python 侧冒出来。

## 误诊

- 该错误初次出现时，同轮还有其他 kernel 侧 C++ 错误；若按"C++ 代码问题"顺序排查会绕远。正确切入点是堆栈里的 `extract_template_tiling_info`——名字直接指明是模板 tiling 解析。
