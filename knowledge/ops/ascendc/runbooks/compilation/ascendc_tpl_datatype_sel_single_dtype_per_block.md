---
type: "Runbook"
title: ASCENDC_TPL_DATATYPE_SEL 须每块单 dtype
description: 把多个 dtype 并列塞进一个 ASCENDC_TPL_DATATYPE_SEL 后 kernel 实例化报 couldn't infer template argument 'D_T_X'，且被 FATBIN ld.lld cannot open .o 掩盖；改为每 dtype 一个 ARGS_SEL 块后编译成功。
tags: [task.debugging, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-16T10:29:17Z'
platforms: ["950"]
---

# ASCENDC_TPL_DATATYPE_SEL 须每块单 dtype

## 现象

在 StridedSlice 类自定义算子的 TilingKey 模板声明从少量组合扩展为「7 dtype × 3 SLICE_MODE = 21 个模板实例」后，执行 `bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8`，Host 侧已通过，Kernel 侧编译到 FATBIN 阶段失败。终端可见的报错是误导性的（.o 没生成，链接器报文件不存在）：

```text
ld.lld: error: cannot open .../kernel_meta_BenchNoHarnessStridedSlice_a2649a32e52c9cdcae0e47f4f47ffba9_0.o: No such file or directory
Exception: An error occurred during compile phases of CompileStage.FATBIN
BenchNoHarnessStridedSlice_a2649a32e52c9cdcae0e47f4f47ffba9.json not generated!
```

真实的编译错误被淹没在保存的完整输出里（6 个 kernel_* 分支同样报错），需要过滤才能看到：

```text
..._kernel.cpp:27:5: error: no matching function for call to 'bench_no_harness_strided_slice_0_tilingkey'
    bench_no_harness_strided_slice(x_in__, y_out_, usrWorkspace, tiling);
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<command line>:16:40: note: expanded from here
#define bench_no_harness_strided_slice bench_no_harness_strided_slice_0_tilingkey
op_kernel/bench_no_harness_strided_slice_arch35.cpp:23:28: note: candidate template ignored: couldn't infer template argument 'D_T_X'
__global__ __aicore__ void bench_no_harness_strided_slice(
```

当时代码状态：tiling_key.h 的 `ASCENDC_TPL_SEL` 把全部 7 个 dtype 塞进了一个选择块（before 见下）。

## 适用场景

- 工程形态：算子仓开发——op_kernel/arch35/*_tiling_key.h 的 ASCENDC_TPL 模板编程（多 dtype × 多模式实例展开）。
- API 路径：`ascendc/host_api/tiling/template_argument.h` 宏族（`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_DATATYPE_DECL` / `ASCENDC_TPL_SEL` / `ASCENDC_TPL_ARGS_SEL` / `ASCENDC_TPL_DATATYPE_SEL` / `ASCENDC_TPL_UINT_SEL`）。
- 验证过的平台与 CANN 版本：ascend950 编译验证 + CANN 9.1.0。
- 适用粒度：platform_wide（ASCENDC_TPL 宏语义是框架级事实）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：旧式 `TILING_KEY_IS` 宏的算子（非 ASCENDC_TPL 模板编程）；其他 SEL 参数类型（UINT/BOOL/FORMAT）与多值并列的组合未逐一验证。多参数 TilingKey 的整体扩展配方见 [ASCENDC_TPL 多参数 TilingKey 卡](ascendc_tpl_multi_param_tilingkey.md)；FATBIN cannot open 报错的另一独立根因见 [ld.lld cannot open _mix_aiv 卡](ld_lld_cannot_open_mix_aiv.md)。

## 根因

ASCENDC_TPL 框架按 `ASCENDC_TPL_SEL` 里每个 `ASCENDC_TPL_ARGS_SEL` 块生成一组模板实例化选择器；`ASCENDC_TPL_DATATYPE_SEL` 的取值参数是"本组选定的一个 dtype"，把多个 dtype 并列塞进同一个 SEL 块后，框架无法为每个 dtype 生成可推导的实例化入口，编译器实例化 kernel 入口函数时报 `couldn't infer template argument 'D_T_X'`（kernel 入口形参全是 `GM_ADDR`，模板参数只能靠框架生成的实例化机制提供）。注意 DECL 与 SEL 的参数语义不同：`ASCENDC_TPL_DATATYPE_DECL` 可以并列声明多个 dtype 表示支持范围，SEL 必须每块一个值。

## 解决配方

before（op_kernel/arch35/bench_no_harness_strided_slice_tiling_key.h）：

```cpp
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT8, C_DT_UINT8, C_DT_INT32, C_DT_INT64,
                                 C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    )
);
```

after（每个 dtype 一个独立 `ASCENDC_TPL_ARGS_SEL` 块，verbatim 取自修复事件；对照参考算子 add_example 的 tiling_key.h 模式——它同样为每个 dtype 单独一块）：

```cpp
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT8),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_UINT8),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT32),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT64),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(SLICE_MODE, ASCENDC_TPL_UI_LIST, 1, 2, 3)
    )
);
```

配套的判别方法（当 FATBIN 报 cannot open .o 时找真实错误）：完整输出被工具保存到文件后，过滤出真正的编译错误行：

```bash
rg -i "error:" <保存的完整输出文件> | rg -v "ld.lld|opc_common|op_compilation|Traceback|File \"|raise|format|RuntimeError|Exception" | head -30
rg -A 20 "no matching function for call to '<op>_0_tilingkey'" <保存的完整输出文件> | head -40
```

## 设计决策

- 为何选每 dtype 一块：对照框架自带参考算子 add_example 的 tiling_key.h（`ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)` 每块单值），这是宏族的既定用法。
- 被放弃的写法：单块多 dtype SEL（即 before，本次失败根源）。
- 实例数量核对：7 dtype × 3 SLICE_MODE = 21 个模板实例；构建产物 binary_info_config.json 中只有 7 个 binary 条目是正常的——每个条目经 simplifiedKey（含 dtype、9 个键位等）覆盖同 dtype 的 3 个 SLICE_MODE，7 个 binary 覆盖 21 实例，不是漏编。

## 预防

写 ASCENDC_TPL 声明时套用参考算子结构：DECL 并列声明支持域，SEL 每组一个取值；编译失败先怀疑自己的宏用法，再对照 add_example。遇到 `ld.lld: cannot open ... .o` 一律先 rg 完整输出里的 `error:`。

## 验证信号

段首来源说明：本结论来自该算子 TilingKey 全量扩展（21 实例）的编译修复过程，修复前后各一次干净全量编译对照。

- 修复前：FATBIN 阶段 `ld.lld: error: cannot open .../kernel_meta_..._0.o: No such file or directory` ×6 个 kernel 分支，`RuntimeError: Kernel Compilation Error`（来源平台：950；证据等级：tool_observed）
- 真实错误定位：`error: no matching function for call to 'bench_no_harness_strided_slice_0_tilingkey'` + `candidate template ignored: couldn't infer template argument 'D_T_X'`（来源平台：950；证据等级：tool_observed）
- 修复后：`bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 输出 `[INFO] Build completed successfully!`，生成 `custom_opp_ubuntu_x86_64.run` 安装包与 7 组 kernel binary（.o + .json）（来源平台：950；证据等级：tool_observed）

## 失效边界

验证于 CANN 9.1.0 + ascend950 + template_argument.h 的 ASCENDC_TPL 宏族；`ASCENDC_TPL_DATATYPE_SEL` 是否在后续版本支持多值并列未验证，升级版本时应重查宏定义。UINT/BOOL 等其他 SEL 类型按"每块一个取值"推定，未逐一验证。

**未决项**：

无（本根因已闭合）。
## 坏实践

把多个 dtype 并列写进一个 `ASCENDC_TPL_DATATYPE_SEL`；被 FATBIN 的 `cannot open .o` 误导去查链接器/文件系统/清理缓存——.o 不存在只是上游编译失败的下游症状。
