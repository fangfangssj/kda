---
type: "Runbook"
title: kernel printf 引用 tiling 形参编译报错
description: "AscendC kernel 里 AscendC::printf 的参数中直接使用名为 tiling 的形参（即使加了 (uint64_t) 强转）会触发 error: unexpected namespace name 'tiling': expected expression；调试打印须避开该标识符。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# kernel printf 引用 tiling 形参编译报错

## 现象

在 Ascend950PR（arch35，CANN 9.1.0）上调试一个自定义 Transpose 算子 kernel：为定位输出错误，在 `Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TransposeTilingData* tilingData)` 里加调试打印，想顺带把 kernel 的 tiling 形参地址打出来：

```cpp
AscendC::printf("DEBUG Init: x=%p, y=%p, workspace=%p, tiling=%p\n", x, y, workspace, tiling);
```

执行 `bash build.sh --soc=ascend950` 后 kernel 编译失败，报错（bisheng/clang 前端，出现在 INFERCHANNEL 编译阶段日志中，同一个错误重复多次）：

```
transpose.h:107:91: error: unexpected namespace name 'tiling': expected expression
    AscendC::printf("DEBUG Init: x=%p, y=%p, workspace=%p, tiling=%p\n", x, y, workspace, tiling);
1 error generated.
RuntimeError: Kernel Compilation Error: OpType BenchNoHarnessTranspose Kernel File .../op_kernel/transpose_arch35.cpp!
```

尝试把 `%p` 换成 `%lu` 并加强转 `(uint64_t)tiling`，同样的错误仍在（行列号随代码移动变为 108:78）。最后从 printf 参数中**移除 tiling 形参**（只打印 x/y/workspace），该错误消失（编译进入下一个错误——bf16 cast，另行处理）。

## 适用场景

- 工程形态：AscendC kernel（`__aicore__` 设备代码）内加 `AscendC::printf` 调试输出；kernel 入口形参按惯例命名为 `tiling`（与模板/样例一致）。
- API 路径：`AscendC::printf`；kernel 入口 `tiling` 形参（GM_ADDR，经框架传入 tiling 数据）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——标识符冲突源于 AscendC 编译链对 `tiling` 的保留命名空间处理，与算子无关。
- 路线优先级：recommended_fastpath（一行规避）。
- 明确不适用 / 未验证场景：host 侧代码（不受影响）；形参改名为其他标识符后打印该新名字（未验证，推断可行——冲突绑定在 `tiling` 这个名字上）。

## 根因

`tiling` 在 AscendC 编译链（bisheng/clang 前端处理 `__aicore__` 代码）中是保留命名空间名；kernel 函数形参名 `tiling` 在普通表达式里可以被形参遮蔽规则解析，但在 `AscendC::printf` 的参数上下文中被解析为命名空间名而非变量，报 `unexpected namespace name 'tiling': expected expression`。加 `(uint64_t)` 强转不能绕过（错误在标识符解析层，不在格式串）。

## 解决配方

修复前（printf 参数引用 tiling 形参，两种写法均失败）：

```cpp
    // 写法一（%p）：
    AscendC::printf("DEBUG Init: x=%p, y=%p, workspace=%p, tiling=%p\n", x, y, workspace, tiling);
    // 写法二（%lu + 强转，同样失败）：
    AscendC::printf("DEBUG Init: x=%lu, y=%lu, ws=%lu, tiling=%lu\n",
                    (uint64_t)x, (uint64_t)y, (uint64_t)workspace, (uint64_t)tiling);
```

修复后（printf 参数中移除 tiling 形参；tiling 内容通过已解引用的 tilingData 字段打印，不受影响）：

```cpp
    // DEBUG: 打印 GM 地址和 tiling 数据
    AscendC::printf("DEBUG Init: x=%lu, y=%lu, ws=%lu\n",
                    (uint64_t)x, (uint64_t)y, (uint64_t)workspace);
    AscendC::printf("DEBUG Tiling: totalNum=%ld, batch=%ld, blkSize=%ld, elemSize=%ld\n",
                    totalNum_, tilingData->batch, tilingData->blkSize, elemSize_);
```

如确需打印 tiling 缓冲区地址，可在 kernel 外（host/plog 侧）从 kernel launch 日志的 args 地址获取，或先把形参值存入局部变量再打印局部变量（本轨迹未验证后一种）。

## 设计决策

- 选择"移除该参数"而非"给形参改名"：kernel 入口形参名 `tiling` 与模板/生成代码约定一致，改名影响面大于少打一个调试值；tiling 内容本身经 `tilingData->字段` 打印信息量更大。
- 当前方案属 recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0，bisheng 编译链（`ascendc_compile_kernel.py` / CompileStage.INFERCHANNEL）。
- 相关文件：`op_kernel/arch35/transpose.h`（Init 函数调试打印）。

## 预防

- kernel 内调试 printf 的参数清单先过一遍"保留字"：`tiling` 是已确认的坑；遇到 `unexpected namespace name 'X': expected expression` 且 X 恰是形参名，直接从表达式里拿掉该标识符，不要在格式串/强转上浪费时间。
- 打印 tiling 信息统一走 `tilingData->field`（结构体字段名无此冲突）。

## 验证信号

以下信号来自该算子迭代一 kernel 调试会话（Ascend950PR 编译环境）：

- 修复前：`%p` 版本编译报 `transpose.h:107:91: error: unexpected namespace name 'tiling': expected expression`，构建失败（gmake Error 2，无 .run 包生成）（来源平台：950；证据等级：tool_observed）。
- 修复前：`%lu`+(uint64_t) 强转版本同样报 `108:78: unexpected namespace name 'tiling'`（来源平台：950；证据等级：tool_observed）。
- 修复后：printf 移除 tiling 参数后该错误消失，编译推进到下一错误（bf16 cast，属另一根因）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 冲突验证于 CANN 9.1.0 bisheng 前端；其他 CANN 版本是否保留该行为未验证（推断可迁移，未验证）。
- "改形参名"与"经局部变量转打印"两条规避路径未在本轨迹验证。

**未决项**：

- `tiling` 被保留的确切编译器机制（宏展开/内建识别）未深挖；如需机制层结论可对最小样例做预处理输出对比。
## 坏实践

- 看到该错误后先去查 include、namespace using 或格式串——错误根因是标识符 `tiling` 在该上下文被当作命名空间解析，与头文件无关。

## 误诊

- 无（两次写法快速对照后即定位到标识符本身）。
