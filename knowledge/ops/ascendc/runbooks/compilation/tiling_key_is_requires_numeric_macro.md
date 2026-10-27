---
type: "Runbook"
title: TILING_KEY_IS 须数值宏，constexpr 分支不注册
description: "TILING_KEY_IS(k) 的 k 用 constexpr 变量时 TBE 预编译器报 \"can not be processed as numeric variables in the precompilation phase. please use numeric constants, macros.\"，对应 TilingKey 分支不注册（kernelList 仅 1 条 _0）；该诊断不使构建失败，构建显示成功但注册缺失；改为 #define 数值宏后 4 个分支全部注册进 kernelList。"
tags: [task.debugging, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# TILING_KEY_IS 须数值宏，constexpr 分支不注册

## 现象

在 Ascend950（CANN 9.1.0）上把一个自定义算子从 ASCENDC_TPL 模板机制重构为 `TILING_KEY_IS` 运行时分发：`op_kernel/arch35/*_tiling_key.h` 里用 `constexpr uint32_t` 定义 4 个 TilingKey 常量（0/2/4/6），kernel 入口用 `if (TILING_KEY_IS(TILING_KEY_FP16_G1_DIL1)) ... else if` 四分支分发。`bash build.sh --soc=ascend950 -j8` 构建能完成，但 TBE 编译日志刷出 16 条 ERROR 级预编译诊断（每个 TilingKey 常量 × 每个模板实例），报错原文：

```
[ERROR] TBE(...): ... [ascend950] Var: TILING_KEY_FP16_G1_DIL1 in TILING_KEY_IS(TILING_KEY_FP16_G1_DIL1) can not be processed as numeric variables in the precompilation phase. please use numeric constants, macros.
```

（对 TILING_KEY_FP16_G1_DIL1 / TILING_KEY_FP16_G1_DILN / TILING_KEY_FP16_GN_DIL1 / TILING_KEY_FP16_GN_DILN 各重复报出。）同时构建产物 kernel JSON 的 kernelList 只注册了 1 个条目（`kernelName: ..._0`），而非 4 个 TilingKey 分支。

## 适用场景

- 工程形态：Ascend C 自定义算子采用 `TILING_KEY_IS` + if-else 运行时分发（单 kernel binary，多 TilingKey 分支），tiling 端 `context->SetTilingKey()` 置键。
- API 路径：`TILING_KEY_IS(k)` 宏（`x86_64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h:445`，定义为 `(TILING_KEY_VAR == (k))`）；TBE 预编译器（asc_op_compile_base）对 kernel 源码做 TilingKey 分支扫描。
- 验证过的平台与 CANN 版本：Ascend950，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——约束来自 TBE 预编译器对宏参数的文本级扫描，与算子无关。
- 路线优先级：recommended_fastpath（一行形式修正）。
- 明确不适用 / 未验证场景：ASCENDC_TPL 模板机制（走 ASCENDC_TPL_UINT_DECL 声明路径，不做 TILING_KEY_IS 文本扫描）；enum/`static constexpr int` 等其他常量形式未验证；其他 CANN 版本未验证。

## 根因

TBE 预编译器在编译前按**文本扫描** `TILING_KEY_IS(k)` 的参数 k 来枚举并注册 TilingKey 分支（生成 kernelList 条目），只认**数值常量与宏**；`constexpr` 变量是一个不可求值的标识符，扫描失败即报 "can not be processed as numeric variables"，对应分支不注册（kernelList 缺条目）。宏展开为字面量后扫描成功。

注意该诊断不会使构建失败（build 显示 `Build completed successfully!`），但功能后果是 TilingKey 分支未注册——本例 kernelList 从期望的 4 条缩水为 1 条（`_0` 后缀），若运行时 `SetTilingKey` 设置了未注册的键，分支匹配依赖的注册信息缺失。

## 解决配方

修复位置：`op_kernel/arch35/*_tiling_key.h`，把 constexpr 改为 `#define` 数值宏。

修复前（预编译报错、分支不注册）：

```cpp
// 采用 TILING_KEY_IS + if-else 运行时分发（参考内置 group_norm_grad / conv3d_backprop_filter_v2），
// 编译为单个 kernel binary，避免 ASCENDC_TPL 多 kernelList 的 binary 缓存复用问题。
constexpr uint32_t TILING_KEY_FP16_G1_DIL1 = 0;  // fp16 + groups=1 + dilation=1
constexpr uint32_t TILING_KEY_FP16_GN_DIL1 = 2;  // fp16 + groups>1 + dilation=1
constexpr uint32_t TILING_KEY_FP16_G1_DILN = 4;  // fp16 + groups=1 + dilation>1
constexpr uint32_t TILING_KEY_FP16_GN_DILN = 6;  // fp16 + groups>1 + dilation>1
```

修复后（告警消除、4 个分支全部注册）：

```cpp
// 注意：必须使用 #define 宏而非 constexpr，因为 TILING_KEY_IS 在预编译阶段
// 要求参数为数值常量或宏（constexpr 变量无法被预编译器识别为数值）。
#define TILING_KEY_FP16_G1_DIL1  0  // fp16 + groups=1   + dilation=1
#define TILING_KEY_FP16_GN_DIL1  2  // fp16 + groups>1   + dilation=1
#define TILING_KEY_FP16_G1_DILN  4  // fp16 + groups=1   + dilation>1
#define TILING_KEY_FP16_GN_DILN  6  // fp16 + groups>1   + dilation>1
```

kernel 入口侧用法不变（`TILING_KEY_IS(TILING_KEY_FP16_G1_DIL1)` 等四处分支无需改动），tiling 端 `context->SetTilingKey(tilingKey)` 的编码值与宏保持一致。

## 设计决策

- 为什么是 #define 而不是 enum：诊断信息明确要求 "numeric constants, macros"；enum 同样是标识符，未验证是否可被识别，宏是诊断文本点名的安全选项。
- 为什么不用 ASCENDC_TPL：该算子只需要 attrs 分支（groups/dilation 组合），无需编译期模板特化；两条路线的取舍见同域《TILING_KEY_IS 运行时分发单 binary》卡。

## 预防

- 定义 TilingKey 常量给 `TILING_KEY_IS` 用时，一律用 `#define` 数值宏，并让 tiling 端 `SetTilingKey` 的编码与宏值一一对应。
- 构建日志里出现 `Var: ... can not be processed as numeric variables` 时不要因 "Build completed" 而忽略——去查 kernel JSON 的 kernelList 条目数是否等于分支数。

## 验证信号

来源说明：该算子 TILING_KEY_IS 重构会话：constexpr 版本构建刷 16 条预编译 ERROR 日志且 kernelList 只有 1 条，改 #define 后干净构建、4 条注册并通过全量测试。

- 修复前：构建日志对 4 个 constexpr 常量各报 `Var: TILING_KEY_FP16_G1_DIL1 in TILING_KEY_IS(TILING_KEY_FP16_G1_DIL1) can not be processed as numeric variables in the precompilation phase. please use numeric constants, macros.`（16 条 ERROR 级日志），构建产物 kernelList 仅 1 条 `kernelName: ..._0`（来源平台：950；证据等级：tool_observed）
- 修复后：#define 版本重新构建，`Build completed successfully!` 且无任何 `Var:` 诊断；kernelList 含 4 个条目（TilingKey 后缀 0/2/4/6）（来源平台：950；证据等级：tool_observed）
- 修复后：UT 12/12 通过（含 tiling 用例的 TilingKey 断言：标准 fp16 用例 TilingKey=0、groups=2 用例 TilingKey=2、dilation=2 用例 TilingKey=4）；ST Real 5/5（MERE=0.00e+00）（来源平台：950；证据等级：tool_observed）

## 失效边界

- "constexpr 导致分支不注册"的功能后果（运行时键值无匹配条目时的具体行为）未直接触发观测——本例修复前未上板跑 constexpr 版本；卡内因果链为：预编译诊断（实测）→ kernelList 缺条目（实测 JSON）→ 分支注册缺失（由前两者推定）。
- enum/`static constexpr int` 等其他常量形式未验证；其他 CANN 版本的预编译器行为未验证（未验证，推断可迁移：同版本 TBE 管线覆盖的所有算子）。

**未决项**：

- 无（根因与修复均为受控对照验证）。
## 坏实践

- 用 `constexpr uint32_t`（或任何 C++ 常量声明）配合 `TILING_KEY_IS`——编译能过，但预编译器静默（仅日志级）丢弃分支注册。

## 误诊

- 首次看到 kernelList 只有 1 条时曾被解读为"成功移除 ASCENDC_TPL、单 binary 符合预期"，未意识到这是 constexpr 导致的分支未注册；直到改 #define 后看到 4 条才确认 1 条是异常态（初始误读后由构建产物对照纠正）。
