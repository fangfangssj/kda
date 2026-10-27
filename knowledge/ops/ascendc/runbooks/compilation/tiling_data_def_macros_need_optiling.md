---
type: "Runbook"
title: "TILING_DATA_DEF 宏须包在 optiling 命名空间"
description: "tiling 头中使用 BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS 宏时，宏展开引用的 TilingDef/FieldInfo/CTilingDataClassFactory 都定义在 namespace optiling 内；宏块裸放全局作用域会报 expected class-name before '{' token，须整体包进 namespace optiling。"
tags: [task.implementation, topic.tiling, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# TILING_DATA_DEF 宏须包在 optiling 命名空间

## 现象

在为 Ascend950 上的 RoPE 类自定义算子实现 TilingData 汇总结构体时，按设计文档给出的宏范式改写 `op_kernel/bench_harness_apply_rotary_pos_emb_tiling.h`：`#include "register/tilingdata_base.h"` 后直接在全局作用域写 `BEGIN_TILING_DATA_DEF(...)` + 18 个 `TILING_DATA_FIELD_DEF(...)` + `END_TILING_DATA_DEF` + `REGISTER_TILING_DATA_CLASS(...)`——与 `tilingdata_base.h` 头文件注释里的示例写法完全一致。随后 source 共享 CANN 安装的 `set_env.sh` 并执行 `cd operators && ./build.sh <package>`（内部 `cmake .. && make -j binary package`），宿主侧 opbuild 编译立即失败。opbuild 用 `/usr/bin/c++ -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -I<ASCEND_HOME>/include` 编 `op_host/bench_harness_apply_rotary_pos_emb_host.cpp`，该文件第 11 行 `#include "../op_kernel/bench_harness_apply_rotary_pos_emb_tiling.h"` 把宏头带进了宿主编译。报错原文：

```
CMake Error at <ASCEND_HOME>/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake:732 (message):
  opbuild run failed! In file included from
  .../op_kernel/bench_harness_apply_rotary_pos_emb_tiling.h:13,
                   from .../op_host/bench_harness_apply_rotary_pos_emb_host.cpp:11:
  <ASCEND_HOME>/include/register/tilingdata_base.h:152:39:
  error: expected class-name before '{' token
    152 |   class class_name : public TilingDef {
         |                                       ^
  .../bench_harness_apply_rotary_pos_emb_tiling.h:19:1:
  note: in expansion of macro 'BEGIN_TILING_DATA_DEF'
  <ASCEND_HOME>/include/register/tilingdata_base.h:155:7:
  error: 'CheckAlignAndGenPlaceHolder' was not declared in this scope
```

报错位置全部指向 `tilingdata_base.h` 头文件内部（宏定义体），而不是用户代码行，第一眼很难看出是命名空间问题。

## 适用场景

- 工程形态：CANN 自定义算子（AscendC kernel + op_host TilingFunc），op_host 侧经 opbuild（g++，`-std=c++11`）编译。
- API 路径：`register/tilingdata_base.h` 的 `BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `END_TILING_DATA_DEF` / `REGISTER_TILING_DATA_CLASS` 宏族。
- 验证过的平台与 CANN 版本：Ascend950（宿主侧编译验证），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（宏与命名空间的关系由 CANN 头文件自身决定，与具体算子无关）。
- 路线优先级：recommended_fastpath（凡使用宏范式即必备）。
- 明确不适用 / 未验证场景：不使用宏范式（平凡 POD TilingData）的工程无此问题；本卡只解决宿主侧命名空间编译错——若同一份 tiling.h 还被 kernel 侧 `.cpp` include，宏范式会在 kernel 侧再报 `DT_*` 宏冲突（另一独立根因），彼时应整体放弃宏范式改纯 POD。

## 根因

`BEGIN_TILING_DATA_DEF` 等宏定义在 `namespace optiling { ... }` 闭合之后（即宏定义本身位于全局作用域），宏体内以未限定名引用的 `TilingDef`、`FieldInfo`、`CheckAlignAndGenPlaceHolder`、`CTilingDataClassFactory` 全部声明于 `namespace optiling` 内；宏块在全局作用域展开时这些名字解析不到，`class class_name : public TilingDef` 即报 expected class-name。

## 解决配方

修复前（宏块裸放全局作用域，`tilingdata_base.h` 注释示例的写法，verbatim 取自修复前的写入事件）：

```cpp
// TilingData 汇总结构体（DESIGN §5.3）。
// 8 分支共用同一 POD，分支差异由字段值（cos/sin stride、tilingKey）表达，不另立 struct。
BEGIN_TILING_DATA_DEF(BenchHarnessApplyRotaryPosEmbTilingData)
  // ---- shape（对齐后）----
  TILING_DATA_FIELD_DEF(uint32_t, B);      // batch
  TILING_DATA_FIELD_DEF(uint32_t, S);      // seq len
  TILING_DATA_FIELD_DEF(uint32_t, N);      // head num
  TILING_DATA_FIELD_DEF(uint32_t, D);      // head_dim（偶数）
  TILING_DATA_FIELD_DEF(uint32_t, Dh);     // D/2
  // ---- 多核切分（B*S*N token-head 行）----
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);       // B*S*N
  TILING_DATA_FIELD_DEF(uint32_t, perCoreRows);     // 主核行数
  TILING_DATA_FIELD_DEF(uint32_t, tailRows);        // 末核尾块
  // ---- UB 切分 ----
  TILING_DATA_FIELD_DEF(uint32_t, ubRows);          // 每 chunk 行数
  TILING_DATA_FIELD_DEF(uint32_t, ubTail);          // 尾 chunk 行数
  TILING_DATA_FIELD_DEF(uint32_t, rowElems);        // = D
  // ---- cos/sin 对齐（NDDMA stride，按分支算）----
  TILING_DATA_FIELD_DEF(uint32_t, cosB);            // cos 的 B（1 或 B）
  TILING_DATA_FIELD_DEF(uint32_t, cosS);            // cos 的 S（1 或 S）
  TILING_DATA_FIELD_DEF(uint32_t, cosLoopSize);     // NDDMA loopSize（=Dh）
  TILING_DATA_FIELD_DEF(uint32_t, cosSrcStrideDh);  // 源 stride（Dh 单位）
  TILING_DATA_FIELD_DEF(uint32_t, cosDstStrideD);   // 目标 stride（D 单位，tile 因子 2）
  // ---- dtype / 分支 ----
  TILING_DATA_FIELD_DEF(uint8_t,  dtypeBytes);      // sizeof(input dtype)
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);       // §5.5
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(BenchHarnessApplyRotaryPosEmb, BenchHarnessApplyRotaryPosEmbTilingData)
```

修复后（宏块整体包进 `namespace optiling { ... }`，verbatim 取自修复编辑）：

```cpp
// TilingData 汇总结构体（DESIGN §5.3）。
// 8 分支共用同一 POD，分支差异由字段值（cos/sin stride、tilingKey）表达，不另立 struct。
// 宏内引用的 TilingDef/FieldInfo/CTilingDataClassFactory 定义在 namespace optiling，须包在该命名空间内展开。
namespace optiling {
BEGIN_TILING_DATA_DEF(BenchHarnessApplyRotaryPosEmbTilingData)
  // ---- shape（对齐后）----
  TILING_DATA_FIELD_DEF(uint32_t, B);      // batch
  TILING_DATA_FIELD_DEF(uint32_t, S);      // seq len
  TILING_DATA_FIELD_DEF(uint32_t, N);      // head num
  TILING_DATA_FIELD_DEF(uint32_t, D);      // head_dim（偶数）
  TILING_DATA_FIELD_DEF(uint32_t, Dh);     // D/2
  // ---- 多核切分（B*S*N token-head 行）----
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);       // B*S*N
  TILING_DATA_FIELD_DEF(uint32_t, perCoreRows);     // 主核行数
  TILING_DATA_FIELD_DEF(uint32_t, tailRows);        // 末核尾块
  // ---- UB 切分 ----
  TILING_DATA_FIELD_DEF(uint32_t, ubRows);          // 每 chunk 行数
  TILING_DATA_FIELD_DEF(uint32_t, ubTail);          // 尾 chunk 行数
  TILING_DATA_FIELD_DEF(uint32_t, rowElems);        // = D
  // ---- cos/sin 对齐（NDDMA stride，按分支算）----
  TILING_DATA_FIELD_DEF(uint32_t, cosB);            // cos 的 B（1 或 B）
  TILING_DATA_FIELD_DEF(uint32_t, cosS);            // cos 的 S（1 或 S）
  TILING_DATA_FIELD_DEF(uint32_t, cosLoopSize);     // NDDMA loopSize（=Dh）
  TILING_DATA_FIELD_DEF(uint32_t, cosSrcStrideDh);  // 源 stride（Dh 单位）
  TILING_DATA_FIELD_DEF(uint32_t, cosDstStrideD);   // 目标 stride（D 单位，tile 因子 2）
  // ---- dtype / 分支 ----
  TILING_DATA_FIELD_DEF(uint8_t,  dtypeBytes);      // sizeof(input dtype)
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);       // §5.5
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(BenchHarnessApplyRotaryPosEmb, BenchHarnessApplyRotaryPosEmbTilingData)
} // namespace optiling
```

## 设计决策

- 判定依据是读头文件本身：`tilingdata_base.h` 内 `TilingDef`/`FieldInfo`/`StructSizeInfoBase`/`CTilingDataClassFactory` 全部在 `namespace optiling` 内（第 21 行开、注释 `// end of namespace optiling` 闭），而宏定义在该命名空间闭合之后；头文件注释里的示例（裸宏写法）没有展示命名空间，按示例照抄即踩坑。
- `REGISTER_TILING_DATA_CLASS` 生成的匿名命名空间辅助类与全局静态注册实例一并放进 `optiling`，与工厂 `CTilingDataClassFactory` 同域，无副作用。
- 本修复只让宿主侧通过；该工程的 tiling.h 同时被 kernel 侧 include，最终整体改用平凡 POD（本卡的场景边界见适用场景）。

## 预防

- 使用 `BEGIN_TILING_DATA_DEF` 宏族时，一律先写 `namespace optiling {`，再放宏块，最后 `} // namespace optiling`；不要按头文件注释示例的裸写法照抄。
- 遇到报错位置落在 `tilingdata_base.h` 内部（`: public TilingDef`、`CheckAlignAndGenPlaceHolder`）时，先怀疑调用点命名空间，而不是怀疑头文件损坏。

## 验证信号

以下信号来自该算子 TilingData 结构体的实现会话：宏范式首写触发宿主侧编译失败，包命名空间后重建对比。

- 修复前：`./build.sh` 宿主侧 opbuild 阶段报 `tilingdata_base.h:152:39: error: expected class-name before '{' token` 与 `error: 'CheckAlignAndGenPlaceHolder' was not declared in this scope`，`opbuild run failed`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一 build 重跑，宿主侧 `ascend_all_ops` 编译通过（无上述报错），失败点后移到 kernel 侧 opc 编译阶段（另一独立问题），证明命名空间修复对宿主侧生效（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论来自 CANN 9.1.0 的 `register/tilingdata_base.h`；其他版本头文件若把宏放进命名空间内或加了 using 声明，问题不复现（未验证，机制以目标版本头文件为准）。
- 只覆盖宿主侧编译错这一层；kernel 侧 include 同一 tiling.h 时的 `DT_*` 宏冲突是另一根因，本卡修不了。

**未决项**：

- 宏生成类与 `GetTilingData<T>()` 的完整可用性未在本工程继续验证（因 kernel 侧冲突最终改用纯 POD）；host-only tiling 头场景下宏范式 + 命名空间包裹的端到端用法留待独立轨迹验证。
## 坏实践

- 照抄 `tilingdata_base.h` 注释示例的裸宏写法（示例本身不可编译）。
- 看到 expected class-name 就去改宏参或怀疑编译器版本，而不检查宏展开依赖的命名空间。

## 误诊

- 无（本闭环归因一次到位：报错 → 读头文件命名空间结构 → 包 namespace 重建验证）。
