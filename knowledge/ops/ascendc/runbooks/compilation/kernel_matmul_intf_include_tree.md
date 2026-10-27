---
type: "Runbook"
title: "kernel Matmul 头在 ascendc 树且别名在 intf"
description: "kernel 声明 Matmul<...> 报 no template named 'Matmul'：kernel 编译（ccec）的 include 基路径是 ascendc 树（Matmul 高阶 API 头在 highlevel_api/lib/matmul/，host 树的 adv_api/matmul/matmul.h 解析不到），且 Matmul 别名在 matmul_intf.h 而非 matmul.h（后者只定义 MatmulImpl）。"
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-09-17T04:05:46Z'
updated_at: '2026-09-17T04:05:46Z'
platforms: ["950"]
---

# kernel Matmul 头在 ascendc 树且别名在 intf

## 现象

在为 MoE 分组矩阵乘算子（Ascend950PR / arch35 / CANN 9.1.0）的 kernel 头里声明高阶 Matmul 对象时，最初照 host 侧文档的路径直觉写：

```cpp
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/matmul/matmul.h"
...
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X, (TRANSPOSE_WEIGHT == 1)>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<AType, BType, CType, BiasType, CFG_NORM> mm;
```

kernel 编译（ccec，include 基路径为 `${ASCEND_HOME_PATH}/x86_64-linux/include/ascendc/`）报错原文：

```text
op_kernel/arch35/bench_no_harness_grouped_matmul.h:59:5: error: no template named 'Matmul'
```

先后尝试 `#include "adv_api/matmul/matmul.h"`、`#include "lib/matmul/matmul.h"` 都仍是同一报错。排查发现两个叠加事实：

1. **路径树不同**：kernel 编译能解析的是 `include/ascendc/` 树（其下有 `highlevel_api/lib/matmul/`），而 `adv_api/matmul/matmul.h` 挂在 host 树 `asc/include/` 下，kernel 的 -I 集合里没有它（include 静默落空后 `Matmul` 名字不存在）。
2. **别名头不同**：`ascendc/include/highlevel_api/lib/matmul/matmul.h` 只定义 `MatmulImpl` 类；`using Matmul = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;` 这组别名在旁边的 `matmul_intf.h`（45/53/63/75/82 行多个条件编译分支）。

最终 `#include "lib/matmul/matmul_intf.h"` 编译通过（kernel -I 已含 `highlevel_api/lib` 的父级解析路径，`lib/matmul/matmul_intf.h` 可解析，且其内部 `#include "include/adv_api/matmul/matmul.h"` 由 asc 树补齐）。

## 适用场景

- 工程形态：registry-invoke 自定义算子 op_kernel 侧使用高阶（模板化）Matmul API：`Matmul<AType, BType, CType, BiasType, CFG_NORM>` + `REGIST_MATMUL_OBJ` + SetTail/SetTensorA/SetTensorB/IterateAll/End。
- API 路径：kernel 侧 `lib/matmul/matmul_intf.h`（`Matmul` 别名）与 `MatmulType`（`lib/matmul/constant_tiling.h`）；host 侧 tiling 对应头是 `asc/include/adv_api/matmul/matmul_tiling.h`（另一棵树）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（CANN 9.1.0 双树 include 布局）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：host 侧 tiling（MatmulApiTiling）的头路径见 [op_host include 根卡](matmul_highlevel_api_include_paths_ini.md)；不同 CANN 版本目录布局未验证；其他构建框架若 kernel -I 不含 ascendc 树而含 asc/include/adv_api，则应改用 adv_api 树的 `matmul_intf.h`（同内容，另见同卡增量）。

## 根因

kernel 与 host 是两棵并行的 include 树：kernel 编译（ccec）以 `include/ascendc/` 为基（Matmul 高阶 API 位于 `ascendc/include/highlevel_api/lib/matmul/`），host 编译以 `asc/include/` 为基（adv_api/...）。`#include "adv_api/matmul/matmul.h"` 在 kernel 翻译单元解析不到（或解析到不含 Matmul 别名的副本）；同时 `lib/matmul/matmul.h` 只含 `MatmulImpl` 实现，用户要的 `Matmul` 模板别名（含 CFG_NORM 等 policy 分支）在 `matmul_intf.h`。两个因素叠加导致 "no template named 'Matmul'"。

## 解决配方

修复前（kernel 头的 include 与声明）：

```cpp
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/matmul/matmul.h"
#include "bench_no_harness_grouped_matmul_tiling_data.h"
#include "bench_no_harness_grouped_matmul_tiling_key.h"

namespace NsBenchNoHarnessGroupedMatmul {

using namespace AscendC;

constexpr int64_t MAX_EXPERT_NUM = 128;

template <typename D_T_X, int HAS_BIAS, int TRANSPOSE_WEIGHT>
class BenchNoHarnessGroupedMatmul {
    // Matmul 类型定义
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X, (TRANSPOSE_WEIGHT == 1)>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<AType, BType, CType, BiasType, CFG_NORM> mm;
    ...
```

修复后（include 改为 kernel 树的 matmul_intf.h；其余声明不变）：

```cpp
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul/matmul_intf.h"
#include "bench_no_harness_grouped_matmul_tiling_data.h"
#include "bench_no_harness_grouped_matmul_tiling_key.h"

namespace NsBenchNoHarnessGroupedMatmul {

using namespace AscendC;

constexpr int64_t MAX_EXPERT_NUM = 128;

template <typename D_T_X, int HAS_BIAS, int TRANSPOSE_WEIGHT>
class BenchNoHarnessGroupedMatmul {
    // Matmul 类型定义
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X, (TRANSPOSE_WEIGHT == 1)>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, D_T_X>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<AType, BType, CType, BiasType, CFG_NORM> mm;
    ...
```

排查用的路径定位命令（可复用）：

```bash
# kernel 树里的 Matmul 头
find $ASCEND_HOME_PATH/x86_64-linux/include/ascendc -name "matmul*" | head
# → .../ascendc/include/highlevel_api/lib/matmul/{matmul.h, matmul_intf.h, ...}

# Matmul 别名在哪
grep -n "using Matmul" $ASCEND_HOME_PATH/x86_64-linux/include/ascendc/highlevel_api/lib/matmul/matmul_intf.h
# → 45: using Matmul = MatmulImpl<...>; （多个条件编译分支）

# matmul.h 里只有实现类
grep -n "class Matmul" .../highlevel_api/lib/matmul/matmul.h
# → 54: class MatmulImpl {
```

## 设计决策

- include `matmul_intf.h` 而不是 `matmul.h`：用户 API 是 `Matmul` 别名模板（带 MM_CFG/回调/policy 分支），intf 头还会按宏选择 `MatmulClient`/`MatmulServiceAux` 等变体；`matmul.h` 只是实现细节。
- 高阶 Matmul（`Matmul<...>` + REGIST_MATMUL_OBJ + IterateAll）而不是手写 Mmad：分组 Matmul 每个 expert 一组 GEMM，高阶 API 内部管理 UB/分块/流水，骨架代码量小；代价是 include 树与注册宏的额外约束（本卡与 [REGIST_MATMUL_OBJ 类型统一卡](regist_matmul_obj_type_unify.md)、[REGIST_MATMUL_OBJ 入口调用卡](regist_matmul_obj_kernel_entry_tpipe_pointer.md)即其成本）。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写 kernel 侧 Matmul 前先 `find $ASCEND_HOME_PATH -path "*ascendc*matmul*"` 区分两棵树：kernel 用 `lib/matmul/matmul_intf.h`（ascendc 树），host tiling 用 `asc/include/adv_api/matmul/matmul_tiling.h`。
- `no template named 'X'` 且 include 的是"看似正确"的头时，优先怀疑：① 头在另一棵 include 树（-I 不含其根）；② 头里只有实现类，别名在兄弟头。

## 验证信号

以下信号来自该算子初版骨架的编译闭环：include 改为 `lib/matmul/matmul_intf.h` 后重新编译，no template named 'Matmul' 消失，kernel 编译推进到 REGIST_MATMUL_OBJ 实参匹配；最终全量编译通过。

- 修复前：`op_kernel/arch35/bench_no_harness_grouped_matmul.h:59:5: error: no template named 'Matmul'`（`adv_api/matmul/matmul.h` 与 `lib/matmul/matmul.h` 两种 include 下均复现，各 5+ 处实例化点）（来源平台：950；证据等级：tool_observed）。
- 修复后：该报错不再出现，kernel 编译推进到 `InitObj` 实参匹配错误（下一根因，见 [REGIST_MATMUL_OBJ 入口调用卡](regist_matmul_obj_kernel_entry_tpipe_pointer.md)）（来源平台：950；证据等级：tool_observed）。
- 全量回归：`build.sh --soc=ascend950 -j8` 生成 `custom_opp_ubuntu_x86_64.run`（694K）与 3 个 kernel .o（binary_info_config 覆盖 12 TilingKey）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- include 路径基于 CANN 9.1.0 x86_64-linux 布局（ascendc 树 vs asc 树）；版本升级后需重新定位（机制推断可迁移）。
- `matmul_intf.h` 内部 `#include "include/adv_api/matmul/matmul.h"` 依赖 asc 树也在 kernel -I 集合中（本工程由 CANN 编译框架自动补齐）；自建编译脚本需自行保证。
- kernel 侧可达的 matmul 头随构建框架 -I 集合而异：本工程解析 ascendc 树的 `lib/matmul/matmul_intf.h`；另一构建框架的 kernel -I 含 `asc/include/adv_api` 时，应写裸 `matmul_intf.h`（与 `asc/include/adv_api/matmul_intf.h` 同内容），不要照抄内置算子的 `lib/matmul_intf.h`——该路径仅内置算子编译可见（见 [TCubeTiling 双定义卡](tcube_tiling_host_kernel_dual_definition.md)与 [Matmul 高阶 API 头不可见卡](matmul_highlevel_api_include_paths_ini.md)）。

**未决项**：

- `matmul_intf.h` 各条件编译分支（MatmulClient/MatmulServiceAux 等）的选择宏与适用场景未逐一验证，本卡仅覆盖默认（MatmulImpl）分支。
## 坏实践

- kernel 头照 host 文档写 `#include "adv_api/matmul/matmul.h"`。
- 只 include `matmul.h` 就想用 `Matmul<...>`——别名不在该头。

## 误诊

- 曾怀疑"kernel 编译器没带 matmul 模块"想换 API 路线；实际是 include 树选择问题，换到 `lib/matmul/matmul_intf.h` 即解。
