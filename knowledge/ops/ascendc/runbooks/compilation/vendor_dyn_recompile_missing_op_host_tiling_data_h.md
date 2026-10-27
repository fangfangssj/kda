---
type: "Runbook"
title: vendor 缺 op_host 头致 DYN 重编译失败
description: 自定义算子包安装规则不把 op_host/tiling_data.h 装进 vendor 动态实现目录，kernel 头相对 include '../../op_host/tiling_data.h' 在部署树解析落空，TTK kernel/aclnn 模式 DYN 动态重编译在 PRECOMPILE 阶段 fatal error；手工补拷会被重装清除，用 __has_include 内联同 guard 副本持久修复。
tags: [task.debugging, topic.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# vendor 缺 op_host 头致 DYN 重编译失败

## 现象

为自定义 softmax 算子包（含 op_kernel/、op_host/，build.sh 装入 CANN vendor 树，Ascend950PR + CANN 9.2.0）跑 TTK 白盒动态编译验证（`ttk kernel -i whitebox_all.csv --plugin golden_plugin.py`，DYN 动态编译路径；算子包已构建并部署进 `$ASCEND_OPP_PATH/vendors/<Op>`，op_api/op_proto/kernel binary 都在），单用例直接失败：

```
Exception: An error occurred during compile phases of CompileStage.PRECOMPILE, msg is
In file included from .../opp/vendors/BenchHarnessSoftmax/op_impl/ai_core/tbe/BenchHarnessSoftmax_impl/dynamic/op_kernel/bench_harness_softmax_apt.cpp:20:
.../dynamic/op_kernel/arch35/bench_harness_softmax_tiling_struct.h:23:10:
fatal error: '../../op_host/tiling_data.h' file not found
#include "../../op_host/tiling_data.h"
```

检查部署树：`op_impl/ai_core/tbe/<Op>_impl/dynamic/` 下只有 `op_kernel/`（含 arch35 源码——DYN 重编译的输入），**没有 `op_host/` 目录**；源码包里的 `op_host/tiling_data.h`（只 `#include <cstdint>`，自包含）没有被安装规则带过去。该错误随后在单用例复现（ttk kernel 与 aclnn 模式触发 tiling 推断重编译时）**第二次出现**——每次重新部署 vendor 都会复现。

此前为排查该问题，曾把源码树 op_host/tiling_data.h 手工拷贝进安装树（`<Op>_impl/dynamic/op_host/`），动态编译一度恢复、74 用例跑通（73 PASS/1 FAIL）；随后重跑一次包构建脚本 `./build.sh <Package>`（重装 vendor 包），再跑 TTK 白盒，动态编译再次失败：

```
[ERROR] [<用例名> OnDynCompilation] [dynamic_compilation.py:182]: Dyn operator dyn_op_<用例名> build artifacts not found
STATUS: FAIL                       PRECISION_STATUS: FAIL
```

排查安装树发现：重装后的 dynamic/ 目录下根本没有 op_host/ 子目录——手工拷贝的 tiling_data.h 被这次重装覆盖清除。手工补拷头文件每次 rebuild 都会被清掉，验收方重跑 build.sh 后问题必然复现。

## 适用场景

- 工程形态：自定义算子包（CMake `npu_op_package`，TYPE RUN 运行时包）部署 + TTK（ops-test-kit）kernel/aclnn 模式评测（对部署 vendor 做 DYN 动态重编译）。
- API 路径：CANN vendor 目录布局 `$ASCEND_OPP_PATH/vendors/<Op>/op_impl/ai_core/tbe/<Op>_impl/dynamic/`；kernel 源内 `#include "../../op_host/tiling_data.h"` 相对解析；TTK DYN 编译（CompileStage.PRECOMPILE）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter platforms 一致）。
- 适用粒度：operator_family（任何"kernel 头相对引用 op_host 共享头 + 构建脚本不装 op_host 头"的自定义算子包）。
- 路线优先级：recommended_fastpath（__has_include 内联回退，一次修复对 rebuild 免疫）。
- 明确不适用 / 未验证场景：源码包内正常构建（不受影响）；其他安装类型（全量/调试包）是否同样缺该头未验证；静态编译链路未单独验证。

## 根因

单一根因：算子包构建脚本的安装规则不把 op_host/tiling_data.h 安装进 vendor 动态实现目录，device TU 的相对 include `../../op_host/tiling_data.h` 在安装树中无法解析，TBE 动态编译缺头失败。该相对路径相对头文件自身位置解析，在源码树有效（op_kernel/arch35/ → ../../op_host/），在安装树中 op_host/ 不存在时无法解析。

## 解决配方

治本修复（`__has_include` 优先源树路径，否则用与 canonical 头同一 include guard 的内联 verbatim 副本）：

before（op_kernel/arch35/*_tiling_struct.h 原状，相对 include 依赖源树 op_host/）：

```cpp
#ifndef BENCH_HARNESS_SOFTMAX_TILING_STRUCT_H
#define BENCH_HARNESS_SOFTMAX_TILING_STRUCT_H

#include "../../op_host/tiling_data.h"

#endif
```

after（修复后完整文件 73 行，verbatim 取自事件流中的文件读取）：

```cpp
#ifndef BENCH_HARNESS_SOFTMAX_TILING_STRUCT_H
#define BENCH_HARNESS_SOFTMAX_TILING_STRUCT_H

#if defined(__has_include)
#  if __has_include("../../op_host/tiling_data.h")
#    include "../../op_host/tiling_data.h"
#  else
#    ifndef BENCH_HARNESS_SOFTMAX_TILING_DATA_H
#    define BENCH_HARNESS_SOFTMAX_TILING_DATA_H
// ── inline fallback (verbatim copy of op_host/tiling_data.h) ──
#include <cstdint>
constexpr int32_t MAX_PATTERN_RANK = 3;
struct BenchHarnessSoftmaxTilingData {
    int32_t axisNum = 0;
    int64_t axisShape[MAX_PATTERN_RANK] = {0};
    int64_t axisStride[MAX_PATTERN_RANK] = {0};
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;
    int32_t aSplitAxisIdx = 0;
    int32_t rSplitAxisIdx = 0;
    int64_t aUbFactor = 0;
    int64_t aUbFactorAlign = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerAProd = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProd = 0;
    int64_t innerRProdAlign = 0;
    int64_t rLoopCntTotal = 0;
    int64_t preReduceUbSize = 0;
    int64_t expBufUbSize = 0;
    int64_t postReduceUbSize = 0;
    int64_t tmpBufUbSize = 0;
    int64_t cacheBufUbSize = 0;
    int64_t rGroupCnt = 0;
};
#    endif // BENCH_HARNESS_SOFTMAX_TILING_DATA_H
#  endif
#else
#  include "../../op_host/tiling_data.h"
#endif

#endif
```

要点：

1. 内联副本与 op_host/tiling_data.h 使用**同一 include guard**（BENCH_HARNESS_SOFTMAX_TILING_DATA_H）——同一 TU 中两条路径都可见时只生效一份，避免 ODR 双定义。
2. 源码树构建（`__has_include` 命中）仍走 canonical 头，单一事实源不破坏；仅安装树动态编译落到内联副本。
3. 修改落在 op_kernel/ 下，"只允许改 op_kernel/op_host 文件"的约束下可用；正道是改构建安装规则，但那要动 CMake/构建脚本。
4. `__has_include` 依赖编译器支持（bisheng/Clang 支持）；不支持时落到 `#else` 分支的相对 include，回到原问题。

临时补拷（评测继续跑的最短路径，不持久）：

```bash
source <工程>/docs/<Op>/develop/env.sh          # 取得 ASCEND_OPP_PATH 等环境
VENDOR="$ASCEND_OPP_PATH/vendors/<Op>"
IMPL="$VENDOR/op_impl/ai_core/tbe/<Op>_impl/dynamic"
PKG=<源码算子包路径>

# 部署补全：让 kernel 自己的 op_host/tiling_data.h 出现在
# #include "../../op_host/tiling_data.h" 期望的位置（dynamic/op_host/）
mkdir -p "$IMPL/op_host"
cp "$PKG/op_host/tiling_data.h" "$IMPL/op_host/tiling_data.h"
```

注意：拷贝目标是 `<Op>_impl/dynamic/op_host/`（与 `dynamic/op_kernel/` 平级）；重新 build/部署 vendor 会清掉手工补的文件，需重做；先确认头文件自包含（本例只依赖 `<cstdint>`），若它还 include 其他 op_host 头需一并补拷。

## 设计决策

- 备选 1：手工拷贝 tiling_data.h 到安装树——被否决：每次 build.sh 重装都会清除，验收方重跑必挂。
- 备选 2：修改 CMakeLists 安装规则把 op_host 头装进安装树——被否决：本任务约束只允许改 op_kernel/op_host 文件。
- 备选 3：完全内联、删掉相对 include（不用 __has_include）——被否决：破坏"单一事实源"，源树侧后续改 tiling_data.h 时内联副本会漂移。
- 采用：__has_include + 同 guard 内联 verbatim 副本，兼容源树构建与安装树动态编译两条链。
- 为什么手工补拷而不是改 kernel 的 include 路径（临时路线）：评测约束"不改 kernel 源码让结果变绿"；缺的是部署产物而非代码缺陷，补齐部署树是基础设施修复。

## 系统信息

- 工具链：CANN 9.2.0（bisheng/ccec TBE 动态编译）、TTK（ops-test-kit）白盒 kernel 模式。
- 目标平台：Ascend 950 系列（ascend950 构建目录）。
- 关键路径：`operators/<Package>/op_host/tiling_data.h`（canonical 定义，77 行 plain C++ POD）；`operators/<Package>/op_kernel/arch35/*_tiling_struct.h`（shim，修复落点）；安装树 `$ASCEND_OPP_PATH/vendors/<Op>/op_impl/ai_core/tbe/<Op>_impl/dynamic/`（build.sh 产物，无 op_host/）。
- 诊断命令：`ls $ASCEND_OPP_PATH/vendors/<Op>/op_impl/ai_core/tbe/<Op>_impl/dynamic/op_host/`（重装后为空/不存在）；`grep -c __has_include <安装树>/op_kernel/arch35/*_tiling_struct.h`（修复后为 3）。

## 预防

- 自定义算子包若 kernel 头通过相对路径引用 op_host 共享头，跑 TBE 动态编译（从安装树重编）前先检查安装树是否有该头。
- 算子包 CMake 安装规则显式 install `op_host/tiling_data.h`（及其他被 kernel 源 include 的 op_host 头）到 `dynamic/op_host/`；TTK 评测环境准备清单加一项：`dynamic/op_host/` 存在性检查 + 单用例冒烟。
- 修改共享 TilingData 头时同步更新内联副本：同一 guard 只防双定义，不防两份内容漂移——结构体布局不一致时 host/device 可能静默错位。

## 验证信号

问题与验证来源：一次自定义算子包白盒修复会话（rebuild 清除手工拷贝的头后动态编译失败）与其前的手工补拷复现（TTK 白盒 74 用例，Ascend 950 + CANN 9.2.0），以下均为该会话内实测：

- 手工补拷前单用例 DYN_COMPILE_FAILURE（`fatal error: '../../op_host/tiling_data.h' file not found`）；补拷后同用例 `precision_status=PASS`、dyn_tiling_key=0、DYN_PERF 正常出数（来源平台：950；证据等级：tool_observed）。
- 手工补拷后全量 74 白盒用例跑通（74/74 执行、73 PASS / 1 FAIL / 0 SKIP，无 SOC_NOT_SUPPORT、无编译失败）（来源平台：950；证据等级：tool_observed）。
- 第二次复现（评测环境重装后单用例 kernel 模式同样报错）再次补拷后，该用例 `STATUS: PASS / PRECISION_STATUS: PASS`（来源平台：950；证据等级：tool_observed）。
- 重装后安装树 dynamic/ 下无 op_host 目录——手工拷贝被重装清除，证明补拷不持久（来源平台：950；证据等级：tool_observed）。
- 应用 __has_include 内联回退并 rebuild 后：安装树 tiling_struct.h 含 3 处 __has_include、op_host 目录仍不存在（来源平台：950；证据等级：tool_observed）。
- 治本修复后构建成功（build.sh SUCCESS）且白盒全量从 TTK 根目录运行 Total 74 / PASS=74 / FAIL=0；从测试目录（配 kernel_meta 符号链接）运行 PASS=74 FAIL=0（来源平台：950；证据等级：tool_observed）。
- 回归：UT 51/51、C++ aclnn ST 46/46 precision PASSED（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只验证了 TBE 动态编译场景；源码树构建走 __has_include 命中分支，行为应不变但未单独实测。
- 内联副本是 verbatim 快照：canonical 头（77 行版本）后续演化时若不同步更新内联副本，动态编译会用旧结构体定义。
- 头文件补拷只修复"部署树缺头文件"这一形态（报错里能直接看到 `fatal error: ... file not found`）。
- 若 DYN 失败表现为 `DYN_OPERATOR_BUILD_LOST`（`build artifacts not found`、precision_metrics 空）且日志无 fatal error，是别的编译期问题（kernel_meta 缓存/并发等），本配方不覆盖。
- 其他平台未验证，推断 CANN 9.x 自定义包场景可迁移（未验证，推断可迁移）。

**未决项**：

- 正道修复（构建脚本把 op_host/tiling_data.h 装入 dynamic/op_host/）未做；后续应推动恢复单一事实源，消除内联副本的维护负担。
- 补拷后曾出现 aclnn 全量重跑 49/74 FAIL、`DYN_OPERATOR_BUILD_LOST`、空 precision_metrics 的框架级失败，清理 kernel_meta 后仍未解释——跨会话未闭环，需单独归因（缓存、并发还是安装规则）。
## 坏实践

- 手工把头文件拷进 CANN vendor 安装树当"修复"：下一次 build.sh 重装即被清除，且验收方不可见该手工步骤，问题必然复现。
- 把 DYN 重编译失败当成算子精度/功能问题去改 kernel——这是部署产物不完整，改代码方向就错了。
- 不确认安装树布局就反复清缓存重跑：本失败链先撞 .ttk 失败缓存、再撞 kernel_meta 路径错位（各有独立表现），多个问题叠加，需逐个剥离才定位到缺头。

## 误诊

- 初期怀疑 TBE 编译器 bug / 编译结果被缓存返回假成功、"op_host 源文件没安装所以 tiling 函数编不了"（tiling 函数实际在宿主 .so 中，动态编译只重编 kernel 二进制）——均无证据，最后由安装树目录清点 + 编译日志证伪。
- 手工直跑 ccec 编 kernel 源时报 `fatal error: 'kernel_operator.h' file not found`——是手工命令缺少 TBE 注入的 -I 路径所致的红鲱鱼，不代表真实编译环境缺头。
