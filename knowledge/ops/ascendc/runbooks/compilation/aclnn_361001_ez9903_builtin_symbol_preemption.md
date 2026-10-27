---
type: "Runbook"
title: 沿用内置算子名被系统库抢占报 EZ9903
description: 自定义算子沿用内置退役算子名（如 MlaProlog）时，系统 libopapi.so 导出同名 aclnn 符号抢占解析，GetWorkspaceSize 返回 361001 并报 EZ9903 no longer supported；LD_PRELOAD 对照可确诊，全链路重命名修复。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# 沿用内置算子名被系统库抢占报 EZ9903

## 现象

在 Ascend950PR（DAV_3510/arch35）+ CANN 9.2.0 环境下开发纯 AIV 自定义算子 MlaProlog（vendor 包 mla_prolog_custom，已通过 `custom_opp_ubuntu_x86_64.run` 安装到 `$ASCEND_HOME_PATH/opp/vendors/mla_prolog_custom/`）。op_host UT 全部通过后，第一次跑 ST 真机测试（`bash tests/st/run.sh --real --level L0`，共 6 条 L0 用例）：CPU golden 自测 7/7 通过，但每条用例都在 `aclnnMlaPrologGetWorkspaceSize` 阶段失败，6/6 用例报：

```
[ERROR] GetWorkspaceSize 失败: 361001
[ERROR DETAIL] [PID: 1578247] 2026-07-23-19:04:00.232.888 AclNN_Runtime_Error(EZ9903): Interface aclnnMlaPrologGetWorkspaceSize are no longer supported on Ascend950.
        Solution: In this scenario, collect the plog when the fault occurs and locate the fault based on the plog.
```

测试代码已正确调用 `aclInit`/`aclrtSetDevice`，算子包也已确认安装成功（include/lib/kernel binary 都在位）。错误码 361001 属 361xxx 运行时错误区间，初看像 tiling 失败，但真实原因与 tiling 无关。

## 适用场景

- 工程形态：算子仓开发（custom opp 包）+ aclnn 两段式 ST 真机测试。
- API 路径：`aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`；`$ASCEND_HOME_PATH/lib64/libopapi.so`（系统库）与 `$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib/libcust_opapi.so`（自定义库）的符号解析。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何自定义算子在命名阶段都适用。
- 路线优先级：recommended_fastpath（命名前 nm 预检；冲突后全链路重命名）。
- 明确不适用/未验证场景：系统库不导出同名符号的算子名（无冲突，不需要本卡）；其他 CANN 版本的内置符号集合可能不同，需重新用 nm 验证（未验证，推断判别方法可迁移）。

## 根因

CANN 9.2.0 的系统库 `$ASCEND_HOME_PATH/lib64/libopapi.so` 内部实现了内置算子 MlaProlog（该实现在 Ascend950 上已退役），并导出与自定义算子完全同名的符号：

```
=== 系统 libopapi.so 中的 aclnnMlaProlog 符号 ===
00000000023abdcb T aclnnInnerMlaProlog
00000000023aa386 T aclnnInnerMlaPrologGetWorkspaceSize
00000000020cd6d0 T aclnnMlaProlog
00000000020cd000 T aclnnMlaPrologGetWorkspaceSize
00000000020cddb0 T aclnnMlaPrologV2WeightNz
...
=== 自定义 libcust_opapi.so 中的 aclnnMlaProlog 符号 ===
000000000000682a T aclnnMlaProlog
0000000000005a6a T aclnnMlaPrologGetWorkspaceSize
```

机制链：测试二进制同时链接 `libopapi.so`（提供 acl 基础符号）与 `libcust_opapi.so`（自定义算子入口）→ 动态链接器按加载顺序把 `aclnnMlaPrologGetWorkspaceSize` 解析到系统库的内置实现 → 该内置实现在 Ascend950 上已下线，直接返回 361001 并打印 EZ9903 "no longer supported on Ascend950"。即报错根本不来自自定义算子，而是同名内置算子的退役路径。`strings libopapi.so` 还能看到同类退役提示（aclnnFusedInferAttentionScore V1-V4、aclnnIncreFlashAttention V1-V4），说明"版本下线即报 no longer supported"是 libopapi 的通用机制，MlaProlog 只是其中之一。

## 判别方法

1. 符号预检/冲突确认：`nm -D $ASCEND_HOME_PATH/lib64/libopapi.so | grep -i <OpName>`——系统库若已导出 `aclnn<OpName>*` 即冲突。
2. 在测试的报错处理中加 `aclGetRecentErrMsg()` 打印 `[ERROR DETAIL]`（本例由此拿到 EZ9903 原文，比裸错误码 361001 明显得多）。
3. 不改代码的对照验证：`LD_PRELOAD=<vendor>/op_api/lib/libcust_opapi.so ./test ...`——若错误码从 361001（系统库退役路径）变为自定义 wrapper 内部的参数错误（本例变为 161002 `token_x dtype not supported: 27`），即证实符号被系统库抢占、自定义实现本身可达。

## 解决配方

唯一可靠修复是给算子起不与内置符号冲突的名字（本例 `MlaProlog` → `BenchNoHarnessMlaProlog`，对齐同环境 GRU/LSTM 的 Bench 前缀约定），并**全链路同步重命名**。修复涉及的全部关键代码点（before → after，均取自当次修复的实际改动）：

`op_graph/mla_prolog_proto.h`（注册名）：

```cpp
// before
REG_OP(MlaProlog)
...
.OP_END_FACTORY_REG(MlaProlog);
// after
REG_OP(BenchNoHarnessMlaProlog)
...
.OP_END_FACTORY_REG(BenchNoHarnessMlaProlog);
```

`op_host/mla_prolog_def.cpp`：

```cpp
// before
class MlaProlog : public OpDef {
public:
    explicit MlaProlog(const char* name) : OpDef(name)
...
OP_ADD(MlaProlog);
// after
class BenchNoHarnessMlaProlog : public OpDef {
public:
    explicit BenchNoHarnessMlaProlog(const char* name) : OpDef(name)
...
OP_ADD(BenchNoHarnessMlaProlog);
```

`op_host/mla_prolog_infershape.cpp` / `op_host/arch35/mla_prolog_tiling.cpp`：

```cpp
// before
IMPL_OP_INFERSHAPE(MlaProlog).InferShape(InferShape4MlaProlog);
IMPL_OP_OPTILING(MlaProlog).Tiling(MlaPrologTilingFunc).TilingParse<MlaPrologCompileInfo>(TilingParseForMlaProlog);
// after
IMPL_OP_INFERSHAPE(BenchNoHarnessMlaProlog).InferShape(InferShape4MlaProlog);
IMPL_OP_OPTILING(BenchNoHarnessMlaProlog).Tiling(MlaPrologTilingFunc).TilingParse<MlaPrologCompileInfo>(TilingParseForMlaProlog);
```

`op_kernel/CMakeLists.txt` 与 `op_kernel/arch35/mla_prolog_tiling_key.h`：

```cpp
// before
    OP_TYPE MlaProlog
ASCENDC_TPL_ARGS_DECL(MlaProlog,
// after
    OP_TYPE BenchNoHarnessMlaProlog
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessMlaProlog,
```

`op_api/aclnn_mla_prolog.cpp`（L2 入口，`extern "C"` 符号即被抢占点；同时把 epsilon 参数 `float`→`double`，对齐 `npu_op_code_gen` 自动生成头文件中标量属性被提升为 double 的签名——op_host def 文件里属性用 `.Float()` 声明，但 autogen aclnn 头文件签名是 `double`，手写层写 `float` 会造成 ABI 级参数不一致）：

```cpp
// before
extern "C" aclnnStatus aclnnMlaPrologGetWorkspaceSize(
    const aclTensor* token_x,
    ...
    const int64_t n_heads,
    const float rmsnorm_epsilon_cq,
    const float rmsnorm_epsilon_ckv,
    ...
// after
extern "C" aclnnStatus aclnnBenchNoHarnessMlaPrologGetWorkspaceSize(
    const aclTensor* token_x,
    ...
    const int64_t n_heads,
    const double rmsnorm_epsilon_cq,
    const double rmsnorm_epsilon_ckv,
    ...
```

`op_api/aclnn_mla_prolog.h`（`ACLNN_API aclnnStatus aclnnMlaPrologGetWorkspaceSize(` → `ACLNN_API aclnnStatus aclnnBenchNoHarnessMlaPrologGetWorkspaceSize(`）、`op_api/mla_prolog.h`/`op_api/mla_prolog.cpp`（L0 API 同步改名与 double 化）、UT 的 `OP_NAME` 字符串、ST 的 aclnn 调用点与 `#include "aclnn_bench_no_harness_mla_prolog.h"`、ST CMakeLists 的 vendor include/lib 路径一并同步；kernel 入口函数还须跟随改名（否则编译报 kernel entry not implement）。重命名后卸载旧包 → `bash build.sh --soc=ascend950` 重建 → 重装 `.run` 包。

## 设计决策

- 为什么选"全链路重命名"而不是 LD_PRELOAD/链接顺序调整：LD_PRELOAD 只是诊断手段且依赖运行环境；链接顺序在 CMake 结构或库加载顺序变化时不稳定；多 vendor 共享环境下同 SONAME 库的解析还有额外变数。改名从根上消除符号抢占，并顺势对齐了同仓 GRU/LSTM 的 `Bench*` 命名约定。
- 放弃的候选：只改 op_api 的 aclnn 函数名、不动 `REG_OP`——注册名、aclnn 符号、kernel 入口、autogen 头文件名全部由算子名派生，只改一层会留下链路不一致（后续 kernel entry 报错正是这个派生关系的体现）。

## 系统信息

- 芯片 Ascend950PR（DAV_3510/arch35）；CANN 9.2.0（`ASCEND_HOME_PATH` 指向 cann-9.2.0 安装）；GCC 13.3.0；CMake 3.20.5。
- 相关文件：`op_graph/mla_prolog_proto.h`、`op_host/mla_prolog_def.cpp`、`op_host/mla_prolog_infershape.cpp`、`op_host/arch35/mla_prolog_tiling.cpp`、`op_kernel/CMakeLists.txt`、`op_kernel/arch35/mla_prolog_tiling_key.h`、`op_kernel/mla_prolog_arch35.cpp`、`op_api/aclnn_mla_prolog.cpp`、`op_api/aclnn_mla_prolog.h`、`op_api/mla_prolog.cpp`、`op_api/mla_prolog.h`、`tests/st/CMakeLists.txt`、`tests/ut/op_host/test_mla_prolog_tiling.cpp`、`tests/ut/op_host/test_mla_prolog_infershape.cpp`、`tests/st/test_aclnn_mla_prolog.cpp`。

## 预防

- 新建自定义算子命名前先跑 `nm -D $ASCEND_HOME_PATH/lib64/libopapi.so | grep -i <候选名>`，命中 `aclnn<候选名>*` 即换名。
- 团队/仓库统一的算子名前缀（如 Bench 基准前缀）可系统性避开内置算子名空间。
- 自定义算子包安装后可用 `nm -D <vendor>/op_api/lib/libcust_opapi.so | grep <OpName>` 与系统库对照，确认无交集。

## 验证信号

来源说明：以下信号来自该算子一次完整 ST 真机联调过程（重命名前后的实测对比），均为命令/测试输出。

- 重命名前：6/6 L0 用例 `GetWorkspaceSize 失败: 361001`，`[ERROR DETAIL] AclNN_Runtime_Error(EZ9903): Interface aclnnMlaPrologGetWorkspaceSize are no longer supported on Ascend950`（来源平台：950；证据等级：tool_observed）。
- `nm -D` 确认系统 libopapi.so 导出 `aclnnMlaProlog`/`aclnnMlaPrologGetWorkspaceSize`，自定义 libcust_opapi.so 导出同名符号（来源平台：950；证据等级：tool_observed）。
- `LD_PRELOAD` 自定义 libcust_opapi.so 后错误码变为 161002（进入自定义 wrapper 的参数校验），证实符号抢占关系（来源平台：950；证据等级：tool_observed）。
- 重命名并重建重装后：autogen 头文件变为 `aclnn_bench_no_harness_mla_prolog.h`，libcust_opapi.so 导出 `aclnnBenchNoHarnessMlaProlog*`；UT 12/12 通过；ST 不再出现 361001（转为自定义 wrapper 内部参数错误，另有根因）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只解决"系统库符号抢占"。重命名后出现的 161002 参数错误与 ST 二进制 double free 崩溃是独立问题，不要归因到符号冲突。
- 系统库符号集合随 CANN 版本变化；换版本后需重新 nm 验证。
- 未验证，推断可迁移：其他芯片平台上的 libopapi 同样存在退役算子符号，判别方法（nm + LD_PRELOAD 对照）应当适用。
- 与"opType 与内置 ops_legacy 同名碰撞（507035，kernel binary 被 built-in 优先加载）"是不同层的同名碰撞：本卡是 aclnn 符号被系统库抢占，那张卡是 kernel 二进制加载优先级；判别信号（EZ9903 vs 507035）与抢占层不同，修复手段同为重命名。

**未决项**：

- 无（本根因已闭合；后续 161002 与 double free 见对应卡片）。
## 坏实践

- 自定义算子直接沿用语义相近的内置算子名（如 MlaProlog），不做符号预检。
- 依赖链接顺序或 LD_PRELOAD 让自定义符号"碰巧"先被解析——环境一变即失效。

## 误诊

- 初判 361001 为 tiling 失败（361xxx 错误码区间直觉），准备查 tiling 实现；实际从错误详情文本（EZ9903 no longer supported）与 nm 符号对照才指向符号抢占。
- 中途怀疑 ops-info.json 的 `dynamicCompileStatic`/`dynamicRankSupport` 配置差异、kernel binary 的 magic/coreType 差异——与同环境正常算子（GRU）逐项对照后排除。
