---
type: "Runbook"
title: ge.jit_compile=2图级闸门挡预编译复用
description: op 侧注册 GenSimplifiedKey 且 key 与 binary_info_config.json 精确匹配、vendor config 被成功读取，GE 图路径仍因 graph 级 ge.jit_compile=2（CheckJitCompileForBinReuse 判 Need to compile、CanReuseBinaryKernel 拒绝复用）回退 TBE 动态编译；op 级 jitCompile 未设置被 graph 级默认覆盖，op 层无法反制，负结果。
tags: [task.debugging, topic.graph_execution, topic.compilation]
status: stable
sources: []
created_at: '2026-09-14T12:16:44Z'
updated_at: '2026-09-14T12:16:44Z'
platforms: ["950"]
---

# ge.jit_compile=2图级闸门挡预编译复用

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义算子 BenchHarnessScatter 做 GEIR 图模式验收时，route 探针与 TTK GEIR 用例全部失败（kernel 执行 EZ9999，判别见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡）。为让 GE 图路径像 ACLNN 路径一样命中预编译二进制（`"blockDim":-1`，运行时 `SetBlockDim` 生效），op_host 侧已经做了三件事：

1. 在 `bench_harness_scatter_def.cpp` 注册 `GenSimplifiedKey` 回调（`IMPL_OP(BenchHarnessScatter).GenSimplifiedKey(GenSimplifiedKeyForBenchHarnessScatter)`）；
2. 把 `TilingParse` 注册放在 `OP_ADD` 之前（避免被 OP_TILING_LIB 模式的 no-op 占位覆盖）；
3. 把算子安装进 OPP vendor 使 `binary_info_config.json` 可被 GE 发现。

验收时用 route 日志核对：GenSimplifiedKey 确实已注册且最先（`RegisterOpImplToRegistry: Op type[BenchHarnessScatter] [GenSimplifiedKey]`、`gen_simplifiedkey func registered first`），回调产出的 key 与 vendor config 精确匹配——route 用例（dim=0, data=f32, indices=i32, updates=f32）产出 `BenchHarnessScatter/d=0,p=0/0,2/3,2/0,2/0,2`，与 `binary_info_config.json` 中 `BenchHarnessScatter_b51b45466550cde4aefe19d6cd25f864.o` 条目的 key 逐字符一致；TEFUSION 也成功读取了该 config（`ReadOpBinaryInfoConfig Start to read binary info file[.../vendors/BenchHarnessScatter/...]`）。

但 GE 图路径仍然没有复用预编译二进制，决策链（route 日志逐行）：

```text
GetContextValue The value of context [ge.jit_compile] is [2].
CheckJitCompileForBinReuse Node(n_route_target), get jit_compile from ge is 2. Need to compile
CanReuseBinaryKernel Cannot reuse binary result for node(n_route_target)
Build options[need_precompile] value is [false].
The [prebuildPattern] for operation [BenchHarnessScatter] is [Opaque].
DoPrebuildOp Node needs to skip prebuild
```

随后走 TBE 动态编译生成 `te_benchharnessscatter_911e6935...` 内核并 EZ9999 崩溃，route 维持 `rungraph_rc=4294967295 ... value=FAIL`。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：GEIR/GE 图模式算子开发与验收（`ge::Session` + `RunGraph`，op_host 注册层可改、上层调用方冻结）。
- API 路径：`IMPL_OP(...).GenSimplifiedKey(...)`、`binary_info_config.json`（vendor `op_impl/ai_core/tbe/kernel/config/<soc>/`）、GE 图编译决策链 `GetContextValue [ge.jit_compile]`/`CheckJitCompileForBinReuse`/`CanReuseBinaryKernel`/`PreBuildTbeOp`、`GetOpJitCompileForBinReuse`（op 级 jitCompile）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——GE 图执行链的二进制复用决策行为，与算子种类无关；对按 `ge::Session` 空选项跑图的所有自定义算子成立。
- 路线优先级：banned——在 Session RunGraph 默认 `ge.jit_compile=2` 的执行链上，"op 侧注册 GenSimplifiedKey 即可让 GE 图路径命中/复用预编译二进制"这条路线不可达；不要在这条路线上继续投入。修复入口在上层（session/graph options 显式设置 `ge.jit_compile=0/1`）或 kernel 侧防护。
- 明确不适用 / 未验证场景：
  - 上层能在 session/graph options 显式设置 `ge.jit_compile=0/1`（或 `need_precompile=true`）的执行链——该场景下预编译匹配能否生效未验证（本轨迹的测试程序是冻结 ST，不可改）；
  - ACLNN 直调路径：不走 GE 图编译决策链，直接加载预编译二进制，本卡不适用（实测正常）；
  - CANN 9.2.0 之外的版本（决策链日志行为可能变化，未验证，推断决策链类似但需重新 grep 确认）；
  - "key 格式写错导致不命中"的场景（那是 op 侧责任，本轨迹 key 是匹配的）；若 key 不匹配，先修 key 再回到本卡看闸门。

## 根因

GE 图执行链是否复用预编译二进制的闸门在 graph 级上下文 `ge.jit_compile`（`ge::Session` 空选项时默认为 2，"总是 JIT 编译"），不在 op 侧注册是否齐全。

op 级 jitCompile 无法反制：日志显示 `GetOpJitCompileForBinReuse Jit compile of op[...] is [0]`（未设置），随后仍被 graph 级值 2 覆盖；算子 ops-info JSON（aic-ascend950-ops-info.json）里没有 `jitCompile` 字段（另有 `GetStrFromOpContent:Op BenchHarnessScatter not found jitCompile in OpContent!`），OpDef API 也没有对应注册项（本算子只用过 `ExtendCfgInfo("opFile.value", ...)`）。因此 key 匹配、config 可读、回调注册先后顺序这些 op 侧条件全部满足时，二进制复用依然被拒——GenSimplifiedKey 的产出根本没有参与命中判定。

## 尝试与观测（负结果证据）

- **attempt 1**（op 侧三件事就位：GenSimplifiedKey 注册且最先 + key 逐字符匹配 + vendor config 读取成功）：决策链仍 `ge.jit_compile] is [2]` → `Cannot reuse binary result` → TBE 动态编译。
- **attempt 2**（fresh staging 重建重跑）：同一失败复现——排除缓存/安装因素。
- **attempt 3**（排查期怀疑 config 可见性：日志有 `GetBinaryConfigFilePath:The binary configuration file path is .` 空路径、`DO_NOT_READ` 警告、`GetCustomBinaryPath Not found custom binary key`）：最终在 vendor 路径找到 `ReadOpBinaryInfoConfig` 成功读取的证据——配置可见性不是问题，"是否允许复用"才是；这些空路径警告是 built-in 路径探测的噪音，容易把人带偏。
- **boundary**：op 级 `jitCompile` 即使存在也会被 graph 级 2 覆盖（日志里其他算子出现过 `not found jitCompile in OpContent`，核实后确认该字段无注册 API）；上层显式 `ge.jit_compile=0/1` 后行为未验证（冻结测试不可改）。

## 设计决策

- 为什么把该结论独立成负结果卡：op 侧三件事（key 格式、config 可读、注册顺序）都是显性成本，且源码注释/既往经验容易让人相信"注册 GenSimplifiedKey 就能命中预编译"；实测证明这些条件全满足时闸门仍在 graph 级，提前止损的价值明确。
- 为什么判定 banned 而非 safety_fallback：本轨迹没有任何一条 op 层路径能在默认 Session 选项下命中预编译二进制（fresh staging 重跑复现同一失败），不存在"较慢但可达"的降级路线。
- 判别手段选"日志差分"而非"改代码二分"：`ge.jit_compile` + `CanReuseBinaryKernel` + `need_precompile` 三条日志证据即可定性，无需破坏性实验。
- 当前方案路线档位：banned（op 层路线）；上层配置路线未验证，不在本卡宣称。

## 系统信息

- 硬件/软件：Ascend950PR_9579（`ai_core_cnt:28`、`vector_core_cnt:56`），CANN 9.2.0，GE route 日志（DEBUG 级，单文件约 35MB）。
- 关键文件：
  - `op_host/bench_harness_scatter_def.cpp`——GenSimplifiedKey/TilingParse 注册（IMPL_OP 在 OP_ADD 之前）；
  - `<staging_vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json`——10 条 simplifiedKey（覆盖 d=0/1 × p=0/1 × 各 dtype 组合）；
  - `<staging_vendor>/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json`——ops-info（无 jitCompile 字段）；
  - route 日志 `/tmp/geir-route.*.log`。
- key 匹配核对命令（本会话走通）：

```bash
grep -n "jit_compile\|CanReuseBinaryKernel\|need_precompile\|DoPrebuildOp" "$ROUTE_LOG" | head -6
# 命中 "ge.jit_compile] is [2]" + "Cannot reuse binary result" 即本卡场景

cat <staging_vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); [print(k) for e in d['binList'] for k in e['simplifiedKey']]" | \
  grep "d=0,p=0/0,2/3,2/0,2/0,2"
# GenSimplifiedKey 产出（def.cpp）："BenchHarnessScatter/d=%lld,p=0/%d,%d/%d,%d/%d,%d/%d,%d" → 与上一步逐字符一致
```

- GenSimplifiedKey 回调 verbatim（`op_host/bench_harness_scatter_def.cpp`，产出 key 与 config 精确匹配、注册本身无缺陷）：

```cpp
static UINT32 GenSimplifiedKeyForBenchHarnessScatter(gert::TilingContext *context, ge::char_t *key_buf)
{
    // Read dim attr (index 0, REQUIRED Int)
    int64_t dim = 0;
    const auto* computeNodeInfo = context->GetComputeNodeInfo();
    if (computeNodeInfo != nullptr) {
        const auto* attrs = computeNodeInfo->GetAttrs();
        if (attrs != nullptr) {
            const int64_t* dimPtr = attrs->GetInt(0);
            if (dimPtr != nullptr) {
                dim = *dimPtr;
            }
        }
    }

    // Read input dtypes/formats: 0=data, 1=indices, 2=updates
    // Output y shares dtype/format with data (y.dtype = data.dtype, y.format = data.format).
    int dType0 = 0, dFmt0 = 2;   // data
    int dType1 = 0, dFmt1 = 2;   // indices
    int dType2 = 0, dFmt2 = 2;   // updates
    const auto* tensor0 = context->GetInputTensor(0);
    if (tensor0 != nullptr) {
        dType0 = static_cast<int>(tensor0->GetDataType());
        dFmt0 = static_cast<int>(tensor0->GetStorageFormat());
    }
    const auto* tensor1 = context->GetInputTensor(1);
    if (tensor1 != nullptr) {
        dType1 = static_cast<int>(tensor1->GetDataType());
        dFmt1 = static_cast<int>(tensor1->GetStorageFormat());
    }
    const auto* tensor2 = context->GetInputTensor(2);
    if (tensor2 != nullptr) {
        dType2 = static_cast<int>(tensor2->GetDataType());
        dFmt2 = static_cast<int>(tensor2->GetStorageFormat());
    }

    std::snprintf(key_buf, 256,
        "BenchHarnessScatter/d=%lld,p=0/%d,%d/%d,%d/%d,%d/%d,%d",
        static_cast<long long>(dim),
        dType0, dFmt0,
        dType1, dFmt1,
        dType2, dFmt2,
        dType0, dFmt0);
    return 0;
}

IMPL_OP(BenchHarnessScatter).GenSimplifiedKey(GenSimplifiedKeyForBenchHarnessScatter);
```

## 预防

- 给算子加 GEIR 图模式支持时，不要把"注册 GenSimplifiedKey + 安装 vendor"当作预编译命中的充分条件；先用一行 grep 确认 `ge.jit_compile` 上下文值再决定投入方向。
- 评估算子能否走预编译二进制时，把"key 匹配"与"复用闸门"分开验证：前者是 op 侧责任，后者由上层执行链决定。
- 排查 binary 选择时忽略 built-in 路径探测的空路径/`DO_NOT_READ` 噪音警告，以 vendor 路径上的 `ReadOpBinaryInfoConfig` 与决策链行为为准。

## 验证信号

来源说明：以下信号来自该算子 GEIR 图模式验收会话（route 日志 grep + vendor config dump，fresh staging 重跑复核），均为 950 上板实测。

- GenSimplifiedKey 已注册且最先：`RegisterOpImplToRegistry: Op type[BenchHarnessScatter] [GenSimplifiedKey]`、`gen_simplifiedkey func registered first`（来源平台：950；证据等级：tool_observed）。
- key 精确匹配：回调产出 `BenchHarnessScatter/d=0,p=0/0,2/3,2/0,2/0,2`，与 binary_info_config.json 中 `BenchHarnessScatter_b51b45466550cde4aefe19d6cd25f864.o` 条目一致；config 共 10 条 binList 条目全部 dump 核对（来源平台：950；证据等级：tool_observed）。
- config 被读取：`ReadOpBinaryInfoConfig Start to read binary info file[.../binary_info_config.json]`（来源平台：950；证据等级：tool_observed）。
- 复用仍被拒（决策链）：`GetContextValue The value of context [ge.jit_compile] is [2].`、`CheckJitCompileForBinReuse ... Need to compile`、`CanReuseBinaryKernel Cannot reuse binary result for node(n_route_target)`、`Build options[need_precompile] value is [false].`、`DoPrebuildOp Node needs to skip prebuild`（来源平台：950；证据等级：tool_observed）。
- op 级 jitCompile 未设置（=0）被 graph 级 2 覆盖；ops-info 无 jitCompile 字段（`GetStrFromOpContent:Op BenchHarnessScatter not found jitCompile in OpContent!`）（来源平台：950；证据等级：tool_observed）。
- 最终仍生成 TBE 动态编译内核 `te_benchharnessscatter_911e6935...` 并执行失败 EZ9999（route value=FAIL，rungraph_rc=4294967295）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定在"`ge::Session` 空选项 + 静态 shape 节点"的 GE 图执行链；上层显式设置 `ge.jit_compile=0/1` 后行为未验证（需非冻结调用方配合）。
- 结论限定 CANN 9.2.0；其他版本的 prebuild 决策链未验证（未验证，推断决策链类似但需重新 grep 确认）。
- `ge.jit_compile=2` 复用闸门与 prebuildPattern=Opaque 跳过链（见同批《GE图prebuildPattern=Opaque跳过预编译匹配》卡）是决策链上两层独立闸门：本卡证明复用被 graph 级值拒绝，不回答"若复用被放行、prebuild 匹配能否命中"。
- 环境变量途径（`GE_OPTIONS="ge.jit_compile=1"`、`GE_JIT_COMPILE=0`）已实测不改变 kernel 选择（同批主诊断卡修复对照），Session options 途径未试。

**未决项**：

- 优先项：在上层可配置的执行链上复现并验证 `ge.jit_compile=0/1`（或 `need_precompile=true`）后预编译二进制能否命中、route/TTK 是否转绿（需非冻结的调用方配合，本轨迹不可达）。
- `DynamicCompileStaticFlag(false)` 会引发 E40021 编译失败的既有记录（上一会话 LOG 转述，本轨迹未复现，agent_reported）——该方向与预编译命中的关系待验证。
- `prebuildPattern=Opaque` 是否可在 op 侧改写（如通过 ExtendCfgInfo 声明非 Opaque pattern）未探索。
## 坏实践

- 在 op 层反复调整 GenSimplifiedKey 的 key 格式、config 路径、注册顺序，试图让 GE 图路径命中预编译二进制——闸门在 graph 级 `ge.jit_compile`，这些调整在本场景全部无效。
- 把"TEFUSION 成功读取了 binary_info_config.json"当作"预编译二进制会被使用"的证据——读取和命中是两个独立环节。
- 把希望寄托在 op 级 `jitCompile` 配置项上——该字段即使存在也会被 graph 级 2 覆盖，且 OpDef 无注册 API。

## 误诊

- 排查初期怀疑 vendor 的 `binary_info_config.json` 根本没被 GE 发现（空路径警告、`DO_NOT_READ`、`GetCustomBinaryPath Not found custom binary key`），沿 built-in OPP 路径与 `ASCEND_CUSTOM_OPP_PATH` 追了多轮；最终在 vendor 路径上找到 `ReadOpBinaryInfoConfig` 成功读取的证据——配置可见性不是问题，"是否允许复用"才是（agent_reported 假设 → tool_observed 修正）。
- 曾怀疑 prebuildPattern=Opaque 是本算子的配置异常：经 grep 对照，内置算子（Abs/Acos 等）同为 Opaque，属默认值，非差异项。
