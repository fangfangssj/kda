---
type: "Runbook"
title: GE图prebuildPattern=Opaque跳过预编译匹配
description: GE 图路径自定义算子预编译二进制不命中：GenSimplifiedKey 已注册、key 与 binary_info_config.json 逐字符一致、config 被成功读取仍不命中——prebuildPattern=Opaque（AscendC 自定义算子全量默认）在 simplifiedKey 匹配之前跳过 prebuild，节点恒回退 te_ 动态编译；op 侧注册顺序、环境变量、TilingParse 调整均不可达。
tags: [task.debugging, topic.compilation, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-14T12:16:44Z'
updated_at: '2026-09-14T12:16:44Z'
platforms: ["950"]
---

# GE图prebuildPattern=Opaque跳过预编译匹配

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义算子（AscendC kernel + 自建 op_host/op_graph 包，scatter 类，BenchHarnessScatter）做 GEIR 图模式验收：算子包已通过 op_build 产出带 10 条 simplifiedKey 的预编译二进制（kernel JSON `"blockDim": -1`，即动态 blockDim，运行时由 TilingFunc `SetBlockDim` 生效），期望 GE 图路径命中预编译二进制、从而避开 TBE 动态编译。为此算子包具备全部"教科书条件"：

- `op_host/bench_harness_scatter_def.cpp` 注册了 `IMPL_OP(BenchHarnessScatter).GenSimplifiedKey(...)` 回调，运行期按实际输入生成形如 `BenchHarnessScatter/d=<dim>,p=0/<dtype>,<fmt>/<dtype>,<fmt>/<dtype>,<fmt>/<dtype>,<fmt>` 的 key；
- vendor 包 `op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json` 中存在与 route 用例（data=f32、indices=i32、updates=f32、dim=0）**逐字符一致**的 key：`BenchHarnessScatter/d=0,p=0/0,2/3,2/0,2/0,2`，映射到 `ascend950/bench_harness_scatter_apt/BenchHarnessScatter_b51b45466550cde4aefe19d6cd25f864.o`；
- `IMPL_OP(...).TilingParse<...>(...)` 注册在 `OP_ADD` 之前（不被 OP_TILING_LIB 模式的 no-op 占位覆盖），`aic-ascend950-ops-info.json` 里 `dynamicCompileStatic.flag` 为 `"true"`。

但 route 每轮执行的 kernel 恒为 TBE 动态编译名 `te_benchharnessscatter_911e6935...`（`FillTaskDefAfterGenTask:Set kernel_name[te_...]`），预编译二进制从未参与匹配。两轮独立验收会话（不同 staging 重建）一致复现，并做了多轮摆放/缓存对照：

1. 隔离 staging（`ASCEND_CUSTOM_OPP_PATH=/tmp/geir_opp.*/vendors/<Op>`）：TBE fusion 日志只解析 built-in 路径的 binary 信息（`GetBinaryInfoConfigPath .../opp/built-in/... does not exist` 后回退 `ops_legacy/ops_cv/ops_nn/ops_math` 子目录），vendor 路径仅被 `SetRelocatableBinaryConfigPaths` 探测 `relocatable_kernel_info_config.json`（多条 `does not exist`）；
2. staging 安装进全局 OPP（`$ASCEND_HOME_PATH/opp/vendors/<Op>`，安装日志 `SUCCESS`）：`ReadOpBinaryInfoConfig Read binary info file[.../opp/vendors/<Op>/.../binary_info_config.json] success`——config 已进内存；
3. 清空 TBE 缓存（`~/atc_data/kernel_cache` 的 `te_*` 与 `tests/geir/kernel_meta`）后 fresh staging 重建、全新动态编译（kernel sha256 变化）：仍 `te_`。

无论哪轮，整份 route 日志 grep 不到任何 GenSimplifiedKey 的**调用**记录（只有 3 条注册记录：`RegisterOpImplToRegistry ... register OP_IMPL : [GenSimplifiedKey]`，两个 LocalRegistry + python3 编译子进程各一条）。同时同一 staging 源码的 ACLNN 直调回归全 PASS（`ACLNN BenchHarnessScatter summary: SUCCESS (overwrite=1 add=1)`）。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：自定义算子（AscendC）GEIR/GE 图模式验收，期望图路径命中 `build.sh` 预编译 binary 的场景；binary 选择问题排查。
- API 路径：`binary_info_config.json`、`relocatable_kernel_info_config.json`、`GenSimplifiedKey`（`IMPL_OP(...).GenSimplifiedKey`）、`TilingParse`（CompileInfo.coreNum）、`ASCEND_CUSTOM_OPP_PATH`、TEFUSION 的 `GenPyOptionPair`/`NeedSkipPrebuild`/`DoPrebuildOp`/`PreBuildTbeOp`/`ParseBinaryInfoFile`/`ReadOpBinaryInfoConfig`/`SetBinaryConfigPaths` 日志链。
- 验证过的平台与 CANN 版本：Ascend950PR（950），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.2.0 GE 图编译管线对自定义 AscendC 算子的 prebuild/binary 选择行为，与具体算子无关。
- 路线优先级：recommended_fastpath（判别链：先用日志五行确认 prebuild 被跳过，避免在 op_host 侧徒劳调整注册）；依赖"GE 图路径命中预编译 binary"的路线在本版本不可达，不要作为前提（fallback：接受 `te_` 动态编译并用 kernel 侧实际网格对齐兜底，但两条路径 blockDim 语义不等价，见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡）。
- 明确不适用 / 未验证场景：
  - ACLNN 直调路径（不走 TEFUSION prebuild，实测正常）；
  - CANN ≠ 9.2.0 的版本（prebuild 判定在闭源 .so 内，跨版本须重验）；
  - relocatable kernel 构建产物（`relocatable_kernel_info_config.json` 从未在本环境出现，能否激活命中未验证）；
  - 内置 TBE 算子（可能经 `check_support` 返回非 Opaque 的 prebuildPattern，行为未验证）。

## 判别方法

**触发**：GE 图路径 route/RunGraph 失败或下发了 `te_` 前缀内核，而 ACLNN 路径同一算子正常——先确认是不是"预编译未命中、回退 TBE 动态编译"，再决定修哪一层。

**探针**（route debug 日志 grep，本会话走通）：

```bash
# 1) prebuild 闸门：是否被 prebuildPattern 跳过（本卡判别点）
grep -n "need_precompile\|DoPrebuildOp\|prebuildPattern\|NeedSkipPrebuild\|SetPrebuildPattern" "$ROUTE_LOG" | head -20
# 2) kernel 选择结果：te_ 前缀 = TBE 动态编译回退
grep -n "FillTaskDefAfterGenTask\|StreamLaunchKernelV2" "$ROUTE_LOG" | head -10
# 3) 二进制索引是否被读取（读取 ≠ 命中）
grep -n "ParseBinaryInfoFile\|ReadOpBinaryInfoConfig\|GetBinaryInfoConfigPath" "$ROUTE_LOG" | head -20
# 4) 对照：GenSimplifiedKey 只有注册没有调用
grep -n "GenSimplifiedKey" "$ROUTE_LOG" | head -10
```

**原始观测**（关键五行，route 日志原文）：

```
GenPyOptionPair Build options[need_precompile] value is [false].
NeedSkipPrebuild Op[n_route_target] core type fixed: 1
UpdateOpTaskForPreBuildCache Node[n_route_target] no need to precompile, kernelName[te_benchharnessscatter_911e..._pre].
NeedSkipPrebuild Op[n_route_target] prebuildPattern jump prebuild, prebuildPattern is Opaque.
DoPrebuildOp Node [0_1_n_route_target] needs to skip prebuild.
```

随后 `FillTaskDefAfterGenTask:Set kernel_name[te_benchharnessscatter_...]` 证实回退 TBE 动态编译。对照检查：

```bash
# 预编译侧状态：10 个 kernel JSON 全部 blockDim=-1 compileInfo={}（对照正常算子 AddCustom 同样如此，是 AscendC 构建常态）
python3 -c "
import json, glob
for jf in sorted(glob.glob('<build>/ascendc_kernels/binary/ascend950/<op>_apt/*.json')):
    with open(jf) as f: kd = json.load(f)
    print(f'{jf.split(\"/\")[-1]}: blockDim={kd.get(\"blockDim\")} compileInfo={kd.get(\"compileInfo\", \"NOT_PRESENT\")}')"
```

ops-info JSON 里 `"prebuildPattern":{"value":"Opaque"}`，且 `InitOpStrItem` 日志显示内置算子（Abs、AccumulateNV2、Acos 等）与其他自定义算子（AddCustom 等）的 prebuildPattern 同为 Opaque——这是**全量默认值**，不是本算子的配置错误。

**解释**：AscendC 自定义算子 op_build 生成的 ops-info JSON 中 `prebuildPattern` 恒为 `Opaque`；TEFUSION prebuild_manager 的 `NeedSkipPrebuild` 命中 Opaque 即跳过 prebuild，二进制匹配（simplifiedKey 命中检查）从未发生，`GenPyOptionPair` 生成的 `need_precompile` 也为 false。因此"GenSimplifiedKey 已注册 + key 逐字符一致 + binary_info_config 可读"**不是**命中预编译二进制的充分条件——prebuild 在 simplifiedKey 匹配之前就被跳过，注册侧任何调整都到不了匹配环节。`DynamicCompileStaticFlag(true)`、`OP_DYNAMIC_COMPILE_STATIC=1`/`GE_DYNAMIC_COMPILE_STATIC=1` 环境变量都不能改变该跳过行为。预编译二进制实际只在 ACLNN 直调路径被使用。同一 CANN 版本上 binary 复用另有 graph 级 `ge.jit_compile` 闸门（见同批《ge.jit_compile=2 图级闸门挡预编译复用》卡），两层闸门独立。

**TilingParse 构建期探针**（验证 CompileInfo 为何为空，before/after verbatim）：

```cpp
// after（探针版本；构建输出零打印即证明回调未被调用）
ge::graphStatus TilingPrepareForBenchHarnessScatter(gert::TilingParseContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    auto* compileInfo = context->GetCompiledInfo<BenchHarnessScatterCompileInfo>();
    fprintf(stderr, "[TilingParse] TilingPrepareForBenchHarnessScatter CALLED platformInfo=%p compileInfo=%p\n",
            (void*)platformInfo, (void*)compileInfo);
    fflush(stderr);
    if (platformInfo == nullptr || compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ap(platformInfo);
    compileInfo->coreNum = 1;
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    fprintf(stderr, "[TilingParse] coreNum=%lu ubSize=%lu\n",
            (unsigned long)compileInfo->coreNum, (unsigned long)compileInfo->ubSize);
    fflush(stderr);
    return ge::GRAPH_SUCCESS;
}
```

完整重建（cust_optiling_obj + kernel 构建）后构建输出零 `[TilingParse]` 打印、10 个 kernel JSON compileInfo 仍 `{}`——TilingParse 在 kernel 构建期不被调用（route 日志已证实注册成功，含 `compile_info_creator func registered first`）。

## 设计决策

- 判别顺序上先 grep 日志确认 prebuild 跳过，再动代码：本会话先花大量轮次怀疑 compileInfo/注册顺序，之后才建立"日志五行定乾坤"的判别链——日志判定成本远低于重建实验。
- 用对照算子排除假设：拿全局 OPP 里正常工作的 AddCustom 的 kernel JSON 对照（同样 `blockDim=-1 compileInfo={}`），一次性排除"compileInfo 为空是本算子缺陷"。
- 判定"命中与否"只看一行：`FillTaskDefAfterGenTask:Set kernel_name[...]` 的前缀（`te_` vs `<OpType>_`）——比追 fusion 内部状态机便宜且可靠。
- 把"装全局 OPP"设计成二分实验：staging-only 与全局 OPP 两种 vendor 摆放各跑一轮，直接用 `ReadOpBinaryInfoConfig` 行回答"binary 信息是否被读"，避免在 25 万行融合日志里盲找匹配逻辑。
- 清缓存轮同时比较编译产物指纹（sha256/length），一轮实验同时排除"binary 信息未读"与"二进制陈旧"两个假设。
- 当前方案路线档位：判别链 recommended_fastpath；"让图路径命中预编译"路线在本版本未见任何生效样例，不推荐作为依赖前提。

## 系统信息

- 硬件/软件：Ascend950PR_9579 单卡 device 0，CANN 9.2.0；GEIR 测试 `operators/<Op>_package/tests/geir`（`run.sh stage/build/route`）；route debug 日志 `/tmp/geir-route.*.log`。
- 关键文件（两轮会话实际读/查）：
  - `<opp>/vendors/<Op>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json`（key→binPath 映射）；
  - `.../config/ascend950/aic-ascend950-ops-info.json`（`dynamicCompileStatic/dynamicRankSupport/dynamicShapeSupport=true`、`prebuildPattern=Opaque`）；
  - `.../config/ascend950/bench_harness_scatter_apt.json`（binList，含 simplifiedKey/staticKey）；
  - `op_host/bench_harness_scatter_def.cpp`（GenSimplifiedKey/TilingParse 注册）、`op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`（TilingParse 回调）。

## 预防

- 不要以"注册了 GenSimplifiedKey + binary_info_config.json 里有匹配 key"作为"GE 图路径会命中预编译 binary"的依据——9.2.0 上实测不成立；先 grep `NeedSkipPrebuild`/`FillTaskDefAfterGenTask` 三处再决定投入方向。
- 不要把"kernel JSON `compileInfo` 为 `{}`"当作异常去修——对照任一正常自定义算子即可确认这是 AscendC 构建常态。
- 隔离验收（`ASCEND_CUSTOM_OPP_PATH` 指向 staging）时，若需要 fusion 读 vendor binary 信息，先在日志里确认 `ReadOpBinaryInfoConfig ... success` 行是否出现；读取路径语义见同批《TEFUSION二进制匹配走全局OPP非staging》卡。
- 验收 GE 图模式时区分两条路径：ACLNN 直调（预编译二进制 + 运行期 SetBlockDim）与 GE 图（TBE 动态编译回退），两者 blockDim 行为不同，ACLNN PASS 不能外推图路径 PASS。

## 验证信号

来源说明：以下信号来自该算子两轮独立的 GEIR 图模式验收会话（不同 staging 重建、多轮 vendor 摆放/缓存对照实验），均为 950 上板 route 日志实测。

- prebuild 闸门链：`GenPyOptionPair Build options[need_precompile] value is [false].` → `NeedSkipPrebuild Op[n_route_target] prebuildPattern jump prebuild, prebuildPattern is Opaque.` → `DoPrebuildOp Node [0_1_n_route_target] needs to skip prebuild.`（来源平台：950；证据等级：tool_observed）。
- prebuildPattern=Opaque 为全量默认：ops-info JSON `"dynamicCompileStatic":{"flag":"true"}` 与 `"prebuildPattern":{"value":"Opaque"}` 并存；`InitOpStrItem` 日志同样打印在内置 Abs、AccumulateNV2、Acos 等算子上（来源平台：950；证据等级：tool_observed）。
- 预编译 key 精确匹配：回调产出 `BenchHarnessScatter/d=0,p=0/0,2/3,2/0,2/0,2` 与 binary_info_config.json 中 `BenchHarnessScatter_b51b45466550cde4aefe19d6cd25f864.o` 条目逐字符一致；该 binary JSON `blockDim: -1`、simplifiedKey 含 4 组 d/p×dtype 组合（来源平台：950；证据等级：tool_observed）。
- GenSimplifiedKey 只注册不调用：route 日志仅 3 条 `RegisterOpImplToRegistry ... register OP_IMPL : [GenSimplifiedKey]`（两个 LocalRegistry + python3 子进程），全日志 0 条调用记录（来源平台：950；证据等级：tool_observed）。
- 摆放对照：staging-only 轮 fusion 只读 built-in 路径/子目录、vendor 路径仅探测 `relocatable_kernel_info_config.json` 且多条 `does not exist`；装入全局 OPP 后 `ReadOpBinaryInfoConfig Read binary info file[.../opp/vendors/<Op>/.../binary_info_config.json] success`——读取成功仍不命中（来源平台：950；证据等级：tool_observed）。
- 清缓存对照：清 `te_*` 缓存后 TBE 全新动态编译（kernel sha256 变为 3e578b16...、GetKernels length 2764→2812），kernel_name 仍恒 `te_`（来源平台：950；证据等级：tool_observed）。
- 环境变量反证：`OP_DYNAMIC_COMPILE_STATIC=1` + `GE_DYNAMIC_COMPILE_STATIC=1` 下 route 仍失败、`The [prebuildPattern] for operation [BenchHarnessScatter] is [Opaque].` 不变；未设置时另有 `InitParamFromEnv:Env[OP_DYNAMIC_COMPILE_STATIC] is not found.`（来源平台：950；证据等级：tool_observed）。
- TilingParse printf 探针：完整重建后构建输出零 `[TilingParse]` 打印、10 个 kernel JSON compileInfo 仍 `{}`——构建期回调不被调用（来源平台：950；证据等级：tool_observed）。
- 对照：同一 staging 源码 ACLNN 回归 `SUCCESS (overwrite=1 add=1)`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定 CANN 9.2.0 + 950PR 的 TEFUSION prebuild 链；`GenPyOptionPair`/`NeedSkipPrebuild` 判定逻辑在闭源 `libop_compile_adapter.so` 内，CANN 升级后须按本卡判别方法重验（未验证，推断机制相同）。
- `need_precompile=false` 与 `prebuildPattern=Opaque` 的因果方向（谁先谁后）基于 `te_fusion/fusion_manager.py` 的 `no_need_do_prebuild(build_mode, need_precompile, is_support_fusion)` 源码阅读推断（`build_mode=="prebuild" and need_precompile=="false" and not is_support_fusion` 即跳过；本算子无 custom_fusion_passes），未在 C++ 层反汇编确认；不影响"prebuild 被跳过、回退 TBE"的结论。
- 同批另一会话观测到 `SetBinaryConfigPaths` 扫描清单不含本算子 vendor（即便重装到全局 OPP），与本会话"全局 OPP 中 config 读取成功"矛盾——扫描清单的建立机制未定案（可能与安装状态有关），见同批扫描清单卡与《TEFUSION二进制匹配走全局OPP非staging》卡；两处观测不改变"读取 ≠ 匹配、prebuild 在匹配前被跳过"的本卡结论。
- relocatable kernel 产物（`relocatable_kernel_info_config.json`）在本环境从未出现，其能否激活命中未验证——该文件名是 fusion 实际探测的（日志证据），值得作为后续实验变量。
- 双 vendor（staging + 全局 OPP）同场时的 registry/路径优先级未验证。

**未决项**：

- 优先项：是否存在官方途径（特定 ops-info 字段、CANN 配置或版本）让 AscendC 自定义算子在 GE 图路径命中预编译二进制：未找到；可用一个最小自定义算子对照实验——提供 `relocatable_kernel_info_config.json`（relocatable 构建产物）前后各跑一轮 route，看 kernel_name 前缀是否翻转。
- `prebuildPattern=Opaque` 是否可在 op 侧改写（如通过 `ExtendCfgInfo` 声明非 Opaque pattern）未探索。
- GenSimplifiedKey 在哪条路径（GERT/aclnn 直调）被实际消费，未在日志中观测到调用点。
## 坏实践

- 在 op_host 侧反复调整 `GenSimplifiedKey`/`TilingParse` 注册顺序、key 格式试图让 GE 图路径命中预编译二进制——prebuild 在 simplifiedKey 匹配之前就被 prebuildPattern 跳过，注册侧任何调整都到不了匹配环节。
- 把"binary_info_config.json 装进全局 OPP 且被成功读取"当作"预编译 binary 已命中"的证据——读取 ≠ 匹配，必须回看 `FillTaskDefAfterGenTask:Set kernel_name` 前缀。
- 把 `compileInfo: {}` 当作根因去修（给 TilingParse 加逻辑、怀疑注册被覆盖）——该字段为空是 AscendC 自定义算子常态。
- 试图用 `OP_DYNAMIC_COMPILE_STATIC=1` 一类环境变量打开 prebuild——实测无效。
- 反复清 TBE 缓存试图"逼"fusion 走预编译——缓存与 binary 选择无关（清后仍动态编译）。

## 误诊

- "GenSimplifiedKey 生成的 key 与 binary_info_config 的 key 格式不一致所以不命中"：实测两边 key 逐字符一致，排除 key 格式问题。
- "staging 没装所以 fusion 读不到 binary 信息"：装入全局 OPP 后确实读到了（`success`），依旧不命中——读取不是瓶颈。
- "kernel JSON `compileInfo: {}` 导致 need_precompile=false"：对照 AddCustom（同样空）后排除；printf 探针进一步证明构建期回调根本不被调用（agent_reported 假设，已由对照实验否定）。
- "IMPL_OP 的 TilingParse 注册被 OP_ADD 的 placeholder 覆盖"：route 日志显示多个 registry 中 `[TilingParse]` 注册成功且有 `compile_info_creator func registered first`——注册无问题，问题在调用时机（agent_reported 假设，日志否定）。
- "TEFUSION 读全局路径是 prebuild 被跳过的原因"：删除全局安装后 prebuild 仍被跳过（prebuildPattern=Opaque 来自 ops-info 判定，与二进制路径无关）——全局路径问题是独立事实（见同批 TEFUSION 路径卡）。
