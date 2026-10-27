---
type: "Runbook"
title: TBE编译期blockDim不受coreNum与缓存影响
description: TilingParse 设 CompileInfo.coreNum=1 想让 TBE 动态编译产出 blockDim=1 的假设被三轮独立验收证伪：TBE 构建选项 core_num 为 None（不读 CompileInfo）、清空 TBE 缓存后 fresh 编译（sha256 变化证明非缓存复用）仍 blockDim=6、运行时 SetBlockDim 也压不下 TBE 编译产物的下发核数；coreNum 只进 TilingFunc 运行时切分，清缓存与调 coreNum 都不是 GE 图路径 blockDim 不匹配的修复手段。
tags: [task.debugging, topic.tiling, topic.compilation]
status: stable
sources: []
created_at: '2026-09-14T12:16:44Z'
updated_at: '2026-09-14T12:16:44Z'
platforms: ["950"]
---

# TBE编译期blockDim不受coreNum与缓存影响

## 现象

在 Ascend950PR（CANN 9.2.0）上为 scatter 类自定义算子（BenchHarnessScatter）排查 GE 图路径 kernel 执行失败（EZ9999，errcode=95 MTE DDR 越界，见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡）时，出现两个非常直觉的归因/修复候选：

1. **缓存陈旧假设**：TBE 编译缓存目录 `~/.atc_data/kernel_cache/Ascend950PR_9579/` 下存在数天前的 `te_benchharnessscatter_<hash>.json/.lock/.o`（json 内 `blockDim: 6`），怀疑陈旧缓存把错误 blockDim 带进本次运行——"清缓存重跑"是这类问题的常规第一反应。
2. **coreNum 驱动假设**：op_host 的 TilingParse 回调（`TilingPrepareForBenchHarnessScatter`）源码里已设 `compileInfo->coreNum = 1`，且注释明确宣称 "Set coreNum=1 so the TBE dynamic compiler produces blockDim=1"——若该机制有效，TBE 动态编译本应产出 blockDim=1，与实际 6 矛盾，怀疑是被缓存掩盖。

三轮独立验收会话（不同 staging 重建）以清缓存 + fresh 编译 + 产物指纹对照的实验一致证伪两条路线：清空 `te_benchharnessscatter_*` 的 `.json`/`.o`/`.lock`（清后计数 0）后 TBE 重新动态编译，fresh 产物（sha256 与缓存版不同、kernel_meta 目录名 hash 变化，证明非缓存复用）仍 `"blockDim":6,"coreType":"MIX"`，route 依旧 `value=FAIL`、`GetArgsInfo ... blockDim=6`。TBE 构建选项日志直接给出解释——传给编译器的 `core_num` 是 `None`（不是 1），编译器侧用平台信息初始化核数：

```text
[INFO] TBE(...): ... [parallel_compilation.py:1284][_update_options] new options{[soc_version:Ascend950PR_9579], [core_type:AiCore]}, [core_num:None], ...
[DEBUG] FE(...): ... [platform_infos_def.cpp:194] SetCoreNumByCoreType:Set PlatFormInfos::core_num_ to ai_core_cnt: [28], core_type: AiCore.
```

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：GEIR/GE 图模式算子开发/验收（GE Session 图执行）走 TBE 动态编译的场合；以及任何"想通过 TilingParse / CompileInfo 控制编译期 blockDim"或"怀疑 TBE 缓存陈旧"的尝试。
- API 路径：`gert::TilingParseContext::GetCompiledInfo`（`CompileInfo.coreNum`）、`gert::TilingContext::GetCompileInfo`、TilingFunc `SetBlockDim`、TBE 构建选项（`parallel_compilation.py _update_options` 的 `core_num`）、kernel_meta json 的 `blockDim` 字段、TBE 缓存 `~/atc_data/kernel_cache/<SoC>/te_<op>_*.json/.o/.lock`。
- 验证过的平台与 CANN 版本：Ascend950PR_9579 + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本下所有自定义 AscendC 算子走 TBE 动态编译时的 blockDim 行为，与算子种类无关。
- 路线优先级：banned——"用 CompileInfo.coreNum 控制 TBE 编译期 blockDim"与"清 TBE kernel cache 修 blockDim 不匹配"两条路线均被实测证伪，不要再走；kernel 需要按"实际下发核数可能大于 numCores"做防护（见同批 GetBlockNum 对齐卡与主诊断卡），而不是试图把编译期 blockDim 压小。
- 明确不适用 / 未验证场景：
  - 预编译二进制路径（`"blockDim":-1`，运行时 `SetBlockDim` 生效）：coreNum 在该路径经 TilingFunc 正常生效，不受本卡结论影响（ACLNN 回归 PASS 为旁证）；
  - 其他 CANN 版本 / 其他 soc 的 TBE 构建选项行为未验证（未验证，推断机制一致）；
  - 缓存确实陈旧导致其他症状的场景（如增量编译产物与源码不一致类问题）不在本卡否定范围。

## 根因

`CompileInfo.coreNum` 只在运行时被 TilingFunc 消费（做多核切分与 `SetBlockDim`），TBE 动态编译器的构建选项根本不读它（`core_num:None`）；TBE 编译期 blockDim 按平台/核型默认固化（本平台 MIX 核型观测值 6，与 `ai_core_cnt:28`/`vector_core_cnt:56` 均无直接换算关系），写进 kernel_meta json 后由 task builder 原样使用。缓存只存编译产物，不决定编译参数——blockDim 在编译期烘焙，清缓存重编译结果相同。

两条独立通道的对应证据：

```text
编译期（coreNum=1 无效）：
  [parallel_compilation.py] new options{...}, [core_num:None]        ← CompileInfo.coreNum 未进构建选项
  kernel_meta json: "blockDim":6,"coreType":"MIX"                    ← 仍固化平台默认 6

运行期（SetBlockDim 压不下 TBE 编译核数）：
  ParseTvmBlockDim:Op[n_route_target], ParseTvmBlockDim: 6.           ← task builder 读编译 json
  Init:Tbe blockdim: 6.
  StreamLaunchKernelV2: kernel_name=te_benchharnessscatter_..., coreDim=6 ← 实际 6 核下发
```

coreNum 的真实消费点——TilingFunc 开头（运行时通道，`op_host/arch35/bench_harness_scatter_tiling_arch35.cpp` verbatim）：

```cpp
// ─── TilingParse callback ───
ge::graphStatus TilingPrepareForBenchHarnessScatter(gert::TilingParseContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    auto* compileInfo = context->GetCompiledInfo<BenchHarnessScatterCompileInfo>();
    if (platformInfo == nullptr || compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ap(platformInfo);
    // Set coreNum=1 so the TBE dynamic compiler produces blockDim=1, matching
    // the kernel's CopyDataToY multi-core split (numCores=1 for small cases).
    // The precompiled binary (ACLNN path) already has blockDim=-1 (dynamic),
    // so SetBlockDim at runtime is unaffected.  This aligns the GE graph path's
    // compiled blockDim with the kernel's numCores expectation.
    compileInfo->coreNum = 1;
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

// ─── TilingFunc entry（运行时）───
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. Get CompileInfo (platform info) — fallback to hardcoded if not available
    int64_t coreNum = 8;   // ascend950 default AIV cores
    int64_t ubSize   = 262144;  // 256KiB
    auto* ci = context->GetCompileInfo<BenchHarnessScatterCompileInfo>();
    if (ci != nullptr) {
        coreNum = static_cast<int64_t>(ci->coreNum);
        ubSize   = static_cast<int64_t>(ci->ubSize);
    }
    if (coreNum < 1) coreNum = 1;
    // ...（读 shape/dtype/attr 后 FlatUBSplit + MultiCoreSplit + SetBlockDim(mc.numCores)）
```

多核切分逻辑（说明 coreNum 即使生效也只影响运行时切分）：

```cpp
static void MultiCoreSplit(int64_t totalTiles, int64_t coreNum, MultiCoreResult& out)
{
    out.totalTiles = totalTiles;
    if (totalTiles == 0) {
        out.numCores  = 1;
        out.tilesMain = 0;
        out.coresTail = 0;
        return;
    }
    out.numCores  = (totalTiles < coreNum) ? totalTiles : coreNum;
    if (out.numCores < 1) out.numCores = 1;
    out.tilesMain = totalTiles / out.numCores;
    out.coresTail = totalTiles % out.numCores;
}
```

注意：`compileInfo->coreNum = 1` 上方的注释宣称"使 TBE 动态编译器产出 blockDim=1"——该宣称与实测相反（fresh TBE 编译仍 blockDim=6），属坏实践（见下）。

## 尝试实现（证伪实验，verbatim 命令）

```bash
# 1) 清空 TBE 缓存（必须连 .lock 一起清，否则旧缓存条目可能继续生效）
rm -f /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_*.json
rm -f /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_*.o
rm -f /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_*.lock
# 清后校验：
ls /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_* 2>/dev/null | wc -l   # → 0

# 同时清 GEIR kernel_meta（另一会话补充的对照）
rm -rf operators/<Op>_package/tests/geir/kernel_meta

# 2) 重跑 route（含环境变量对照轮，证明 env 也不影响）
cd "$GEIR_DIR" && GE_DYNAMIC_COMPILE_STATIC=1 OP_DYNAMIC_COMPILE_STATIC=1 ./run.sh route > "$RAW_ROUTE" 2>&1

# 3) 判读 fresh 产物
grep -n "GetJsonValueFromJsonFile" "$ROUTE_LOG" | grep kernel_meta | head -2   # jsonValue 内 "blockDim":6
grep -n "ParseTvmBlockDim\|Init:Tbe blockdim\|LaunchKernelByArgsWithType" "$ROUTE_LOG" | head -4
grep -n "core_num\|ai_core_cnt\|SetCoreNumByCoreType" "$ROUTE_LOG" | head -4
```

清缓存前缓存现状（证明缓存确实存在且含 blockDim:6）：

```
-rw-r-----. 1 <user> <group>  612 Sep  4 23:55 /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_37a2c88c....json
-rw-r-----. 1 <user> <group>    0 Sep  4 23:55 /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_37a2c88c....lock
-rw-r-----. 1 <user> <group> 280424 Sep  4 23:55 /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_37a2c88c....o
...
blockDim: 6
```

fresh 编译产物（清缓存后）：`kernel_meta_3400381921362194460/kernel_meta/te_benchharnessscatter_911e....json` 仍 `"blockDim":6,"coreType":"MIX","intercoreSync":1`（`"sha256":"00359f270b432ba70ee3147ad7ff4d246541a975694610ebed4c58ef307b2278"`，与缓存版 `a09bb5de...` 不同，证明非缓存复用），`GetArgsInfo ... blockDim=6`，route `value=FAIL` 与清缓存前逐项一致。

## 设计决策

- 实验设计要点一：必须连 `.lock` 一起清——残留 lock 文件会让"清了缓存还是旧结果"的判定不可靠。
- 实验设计要点二：用 fresh 产物 sha256 ≠ 缓存版 sha256 证明"确实重新编译了"，排除缓存复用对结论的干扰；评估任何修复前先确认"新代码已编入"（比较编译产物指纹），再看失败是否复现——一次实验同时排除"旧二进制"假设。
- 对照组：预编译二进制 json `"blockDim":-1` + ACLNN 回归 overwrite/add 均 PASSED，隔离出"编译期 vs 运行时"两条通道的差异——coreNum 经 SetBlockDim 在运行时通道正常生效的旁证。
- 为什么把该结论独立成负结果卡：源码注释把 coreNum=1 描述为"使 TBE 动态编译器产出 blockDim=1"，这是对本卡结论的直接反例声明——下一个 agent 读到这段注释会信以为真并在此路线上浪费轮次；用构建选项日志（`core_num:None`）+ kernel_meta（`blockDim:6`）+ sha256 指纹三重证据把它纠正为负结果。
- 排障顺序上先做低成本假设的否定实验（清缓存是分钟级操作），再进入机制层排查（二进制选择路径），避免直接改 kernel 或重装环境这类高成本动作。
- 当前路线档位：banned（两条路线均作为修复手段被证伪）。

## 系统信息

- 硬件/软件：Ascend950PR_9579 单卡（npu-smi 25.7.rc1.6，`ai_core_cnt:28`、`vector_core_cnt:56`），CANN 9.2.0；TBE 缓存目录 `~/atc_data/kernel_cache/Ascend950PR_9579/`。
- 相关文件（三轮会话实际读/查）：
  - `op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`——TilingParse（coreNum=1 写入点）与 TilingFunc（coreNum 消费点）；
  - `op_kernel/arch35/bench_harness_scatter_tiling_data.h`——`MultiCoreResult`（numCores 字段）；
  - `tests/geir/kernel_meta/kernel_meta_*/kernel_meta/te_benchharnessscatter_*.json`——fresh TBE 编译产物；
  - `<package>/build/ascendc_kernels/binary/ascend950/<op_dir>/BenchHarnessScatter_*.json`——预编译二进制元数据（blockDim=-1 对照）；
  - route 原始日志（`./run.sh route > impl-route.raw.log 2>&1` 完整重定向，DEBUG 级）。

## 预防

- 不要在 TilingParse 的注释、文档或设计里宣称 `CompileInfo.coreNum` 会改变 TBE 编译期 blockDim；只描述其运行时语义（TilingFunc 核数来源）。写代码注释宣称这种因果关系前先实测。
- 验收/联调 GE 图路径时，把"kernel_meta json 的 blockDim 与 TilingData numCores 一致性"列入检查清单（grep `jsonValue: \[.*blockDim` 或直接读 fresh 产物 json）。
- 清 TBE 缓存做对照实验时，`.json`/`.o`/`.lock` 三类都要清，并用 sha256 差异确认 fresh 编译；清 kernel cache 是有代价动作（后续编译变慢），把它当"万能重试"前先用一次 fresh 编译对照证明缓存确实影响了结果。
- 遇 GE 图路径 kernel blockDim 异常，先 grep fresh 产出的 kernel_meta json 的 `blockDim` 字段再怀疑缓存——缓存命中与 fresh 编译产出完全相同，清缓存不改变结果。

## 验证信号

来源说明：以下信号来自三轮独立 GEIR 图模式验收会话（各自 fresh staging 重建后）针对"TBE 编译期 blockDim 是否受 CompileInfo.coreNum / 缓存影响"的对照实验，均为 950 上板实测。

- TBE 构建选项未消费 CompileInfo：`new options{...}, [core_num:None]`、`SetCoreNumByCoreType ... ai_core_cnt: [28]`（来源平台：950；证据等级：tool_observed）。
- 清缓存前：`kernel_cache/Ascend950PR_9579/te_benchharnessscatter_*.json` 存在（历史产物），json 内 `blockDim: 6`；缓存版 sha256 `a09bb5de...`（来源平台：950；证据等级：tool_observed）。
- 清空 `.json`/`.o`/`.lock` 后目录计数 0（来源平台：950；证据等级：tool_observed）。
- 清缓存后 fresh TBE 编译（含 `GE_DYNAMIC_COMPILE_STATIC=1` + `OP_DYNAMIC_COMPILE_STATIC=1` 对照轮）：kernel_meta json 仍 `"blockDim":6`，sha256 `00359f270b432ba70ee3147ad7ff4d246541a975694610ebed4c58ef307b2278`（与缓存版不同，证明非缓存复用）；`LaunchKernelByArgsWithType ... blockDim=6`；新 kernel_meta 目录名 hash 变化（来源平台：950；证据等级：tool_observed）。
- fresh 重跑 route：`route exit=1`；`GRAPH-RUN ... rungraph_rc=4294967295 ... value=FAIL`；`GetArgsInfo ... blockDim=6`——与清缓存前逐项一致（来源平台：950；证据等级：tool_observed）。
- task builder 与下发核数：`ParseTvmBlockDim: 6.`、`Init:Tbe blockdim: 6.`、`StreamLaunchKernelV2 ... coreDim=6`（来源平台：950；证据等级：tool_observed）。
- coreNum=1 与 blockDim=6 共存：源码 `compileInfo->coreNum = 1` 在案（TilingPrepareForBenchHarnessScatter），而 fresh 编译产物 blockDim=6——同时存在即证明无驱动关系（来源平台：950；证据等级：tool_observed）。
- 对照组：预编译二进制 json `"blockDim":-1`、`workspace.size:[-1]`，ACLNN 回归 overwrite/add 均 PASSED——coreNum 经 SetBlockDim 在运行时通道正常生效的旁证（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 Ascend950PR_9579 + CANN 9.2.0 上对 TBE 动态编译路径实测；其他 CANN 版本的 TBE 构建选项行为未验证（未验证，推断机制一致）。
- 结论限定"编译期 blockDim"：coreNum 在运行时通道（TilingFunc/SetBlockDim）的作用未被证伪，且被 ACLNN PASS 旁证正常。
- "6"是平台观测值（MIX 核型编译默认）；tiling 源码注释写 ascend950 默认 AIV=8、平台日志 `ai_core_cnt:28`/`vector_core_cnt:56`，均与 6 不一致——TBE 固化 blockDim 的确切决定逻辑在闭源编译器内未追到，其他 SoC 上数值可能不同。
- kernel 侧对 blockDim > numCores 的守卫方案（按 GetBlockIdx/实际 blockDim 裁剪）的充分性另见同批 GetBlockNum 对齐卡（该卡证明对齐后 core 0 仍崩）；本卡只否定 coreNum 与缓存两条路线。

**未决项**：

- 优先项：在 CANN 9.2.0 上确认 TBE 动态编译 blockDim 的精确决定机制（MIX 核型默认值的编译器来源），以及是否存在任何 op 侧可影响的编译期 blockDim 覆盖途径（本卡仅排除了 CompileInfo.coreNum 与缓存）。
- kernel 加 `cid >= numCores` 防护 / `GetBlockNum()` 对齐后 GE 图路径能否恢复（同批 GetBlockNum 卡证明对齐后 core 0 仍崩，第二层机制未定位）。
- 使 GE 图路径命中预编译二进制的正解未找到（见同批 prebuildPattern/jit_compile 两卡）。
## 坏实践

- 在 TilingParse 里设置 coreNum 并在注释中宣称它控制 TBE 编译期 blockDim（本例源码注释即此坏实践，已被证伪）——误导后续维护者。
- 把 `rm -rf kernel_cache` 当 GEIR blockDim 问题的修复动作——实测无效，且拖慢后续编译。
- 清 TBE 缓存时遗漏 `.lock` 文件，导致"清了缓存还是旧结果"的误判。
- 用缓存复用路径的 json 直接下"编译行为"结论——必须先证明是 fresh 编译（sha256 对照）。

## 误诊

- 早前会话（仓库内 LOG/验收报告转述）曾把"TilingData<8> 统一 + coreNum=1 对齐 blockDim"当作 GE 图路径的修复思路（agent_reported 转述）；三轮独立验收以清缓存 fresh 编译实验证伪其编译期部分——coreNum=1 一直在源码中，blockDim 仍为 6。
- 初期假设"TBE 缓存里残留旧 blockDim"或"vendor 未装好导致行为异常"；fresh staging 重建 + 重装 + 清缓存后失败逐字复现，环境归因排除。
- 曾尝试以 `OP_DYNAMIC_COMPILE_STATIC=1`、`GE_DYNAMIC_COMPILE_STATIC=1` 间接影响编译行为——两轮均不改变 `need_precompile=false` 与 blockDim=6。
- 曾沿"blockDim 会不会等于 ai_core_cnt=28"推演（日志 `ai_core_cnt:28` vs 实际 blockDim=6），确认两者无直接换算关系。
