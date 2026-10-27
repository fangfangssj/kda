---
type: "Runbook"
title: GEIR 图路径不消费 TilingParse/GenSimplifiedKey
description: op_host 注册的 TilingParse 与 GenSimplifiedKey 回调在 GEIR 图模式 TBE 动态编译路径下注册成功但从不被调用（compile_info 仍空、binary matching 不发生）；回调健壮化、IMPL_OP 前置注册、GenSimplifiedKey 返回键长修复、GE_JIT_COMPILE=0、清缓存五类 host 侧修复实测全部无效——注册域与消费域不重叠，勿在此层反复投入。
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# GEIR 图路径不消费 TilingParse/GenSimplifiedKey

## 现象

自定义 scatter 类算子（AscendC kernel + GERT host 侧 TilingFunc/TilingParse + 预编译包）的 GEIR 图模式验收：图执行稳定复现 EZ9999/aivec error 95（根因判别见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡：GE 图路径未命中预编译二进制，回退 TBE 动态编译）。op_host 侧 def.cpp 已包含两个针对性修复回调（源码确认在、随 staging 重建生效、注册日志可见）：

1. never-fail 的 TilingParse（目的：TBE 编译期总能产出 compile_info，避免 `Compile info json after compiling is []`），CompileInfo 回退 `coreNum=1 / ubSize=256KiB`；
2. GenSimplifiedKey 回调（目的：运行时产出 binary matching 用的 simplified key，格式与 op_build 预生成键一致并返回键长度）。

注册日志显示两个回调都注册成功：

```
RegisterOpImplToRegistry:LocalRegistry[...] Op type[BenchHarnessScatter] register OP_IMPL : [Tiling][TilingParse]
RegisterOpImplToRegistry:LocalRegistry[...] Op type[BenchHarnessScatter] register OP_IMPL : [GenSimplifiedKey]
```

但目标都没有发生——

```
GetCompileInfo There is no compileInfo in jsonFilePath .../kernel_meta/te_benchharnessscatter_911e....json.
SetNodeCompileInfoAttr Op[name=n_route_target]: compile_info_key is empty, returning.
SetOpCompileInfo:Compile info json after compiling is [].
CheckArgs:hostInputInfoNum=0, isNoNeedH2DCopy=1, argsSize=40, hasTiling=0
```

compile_info 仍为空、kernel json 无 compileInfo 字段；binary matching 仍不发生、kernel 仍为 `te_` 前缀、blockDim 仍 6、EZ9999 逐字不变。为关闭该失败所做的五类 host 侧修复（详见「尝试与观测」）全部无效。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：aclnn 算子仓自定义算子（AscendC kernel + GERT host 侧 TilingFunc/TilingParse/GenSimplifiedKey）在 GEIR 图路径上遇到"kernel 收不到 tiling / binary matching 不命中"类失败，评估修复投入方向。
- API 路径：`IMPL_OP(<Op>).TilingParse<T>(...)` / `IMPL_OP(<Op>).GenSimplifiedKey(...)`（`register/op_impl_registry.h` 的 `OpImplRegisterV2`）；`gert::TilingParseContext`；tbe_op_store_adapter 的 `SetOpCompileInfo`；GE 选项 `ge.jit_compile`；TBE kernel cache（`~/atc_data/kernel_cache/<SOC>/te_<op>_*.json`）。
- 验证过的平台与 CANN 版本：Ascend950PR（NpuArch 3510）+ CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`——结论关于该平台/CANN 版本的 GE 图执行框架行为，与具体算子无关。
- 路线优先级：`banned`——对"GEIR TBE 动态编译路径不消费回调"这一失败，以下五类 host 侧修复是已验证无效的路线，不要再走；它们不是替代修复，只是排除项。
- 明确不适用 / 未验证场景：这些修复对其他问题仍有价值（注册顺序防占位符覆盖、cache 陈旧的真旧二进制问题）；本卡的"无效"仅指对"图路径不消费回调"这一根因。ACLNN 直调路径不受影响。binary 配置加载问题若被解决，GenSimplifiedKey 是否随之生效需重测（届时本卡结论更新）。相邻机制的判别见 [te_fusion 未调 TilingPrepare 致 GEIR FPE](te_fusion_tilingprepare_geir_fpe.md)——彼卡是编译子路径不调 TilingPrepare 致 compileInfo=0 除零 FPE，本卡是图路径整体不消费回调且修复路线证伪。

## 尝试与观测（负结果）

### 尝试 1：TilingParse 健壮化 + IMPL_OP 链前置注册（真实代码改动，无效）

修复前——tiling 文件里的原回调（platformInfo/compileInfo 为 null 时返回 `GRAPH_FAILED`，注释里带着"TilingParse 设 coreNum=1 可让 TBE 产出 blockDim=1"的假设）：

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
```

修复前——def.cpp 的注册链（仅 TilingParse，`Tiling` 依赖 OP_ADD 从 OpDef 取）：

```cpp
IMPL_OP(BenchHarnessScatter)
    .TilingParse<optiling::BenchHarnessScatterCompileInfo>(optiling::TilingPrepareForBenchHarnessScatter);
OP_ADD(BenchHarnessScatter);
```

修复后——def.cpp 新增永不失败的 Safe 回调，并在 `IMPL_OP` 链同时显式注册 `Tiling` 与 `TilingParse`、置于 `OP_ADD` 之前：

```cpp
static ge::graphStatus TilingParseSafeForBenchHarnessScatter(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<optiling::BenchHarnessScatterCompileInfo>();
    if (compileInfo == nullptr) {
        // Cannot set CompileInfo, but return SUCCESS to avoid empty compile_info.
        return ge::GRAPH_SUCCESS;
    }
    compileInfo->coreNum = 1;
    compileInfo->ubSize = 262144;  // 256 KiB default for Ascend950

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC ap(platformInfo);
        ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BenchHarnessScatter)
    .Tiling(optiling::TilingFunc)
    .TilingParse<optiling::BenchHarnessScatterCompileInfo>(TilingParseSafeForBenchHarnessScatter);
OP_ADD(BenchHarnessScatter);
```

该尝试依赖的注册机制（`register/op_def_registry.h`，OP_TILING_LIB 模式下 OP_ADD 自动注册占位回调，merge 逻辑先注册者胜出）：

```cpp
#elif defined(OP_TILING_LIB)

#define OP_ADD(opType, ...)                                                                         \
  struct OpAddCompilerInfoPlaceholder##opType {};                                                   \
  static ge::graphStatus TilingPrepare##opType(gert::TilingParseContext *context) {                 \
    return ge::GRAPH_SUCCESS;                                                                       \
  }                                                                                                 \
  static int g_##opType##_added = [](const char *name) {                                            \
    opType op(#opType);                                                                             \
    gert::OpImplRegisterV2 impl(#opType);                                                           \
    impl.Tiling(op.AICore().GetTiling());                                                           \
    impl.TilingParse<OpAddCompilerInfoPlaceholder##opType>(TilingPrepare##opType);                  \
    ...
```

客观结果：构建成功；注册确实生效（路由日志 `MergeTypesToImpl:op type BenchHarnessScatter tiling_parse func registered first.`，占位符被自定义回调压住）；但 `SetOpCompileInfo:Compile info json after compiling is [].`、`hasTiling=0`、`blockDim=6` 与 EZ9999 逐字不变，TTK GEIR 静态用例仍 0/8。

### 尝试 2：GenSimplifiedKey 返回值修复（真实代码改动，必要的合同修正，但救不了 binary matching）

排查中读 def.cpp 发现回调在把键串 `snprintf` 进 `key_buf` 后**返回 0**，而 CANN 头文件（`op_impl_kernel_registry.h`）的签名是 `using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext *, ge::char_t *);`——按回调合同应返回写入的键串长度，返回 0 意味着"空键"。据此修复：

**before**（回调尾部，verbatim）：

```cpp
    std::snprintf(key_buf, 256,
        "BenchHarnessScatter/d=%lld,p=0/%d,%d/%d,%d/%d,%d/%d,%d",
        static_cast<long long>(dim),
        dType0, dFmt0,
        dType1, dFmt1,
        dType2, dFmt2,
        dType0, dFmt0);
    return 0;
```

**after**（返回键长，verbatim）：

```cpp
    int keyLen = std::snprintf(key_buf, 256,
        "BenchHarnessScatter/d=%lld,p=0/%d,%d/%d,%d/%d,%d/%d,%d",
        static_cast<long long>(dim),
        dType0, dFmt0,
        dType1, dFmt1,
        dType2, dFmt2,
        dType0, dFmt0);
    // GenSimplifiedKey must return the length of the key string written to
    // key_buf.  Returning 0 makes the framework treat the key as empty and
    // skip binary matching, falling back to TBE dynamic compilation which
    // bakes blockDim=<platform AIV cores> (6) into the binary — ignoring
    // SetBlockDim(1) from TilingFunc and causing multi-core out-of-bounds.
    return static_cast<UINT32>(keyLen);
```

随后按"排除一切干扰"流程验证：重新构建、把新包装到全局 OPP（installed vendor 与 staging 的 SO 都会注册该回调，merge 规则 "registered first" 先到先得）、清 `~/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter*` 缓存后重跑 route——三行关键日志（kernel_name、ParseTvmBlockDim、coreDim）与修复前**逐字相同**，TBE 回退与 blockDim=6 纹丝不动；route 日志中 GenSimplifiedKey 仍只有注册行，无任何调用行。结论：返回值缺陷真实存在且值得修（回调一旦被调用，`return 0` 仍是空键 bug），但它不是 binary matching 不发生的原因（回调根本没被调用）。

### 尝试 3：GE_JIT_COMPILE=0 环境变量（无效）

```bash
GE_JIT_COMPILE=0 ./build/geir_route > /tmp/route_jit0.log 2>&1; echo "EXIT=$?"
```

客观结果：日志仍为 `GetJitCompileDefaultValue:Current NpuArch = 3510, jit_compile = 2`、`The value of param[ge.jit_compile] is [2]`，仍 EZ9999——GE 选项不读该环境变量。

### 尝试 4：清空 TBE kernel cache 强制重编译（无效）

```bash
rm -f /home/<user>/atc_data/kernel_cache/Ascend950PR_9579/te_benchharnessscatter_*
```

客观结果：重编译后 kernel sha256 变化（证明确实是新二进制），但 `blockDim` 仍 6、仍 EZ9999——不是 cache 陈旧问题（独立负结论详见同批清缓存卡）。

### 尝试 5（前置假设，被证伪）："TilingParse 设 coreNum=1 会让 TBE 产出 blockDim=1"

原回调注释声称此机制。客观结果：TBE 动态编译产物的 `blockDim` 恒为平台 AIV 核数 6（`ParseTvmBlockDim:Op[n_route_target], ParseTvmBlockDim: 6.`），不读 TilingParse 的 coreNum。

### 边界（负结果的适用边界）

- ACLNN 直调路径在改动后回归仍 SUCCESS（overwrite=PASS、add=PASS）——改动无回归也无收益。
- GEIR infer/reject 检查（InferDataType、InferShape、reject 全 PASS）不受影响。
- TTK GEIR 静态用例前后均全失败、dynamic 用例结果不变——用例级结果无任何变化。

## 根因

单句归因：图模式 TBE 动态编译路径下，compile_info_key 为空使 TBE adapter 不产出 compile_info（TilingParse 回调注册了也不被该路径消费），binary 配置又未被加载/匹配（GenSimplifiedKey 无从被调用）——两个回调的**注册域与消费域不重叠**，host 侧回调写得再健壮、注册得再靠前都改变不了这条链路。

证据与边界：

1. 注册域实证：route 日志 `[TilingParse]`、`[GenSimplifiedKey]` 均注册成功（RegisterOpImplToRegistry），且 `compile_info_creator/deleter`、`gen_simplifiedkey` 均按 "registered first" 合并生效；staging `libcust_opmaster_rt2.0.so` 的 nm 输出含 `OpImplRegisterV2::TilingParse<optiling::BenchHarnessScatterCompileInfo>` 与 `CreateCompileInfo<optiling::BenchHarnessScatterCompileInfo, 0>` 模板符号。
2. 消费域实证：强制重新编译后 `SetOpCompileInfo:Compile info json after compiling is []`、kernel json 无 compileInfo 字段——TilingParse 的产物没有出现；GenSimplifiedKey 全日志零调用行。
3. CANN 头文件核对：`OpImplRegisterV2::TilingParse<T>` 只注册回调 + `CreateCompileInfo<T>`（`new T()` 默认构造）/ `DeleteCompileInfo<T>`；头文件中未见 compile_info 的 JSON 序列化接口——序列化由框架侧完成，回调"填结构体"并不保证"产出 json"。
4. 对照组：AddCustom / EuclideanNorm 的 ops-info.json 同样没有 compileInfo 字段、simplifiedKeyMode 同为 0，但 AddCustom 在 GEIR 图模式可正常运行——空 compile_info 对不依赖 compile_info 的简单 kernel 不致命；致命的是 blockDim 烘焙（见同批判别卡）。
5. 未闭合环节：compile_info_key 的确切生成机制（是否由 ops-info.json 的 compileInfo 字段驱动、由谁写入）未追到机制层——"TBE adapter 因 key 为空跳过 TilingParse"为机制推断而非直接日志观测。

## 设计决策

- 为什么先做 host 侧注册修复：任务约束"只修改 GEIR 实现、不修改 Kernel/Tiling 计算语义"，host 注册层是可动范围内最接近根因的层；注册机制（OP_ADD 占位符 + registered-first）有头文件宏证据支撑，是合理的排除项。
- 为什么保留 GenSimplifiedKey 返回值修复与 IMPL_OP 前置注册：即使不解决图路径问题，它们分别是回调合同正确性与防占位符覆盖的正确姿势，对其他路径/问题有效。
- 为什么停在五类尝试：每类尝试都有客观信号判定无效（注册日志/日志逐字对比/新 sha256 仍失败），继续在同一层迭代没有新信息；剩余可行方向（GE 会话 options 设置 `ge.jit_compile=0`）需要改冻结测试程序，超出验收可动范围。
- 当前方案路线档位：banned（对这一根因，host 侧回调修复路线已验证无效）；正确去向是框架层修复或在会话/图编译选项层关闭动态编译（未验证）。

## 系统信息

- 硬件/软件：Ascend950PR_9579 单卡（npu-smi 25.7.rc1.6），CANN 9.2.0，NpuArch=3510。
- 相关文件：`op_host/bench_harness_scatter_def.cpp`（回调注册与实现，本组会话唯一代码改动点）、`op_host/arch35/bench_harness_scatter_tiling_arch35.cpp/.h`（TilingFunc 与原 TilingParse 回调，`struct BenchHarnessScatterCompileInfo { uint64_t coreNum; uint64_t ubSize; };` 纯 POD 无序列化方法）、CANN 头 `register/op_def_registry.h` / `register/op_impl_registry.h` / `exe_graph/runtime/tiling_parse_context.h`、`op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json`（无 compileInfo 字段）、`op_impl/ai_core/tbe/<Op>_impl/dynamic/<op>_apt.py`（APT 动态编译入口，grep 无 TilingParse/compile_info 相关代码）、staging `libcust_opmaster_rt2.0.so`（nm 验证）、TBE cache `~/atc_data/kernel_cache/Ascend950PR_9579/`。
- 重构建命令：`export ASCEND_HOME_PATH=<cann-9.2.0>; bash operators/build.sh BenchHarnessScatter_package`（fresh staging 由 `tests/geir/run.sh stage` 生成）。

## 预防

- 给图模式自定义算子写 TilingParse / GenSimplifiedKey 前，先用最小 route 日志确认该路径会消费这些回调（`SetOpCompileInfo` 非 `[]`、binary matching 有命中记录）；注册成功（RegisterOpImplToRegistry）≠ 被调用。
- op_host 注册链保持"自定义 Tiling/TilingParse 经 IMPL_OP 链注册在 OP_ADD 之前"的写法——虽然对本根因无效，但它是防占位符覆盖的正确姿势，对其他注册问题有效。
- 写 GenSimplifiedKey 回调时返回值一律用 `static_cast<UINT32>(std::snprintf(...))`（snprintf 恰好返回写入长度），不要写死 `return 0`；回调键格式必须与打包期 binary_info_config.json 的键逐字符对齐（含 precision_reduce 位与输出张量的 dtype/format 重复段）。

## 验证信号

来源说明：以下信号来自该算子多轮 GEIR 图模式验收会话（含回调注册源码重建 staging、清空 TBE kernel cache 强制重编译、逐项核对注册日志/.so 符号/kernel json/CANN 头文件，以及修复前后全流程对照），失败信号在各会话间逐字一致复现。

- `[TilingParse]`、`[GenSimplifiedKey]` 注册成功（route 日志 RegisterOpImplToRegistry）（来源平台：950；证据等级：tool_observed）。
- staging .so nm 含 `TilingParse<optiling::BenchHarnessScatterCompileInfo>` / `CreateCompileInfo<optiling::BenchHarnessScatterCompileInfo, 0>` 模板符号（来源平台：950；证据等级：tool_observed）。
- 强制重编译后 kernel json 仍无 compileInfo：`GetCompileInfo There is no compileInfo in jsonFilePath .../kernel_meta/te_benchharnessscatter_911e....json`、`SetOpCompileInfo:Compile info json after compiling is [].`（来源平台：950；证据等级：tool_observed）。
- 注册生效但结果不变：改动后日志出现 `MergeTypesToImpl:op type BenchHarnessScatter tiling_parse func registered first.`，同时 `SetOpCompileInfo:Compile info json after compiling is [].`、`hasTiling=0`、`block_dim = 6`、`rungraph_rc=4294967295 value=FAIL` 逐字保持（来源平台：950；证据等级：tool_observed）。
- GenSimplifiedKey 返回值修复 + 全局重装 + 清缓存后重跑：kernel_name 仍 `te_benchharnessscatter_911e6935...`、`ParseTvmBlockDim: 6`、`Tbe blockdim: 6`、`coreDim=6`、`rungraph_rc=4294967295 value=FAIL`，与修复前逐字一致；GenSimplifiedKey 仍仅注册行无调用行（来源平台：950；证据等级：tool_observed）。
- GE_JIT_COMPILE=0 无效：环境变量方式运行后日志仍 `jit_compile = 2` 且 EZ9999 复现（来源平台：950；证据等级：tool_observed）。
- 清 cache 无效：删除 `te_benchharnessscatter_*` 后重编译，kernel sha256 变化，`ParseTvmBlockDim: 6.` 与 EZ9999 不变（来源平台：950；证据等级：tool_observed）。
- 对照：AddCustom / EuclideanNorm 的 ops-info.json 无 compileInfo 字段、simplifiedKeyMode=0（来源平台：950；证据等级：tool_observed）。
- 无回归：改动后 ACLNN 回归 SUCCESS（overwrite=PASS、add=PASS）；GEIR infer V2 10/10 PASS、reject 1/1 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "无效"结论仅对 arch3510 + CANN 9.2.0 的 GEIR TBE 动态编译路径实测；其他 CANN 版本若修复了该路径的回调消费，本卡结论过期（未验证）。
- 五类修复各自针对的问题类型（回调失败返回、占位符覆盖、cache 陈旧、coreNum→blockDim 假设、返回值合同）在其他场景仍可能是真根因——本卡只否定它们对"图路径不消费回调"的效力。
- `TilingParseSafe` 的回退值（coreNum=1、ubSize=256KB）是保守默认，未经全 shape 验证；不推荐当通用模板。
- 返回键长是否足以让 binary matching 命中，在本组会话中未被验证（回调从未被调用的前提下无从验证）；键格式仅验证了与 binary_info_config.json 静态一致，未做过运行时键匹配成功的正例。
- compile_info_key 生成机制未追明（推断与 ops-info.json 的 compileInfo 字段相关）；"TBE adapter 因 key 为空跳过 TilingParse"为机制推断。

**未决项**：

- （优先项）compile_info_key 的生成与消费机制：tbe_op_store_adapter / python_adapter_manager 侧 key 从何而来（ops-info.json？op_build 产物？）、为空的判定链是什么——追到机制层才能找到让 TBE 动态编译产出 compile_info 的正确入口。
- （优先项）GenSimplifiedKey 在 TBE 动态编译路径下未被调用的确切条件（`simplifiedKeyMode` 取值语义、dynamicCompileStatic 二进制匹配流程入口），以及让 binary 匹配生效的正确注册/配置方式。
- 验证 GE 会话 options 显式设置 `ge.jit_compile=0`（REUSE_BINARY）能否让图路径命中预编译 binary（需可修改的测试程序，本组会话受冻结 ST 限制未尝试）。
- binary 配置加载问题解决后重测 GenSimplifiedKey 是否生效（与同批扫描清单负结果卡联动）。
## 坏实践

- 见注册成功的日志（RegisterOpImplToRegistry）就认定回调会在所有路径被调用：注册域与消费域是两回事，图模式与 ACLNN 的消费链路不同。
- 给 CompileInfo 结构加序列化方法/更多字段试图修复空 compile_info：`OpImplRegisterV2` 的 `TilingParse<T>` 合同只要求默认构造，问题不在这个结构体上。
- 在同一层用"再改一版回调"迭代无效修复：注册已生效而信号逐字不变时，应立即换层（框架/选项/执行路径）。
- GenSimplifiedKey 回调 `return 0`（空键合同违反）；只改 staging 不重装全局 OPP 就验证 host 侧回调修改——installed vendor 的旧 SO 会以 "registered first" 抢先注册，修改可能被掩盖。

## 误诊

- def.cpp 注释里的机制叙述（"TilingParse 失败 → compile_info 空 → GE 跳过 runtime tiling → kernel 读垃圾 → 崩溃"）被验收部分证伪：tiling 数据实际有传（一条 trace 以 opParaSize=312 佐证），回调也非返回失败（never-fail 已保证），而是该路径根本不产出 compile_info（agent_reported 旧叙述 → tool_observed 修正）。
- 会话早期一度因 nm 未命中 GenSimplifiedKey 符号而怀疑"没编进 .so"：静态函数不导出动态符号，运行时注册日志证明回调在（agent_reported 误判 → tool_observed 排除）。
- "GenSimplifiedKey 返回 0 → 空 key → 跳过 binary matching → TBE 回退"作为完整因果链是首个归因假设：返回值修复的对照实验证明它不充分（回调未被调用时返回值无关紧要）。该假设最初以确定性口吻写入代码注释，后被行为对照修正为"必要的回调合同修正，但非 TBE 回退根因"。
- "占位符覆盖了自定义回调导致 tiling 不跑"：注册顺序修复后 `registered first` 日志证实自定义回调已占先，但 hasTiling 仍 0——覆盖假设不成立。
