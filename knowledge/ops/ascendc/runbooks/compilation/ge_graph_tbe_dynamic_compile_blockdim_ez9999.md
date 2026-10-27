---
type: "Runbook"
title: GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999
description: 自定义算子 GE 图路径 kernel 执行失败（EZ9999 / rungraph_rc=4294967295）而 ACLNN 直调通过时，用 route 日志 grep 组合判别 TBE 动态编译回退：kernel_name 带 te_ 前缀 + ParseTvmBlockDim/coreDim 固化为平台 AIV 核数，预编译二进制（blockDim=-1）不参与匹配，SetBlockDim 被无视；返回值修复、jitCompile 字段、环境变量、清缓存、重装等修复路线全部实测无效。
tags: [task.debugging, topic.compilation, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999

## 现象

在 Ascend950PR（CANN 9.2.0）上为一个自定义 scatter 类算子（AscendC kernel + op_host tiling + op_build 预编译包，3 输入 data/indices/updates + 1 输出 y，ND 格式）做 GEIR 图模式验收。宿主侧 TilingFunc 对 route 验证用的小 shape（data=[4,3] f32、indices=[2,3] i32、updates=[2,3] f32）算出 numCores=1 并调用 `SetBlockDim(1)`；ACLNN 单算子直调路径数值全部通过。而 GE 图路径（ge::Session AddGraph/RunGraph 的 route 探针、TTK GEIR static/dynamic 用例）kernel 执行持续失败：

- route 探针输出（原文）：

```
GRAPH-RUN mode=route case=route-target node=n_route_target gid=1 feed={data:[4,3]/DT_FLOAT,indices:[2,3]/DT_INT32,updates:[2,3]/DT_FLOAT} rungraph_rc=4294967295 outputs=0 expected_numel=12 actual_numel=-1 value=FAIL
```

- route 日志显示实际选中的 kernel 是 `te_benchharnessscatter_911e6935...`（TBE 动态编译产物，`te_` 前缀），且核数被烤死为平台 AIV 核数 6，而非 TilingFunc 里 `SetBlockDim` 想要的值：

```
[task_builder.cc:432] FillTaskDefAfterGenTask:Set kernel_name[te_benchharnessscatter_911e693525cbf5d06f5e9a8252bf9902b81ab6cf6292a0f8f44393e1eb95f4c3] for the kernel_def of node[n_route_target, BenchHarnessScatter].
[tbe_json_parse.cc:231] ParseTvmBlockDim:Op[n_route_target], ParseTvmBlockDim: 6.
[tbe_task_builder_adapter.cc:524] Init:Tbe blockdim: 6.
[aix_starsv2.cc:575] StreamLaunchKernelV2: ... coreDim=6, ...
```

- 6 核执行触发设备异常并上报 EZ9999：

```
[ERROR] RUNTIME(...,geir_route): ... [device_error_proc_c.cc:716] ... ProcessDavidStarsCoreErrorInfo:The error from device(chipId:0, dieId:0), serial number is 289, there is an aivec error exception, core id is 0, error code = 95, ... mte error info: 0xdf73d5a800020265, ... mte error: 0x80000000, ...
[INFO] GE(...,geir_route):... ReportInterErrMessage:report error_message, error_code:EZ9999, ...
The extend info: errcode:(95) errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4.
```

- TTK GEIR 静态用例分化为两种失败形态：浮点 dtype 用例（f16/f32/bf16）`EXEC_FAILURE`（同一 EZ9999 aivec error）；整数 dtype 用例（i32/i64）kernel 跑完不崩但逐位失配（bitwise mismatch）；动态用例 kernel 执行挂起至整批超时。
- 对照组：同算子同安装包 ACLNN 直调回归 `Test overwrite: PASSED` / `Test add: PASSED`；InferShape/InferDataType 回调全 PASS；非法输入 reject 用例 PASS——失败集中在 GE 图路径的 kernel 编译/下发层。

而同一算子包里**预编译二进制是存在且可用的**：`binary_info_config.json` 中该算子的 `binPath` 指向 `ascend950/<apt>/<Op>_<hash>.o`，其 json 里 `blockDim: -1`（动态，运行时尊重 SetBlockDim），`simplifiedKey` 列表与 host 侧 GenSimplifiedKey 回调生成的键格式逐字段一致。当时代码状态：op_host 已注册 `IMPL_OP(<Op>).GenSimplifiedKey(...)` 回调（返回键长度）、never-fail `TilingParse`（CompileInfo 回退 coreNum=1/ubSize=256KiB），kernel 三处切分函数已含 `AscendC::GetBlockNum()` 网格对齐；即"回调注册 + 键格式对齐 + kernel 网格自适配"三层防御都在，GE 图路径仍失败。

注意：同一 error 95 / MTE DDR out of range 签名在 ACLNN 直调路径上另有独立根因（workspace 未含平台保留区），判别时先按本卡探针确认执行路径，见[507035 MTE DDR 越界：workspace 须含 16MB 保留区](../runtime/mte_ddr_507035_workspace_reserved_region.md)。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：AscendC 自定义算子包（op_kernel/op_host/op_graph + CPack 安装）+ GEIR 图模式验收（tests/geir 的 route/infer/reject 探针、TTK GEIR static/dynamic 评测）；"ACLNN 直调正常、图路径崩溃/错值"的对照排查。
- API 路径：GE 图编译 binary matching（`binary_info_config.json` 的 `simplifiedKeyMode`/`binaryList`）、`IMPL_OP(<Op>).GenSimplifiedKey(...)`、`OpAICoreConfig::ExtendCfgInfo`、`aic-<soc>-ops-info.json` 的 `jitCompile` 字段、GE `StreamLaunchKernelV2`/`FillTaskDefAfterGenTask`/`ParseTvmBlockDim`、TBE 动态编译缓存 `~/atc_data/kernel_cache/<SoC>/te_*`。
- 验证过的平台与 CANN 版本：Ascend950PR（Ascend950PR_9579，NpuArch 3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`——所有走 GE 图路径（ge::Session/RunGraph）加载自定义算子的场景，与具体算子无关。
- 路线优先级：`recommended_fastpath`（只读日志诊断，无副作用；缓存清理步除外，需重跑成本）。本卡是判别规则：确认 TBE 回退后，修复方向见「遗留问题」与同批负结果卡——host 侧/配置侧修复路线在本版本全部实测无效。
- 明确不适用 / 未验证场景：aclnn 直调路径（不走 GE binary matching，实测正常）；其他 CANN 版本的 jitCompile 行为与日志组件措辞；`simplifiedKeyMode` 非 0 的算子包。

## 判别方法

对 route 失败先判定"kernel 到底走的哪条编译路径"。以下探针套件全部在 route 日志上 grep（日志由 `./run.sh route > $ROUTE_LOG 2>&1` 生成，需 DEBUG 级）：

1. **kernel 选择结果**——`te_` 前缀即 TBE 动态编译，预编译二进制名形如 `<Op>_<hash>.o`：

```bash
grep "kernel_name\[" "$ROUTE_LOG" | head -3
# 命中 te_benchharnessscatter_911e6935... → TBE 动态编译回退
grep "Tbe blockdim\|ParseTvmBlockDim\|coreDim=" "$ROUTE_LOG" | head -5
# 命中 ParseTvmBlockDim: 6 / Tbe blockdim: 6 / coreDim=6 → blockDim 烤死为平台核数
grep "StreamLaunchKernelV2" "$ROUTE_LOG" | head -3
# 该行含 arg_size（GM 参数字节数）、addr1/addr2（kernel/tiling 基址）、coreDim（实际核数），是下发参数的最终事实源
```

2. **host 回调注册 vs 调用**——注册成功但零调用行：

```bash
grep "GenSimplifiedKey" "$ROUTE_LOG"
# 仅有 RegisterOpImplToRegistry ... register OP_IMPL : [GenSimplifiedKey]
# 与 MergeTypesToImpl ... gen_simplifiedkey func registered first.
# 无任何调用/生成键的日志行 → 键未参与匹配
grep "RegisterOpImplToRegistry.*TilingParse\|MergeTypesToImpl.*tiling_parse" "$ROUTE_LOG" | head -10
```

3. **ops-info 编译模式字段与 tiling 传递**——框架查 jitCompile 未命中 + 运行时下发参数里无 tiling：

```bash
grep "jitCompile\|jit_compile" "$ROUTE_LOG" | head -10
# 命中 GetStrFromOpContent:Op <Op> not found jitCompile in OpContent!
# 或 GetJitCompileDefaultValue:Current NpuArch = 3510, jit_compile = 2（平台默认 AUTO）
grep "hasTiling\|infoAddr\|argsSize" "$ROUTE_LOG" | head -5
# 命中 CheckArgs:... argsSize=40, hasTiling=0 与 StreamLaunchKernelV2 ... infoAddr=(nil)
# → launch 参数只有裸指针宽度总和（无 tiling 段），TilingData 未随下发传递
```

4. **binary_info 读取路径与键对照**——确认配置文件被读、键空间一致：

```bash
grep "ReadOpBinaryInfoConfig\|binary_info_config\|SetBinaryConfigPaths" "$ROUTE_LOG" | head -10
# 注意读取路径：installed vendor（$ASCEND_HOME_PATH/opp/vendors/<Op>/...）而非 staging
python3 -c "import json; d=json.load(open('<vendor>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json')); print(d['<Op>']['simplifiedKeyMode'], d['<Op>']['binaryList'][0]['simplifiedKey'][:2])"
```

5. **TBE 缓存 json 实锤"烘焙"**——对比两类编译产物的 blockDim（一个字段即可分辨编译路径）：

```bash
python3 -c "import json;d=json.load(open('~/atc_data/kernel_cache/<SoC>/te_<op>_<hash>.json'));print('TBE blockDim:',d['blockDim'])"
# TBE 动态编译产物：blockDim=6, coreType=MIX, opParaSize=312
python3 -c "import json;d=json.load(open('<vendor>/.../<Op>_<hash>.json'));print('预编译 blockDim:',d['blockDim'])"
# 预编译二进制：blockDim=-1（动态，运行时尊重 SetBlockDim）
```

6. **排除环境干扰（对照轮前执行）**：

```bash
# 清 TBE kernel cache（否则 te_* 产物缓存会掩盖"重新打包后行为未变"的判断）
rm -rf ~/atc_data/kernel_cache/<SoC>/te_<opname>* 2>/dev/null
# installed 与 staging 的 binary_info 一致性对照
diff <installed>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json \
     <staging>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json
```

若修改过 host 代码：重建后**同时重装全局 OPP**（installed vendor 的旧 SO 会以 "registered first" 抢先注册回调），再清缓存重跑。

判读组合：**kernel_name 带 te_ 前缀 + blockdim/coreDim=平台核数 + GenSimplifiedKey 零调用行 + 预编译 binary 存在且 blockDim=-1 + ACLNN 直调对照通过** ⇒ 预编译二进制未参与匹配、kernel 被 TBE 动态编译且 blockDim 被烤死；此时无论 `SetBlockDim` 写什么、二进制键多全，GE 图路径都不会用预编译产物。

## 根因

**已验证的核心因果链**：GE 图路径下该算子的预编译二进制匹配未发生（kernel 名恒为 `te_` 前缀、无任何 binary matching 命中记录、预编译条目存在且键格式与回调一致），编译器回退 TBE 动态编译；TBE 动态编译把 blockDim 烤为平台 AIV 核数（本机 6）写进编译产物 json，运行期 `TilingFunc` 的 `SetBlockDim(1)` 与 `TilingParse` 的 coreNum=1 均不生效；kernel 以 6 核执行按 1 核规划的 TilingData，浮点用例触发 MTE DDR 越界（aivec error 95，core 0，`mte error: 0x80000000`）上报 EZ9999，整数用例表现为逐位失配。一条 trace 的运行时日志另观测到 `argsSize=40, hasTiling=0, infoAddr=(nil)`——launch 参数未含 tiling 段；而 TBE 产物 json 元数据又声明 `opParaSize: 312`（与 TilingData 结构大小一致）。TilingData 究竟是否随下发到达 kernel，各 trace 观测不一致，未在机制层定案（见失效边界）。

**上游"为何 binary matching 不发生"未闭合，各会话提出的候选归因（均为黑盒推断，无一验证到机制层）**：

1. ops-info（`aic-ascend950-ops-info.json`）无 `jitCompile` 字段：`GetStrFromOpContent:Op <Op> not found jitCompile in OpContent!` 后框架按默认编译模式处理（日志原文 tool_observed；ops-info dump 证实确无该字段）。经 `ExtendCfgInfo` 补该字段的路线实测被值校验拒绝（见同批 jitCompile 负结果卡）。
2. 平台默认 `ge.jit_compile=2`（AUTO）覆盖算子级复用配置：`GetJitCompileDefaultValue:Current NpuArch = 3510, jit_compile = 2`、`CheckJitCompileForBinReuse ... opJitCompile configured flag[0]`——算子级"复用 binary"被全局默认胜出。
3. TBE fusion 的 `SetBinaryConfigPaths` 扫描清单不含该算子 vendor：一条 trace 观测到清单只含固定 vendor 与 built-in（见同批扫描清单负结果卡）；但其他 trace 观测到 `ReadOpBinaryInfoConfig Read binary info file[.../opp/vendors/<Op>/...] success`——两处观测矛盾，可能与安装状态有关，未定案。

框架侧（libopskernel.so / op_kernel_info_constructor）为闭源二进制，行为证据链完整（键全一致 + 配置读取成功 + 回调已注册仍 te_ 回退 + 修复对照逐字不变），但启用 binary matching 的确切条件（jitCompile 合法值集、simplifiedKeyMode=0/1 语义、GenSimplifiedKey 被调用的前提）未追到机制层。

## 设计决策

- 为什么用"日志 grep 组合"而不是只看退出码：`rungraph_rc=4294967295` 与 EZ9999 只说明 kernel 执行失败，不区分"TBE 回退"与"二进制匹配后仍失败"；kernel_name 前缀与 blockdim 日志才是路径判别的决定性证据。判别顺序按成本递增：先看 kernel 名前缀与 coreDim（一次 grep），再查缓存 json 的 blockDim 字段（实锤"烘焙"），最后才查 binary 配置加载链（最贵、涉及框架）。
- 为什么以 route 日志为主源而不是 TTK 结果：TTK 的 EXEC_FAILURE/位失配只给现象；route 单用例日志量小（单 node 单 feed）且全链路 DEBUG 级日志齐全；定位后再回 TTK 验证面。
- 为什么同时对照 installed 与 staging 的 binary_info_config.json：框架从 **installed** vendor 路径读 binary_info，而 kernel/SO 可从 staging（ASCEND_CUSTOM_OPP_PATH）加载；两处不一致时会误判"键不存在"。
- 为什么清 TBE kernel cache 后再判：`~/atc_data/kernel_cache/<SoC>/te_<op>*` 缓存会掩盖"重新打包后行为未变"的判断，重跑前先清（其无效性本身是独立负结论，见同批清缓存卡）。
- 为什么保留 ACLNN 直调对照：两条路径一个走预编译二进制（blockDim=-1）、一个可能走 TBE 动态编译，对照结果能立刻把问题域缩小到二进制选择。
- 当前方案路线档位：recommended_fastpath（这是判别规则，不是修复）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（Ascend950PR_9579，NpuArch 3510，npu-smi 25.7.rc1.6），CANN 9.2.0；GEIR 验收套件（tests/geir 的 run.sh route/infer/reject）、TTK（/tmp/ops-test-kit）。
- 目录布局（诊断涉及）：
  - `op_host/<op>_def.cpp`——GenSimplifiedKey/TilingParse/OpAICoreConfig 注册。
  - `build/autogen/aic-ascend950-ops-info.ini` / 安装后 `aic-ascend950-ops-info.json`——框架读取的 op 配置。
  - `$ASCEND_HOME_PATH/opp/vendors/<Op>/op_impl/ai_core/tbe/...`——installed vendor（binary_info_config.json 实际读取处）。
  - `/tmp/geir_opp.*/vendors/<Op>/...`——staging vendor（ASCEND_CUSTOM_OPP_PATH）。
  - `~/atc_data/kernel_cache/Ascend950PR_9579/`——TBE 动态编译缓存。
  - `/tmp/geir-route.*.log`——route 日志（诊断主源）。

## 预防

- 交付自定义算子 GEIR/图模式前，把"kernel 选择路径"检查纳入验收：route 日志里 kernel_name 必须是非 `te_` 前缀的预编译名（或明确接受 TBE 路径并让 kernel 网格无关——但注意网格无关化也不充分，见同批运行时负结果卡）。
- 打包时检查生成的 ops-info 字段集，确认编译模式字段（jitCompile 等）与预期一致；缺失时先弄清框架默认行为再上线。
- 图路径验收时把"实际 coreDim"（StreamLaunchKernelV2 日志）与 kernel 假设的网格数对照；coreDim≠预期时先解决编译路径问题，不要在错误网格下调试 kernel 语义。
- 报 `EZ9999 ... error code = 95, The DDR address of the MTE instruction is out of range` 时，优先核对 blockDim 一致性（缓存 json 的 blockDim vs tiling 的 SetBlockDim）再查 kernel 地址计算。

## 验证信号

来源说明：以下信号来自同一自定义 scatter 算子在 Ascend950PR + CANN 9.2.0 上的多轮独立 GEIR 图模式验收会话（隔离 staging 重建 + route/TTK 用例 + 日志逐项核对 + 修复对照实验），判别结论在各会话间一致复现。

- 复现广度：该判别三联征（`te_` 前缀 kernel 名 + `coreDim/blockDim=6` 与单核 tiling 矛盾 + `EZ9999 aivec error 95`）在本批共 15 条独立验收轨迹中逐条复现，无一例外；各轨迹的清缓存/改注册/改 tiling 尝试均未能改变该路径选择（来源平台：950；证据等级：tool_observed）。

- route 正向失败：`rungraph_rc=4294967295 outputs=0 expected_numel=12 actual_numel=-1 value=FAIL`（12 元素小图）；设备错误 `aivec error exception, core id is 0, error code = 95, ... mte error: 0x80000000`；上层 `EZ9999` 与 `rtStreamSynchronizeWithTimeout:ErrCode=507011`（来源平台：950；证据等级：tool_observed）。
- kernel 选择：`Set kernel_name[te_benchharnessscatter_911e6935...]`、`ParseTvmBlockDim: 6`、`Init:Tbe blockdim: 6`、`StreamLaunchKernelV2 ... coreDim=6`，与 TilingFunc `SetBlockDim(1)` 矛盾（来源平台：950；证据等级：tool_observed）。
- TBE 缓存 json 含 `"blockDim":6,"coreType":"MIX","taskRation":"0:1"`；预编译 binary_info_config 条目 `blockDim=-1`、键空间完整（如 `BenchHarnessScatter/d=0,p=0/0,2/3,2/0,2/0,2`）；`opParaSize=312` 与 TilingData 结构大小一致（来源平台：950；证据等级：tool_observed）。
- 二进制未命中证据：`no need to precompile` + `IMPLY_TYPE [TVM]` + `is_dynamic_impl=1`；`KernelName attr of node[...] is []` 后设为 te_* 名（来源平台：950；证据等级：tool_observed）。
- GenSimplifiedKey 注册成功但零调用行：`RegisterOpImplToRegistry ... register OP_IMPL : [GenSimplifiedKey]` + `MergeTypesToImpl ... registered first.`，排除 register/merge 行后无任何调用痕迹（来源平台：950；证据等级：tool_observed）。
- jitCompile 查询落空：`GetStrFromOpContent:Op BenchHarnessScatter not found jitCompile in OpContent!`（ops-info dump 证实无该字段）；另一会话观测 `GetJitCompileDefaultValue:Current NpuArch = 3510, jit_compile = 2` 与 `CheckJitCompileForBinReuse ... opJitCompile configured flag[0]`（算子级复用配置被全局默认覆盖）（来源平台：950；证据等级：tool_observed）。
- 运行时 tiling 传递观测（单会话）：`CheckArgs:hostInputInfoNum=0, isNoNeedH2DCopy=1, argsSize=40, hasTiling=0`、`StreamLaunchKernelV2 ... arg_size=40 ... infoAddr=(nil)`（来源平台：950；证据等级：tool_observed）。
- binary_info 读取：`ReadOpBinaryInfoConfig Read binary info file[.../opp/vendors/BenchHarnessScatter/...] success`（installed 路径）；`simplifiedKeyMode: 0`；staging 与 installed diff 为空（来源平台：950；证据等级：tool_observed）。
- 修复对照（多会话一致）：GenSimplifiedKey 返回值修复 + 全局重装 + 清 TBE 缓存后重跑，三行 kernel 选择日志逐字不变；`GE_OPTIONS="ge.jit_compile=1"` 与 `GE_JIT_COMPILE=0` 环境变量均不改变 kernel 选择；清缓存后 fresh 重编译产物仍 `blockDim: 6`（来源平台：950；证据等级：tool_observed）。
- TTK GEIR 静态 8/8 失败：6 例 EXEC_FAILURE（EZ9999）+ 2 例整数用例 bitwise mismatch（56.25%~68.75% 区间，各会话略有差异）；动态 4 例挂起至 600s 超时（另一会话观测 dynamic 4/4 PASS，执行路径差异未解释）（来源平台：950；证据等级：tool_observed）。
- ACLNN 直调对照：`Test overwrite: PASSED` / `Test add: PASSED`；Infer V2 10/10 PASS、reject 1/1 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 判据"零调用行 ⇒ 回调未参与"依赖日志覆盖面：若某版本框架调用 GenSimplifiedKey 但不打日志，此判据失效，需改用键生成侧的间接证据（如故意生成错误键观察行为变化，未做）。
- TilingData 是否随 TBE 动态编译产物下发未定案：一条 trace 的运行时日志观测 `hasTiling=0 / infoAddr=(nil) / argsSize=40`，另一条 trace 以产物元数据 `opParaSize=312` 推断 tiling 有传——两者观测层面不同（launch 参数 vs 编译产物元数据），机制层未闭合，本卡不对此下结论。
- `ReadOpBinaryInfoConfig ... success`（installed vendor 读取成功）与 `SetBinaryConfigPaths` 清单不含该 vendor 的观测来自不同会话，安装状态可能不同；不能合并成单一结论。
- 仅在 Ascend950PR + CANN 9.2.0 观测；其他 CANN 版本的 jitCompile 合法值、binary matching 启用条件与日志组件措辞可能不同（未验证，推断判别思路可迁移）。
- `simplifiedKeyMode` 语义为推断；若 1=使用回调，则把该字段改为 1 可能是另一条修复路线，未验证。
- 本卡不解决 TBE 回退本身：返回值修复、jitCompile 字段、环境变量、清缓存、重装、手改 simplifiedKeyMode 均被证伪（见同批各负结果卡）。

**未决项**：

- **优先项：复现并追根因到框架机制层**——确定 binary matching 的启用条件（jitCompile 合法值集、simplifiedKeyMode=0/1 语义、GenSimplifiedKey 被调用的确切前提）。可行入口：反汇编/strings libopskernel.so 定位 op_kernel_info_constructor 的合法值表；或用最小算子样例逐字段做对照实验。
- 验证 GE 会话 options 显式设置 `ge.jit_compile=0`（REUSE_BINARY）能否让图路径命中预编译 binary（route_main.cpp 以空 options 创建 ge::Session，环境变量途径已证伪，Session options 途径未试）。
- TBE 动态编译 `te_` kernel 的编译来源（TBE DSL impl vs AscendC 源码）未确认—— AscendC 侧 GetBlockNum 修复存在却仍 EZ9999 的矛盾由此而来；6 核下 MTE error 95 的具体越界访存指令（PC 0x58c 未反汇编）与整数用例位失配的逐元素归因见同批运行时/精度卡。
- TTK GEIR dynamic 用例在不同会话中分别表现为挂起超时与 4/4 PASS，执行路径差异未查明。
## 坏实践

- 只看 `rungraph_rc`/EZ9999 就开始改 kernel——先确认 kernel 走的是哪条编译路径，否则修的是错误层。
- 看到 EZ9999 MTE 越界就先怀疑 kernel 地址计算/tiling 尾块：本例 ACLNN 直调通过已把 kernel/tiling 本体排除，问题在执行配置（blockDim）而非数据规划。
- 改了 host 代码只重建 staging 就重跑——binary_info 从 installed vendor 读，旧 installed SO 会以 "registered first" 抢先注册回调；必须同步重装全局 OPP 并清 TBE kernel cache。
- 用 `GE_OPTIONS`/`GE_JIT_COMPILE` 环境变量试图翻转编译模式（本组多会话实测无效）。
- 用"多跑几次/清缓存重试"处理确定性编译配置问题：清 cache 强制重编译后产物仍是 blockDim:6，结果逐位不变。
- 把 built-in 路径 `binary_info_config.json does not exist` 的 WARNING 当作"配置不可见"的证据——要看到最终的 `Read ... success` 行才知道实际读了哪里。

## 误诊

- 首个归因假设"GenSimplifiedKey 返回 0 导致空键、跳过 binary matching"：修复返回值为键长度后（重建+安装+清缓存），route 三行关键日志逐字不变，证明返回值不是（充分）根因——回调根本没有被调用（agent_reported 初始归因，被对照实验证伪；返回值修复作为回调合同修正保留）。
- "binary_info_config.json 没被读到"：早期日志片段显示框架先找 built-in 路径不存在、转读 ops_legacy，一度被怀疑配置不可见；全量 grep 证实最终从 installed vendor 路径读取成功，键空间完整。
- "TBE kernel cache 陈旧导致"：清空缓存后 fresh 重编译，sha256 变化但 `blockDim: 6` 与失败信号逐字不变——排除陈旧缓存单因。
- "TilingParse 设 coreNum=1 会让 TBE 产出 blockDim=1"：TBE 动态编译产物的 blockDim 恒为平台核数，不读 TilingParse 的 coreNum（源码注释假设被实测证伪）。
- "compile_info 为空是因为 TilingParse 返回失败"：never-fail 回调保证后 compile_info 仍为 `[]`——回调返回值不是空 compile_info 的原因。
- 首次运行 route 探针时把 `path=UNKNOWN status=NOT_UNIQUE` 当作路由问题排查：实际是外层预导出的 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 污染了包装脚本的命令替换捕获，清洁重跑后 path=V2。
