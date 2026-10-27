---
type: "Runbook"
title: GetCoreNumAiv运行时少于INI核数，分支边界漂移
description: Ascend950PR 实机 aclnn 运行时 GetCoreNumAiv()≈56-57（平台 INI vector_core_cnt=64 为物理核数），含 coreNum/2 阈值的 tiling 分支谓词实际边界从 32 收窄到 28；用未实现分支桩的毒化输出做边界 shape 扫描实测标定。
tags: [task.debugging, topic.architecture]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# GetCoreNumAiv运行时少于INI核数，分支边界漂移


## 现象

在按 ops-registry-invoke 算子仓流程开发一个 GRU Cell 反向融合类自定义算子（aclnn 调用链：`aclnn<Op>GetWorkspaceSize → aclnn<Op>`）时，为其 group（2D 分核，tilingKey=1）分支做测试先行的 ST 红基线：该分支 kernel 数值链尚未实现，是只编译、不写输出的接口桩（输出保持毒化值），因此本分支全部数值用例期望 0 条通过（红基线属预期）。

选例阶段，设计文档的"平台参数实据"记载实机 coreNum=64（来源 npu-smi + 平台 INI `vector_core_cnt=64`），开发者据此用 host tiling 公式复算了 12 个候选 shape，全部满足分支进入条件 `aLoopCntTotal ≤ coreNum/2 = 32`（其中一个 fp16 大 shape 用例 B=1024、H=2048，经算子内 cacheBuf 容量钳制 aUbFactor 128→64 后 aLoopCntTotal = CeilDiv(2048, 64) = 32，恰好压在相等边界上），遂 12 条全部收入 group 分支 ST 清单。

实机构建（`./build.sh <op>_package` 退出码 0）并运行 ST 后：

- 11 条用例如期以毒化量级失败，例如 `grad_input_gates FAIL mere=1.293e+35 mare=2.991e+37`、`grad_hx FAIL mere=1.461e+06 mare=5.741e+09`（mere/mare 为统计相对误差口径，`rel_err=|a-g|/(|g|+1e-7)`）——这是 group 桩不写输出的典型特征；
- 唯独那个 fp16 大 shape 用例（B=1024, H=2048）五路输出全部 PASS，且三路逐元素输出 `mere=0.000e+00`、两路 bias 输出 `mare≤1.178e-03`（预算内）——与已实现的 base 分支 kernel 的正确结果完全一致。

实跑日志原文（该用例段）：

```text
[ST] N2 fp16 B=1024 H=2048 bias=1 src=…非对齐与大 shape…（实机 64 核经 cacheBuf 钳制 aUbFactor 128→64 后
     aLoopCntTotal=32≤coreNum/2=32（相等边界）且 rLoopCntTotal=22 → usedCoreNum=64、rGroupCnt=2…）
     grad_input_gates   PASS mere=0.000e+00 mare=0.000e+00 mare_budget=1.000e-02 [10*th] th=1.0e-03 nan_inf_mismatch=0
     grad_hidden_gates  PASS mere=0.000e+00 mare=0.000e+00 mare_budget=1.000e-02 [10*th] th=1.0e-03 nan_inf_mismatch=0
     grad_hx            PASS mere=0.000e+00 mare=0.000e+00 mare_budget=1.000e-02 [10*th] th=1.0e-03 nan_inf_mismatch=0
     grad_input_bias    PASS mere=1.165e-06 mare=7.610e-04 mare_budget=1.000e-02 [max(10*th,10*noise=4.844e-04)] th=1.0e-03 nan_inf_mismatch=0
     grad_hidden_bias   PASS mere=7.598e-07 mare=1.178e-03 mare_budget=1.504e-02 [max(10*th,10*noise=1.504e-03)] th=1.0e-03 nan_inf_mismatch=0
     determinism OK(bitwise,2runs) | verdict PASS
```

汇总行：`Branch-1 ST summary: PASSED=1 FAILED=11 ERROR=0 TOTAL=12`。

**判别信号**：红基线阶段"本应全 FAIL 的分支清单里出现 PASS"＝该用例根本没进 group 桩分支，而是路由到了已实现的 base 分支——分支路由与选例口径不符。mere=0（与 base kernel 位级一致）排除"桩巧合算对"的可能。

## 适用场景

- **工程形态**：aclnn 算子仓（ops-registry-invoke 流程）ST 用例选型与红基线建设；host TilingFunc 分支谓词（tilingKey 路由）开发与核对。
- **API 路径**：`platform_ascendc::PlatformAscendC::GetCoreNumAiv()`、`gert::TilingContext::GetPlatformInfo()`、`GetCoreMemSize(CoreMemType::UB)`；平台 INI `<CANN>/x86_64-linux/data/platform_config/<SoC>.ini` 的 `vector_core_cnt`/`ai_core_cnt`/`cube_core_cnt`；任何形如 `xxxCnt <= coreNum/2`（或以核数为阈值）的分支进入谓词。
- **验证过的平台与 CANN 版本**：Ascend950PR（注册词表 `950`）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致；仅此组合实测）。
- **适用粒度（transfer_scope）**：`platform_wide`——该平台上一切以运行时核数为阈值的 tiling 谓词与按分支组织的用例选型。
- **路线优先级**：`recommended_fastpath`（边界 shape 扫描 + 桩毒化二值判定的实测定界法，一次约 7 个 shape 即可定界）。
- **明确不适用 / 未验证场景**：其他芯片（a2/a3/310p/910 等）未实测——推断可迁移但边界值必须现测；运行时核数可能随 CANN 版本、容器部署、预留策略变化而漂移，**边界值 28 不可硬编码外推**；谓词阈值不是 `coreNum/2` 形式（固定常数、与 cube 核数联动等）时探针变量需相应替换。

## 根因

**单句归因**：实机 aclnn 运行时 `PlatformAscendC::GetCoreNumAiv()` 返回的可用 AIV 核数（实测反推约 56–57）少于平台 INI `vector_core_cnt=64` 的物理核数，使含 `coreNum/2` 阈值的分支进入谓词的实际边界从 32 收窄到 28。

机制链与算式：

1. host TilingFunc 的分支谓词（源码 verbatim，见「完整代码」）：`isGroup = (aLoopCntTotal <= coreNum/2) && (rLoopCntTotal >= 2)`，其中 `coreNum` 在 `GetPlatformInfo()` 里经 `ascendcPlatform.GetCoreNumAiv()` **运行时获取**（逐项非 0 校验，不写死）。
2. 选例口径用的是 INI/npu-smi 的 64：`64/2 = 32`，大 shape 用例 `aLoopCntTotal = CeilDiv(2048, 64) = 32 ≤ 32` → 判 group。
3. 探针实测边界：扫 `aLoopCntTotal ∈ {24, 28, 29, 30, 31, 32, 40}`，24/28 落 group、29–40 落 base → group 当且仅当 `aLoopCntTotal ≤ 28`（在 `rLoopCntTotal ≥ 2` 前提下）→ `coreNum/2 = 28` → `coreNum ∈ {56, 57}`（整除口径：56/2=28、57/2=28）。
4. 即运行时可用 AIV 核数比 INI 物理核数少 7~8 核（64−56=8 或 64−57=7），谓词边界漂移量 = 32−28 = 4；`aLoopCntTotal=32` 的相等边界用例因此翻到 base 分支（`32 > 28`）。
5. 佐证：平台 INI 中 `ub_size=253952` 等内存容量类参数与运行时一致（该类参数可靠），**只有核数一类存在"物理 vs 运行时可用"差**——所以受影响的是所有以核数为阈值的谓词，而不是 UB 预算类计算。

## 判别方法

**触发信号**：按分支组织的红基线里出现"本应 FAIL 却 PASS（mere≈0）"的用例；或按文档核数预算的分支归属与实机行为不符。

**探针（stub 毒化二值判定 + 边界扫描）**，前提是未实现分支为不写输出的桩、已实现分支正确：

1. 单用例 verdict 即路由二值信号：FAIL（毒化量级 mere 1e+30+）＝进了桩分支；PASS（mere≈0）＝进了已实现分支。同输入连跑 2 次位级一致（determinism OK(bitwise,2runs)），信号确定。
2. 构造只扫谓词变量的 shape 序列：固定 dtype 与 B，把 `aUbFactor` 钉在 cacheline 上限（fp32 时 `cachelineTmp = FloorAlign(cacheLineSize/dtypeSize, bsElem) = FloorAlign(256/4, 8) = 64`），令 `aLoopCntTotal = CeilDiv(H, aUbFactor)` 精确可控地取到边界两侧整数：H=1536→24、1792→28、1856→29、1920→30、1984→31、2048→32、2560→40。
3. 健壮性（变量解耦）：取 B=1024，任何合理 ubSize 下 `rLoopCntTotal ≥ 2` 恒成立（ub=253952 时 rIMax=48、rLoop=CeilDiv(1024,48)=22；即使 ub 小到 65536，rLoop=CeilDiv(1024,9)=114 仍 ≥2），且 cacheBuf 容量校验 `CacheCount(22)×CeilAlign(6×64,8)×4 = 5×384×4 = 7680 ≤ 16384` 不触发二次钳制——探针把"核数"与"UB 容量"两个变量解耦，分支判定只剩 `aLoop` 与 `coreNum/2` 的比较。
4. 用独立 oracle 生成期望值后逐个实跑，二分定位最大 group `aLoopCntTotal`，×2 即运行时 `coreNum`（整除口径）。

## 解决配方

1. **修正用例归属**：把实测落在 base 的用例移出 group 分支清单（本例 12→11 条），分支清单内每条用例以实机路由证据（毒化 FAIL）背书；被移出用例的覆盖缺口**上报设计侧裁定，不得静默丢弃**（本次在开发日志中登记"覆盖归属交裁定"）。
2. **平台口径修订**：凡文档写 `coreNum=N` 的地方注明取值来源与口径（INI 物理核数 vs 运行时 `GetCoreNumAiv()`）；含核数阈值的谓词以运行时口径复核。
3. **反投机边界（重要）**：移出依据是实测路由证据（mere=0 与探针边界），**不是**"删失败用例凑红基线"——被移出的是一条 PASS 用例且其覆盖缺口被显式上报；若因用例 FAIL 而删除/放宽容差/改期望，则属测试侧放松，禁止作为解决配方。

## 设计决策

- **为何用桩毒化输出做路由探针而不是读日志**：读运行时 tiling 日志的 slog/plog 路线实测不可行（详见同问题域负结果卡「自定义算子 host 的 OP_LOGI 不进 stdout 与 plog」）；行为二值信号确定性强——毒化输出量级 1e+30+ 与正确输出 mere=0 之间无混淆可能。
- **为何探针固定 fp32、B=1024、只扫 H**：把谓词变量隔离到 `aLoopCntTotal` 一个维度（推导见「判别方法」第 3 步）；fp32 的 aUbFactor 钉死 64，`aLoopCntTotal=CeilDiv(H,64)` 精确可控；`rLoopCntTotal≥2` 恒成立使第二谓词条件不干扰。
- **被放弃的备选**：① 直接写小程序打印 `GetCoreNumAiv()`——`PlatformAscendC` 构造需要 `TilingContext`，脱离 aclnn 调用链构造成本高；② 用 tiling UT 桩查询——UT 喂的是 mock 平台（fixture 固定 coreNum），测不到真实运行时值；③ 假设"少 1 核（63）"直接改口径——方向对但数值错（实测少 7~8 核），不实测会再次选错边界。
- **当前方案属哪档路线**：`recommended_fastpath`。

## 系统信息

- 设备：Ascend950PR（npu-smi 25.7.rc1.6，NPU ID 0）；平台 INI `Ascend950PR_9589.ini` 实读：`ai_core_cnt=32`、`cube_core_cnt=32`、`vector_core_cnt=64`、`ub_size=253952`、`ubblock_size=32`、`ubbank_size=4096`。
- CANN：9.0.0-beta.2（`ASCEND_HOME_PATH=/home/<user>/Ascend/cann-9.0.0-beta.2`）。
- 工具链：ops-registry-invoke 算子仓流程；`./build.sh <op>_package` 安装自定义算子包后，`tests/aclnn/run.sh` 驱动（cmake 编译 `test_aclnn` → 独立 oracle `golden.py` 生成期望数据 → aclnn 两段调用链实跑，`LD_LIBRARY_PATH` 指向 `${ASCEND_OPP_PATH}/vendors/<op>/op_api/lib/`）。
- 相关文件（与本次事件流实际读写一一对应）：
  - `op_host/arch35/<op>_tiling_arch35.cpp`（`GetPlatformInfo`/`ShouldUseGroup`/`FillAndLogTilingData`，只读核对）
  - `op_kernel/arch35/<op>_group.h`（group 分支接口桩，只读核对）
  - `tests/aclnn/st_cases_branch1.json`（分支 ST 用例清单：本次由 12 条修订为 11 条）
  - `tests/aclnn/gen_st_oracle.py`、`tests/aclnn/test_aclnn.cpp`、`tests/aclnn/run.sh`（ST 基建）
  - `/home/<user>/Ascend/cann-9.0.0-beta.2/x86_64-linux/data/platform_config/Ascend950PR_9589.ini`（平台参数）

## 完整代码

**① 被核对的 host 谓词与平台取值（op_host tiling 源码，事件流 verbatim）**：

```cpp
// ─── 硬件参数全走 platform 接口，逐项非 0 校验 ───
ge::graphStatus GetPlatformInfo(gert::TilingContext* context, ThnnFusedGruCellGradCtx& ctx)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ctx.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ctx.coreNum == 0, OP_LOGE(context, "CoreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ub = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ctx.ubSize = static_cast<int64_t>(ub);
    OP_CHECK_IF(ctx.ubSize == 0, OP_LOGE(context, "UbSize is 0"), return ge::GRAPH_FAILED);
    ctx.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));      // 不写死 32
    ctx.cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context)); // 不写死 256
    OP_CHECK_IF(ctx.blockSize == 0 || ctx.cacheLineSize == 0,
                OP_LOGE(context, "BlockSize=%ld or cacheLineSize=%ld is 0", ctx.blockSize, ctx.cacheLineSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ─── group 触发判定（含 coreNum/2 阈值——运行时口径）───
bool ShouldUseGroup(const ThnnFusedGruCellGradCtx& ctx)
{
    return (ctx.aLoopCntTotal <= ctx.coreNum / 2) && (ctx.rLoopCntTotal >= 2);
}
```

**② 边界扫描探针清单构造（事件流 verbatim）**：

```python
import json
base = json.load(open('operators/ThnnFusedGruCellGrad_package/tests/aclnn/st_cases_branch1.json'))
base['cases'] = []
base['task'] = 'PROBE: runtime coreNum/2 boundary (aLoopCntTotal sweep, fp32 B=1024)'
# aUbFactor=64 (min(64,H) for H>=64, fp32); aLoop = CeilDiv(H,64)
for aloop, H in [(24,1536),(28,1792),(30,1920),(31,1984),(32,2048),(40,2560)]:
    base['cases'].append({
        "case_id": f"PROBE_aLoop{aloop}",
        "source": f"probe aLoopCntTotal={aloop} (H={H}, aUb=64)",
        "B": 1024, "H": H, "dtype": "float32", "has_bias": True,
        "inputs": {"grad_hy": {"pattern":"uniform","lo":-2.0,"hi":2.0},
                   "storage": {"pattern":"forward_reachable"}},
        "assert_mode": "stat_rel_err", "threshold": 1e-5})
json.dump(base, open('/tmp/opencode/manifest_probe.json','w'), ensure_ascii=False, indent=1)
print("probe manifest: 6 cases")
## 收窄边界时同式只留 [(29,1856)] 写入 manifest_probe29.json
```

**③ 期望值生成（独立 oracle，禁被测 kernel 自证）与实跑（事件流 verbatim）**：

```bash
## 生成探针期望值（--golden 必须给绝对路径；相对路径按 cwd 解析会 FATAL）
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 gen_st_oracle.py \
    --manifest /tmp/opencode/manifest_probe.json --out /tmp/opencode/probe_data \
    --golden /workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/docs/ThnnFusedGruCellGrad/develop/golden.py

## 实跑（已安装 vendor 库须在 LD_LIBRARY_PATH）
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/ThnnFusedGruCellGrad/op_api/lib/:${LD_LIBRARY_PATH}
./test_aclnn /tmp/opencode/manifest_probe.json /tmp/opencode/probe_data \
  | grep -E "^\[ST\] PROBE|verdict"
```

**④ 探针原始输出（事件流 verbatim）**：

```text
PROBE_aLoop24 fp32 B=1024 H=1536 bias=1 => FAIL      # FAIL = group 桩（毒化）
PROBE_aLoop28 fp32 B=1024 H=1792 bias=1 => FAIL
PROBE_aLoop30 fp32 B=1024 H=1920 bias=1 => PASS      # PASS = base（已实现，mere≈0）
PROBE_aLoop31 fp32 B=1024 H=1984 bias=1 => PASS
PROBE_aLoop32 fp32 B=1024 H=2048 bias=1 => PASS
PROBE_aLoop40 fp32 B=1024 H=2560 bias=1 => PASS

## 收窄单点（aLoop=29, H=1856）：
[ST] PROBE_aLoop29 fp32 B=1024 H=1856 bias=1 src=probe aLoopCntTotal=29 (H=1856, aUb=64)
     determinism OK(bitwise,2runs) | verdict PASS
```

## 预防

- 冻结按分支组织的 ST/白盒用例清单前，先在目标机实测谓词变量的运行时口径（本探针法一轮即可）；**相等边界用例（`aLoopCntTotal == coreNum/2`）最易因口径差翻面**，优先复核。
- 设计/验证文档中的平台参数逐项注明来源与口径（"INI 物理核数"与"运行时 GetCoreNumAiv()"分开记载）。
- 红基线阶段把"意外 PASS"当作路由错位告警处理，而不是当作部分实现已生效的好消息。
- 以核数为阈值的 host 谓词，配一个可从用例侧复算的口径说明（阈值 = 运行时核数 / 2，非文档核数）。

## 验证信号

以下信号来自一次自定义算子 group 分支 ST 红基线的实机排查（Ascend950PR，CANN 9.0.0-beta.2，ops-registry-invoke 算子仓流程）：开发者按文档核数口径选例后实机运行发现分支路由不符，随后构造边界 shape 探针在真实 aclnn 调用链上实测定界并修正用例归属。

- 探针 `aLoopCntTotal=24`（H=1536）与 `=28`（H=1792）verdict FAIL（group 桩毒化输出）（来源平台：950；证据等级：tool_observed）
- 探针 `aLoopCntTotal=29`（H=1856）/30/31/32/40 verdict PASS（base 分支，已实现）（来源平台：950；证据等级：tool_observed）
- 大 shape 用例（fp16，B=1024，H=2048，aLoopCntTotal=32）实跑五路输出 PASS：三路逐元素 `mere=0.000e+00`、两路 bias `mare≤1.178e-03`（预算内）——证实路由 base 而非 group（来源平台：950；证据等级：tool_observed）
- 其余 11 条 group 用例以毒化量级 FAIL（mere 1e+05~1e+35）；修正清单后终验：构建退出码 0、冒烟 1/1、Branch-0 20/20 不回归、Branch-1 `PASSED=0 FAILED=11 ERROR=0 TOTAL=11`（红基线符合预期，run.sh 退出码 255 属该阶段预期）（来源平台：950；证据等级：tool_observed）
- 平台 INI `Ascend950PR_9589.ini` 实读：`vector_core_cnt=64`、`ai_core_cnt=32`、`cube_core_cnt=32`、`ub_size=253952`（来源平台：950；证据等级：tool_observed）
- 全部用例同输入连跑 2 次五路输出位级一致（`determinism OK(bitwise,2runs)`），路由判定可重复（来源平台：950；证据等级：tool_observed）

## 失效边界

- **边界值 28 是本机本次运行的实测值，不可硬编码外推**：CANN 版本升级、容器部署、核预留策略变化都会移动边界——本卡给的是定界方法，不是常数。
- 未验证，推断可迁移：其他 Ascend 平台若同样存在"INI 物理核数 > 运行时可用核数"，方法适用，但边界值须现测。
- 探针法前提：未实现分支为**不写输出**的桩 + 已实现分支正确。若桩会写部分输出、或两分支都有缺陷，FAIL/PASS 二值判定不成立。
- 探针推导假设 `ubSize`/`blockSize`/`cacheLineSize` 与 INI 一致（内存容量类参数可靠，本次核对 `ub_size=253952` 一致）；若某平台这三者运行时也与 INI 不符，`aUbFactor` 钉值推导需重做。
- 谓词阈值不是 `coreNum/2` 形式时，扫描变量与换算式须相应替换。

## 坏实践

- 用 npu-smi 或平台 INI 的 `vector_core_cnt` 代入含 `GetCoreNumAiv()` 的运行时谓词做选例/分支判定——本例因此把一条 base 用例收进了 group 清单，红基线出现假"部分通过"。
- 红基线里看到某用例 PASS 就归因"实现已部分生效"，不查路由。
- 试图经 slog/plog 读自定义算子 host 的 OP_LOGI 拿运行时 tiling 值——实测不可行（见同问题域负结果卡），应在日志路线上止损。
- 把"实测路由不符的用例"静默删掉了事而不登记覆盖缺口——正确做法是移出 + 上报归属裁定。

## 误诊

- 初始假设"运行时 `ubSize` 与 253952 不同导致 aUbFactor 多折半一次（32）"：被排除——对探针 shape，分支判定只依赖 `aLoop` 与 `coreNum/2`，钳制校验 `5×384×4=7680 ≤ 16384` 不触发；INI 复核 `ub_size=253952` 一致。其中排除推导为 agent 分析（非直接实测运行时 UB），如实标注；INI 复读为实测。
- 初始假设"运行时 coreNum=63（单核预留）"：方向正确（运行时核数 < INI）但数值错误——探针显示边界在 28 而非 31，即少 7~8 核而非 1 核。
- 初始把 `ASCEND_GLOBAL_LOG_LEVEL=3` 当作 INFO 级使用：3 实为 ERROR-only；该误诊浪费一轮日志探测（详见负结果卡）。

## 未决事项

- **优先项：复现并追根因到机制层**——`GetCoreNumAiv()` 的确切返回值（56 还是 57）与"哪些核被预留、为何预留"未验证（边界探测只能定出 `coreNum/2=28`）。接续点：写一个能拿到真实 `TilingContext`/平台句柄的探针直接打印该值，或查 CANN 该版本的核预留策略。
- 设计侧"平台参数实据"（coreNum=64 口径）待修订；被移出用例的 ST 覆盖归属待设计侧裁定（本 trace 只登记未裁定）。
- 运行时核数是否随部署形态（容器、多 stream、负载）漂移未验证。
- 未尝试：用 tiling UT harness 以真实平台（非 mock fixture）加载已安装 tiling `.so` 查询平台值。
