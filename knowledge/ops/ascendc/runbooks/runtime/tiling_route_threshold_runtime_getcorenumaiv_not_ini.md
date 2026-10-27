---
type: "Runbook"
title: GetCoreNumAiv 实测≈56-57，group 路由分界 28
description: host tiling 分支路由按运行时 GetCoreNumAiv() 而非平台 INI 核数整除判定；实机 group 进入分界 aLoopCntTotal≤28（coreNum/2，非按规格 64 核算的 32），用桩/实现不对称的二值边界探针实机定位。 R2（2026-09，CANN 9.1.0/9.2.0）：路由触发还含 R 维条件（rLoopCntTotal>1，R 超单个 UB chunk，实测翻转点 R=512–1024）；设计矩阵/用例注释的静态 key 标注可与真机路由全不符——红基线意外 PASS＝用例被路由到已实现分支，覆盖计数必须按运行时路由，配哨兵探针实测。
tags: [task.testing, topic.tiling, topic.architecture]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# GetCoreNumAiv 实测≈56-57，group 路由分界 28

## 现象

在一个对标 torch GRU Cell 反向的融合算子仓（aclnn 接口，host tiling 按 tilingKey 分发 base/group/empty 三路模板）做 group 分支的 Kernel ST 验收时遇到的问题。该算子处于"测试先行"阶段：group 模板 kernel 还是占位 stub（入口可编译但 ProcessGroup 全核早退、零计算零 IO），base 模板已实现并通过 20 条 ST。验收任务要求核对 ST 清单的分支覆盖，并抽查执行者的一条关键报告：性能关键 shape 用例 N2（fp16，B=1024，H=2048）在实机运行时路由到 base 而非 group，因此已从 group 分支的 ST 清单中剔除。

这条报告的依据是一个平台差异：仓库的平台实据记录（npu-smi + 平台 INI）写着 vector_core_cnt=64（物理核数）。host tiling 的分支判定是

```cpp
bool ShouldUseGroup(const ThnnFusedGruCellGradCtx& ctx)
{
    return (ctx.aLoopCntTotal <= ctx.coreNum / 2) && (ctx.rLoopCntTotal >= 2);
}
```

其中 `coreNum` 来自运行时 `ascendcPlatform.GetCoreNumAiv()`。按 64 核推算，分界是 aLoopCntTotal ≤ 32；N2 经 cacheBuf 钳制后 aUbFactor=64、aLoopCntTotal=CeilDiv(2048,64)=32 ≤ 32，应落 group。但执行者报告实机 GetCoreNumAiv() 只有约 56-57，整除后 coreNum/2=28，于是 32 > 28，N2 实际落 base。剔除处理是否如实、探针证据是否可信，直接决定这份 ST 清单的分支覆盖是否成立——如果分界其实在 32，N2 就该留在 group 清单里。

验收方没有引用执行者的日志，而是自己重打了一遍探针：自建一个 5 例探针清单（B=1024 fp32、aUbFactor=64 下 aLoopCntTotal 跨 28/29 分界的四个 H 点 + N2 本体），用独立 oracle golden.py 生成期望值，直接用已安装 vendor 包的 test_aclnn 二进制实机跑。结果是 aLoopCntTotal=24/28 的两例 FAIL（路由 group，stub 不写输出、毒值保持）、=29/32 的两例 PASS（路由 base）、N2 本体 PASS（base），两个套件槽位结果逐条一致——分界被钉死在 28/29 之间，执行者的剔除处理被独立复证。

## 适用场景

- 工程形态：aclnn 算子仓（ops-registry-invoke 流程）ST 用例的分支归属判定与验收；白盒用例安全盒声明；host tiling 多模板分支路由的预判与复核。
- API 路径：op_host tiling 中 `platform_ascendc::PlatformAscendC::GetCoreNumAiv()` / `GetCoreMemSize(UB)`；ST harness（test_aclnn 双 manifest 槽位 + gen_st_oracle.py）。
- 验证过的平台与 CANN 版本：Ascend 950（ascend950 构建目标、实机 aclnn 运行）+ CANN 9.0.0-beta.2（与 frontmatter platforms 一致）。
- 适用粒度（transfer_scope）：operator_family——凡 host tiling 用 GetCoreNumAiv() 做分支/切分判定的算子（探针方法依赖"被验分支为桩 + 对照分支已实现"的不对称，跨算子族可迁移但需各自具备该前提）。
- 路线优先级：recommended_fastpath（先实测路由分界再定用例归属；不按规格核数静态预判）。
- 明确不适用/未验证场景：单模板算子（无分支路由）；被验分支已实现后 FAIL 不再等价于"路由到该分支"（信号消失，需改用日志/tiling 输出判定）；其他芯片/机器/CANN 版本未验证（须用本卡探针法重测）。

## 根因

host tiling 的 base/group 路由判定中 `coreNum = GetCoreNumAiv()` 的运行时返回值（实测反推 56-57）小于平台规格值（INI vector_core_cnt=64），使实际路由分界 coreNum/2=28 比按规格计算的 32 更紧。

机制推导（不跳步）：

- 判定式：`aLoopCntTotal <= coreNum/2 && rLoopCntTotal >= 2`，C++ 整数除法向下取整：coreNum=56 → 56/2=28；coreNum=57 → 57/2=28。两者分界同为 28。
- 探针二值观测：aLoop=24/28 → FAIL（group 桩不写输出）；aLoop=29/32 → PASS（base 已实现、数值正确）⟹ 28 ≤ coreNum/2 < 29 ⟹ 56 ≤ coreNum ≤ 57。
- 既有套件旁证（同一次官方运行）：P2（aLoop=16）FAIL ⟹ 16 ≤ 分界；B0-a（aLoop=34）、B0-c（aLoop=65）PASS ⟹ 分界 < 34。官方用例只能把分界夹在 [16,34)，不足以判定 aLoop=32（N2）的归属——这正是必须打跨界探针的原因。
- 后果量化：按 64 核口径预判会把 aLoop∈[29,32] 的用例错归 group（N2 的 aLoop=32 即例：64 核口径 32≤32 → group；实机 32>28 → base）。

## 判别方法（边界二分探针）

前提检查（信号可用性）：被验收分支 kernel 为桩（不写输出），对照分支已实现且全绿——桩分支路由 → 输出保持预填毒值 → verdict FAIL（mere/mare 达毒值量级、nan_inf_mismatch=0、determinism OK）；对照分支路由 → PASS。

1. 固定 B 使路由条件化简为单变量：rLoopCntTotal ≥ 2 且 R 不全驻（R 全驻会触发扩 A、改变 aUbFactor）。本例 B=1024 fp32：rUbFactor=48 → rLoopCntTotal=CeilDiv(1024,48)=22 ≥ 2。
2. 按 aUbFactor 换算 H 使 aLoopCntTotal=CeilDiv(H,aUbFactor) 扫过预期分界两侧：H=1536/1792/1856/2048 → aLoop=24/28/29/32。
3. 写临时探针 manifest（与 ST 清单同 schema：case_id/B/H/dtype/has_bias/inputs/assert_mode/threshold），gen_st_oracle.py 生成数据（--golden 用绝对路径），test_aclnn 双槽实跑（同一 manifest 填 Branch-0/Branch-1 两槽，输出重复两遍作内部一致性校验）：

```bash
cd operators/<pkg>/tests/aclnn && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 gen_st_oracle.py \
    --manifest /tmp/probe_manifest.json --out /tmp/probe_data \
    --golden /abs/path/to/repo/docs/<op>/develop/golden.py
cd operators/<pkg>/tests/aclnn/build && \
    export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/<Vendor>/op_api/lib/:${LD_LIBRARY_PATH} && \
    ./test_aclnn /tmp/probe_manifest.json /tmp/probe_data /tmp/probe_manifest.json /tmp/probe_data
```

4. 判读：FAIL=路由桩分支、PASS=路由对照分支；分界 = 最后一个 FAIL 的 aLoop 值；反推 coreNum ∈ {2×分界, 2×分界+1}。
5. 交叉核验：与官方清单既有用例的路由旁证对账（本例 [16,34) 框与探针 28 一致）。

## 设计决策

- 为何以 PASS/FAIL 二值信号而非读 tiling 输出/加日志：验收会话不改产品代码；桩/实现不对称天然提供可观测差异；双槽重复跑提供内部一致性校验。
- 为何固定 B=1024：使 rLoopCntTotal=22 ≥ 2（双条件化简为单变量）且 R 不全驻（避开 Step-3 扩 A 改变 aUbFactor 的干扰）。
- 放弃的候选：在 host tiling 里加 OP_LOGE 打点打印 coreNum（要改产品代码并重编，验收侧不该做）；按 npu-smi/INI 规格值静态推算（正是被证伪的口径）。
- 当前方案属哪档路线：recommended_fastpath。
- 同类漂移备注：实机 ub=253952 vs 设计文档典型 256KB（rUbFactor 195 vs 201、尾块 rLen 122 vs 110）——本例未改变 rLoopCntTotal=3 与路由结论，但手算切分时应一律用运行时接口值复算。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（ASCEND_OPP_PATH=<home>/Ascend/cann-9.0.0-beta.2/opp），GCC 13.3.0，python3 3.12.9 + torch 2.7.1+cpu（数据生成侧）。
- 目标平台：ascend950（构建产物 aic-ascend950-ops-info.json、kernel binary ascend950/；实机 ACLNN 运行）。
- 目录布局：operators/<pkg>/{op_host/arch35, op_kernel/arch35, tests/aclnn}；docs/<op>/develop/{golden.py, LOG.md, STATE.md}。
- 相关文件（与事件流实际读写一一对应）：op_host/arch35/<op>_tiling_arch35.cpp/.h（路由判定与平台参数查询）、op_kernel/arch35/<op>_group.h（stub）、tests/aclnn/{st_cases_branch1.json, run.sh, test_aclnn.cpp, gen_st_oracle.py}、docs/<op>/develop/golden.py（独立 oracle）、/tmp 临时探针清单与数据。

## 完整代码

路由判定与 A 向切分（op_host/arch35 tiling，verbatim）：

```cpp
void ComputeFusedALoopSplit(ThnnFusedGruCellGradCtx& ctx)
{
    ctx.aSplitChunkCnt = CeilDiv(ctx.H, ctx.aUbFactor);
    ctx.aLoopCntTotal = ctx.aSplitChunkCnt;  // outerAProd = 1（前导 A=1 不贡献）
    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;
    ctx.aBigCoreCnt = ctx.aLoopCntTotal % ctx.coreNum;  // 前 aBigCoreCnt 核多处理 1 块
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + ((ctx.aBigCoreCnt > 0) ? 1 : 0);
    ctx.usedCoreNum = (ctx.aSmallCoreLoopCnt > 0) ? ctx.coreNum : ctx.aBigCoreCnt;
    ctx.usedCoreNum = std::max(ctx.usedCoreNum, int64_t(1));  // blockIdx ≥ usedCoreNum：idle
}

// ─── Group 触发判定（§6.1；BranchRoute.md §2 触发条件逐字一致）───
bool ShouldUseGroup(const ThnnFusedGruCellGradCtx& ctx)
{
    return (ctx.aLoopCntTotal <= ctx.coreNum / 2) && (ctx.rLoopCntTotal >= 2);
}
```

平台参数来源（op_host/arch35 tiling，verbatim；注意硬件参数全走 platform 接口、不写死）：

```cpp
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
```

探针清单（/tmp 临时产物，verbatim 全文）：

```json
{
  "schema_version": 1,
  "op_name": "ThnnFusedGruCellGrad",
  "api_name": "aclnnThnnFusedGruCellGrad",
  "task": "kernel-verifier 独立探针（Task 28.2.3 验收抽查）：N2 运行时路由 base 的执行者报告复核——B=1024 fp32 aUbFactor=64 下 aLoopCntTotal 跨 28/29 分界的二值判定 + N2 本体（fp16）",
  "probe_note": "group kernel 现为接口桩（Task 28.2.4 未实现）：路由 group → 输出保持 poison → FAIL；路由 base → Branch-0 kernel 已实现 → PASS。期望（执行者报告口径）：PRB-G24(aLoop=24)/PRB-G28(aLoop=28) FAIL(group)、PRB-B29(aLoop=29)/PRB-B32(aLoop=32) PASS(base)、PRB-N2(fp16,B=1024,H=2048,经 cacheBuf 钳制 aUbFactor=64→aLoop=32) PASS(base)。",
  "golden": {
    "path": "docs/ThnnFusedGruCellGrad/develop/golden.py",
    "independence": "期望值由独立参考 oracle golden.py（纯 torch/numpy）生成，不 import 被测 kernel 任何产物"
  },
  "cases": [
    {
      "case_id": "PRB-G24",
      "source": "kernel-verifier probe: B=1024 fp32 H=1536 → aUbFactor=64, aLoopCntTotal=24（执行者报告：24 落 group）",
      "B": 1024, "H": 1536, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    },
    {
      "case_id": "PRB-G28",
      "source": "kernel-verifier probe: B=1024 fp32 H=1792 → aUbFactor=64, aLoopCntTotal=28（执行者报告：28 落 group）",
      "B": 1024, "H": 1792, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    },
    {
      "case_id": "PRB-B29",
      "source": "kernel-verifier probe: B=1024 fp32 H=1856 → aUbFactor=64, aLoopCntTotal=29（执行者报告：29 落 base）",
      "B": 1024, "H": 1856, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    },
    {
      "case_id": "PRB-B32",
      "source": "kernel-verifier probe: B=1024 fp32 H=2048 → aUbFactor=64, aLoopCntTotal=32（执行者报告：32 落 base）",
      "B": 1024, "H": 2048, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    },
    {
      "case_id": "PRB-N2",
      "source": "kernel-verifier probe: N2 本体 B=1024 fp16 H=2048（Validation.md N2 行；执行者报告：经 cacheBuf 钳制 aUbFactor 128→64 → aLoopCntTotal=32 实机落 base，剔出 branch-1 manifest）",
      "B": 1024, "H": 2048, "dtype": "float16", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-3
    }
  ]
}
```

探针实跑结果（verbatim，判读行；两槽重复一致，PROBE_EXIT=255 因含 FAIL 属预期）：

```text
[ST] PRB-G24 fp32 B=1024 H=1536 bias=1 …
     determinism OK(bitwise,2runs) | verdict FAIL
[ST] PRB-G28 fp32 B=1024 H=1792 bias=1 …
     determinism OK(bitwise,2runs) | verdict FAIL
[ST] PRB-B29 fp32 B=1024 H=1856 bias=1 …
     determinism OK(bitwise,2runs) | verdict PASS
[ST] PRB-B32 fp32 B=1024 H=2048 bias=1 …
     determinism OK(bitwise,2runs) | verdict PASS
[ST] PRB-N2 fp16 B=1024 H=2048 bias=1 …
     determinism OK(bitwise,2runs) | verdict PASS
```

## 预防

- 设计/验收 ST 或白盒用例的分支归属前，先实测（或复用本卡探针法确认过的）运行时路由分界，不要用 npu-smi/INI 物理核数推算 coreNum/2。
- 手算切分/路由一律用运行时接口值（GetCoreNumAiv/GetCoreMemSize），文档典型值仅作参考并标注差异。
- 清单中"base/group（HostTiling 实算）"类机器依赖行，注明实测口径与探针证据；换平台后重测。

## 验证信号

来源说明：以下信号来自对该 group 分支 Kernel ST 的一次独立验收会话——验收者不引用执行者日志，自建边界探针清单、用独立 oracle 生成期望值、在实机上用已安装 vendor 包的测试二进制重跑，全部为本会话内的命令输出。

- 探针二值判定：aLoopCntTotal=24/28 → verdict FAIL（group 路由），29/32 → verdict PASS（base 路由），两槽重复一致（来源平台：950；证据等级：tool_observed）
- N2 本体（fp16 B=1024 H=2048，钳制后 aLoopCntTotal=32）→ verdict PASS（base）（来源平台：950；证据等级：tool_observed）
- 官方 ST 运行旁证：P2（aLoop=16）FAIL、B0-a（aLoop=34）与 B0-c（aLoop=65）PASS ⟹ 16 ≤ 分界 < 34，与探针钉死的 28 一致（来源平台：950；证据等级：tool_observed）
- 红基线：Branch-1 0/11 全 FAIL 且 mere/mare 达毒值量级（如 4.376e+33 / 1.330e+37）、ERROR=0、determinism OK——FAIL 确系"桩未写输出"而非框架错误，二值信号有效（来源平台：950；证据等级：tool_observed）
- 构建退出码 0、目标 ascend950（来源平台：950；证据等级：tool_observed）

- （R2 增补，Ascend950PR + CANN 9.1.0，EuclideanNorm group 分支 ST 设计）设计矩阵 8 个原始 shape 探针实测全部路由到已实现分支（key=0/1），无一命中 key=4；`[2,R,3] axes=[1]` R 扫描 R≤512 全 normal、R≥1024 全 group，翻转点在 512–1024 之间（来源平台：950；证据等级：tool_observed）。
- （R2 增补）shape 适配后 7 个用例探针实测全部命中 key=4（G2 反向用例正确保持已实现路由）；ST 基线（kernel 未接线）Branch-4 0/8 全失败、dev 恒为 0x41 哨兵位型（证明用例确实走到 key=4）；kernel 接线后同 8 用例 8/8 PASSED（fp32 maxRel≤4.038e-07）（来源平台：950；证据等级：tool_observed）。
- （R2 增补，softmax GroupTailA 红基线）`#421 groupa_copyout_p3 [64,4096,4] dim=1`（用例注释标注 key=4）stub 期三 dtype [PASS]、kernel 实现后仍 [PASS] 无变化；同期 31 条 key=4 用例 361001 失败——意外 PASS＝被运行时路由到已实现 key=1（来源平台：950；证据等级：tool_observed）。
- （R2 增补）白盒产物字段佐证：group 用例 `dyn_tiling_key=4`、`dyn_block_dim=56` 可从 result.csv 直接读出（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 分界 28 是单机单会话实测，绑定该机器运行时 GetCoreNumAiv() 返回值；不同机器/CANN 版本/容器核隔离可能不同，迁移必须用本卡探针法重测。
- 56-57 是由分界反推的区间，GetCoreNumAiv() 精确返回值未直接观测（未加日志打点）。
- FAIL=group 信号仅在"被验分支为桩 + 对照分支已实现且正确"时成立；被验分支实现后信号消失。
- 前提 rLoopCntTotal ≥ 2：B 须使 R 不全驻（R 全驻 → rLoopCntTotal=1 → 恒 base，且触发扩 A 改变 aUbFactor）。
- 未验证，推断可迁移：同机同 CANN 下其他用 GetCoreNumAiv() 做切分/分支判定的算子（核数漂移机制上是平台级现象，但未在其他算子实测）。

- （R2 增补）R 维翻转点（≈1024）绑定该算子的 UB 因子推导与本机平台参数；跨算子/跨平台必须重扫，只能照搬方法不能照搬数值。
- （R2 增补）探针判据依赖"已实现 vs stub"的输出差异；目标分支实现后信号消失，且两个分支都已实现时无法区分路由归属（需 tilingKey 级观测）。
- （R2 增补）"路由到已实现 key"的结论由反证+触发公式推导支撑（强推断）；softmax `#421` 具体哪个条件不满足未与实测 tilingKey 对账闭合，以 dyn_tiling_key 实测为准。

## 坏实践

- 按平台规格/INI 物理核数静态预判路由归属并直接定用例清单——本例被证伪的做法（会把 aLoop∈[29,32] 的用例错归 group）。
- 设计期手算切分用文档典型值（64 核、256KB UB）不复算实机接口值（同类：rUbFactor 201→195）。

- （R2 增补）照抄设计文档/评审矩阵的 shape 写多分支 ST 而不实测路由；按用例注释手写的"预期 key"统计分支覆盖——注释不经过运行时，coreNum/UB 预算变了路由就变。
- （R2 增补）在 host 侧探针里花时间找 aclrt 运行时核数查询接口——该 CANN 版本的 aclrtGetDeviceCapability 枚举没有核数项。

## 误诊

- "Branch-1 11/11 FAIL 是测试框架坏了"——实际是 TDD 红基线的预期数值红（毒值保持、ERROR=0、determinism OK）。
- "运行日志 grep 到 2 处 ERROR 说明有框架错误"——实为汇总行 "ERROR=0" 字样。
- 执行者报告的"GetCoreNumAiv()≈56-57"在本次验收前仅是其单方陈述；本卡的分界 28 为验收方独立探针实测反推（执行者口径已被复证，但前序日志本身不作为本卡证据）。

## 增补：R 维触发、设计矩阵落空与红基线意外 PASS（R2，2026-09）

两个不同算子（softmax 类 BenchHarnessSoftmax、归约 EuclideanNorm）独立复现同一规律：tilingKey 路由不只由 shape 模式决定，还依赖运行时平台参数（coreNum、ubSize 推出的 UB 因子），静态预期（设计文档用例矩阵、用例注释里的"预期 key"）与真机路由可以全部不符。

**R 维触发条件**：group 触发除 A 侧判定（本卡 aLoopCntTotal ≤ coreNum/2；softmax 实例为 aLoopCntTotal < coreNum——公式随算子而异，运行时参数依赖不变）外还要求 `rLoopCntTotal > 1`——R 维超过单个 UB chunk（由 ubSize 推出的 rUbFactor 决定）。对 `[2,R,3] axes=[1]`（ARA、tail-A）扫 R：R≤512 全部路由 normal（key=0），R≥1024 全部路由 group（key=4），翻转点在 512–1024 之间；设计矩阵里的小 R（8/17/64）在一个 UB chunk 内（rLoopCntTotal=1）永不触发 group。

**设计矩阵落空**：设计文档给的 8 个 group 用例 shape 在真机全部路由到已实现分支（key=0/1）——照抄写 ST 将永不执行目标分支，"全绿"毫无含金量。shape 适配原则（保持语义意图、放大触发维度）：`[2,3,4] ax=[1,2]`→`[2,1024,3] ax=[1]`（原形合轴后是 tail-R，改单轴归约+大 R）；`[1,64,1]`→`[1,4096,2]`（尾部 A 必须 >1，否则掉成 tail-R）；`[2,17,3]`→`[2,1045,3]`（保非对齐同时 R 超单 chunk）；反向用例（验证偶数 axisNum 不误触发）保留原样 DEFERRED。

**红基线意外 PASS 的反证法**：no-op stub 不可能产出通过精度比对的输出——红基线（新分支 kernel 为 stub）中"本应失败却 PASS"的用例只可能是被运行时路由到了已实现分支；kernel 实现后该用例结果不变进一步确认与新增代码无关。红基线的"本分支期望通过数"必须按运行时路由统计，不能按用例注释分组。终极确认：TTK 白盒产物的 `dyn_tiling_key` 字段直接给出实际命中 key（group 用例 `dyn_block_dim=56` 即 coreNum 补满后的真实核数）。

**独立哨兵路由探针**（aclnn 直连，与上文 manifest 探针同判据）：输出 buffer `aclrtMemset` 0xFF 哨兵 → 跑 `aclnn<Op>GetWorkspaceSize + aclnn<Op>` → 与独立 CPU golden 比对——匹配 ⇒ 路由到已实现分支；输出保持哨兵 ⇒ 路由到 stub 分支。判据无需读 tilingKey（aclnn 接口不暴露），顺带验证 golden 复刻正确性；独立编译（g++ 直链 op_api/libascendcl/libnnopbase），不入算子包。

**host 侧运行时平台查询是死胡同**：`aclrtGetDeviceName` 不存在（编译错）；`aclrtGetDeviceCapability` 的 `aclrtDevFeatureType` 枚举仅 2 个 feature flag（TSCPU_TASK_UPDATE_SUPPORT_AIC_AIV=1、SYSTEM_MEMQ_EVENT_CROSS_DEV=21），查不到核数——ST oracle 改用与 TilingUT 一致的固定平台常数（coreNum=8、ubSize=253952），以探针实测路由互证。
