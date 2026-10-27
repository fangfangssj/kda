---
type: "Runbook"
title: "GEIR 图执行 EZ9999/95 MTE DDR 越界与挂起"
description: "aclnn 直调路径已通过的算子在 TTK GEIR 图执行路径上浮点用例集中报 EZ9999（error code 95，MTE 指令 DDR 地址越界）、动态 shape 用例挂起时，用 ttk geir --proc-timeout 探针把挂起转成完整 ERROR 结果行再分类判别；TTK 汇总行的 SKIP/NA 计数不可信，判定须读结果 CSV 的 precision_status。"
tags: [task.debugging, topic.graph_execution, topic.memory]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# GEIR 图执行 EZ9999/95 MTE DDR 越界与挂起

## 现象

在 Ascend950PR（CANN 9.2.0）上对自定义算子 BenchHarnessScatter（scatter / scatter-reduce，支持 update/add/multiply/amax/amin）做 GEIR ST 验收时，图执行路径出现了与 aclnn 直调路径完全不同的行为。该算子此前已在 aclnn 黑盒回归中全量通过（项目 LOG.md 记录 399/399 PASS，本会话未复现该回归），kernel 侧含 fp32 workspace 累加路径（workspace 经 GetUserWorkspace 取、host 声明 16MB 保留区 + 用户大小，update 模式不声明 workspace）。验收当天的图路径观测：

- 路由探测二进制 geir_route 的 `GRAPH-RUN` 即失败：`rungraph_rc=4294967295 outputs=0 expected_numel=12 actual_numel=-1 value=FAIL`（probe_exit=1）。kernel 二进制 `te_benchharnessscatter_..._mix_aiv` 能从 staging 正常加载（ParseKernelMetaData/BuildNewKernel 成功），失败发生在执行阶段。
- TTK GEIR 静态 8 用例中，全部 6 个浮点 dtype 用例（f16/f32/bf16，覆盖 update/add/dim=1）EXEC_FAILURE，错误原文（取自结果 CSV 的 precision_status）：
  `ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error!`，扩展信息为 aivec 异常、`error code = 95`、`The DDR address of the MTE instruction is out of range. subErrType: 0x4`。
- TTK GEIR 动态 4 用例（dyn_input_shapes 全为 (-1,-1)）：`dynamic_f16_small` 与 `dynamic_f32_large` 卡在 `OnGeirExecute` 800 秒以上不返回（进度条 `Progress: 0% 0 / 4 ET: 801s`，单用例 `RUNNING 800s`）；`dynamic_f16_large` 与 `dynamic_f32_small` 则在十余秒内以同样的 EZ9999 code 95 快速失败。
- 由于 `run.sh test` 内部调用 `ttk geir` 时未设 `--proc-timeout`，整轮测试 900 秒被外层超时杀死，`dynamic-result.csv` 只剩表头（0 行数据）——挂起用例既不是 SKIP 也没有 FAIL 结果，验收无法对账。

当时的代码状态：验收会话不修改 Kernel/Tiling/OpAPI（out-of-scope 约束），kernel 与 aclnn 验收通过时一致；上述图路径失败按 ST 规则记为"业务断言可红"放行，根因未诊断。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke 流程）中的 GEIR ST 验收与图路径联调排查（`ttk geir` 图执行模式 + geir_route 路由探测）。
- API 路径：`ttk geir -i ttk/<static|dynamic>.csv -c=false -d=<true|false> --plugin <golden.py> --proc-timeout N`；GE `RunGraph` / `InferShapeFuncV2`；结果判定脚本按结果 CSV 的 `precision_status` 字段分类。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0，TTK 3.0.0（ops-test-kit）（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何"aclnn 直调路径通过、GE 图执行路径失败/挂起"的自定义算子排查。
- 路线优先级：safety_fallback（本卡是取证与判别路线，不修复 kernel、不放宽任何断言；只保证挂起用例也能产出完整结果行）。
- 明确不适用 / 未验证场景：
  - aclnn 直调路径上的同类 EZ9999/507035 错误（本会话未复现，不适用本卡图路径结论）。
  - 图路径失败的修复方案（根因未追到机制层，本卡不提供修复配方）。

## 判别方法

**触发**：`./run.sh test`（或直接 `ttk geir`）执行静态+动态图用例时，整轮 900s 超时被杀，或浮点用例集中报 EZ9999 设备错误，而同算子 aclnn 直调用例为通过状态。

**可执行探针**（完整命令，verbatim 取自会话；关键是补上 `--proc-timeout`，把挂起用例转成超时结果行）：

```bash
# 动态用例单独跑，带 proc-timeout（90s 足够区分"慢"与"挂起"：正常/快速失败用例 12s 内完成）
timeout 600 env PYTHONPATH="${TTK_SPEC_DIR}:${TTK}:${ASCEND_HOME_PATH}/python/site-packages" \
  "$PY" -m ttk geir \
  -i "$GEIR_DIR/ttk/dynamic.csv" \
  -c=false -d=true \
  --plugin "$TTK_SPEC_DIR/golden.py" \
  --dev=1 --pc=1 --task-prof=false --run=1 \
  --proc-timeout=90 \
  -o "$RESULT"
```

结果行提取（按 `precision_status` 分类，不采信 TTK 汇总行）：

```bash
"$PY" -c "
import csv
with open('$RESULT') as f:
    for r in csv.DictReader(f):
        ps = r.get('precision_status','')
        print(f\"{r.get('testcase_name','?'):30s} {ps[:60]}\")
"
```

**原始观测**（verbatim）：

1. 动态 4 用例带 `--proc-timeout=90` 的结果行：

```
dynamic_f16_small              ERROR: GEIR execution timed out
dynamic_f16_large              ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph f
dynamic_f32_small              ERROR: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph f
dynamic_f32_large              ERROR: GEIR execution timed out
```

2. EZ9999 完整错误文本（`dynamic_f32_small` 单跑 12 秒即失败）：

```
RuntimeError: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1, error: EZ9999: Inner Error!
EZ9999[PID: 2977333] 2026-09-04-22:11:20.230.335 (EZ9999):  The error from device(chipId:0, dieId:0), serial number is 176, there is an aivec error exception, core id is 0, error code = 95, dump info: pc start: 0x120041000000, current: 0x120041000844, sc error info: 0xffffffffffff, su error info: 0xaaa4be403eba026c,0x7fff7fff18000000, mte error info: 0xdff88c6800020206, vec error info: 0x4c601748003101ee, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x12004cc00000, mte error: 0x80000000, aic cond: 0.
The extend info: errcode:(95) errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4.
        rtStreamSynchronizeWithTimeout execution failed, reason=the model stream execute failed
        GraphManager RunGraph failed, session_id:0 graph_id=0.
```

3. 挂起现场（st-test.log 尾部，无 --proc-timeout 的原始运行）：

```
| Version: 3.0.0 Summary (Device Total: 1) Progress: 0% 0 / 4 ET: 801s Rev: b31db58      |
| NPU:0 Ascend950PR_9579  | 2972480 ynamic_f16_small OnGeirExecute        RUNNING  800s  |
```

4. 路由探测同签名失败（同一失败的旁证）：

```
GRAPH-RUN mode=route case=route-target node=n_route_target gid=1 feed={data:[4,3]/DT_FLOAT,indices:[2,3]/DT_INT32,updates:[2,3]/DT_FLOAT} rungraph_rc=4294967295 outputs=0 expected_numel=12 actual_numel=-1 value=FAIL
```

**解释（判别规则）**：

- 同一 kernel 二进制在 aclnn 直调路径通过（项目 LOG.md 记录）而在图路径浮点用例上触发设备侧 aivec 异常（code 95，MTE 指令 DDR 地址越界）→ 归因方向是 **GE 图执行路径下的 kernel 访存/工作区行为差异**，而不是 aclnn 路径已验证过的算法逻辑本身。
- "挂起 vs 快速失败"与 dtype 无单调关系：同为 f16/f32 的动态用例中 small/large 两档各有挂起与快速失败——说明分界不在 dtype 分支，而在动态 shape 编译产物或 shape 相关的执行路径上，需按用例逐个取证。
- TTK 的汇总行（`TTK Test Summary`）会把 ERROR 结果计入 `SKIP/NA`（如 `SKIP/NA: 4`），但结果 CSV 的 `precision_status` 是 `ERROR: ...`；判定用例是否跳过必须读 CSV 字段（`parse_result.py` 口径：`ERROR*` 或含 `FAIL` → 业务失败；`SKIP`/`SUPPRESSED` → 跳过）。
- `--proc-timeout` 超时产生的 `ERROR: GEIR execution timed out` 行同样计入"已执行、业务失败"，与 EXEC_FAILURE 同类；因此补测后 4/4 动态用例都有结果行、0 SKIP，验收对账才成立。
- 根因未追到机制层：本会话按 ST 规则放行上述失败，未定位图路径下 workspace/tiling/launch 的确切差异（见遗留问题）。

## 设计决策

- 诊断路线选择：受"验收不修改 kernel/tiling/测试文件"约束，放弃"清空 dynamic.csv 把动态标 N/A"的替代方案（会掩盖真实执行信号），改用"直接调用 `ttk geir` 并加 `--proc-timeout`"补齐动态结果行——这是唯一既不改被测物、又能拿到完整证据的路径。
- 探针超时取 90s 的依据：单用例正常执行/快速失败在 12s 内完成，90s 足以把"真挂起"转成超时行而不会误杀慢用例。
- 当前方案属 safety_fallback 档：只保证取证完整，不宣称任何修复；相对代价是补测一轮动态用例约多花 4×90s（挂起用例吃满超时）。

## 系统信息

- 硬件/软件：Ascend950PR_9579 单卡（device 0），CANN 9.2.0，TTK 3.0.0（ops-test-kit）。
- 算子包布局：`operators/BenchHarnessScatter_package/tests/geir/`（`run.sh`、`ttk/static.csv` 8 用例、`ttk/dynamic.csv` 4 用例、`cases.tsv` 23 用例清单、`build/` 下 geir_route/geir_suite/geir_infer 二进制）。
- 关键产物文件：`build/ttk/static-result.csv`、`build/ttk/dynamic-result.csv`（结果 CSV，`precision_status` 字段为判别源）、staging vendor `/tmp/geir_opp.dcYUEp/vendors/BenchHarnessScatter`。
- 相关日志：`st-route.raw.log`（路由探测原始日志）、`st-test.log`（挂起现场）、`st-dynamic.log`（补测日志）、`docs/develop/GEIR_VERIFY.md`（验收报告）。

## 完整代码

（诊断卡：核心 artifact 为「判别方法」中的探针命令与原始观测，无 kernel 修复代码；kernel 侧根因代码未在本会话定位。）

## 预防

- 图执行路径验收/联调前，给所有 `ttk geir` 调用默认加 `--proc-timeout`（run.sh 脚本未内置该参数），避免单个挂起用例拖死整轮、结果 CSV 0 行无法对账。
- 把"aclnn 直调通过 ≠ 图路径通过"当作默认假设：两条路径的 kernel 启动与内存布局不同，需分别取证后再下结论。

## 验证信号

以下信号来自该算子的 GEIR ST 验收会话：路由探测 + TTK GEIR 静态全量执行 + 带 `--proc-timeout=90` 的动态补测，结果取自结果 CSV 与原始日志。

- 路由探测 GRAPH-RUN：`rungraph_rc=4294967295 outputs=0 value=FAIL`（probe_exit=1；kernel 二进制从 staging 正常加载后执行失败）（来源平台：950；证据等级：tool_observed）。
- TTK GEIR 静态 8 用例：6 个浮点用例全部 EXEC_FAILURE（EZ9999，RunGraph failed -1）；`parse_result.py` 判定 `RESULT total=8 pass=0 fail=8 skip=0`（来源平台：950；证据等级：tool_observed）。
- TTK GEIR 动态 4 用例（--proc-timeout=90 补测）：dynamic_f16_small、dynamic_f32_large 超时（`ERROR: GEIR execution timed out`），dynamic_f16_large、dynamic_f32_small EZ9999 EXEC_FAILURE；`RESULT total=4 pass=0 fail=4 skip=0`（来源平台：950；证据等级：tool_observed）。
- EZ9999 扩展信息：`errcode:(95) errorStr: The DDR address of the MTE instruction is out of range. subErrType: 0x4`，aivec 异常 core id 0，伴随 `rtStreamSynchronizeWithTimeout execution failed`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 Ascend950PR + CANN 9.2.0 + TTK 3.0.0 的图执行模式实测；其他架构/版本未验证（未验证，推断可迁移：`--proc-timeout` 是 TTK 通用参数，EZ9999 code 95 是设备侧通用错误语义）。
- `--proc-timeout` 只把挂起转成超时结果行，不解决挂起本身；修复需另行定位图路径 kernel 行为（本卡不含修复）。
- 超时阈值 90s 基于本算子正常用例 ≤12s 的观测；对慢算子需按实际耗时放宽，否则会把慢用例误判为挂起。
- 本卡判别的是图路径执行失败签名；aclnn 直调路径的同类 EZ9999 错误不适用本卡结论。

**未决项**：

- **优先项：复现并追根因到机制层**——图路径下浮点用例 MTE DDR 越界（code 95）的确切原因（workspace 布局 / launch 参数 / 动态编译产物差异）未定位；dynamic_f16_small、f32_large 挂起与 dynamic_f16_large、f32_small 快速失败的分界机制未验证。下一个接续点：对挂起用例抓 plog/device dump，对照 aclnn 路径同 shape 的 workspace 与 launch 参数。
- ST 验收规则允许业务断言为红，本会话据此放行；若实现阶段要求图路径全绿，上述失败必须逐个闭环。
- `ttk kernel` 直调模式 VEC_ERROR（LOG.md 记录）与图路径 EZ9999 是否同根因，未验证。
## 坏实践

- 不带 `--proc-timeout` 直接跑含动态用例的 `ttk geir` 全量套件——一个挂起用例拖死整轮，`dynamic-result.csv` 0 行，SKIP/FAIL 均无法判定。
- 用 TTK 汇总行的 `SKIP/NA` 计数当"用例被跳过"的依据（它把 EXEC_FAILURE/超时也计入 SKIP/NA；判跳过必须读结果 CSV 的 `precision_status`）。
- 挂起的动态用例既无 SKIP 也无 FAIL 结果行时，直接按"已执行"对账放行——本会话是通过补测拿到结果行后才放行的。

## 误诊

- 本会话未对 EZ9999 给出任何归因结论（根因明确未追）；下述为归因背景转述。
- 对照背景（hearsay，agent_reported 转述，非本会话实测）：项目 LOG.md 记录同算子 aclnn 黑盒回归 399/399 PASS、`ttk kernel` 直调模式全 shape VEC_ERROR——仅作归因背景，不作为实测信号。
