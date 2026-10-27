---
type: "Runbook"
title: "te_fusion 未调 TilingPrepare 致 GEIR FPE"
description: "GEIR 图测试全 case 返回 GE error 4294967295 并抛 Python FPE；根因是 GE te_fusion/build_single_op 编译子路径不调 TilingParse<TilingPrepare> 填 compileInfo，tiling 函数读 compileInfo=0 除零。用 fprintf 探针判别三条 run 路径的 TilingPrepare 调用语义。"
tags: ["task.debugging", "topic.compilation", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-18T07:12:38Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["GetPlatformInfo"]
---

# te_fusion 未调 TilingPrepare 致 GEIR FPE

## 现象

在 Ascend950 / CANN 9.1.0 上对迁移自 ops-math 的逐元素算子 Xdivy（ops-registry-invoke 算子仓）跑 GEIR 图测试（`tests/geir/run.sh` → `./build/test_geir`，`session->RunGraph` dry-run 编译）时，全部 case 返回 `GE error 4294967295`，无算子层报错。同一环境里 `add_custom_package` 的 GEIR 测试 56/56 全部 PASSED、0 个 Python 异常，对照证明失败与 Xdivy 算子侧相关，而非通用环境问题。

重跑 Xdivy GEIR 捕获完整异常，得到 Python 浮点异常栈：

```
Fatal Python error: Floating-point exception
  File ".../asc_op_compile_base/asc_op_compiler/op_tiling.py", line 380 in do_op_tiling
  File ".../asc_op_compile_base/asc_op_compiler/get_op_tiling.py", line 1450 in get_tiling_info_v2
  ...
  File ".../opp/vendors/Xdivy/op_impl/ai_core/tbe/Xdivy_impl/dynamic/xdivy_apt.py", line 248 in xdivy
```

调用链是：GE 编译图 → FE → TBE → vendor 安装的 `xdivy_apt.py:248 xdivy()` → `op_tiling.py:380 do_op_tiling()` → 通过 ctypes 调 C++ `TilingFuncXdivy`（在 `libcust_optiling.so`）→ 除以零。Xdivy 的 tiling 在 `DoTilingAndSet`/`MultiCoreSplit` 里有 `total_tiles / num_cores`、`ub_per_core / phys_nodes_` 之类的整数除法；当 `compileInfo->ubSize`、`compileInfo->coreNum` 为 0 时即触发 FPE。

## 适用场景

- **工程形态**：ops-registry-invoke 算子仓开发 + GEIR 图测试（`session->RunGraph` dry-run 编译）。
- **API 路径**：`IMPL_OP_OPTILING(Xdivy).Tiling(TilingFuncXdivy).TilingParse<XdivyCompileInfo>(TilingPrepareForXdivy)` 注册链；tiling 函数内 `ctx_->GetCompileInfo()` 读 `ubSize`/`coreNum` 做整数除法。
- **验证过的平台与 CANN 版本**：Ascend950、CANN 9.1.0（vendor 配置 `aic-ascend950-ops-info.json`，coreNum=56，ubSize=253952）。
- **适用粒度**：`operator_family`——任何用 `TilingParse<XxxCompileInfo>` 注册且 tiling 函数读 compileInfo 做整数除法的逐元素/broadcast 算子，迁移到 ops-registry-invoke 仓跑 GEIR 时都可能踩到。
- **路线优先级**：`recommended_fastpath`——诊断成立后用干净重建使 GE 走 PreRun 复用预编译 kernel `.o`，是首选处置方向；不要靠放宽测试或改测试期望绕过。
- **明确不适用/未验证场景**：未在 a2/a3 上验证；纯 aclnn 直调路径（aclnnXdivy）会调用 TilingPrepare，不在本卡判别范围内（仅作对照）。

## 判别方法

判别三条 run 路径是否调用 `TilingPrepareForXdivy`，定位 FPE 属于哪条 GE 编译子路径。

**触发条件**：用 `TilingParse<XxxCompileInfo>(XxxTilingPrepare)` 注册、tiling 函数读 compileInfo 做整数除法的算子，在 ops-registry-invoke 仓跑 `session->RunGraph` GEIR 图测试时，全部 case 返回 `GE error 4294967295` 并抛 `Fatal Python error: Floating-point exception` at `op_tiling.py:380 do_op_tiling`。

**可执行探针**：在 `TilingPrepareForXdivy` 入口加 `fprintf(stderr, ...)` 直打平台信息（`OP_LOGI` 会被 SLOG 默认过滤，看不到，必须用 `fprintf(stderr, ...)`），重新 build+install 后分别跑三条路径并 grep 探针输出与 GE 编译路径行：

```bash
# ACLNN E2E（对照）
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/Xdivy/op_api/lib/:${LD_LIBRARY_PATH}
./tests/aclnn/build/test_aclnn 2>&1 | grep -E "TilingPrepareForXdivy CALLED|PASSED|FAILED"

# GEIR 图测试 + GE 编译路径判别
cd tests/geir && ./build/test_geir 2>&1 \
  | grep -E "TilingPrepareForXdivy CALLED|PASSED|FAILED|GE error|Floating"

# GE 日志路径分流判定（PreRun vs te_fusion/build_single_op）
grep -iE "PreRun|build_single_op|fusion_manager|graph_manager" /tmp/geir_check.log | head
```

> 注：fprintf 探针的 C++ 源码（对 `TilingPrepareForXdivy` 入口的 edit）未在本 trace 逐字恢复（markdown_transcript 适配器对 edit/file_write 事件记为占位）；上方命令体与 grep 模式取自 trace 内 verbatim 工具输出与 agent 复述，探针已被实际执行且输出版本可核（见原始观测）。

**原始观测**：

| run 路径 | 是否调用 TilingPrepareForXdivy | compileInfo 状态 | tiling 函数是否被调 | 现象 |
|---|---|---|---|---|
| **ACLNN**（`aclnnXdivy`） | 调用（探针打印 `TilingPrepareForXdivy CALLED coreNum=56 ubSize=253952`） | 有值 | 调用、读 compileInfo 正常 | 16/16 PASSED |
| **GEIR 干净 build**（`session->RunGraph`） | 不调用（探针无输出） | 0，但不影响 | 不调用 tiling，走预编译 kernel `.o` | 12/12 PASSED |
| **GEIR 触发 te_fusion/build_single_op** | 不调用 | 0 | `compile_op`→`do_op_tiling` 调 tiling、读 compileInfo=0 | FPE / GE error 4294967295 |

GE 日志路径分流：干净 build 走 `PreRun`（`graph_manager.cc:1335` 附近）；FPE 失败时走 `te_fusion/fusion_manager.py:build_single_op`。

**解释**（agent 根因分析）：GE 对 vendor 算子有两条编译路径——(1) **PreRun 路径**（GE native）：直接复用 `build.sh` 阶段已预编译好的 kernel binary（`.o`），不触发运行时 tiling，因此**不调用** `TilingParse<TilingPrepare>`，compileInfo 即使为 0 也不进 tiling 除法 → GEIR 干净重建后 12/12 PASSED 且探针无输出；(2) **te_fusion/build_single_op 路径**（Python TBE 编译）：`xdivy_apt.py:248` 的 `compile_op` → `op_tiling.py:380 do_op_tiling` → ctypes 调 C++ `TilingFuncXdivy`，该子路径**不会**先调 `TilingPrepareForXdivy` 填充 compileInfo，于是 tiling 函数读到 `compileInfo->ubSize=0` / `compileInfo->coreNum=0`，在整数除法处除以零 → Python `Floating-point exception` → GE 返回 `4294967295`。即 `TilingParse<XxxCompileInfo>(XxxTilingPrepare)` 注册的 prepare 回调只在 aclnn 运行时路径被调用，GE 编译 dry-run 的 te_fusion 子路径并不会调用它来填充 compileInfo。FPE 表面是 tiling 除零，实质是"tiling 函数假定 compileInfo 已被 prepare 填充"这一前提在 te_fusion 编译子路径不成立。触发 te_fusion 而非 PreRun 的诱因是 build/install 状态不干净（残留旧 `.o`/旧 `.so` 与新源码不一致时 GE 退回 Python TBE 重编译）。compileInfo=0 这一中间环节为推断（tiling 函数体 verbatim 未从 trace 恢复，无法逐行核验除零确切行），但 FPE 栈与探针输出已 tool_observed。

## 验证信号

来源：Xdivy 算子 ops-registry-invoke 仓 GEIR 图测试开发会话，fprintf 探针 + GE 日志路径分流判别。

- GEIR 首跑全部 case 返回 `GE error 4294967295`（来源平台：950；证据等级：tool_observed）。
- FPE 栈：`Fatal Python error: Floating-point exception` at `op_tiling.py:380 do_op_tiling` ← `xdivy_apt.py:248 xdivy` ← `get_op_tiling.py:1450 get_tiling_info_v2`（来源平台：950；证据等级：tool_observed）。
- 对比基线 `add_custom_package` GEIR 56/56 PASSED、0 Python 异常（来源平台：950；证据等级：tool_observed）。
- fprintf 探针：ACLNN 路径打印 `TilingPrepareForXdivy CALLED coreNum=56 ubSize=253952`（来源平台：950；证据等级：tool_observed）。
- GEIR 路径探针无输出 → 不调 TilingPrepare，12/12 PASSED（来源平台：950；证据等级：tool_observed）。
- GE 日志路径分流：干净 build 走 `PreRun`（`graph_manager.cc:1335`）；FPE 失败时走 `te_fusion/fusion_manager.py:build_single_op`（来源平台：950；证据等级：tool_observed）。
- `rm -rf build` 重跑 build.sh + GEIR：12/12 PASSED、0 errors（来源平台：950；证据等级：tool_observed）。

## 预防

- 迁移用 `TilingParse<XxxCompileInfo>` 注册且 tiling 读 compileInfo 做整数除法的算子后，GEIR 首跑报 `GE error 4294967295` 时，先 `rm -rf build && bash build.sh` 干净重建再判，避免被 stale build 引入的 te_fusion 路径误导。
- tiling 函数里凡读 compileInfo 字段做除法的地方，先判 0 或保证 prepare 已填充，避免 te_fusion 编译子路径 compileInfo=0 时除零。
- 判别 TilingPrepare 是否被调用用 `fprintf(stderr, ...)`，不要用 `OP_LOGI`（被 SLOG 默认过滤，看不到）。

## 失效边界

- **未验证平台**：a2、a3 未在本 trace 验证；推断可迁移（te_fusion/PreRun 分流是 GE 框架级行为，非 950 专有），但 coreNum/ubSize 数值与 TilingPrepare 调用时机需在目标平台重新探针确认。不得据推断写入 frontmatter platforms。
- **未证实**：GE 选择 PreRun vs te_fusion 的精确状态判定条件（仅据日志路径行推断，未由 GE 源码证实）。
- **不覆盖**：aclnn 直调路径的 TilingPrepare 调用已验证为"调用"，但本卡判别重点是 GEIR/te_fusion 子路径，aclnn 仅作对照。
- **中间推断**：compileInfo=0 触发除零为推断环节（tiling 函数体 verbatim 未恢复），FPE 栈本身为 tool_observed。

## 误诊

- 首跑看到 `multiprocess_util.py:68 task_q.get()` 抛 `EOFError`，一度归因为"TBE Python 多进程框架崩溃、Python 3.12 兼容性问题、CANN SDK 问题、与算子无关"（agent_reported）——被 `add_custom_package` GEIR 56/56 PASSED 的对照实验推翻，同环境 add_custom 不崩，证明根因在 Xdivy 算子侧。该 EOFError 是 te_fusion 子路径崩溃的伴生现象，不是根因。
- 一度认为"GE 编译 dry-run 传 -1 动态 shape 触发除零"（agent_reported）——后查 GEIR 测试 shape 是 (4,8) FP16 正常 static shape，排除动态 shape 假设。
- 早期结论"算子本身正确、GEIR 失败是 TBE 多进程框架/Python 兼容性问题"（agent_reported，非实测根因）——已被 add_custom 对照实验推翻，不进入验证信号。