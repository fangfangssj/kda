---
type: "Runbook"
title: GE 图执行 0x50100064：TBE 子进程 Python3.13 崩溃
description: 自定义算子 GE IR 示例运行时 RunGraph/BuildGraph 报 1343225956(0x50100064)、aclgrphBuildModel 报 50331649，TBE 子进程在 Python 3.13 下崩溃 "main process disappeared"；切 Python 3.12 后挂起超 10 分钟。收尾：示例保留编译验证，run.sh 自动检测 Python 版本。
tags: [task.debugging, topic.compilation, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-16T10:29:17Z'
platforms: ["950"]
---

# GE 图执行 0x50100064：TBE 子进程 Python3.13 崩溃

## 现象

自定义 LSTM 算子（已用 `bash build.sh --soc=ascend950` 安装为 vendor 算子包，kernel 二进制在 `<CANN>/opp/vendors/<vendor>_custom/` 下）开发 GE IR 调用示例：示例先按 `ge::Session` 路线写（`GEInitialize` → 构图 → `AddGraph` → `CompileGraph` → `RunGraph`），链接问题修好后运行：

```
./test_geir_<op>   # LD_LIBRARY_PATH 含 vendor op_api/lib 与 op_proto/lib/linux/x86_64
```

构图与 `CompileGraph` 均成功，随后：

```
[FATAL] RunGraph failed: 1343225956
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
（共 8 条同样的 TBE 子进程错误）
start compile Ascend C operator Cast. kernel name is te_cast_e548cb8d...
start compile Ascend C operator Cast. kernel name is te_cast_7c650f93...
<conda 环境>/lib/python3.13/multiprocessing/resource_tracker.py:479:
UserWarning: resource_tracker: There appear to be 9 leaked semaphore objects to clean up at shutdown: {'/mp-e4_40l39', ...}
```

错误码换算：`printf "0x%X" 1343225956` → `0x50100064`。关键线索：GE 在线编译的是图优化插入的 **Cast** 算子（`te_cast_*`），不是 LSTM 算子本身；崩溃的 TBE 子进程跑在 miniconda **Python 3.13.14** 上。同一环境里 ACLNN 路线完全正常（示例运行 PASS、ST 44/44），因为 ACLNN 用的是已安装的预编译 kernel，不触发 TBE 在线编译。

当时代码状态：示例代码构图/API 调用本身正确（编译通过、AddGraph/CompileGraph 成功），失败全部发生在图执行触发的 TBE 在线编译子进程。

## 适用场景

- 工程形态：自定义算子的 GE IR（`ge::Session` 或 `aclgrphBuildModel` + `aclmdlExecute`）调用示例，CANN cann-9.2.0 + 默认 Python 3.13（miniconda）。
- API 路径：`ge::Session::RunGraph/BuildGraph`、`aclgrphBuildInitialize/aclgrphBuildModel`、`aclmdlLoadFromMem/aclmdlExecute`；TBE 在线编译（`task_distribute` 子进程）。
- 验证过的平台与 CANN 版本：Ascend950PR + cann-9.2.0（Python 3.13.14 与系统 Python 3.12.3 均实测）（与 frontmatter `platforms` 一致）。
- 适用粒度：`platform_wide`（GE/TBE 工具链行为，与算子无关）。
- 路线优先级：banned（在本环境把 GE IR 在线编译当可依赖路径是已证伪路线；ACLNN 两段式是可行替代）。
- 明确不适用/未验证场景：满足 Python ≥3.8 且 <3.13 的环境上 GE IR 是否可端到端跑通未验证；`ge.socVersion` 选项在正确取值下是否可绕过在线编译未验证。

## 根因

本卡为失败路线记录，根因链未完全追到机制层：GE 图执行（RunGraph/BuildGraph/aclgrphBuildModel）会对图优化插入的算子（本例 Cast）触发 TBE 在线编译；TBE 编译子进程（`task_distribute`，跑在系统默认 Python 上）在 Python 3.13.14 环境下崩溃（"main process disappeared"，multiprocessing semaphore 泄漏），GE 以错误码 1343225956（0x50100064）/ 50331649 上抛失败。Python 3.13 与 CANN 9.2.0 TBE 的不兼容的确切机制（哪个依赖/语法不兼容）未验证。

## 解决配方

本轨迹内该路线**没有修成**——以下为尝试记录与最终收尾处置；不要把"切 Python 3.12"当成已验证修复。

尝试 1：`BuildGraph` 后再 `RunGraph`（怀疑输入未定型）——同样 1343225956 + TBE 子进程崩溃：

```cpp
    LOG_PRINT("BuildGraph...");
    geRet = session.BuildGraph(graphId, inputs);
    CHECK_RET(geRet == ge::SUCCESS, "[FATAL] BuildGraph failed: %u", geRet);
    LOG_PRINT("BuildGraph 完成");

    LOG_PRINT("执行计算图...");
    std::vector<ge::Tensor> outputs;
    geRet = session.RunGraph(graphId, inputs, outputs);
```

尝试 2：整段改写为 `aclgrphBuildModel` 离线模型路线（`ge::op::Data` 构图不变）：

```cpp
    // ------------------------------------------------------------------
    // 5. 构建离线模型
    // ------------------------------------------------------------------
    LOG_PRINT("构建离线模型 (aclgrphBuildModel)...");
    ge::ModelBufferData modelBufferData;
    std::map<ge::AscendString, ge::AscendString> buildOptions;
    buildOptions[ge::AscendString("ge.device")] = ge::AscendString(std::to_string(deviceId).c_str());
    buildOptions[ge::AscendString("ge.runMode")] = ge::AscendString("0");

    geRet = aclgrphBuildModel(graph, buildOptions, modelBufferData);
    CHECK_RET(geRet == ge::GRAPH_SUCCESS, "[FATAL] aclgrphBuildModel failed: %u", geRet);
    LOG_PRINT("模型构建成功, 大小: %lu bytes", modelBufferData.length);

    aclgrphBuildFinalize();

    // ------------------------------------------------------------------
    // 6. 加载模型并执行
    // ------------------------------------------------------------------
    LOG_PRINT("加载模型...");
    uint32_t modelId = 0;
    ret = aclmdlLoadFromMem(modelBufferData.data.get(), modelBufferData.length, &modelId);
    CHECK_RET(ret == ACL_SUCCESS, "[FATAL] aclmdlLoadFromMem failed: %d", ret);
    ...
    ret = aclmdlExecuteAsync(modelId, inputDataset, outputDataset, stream);
```

——`aclgrphBuildModel failed: 50331649`，同样 TBE 子进程崩溃（构建期同样触发在线编译）。

尝试 3：给 `aclgrphBuildInitialize`/`aclgrphBuildModel` 的 options 加 `ge.socVersion`（先后试 `Ascend950PR` 与 `Ascend950`）——更糟：`aclgrphBuildInitialize failed: 4294967295`，两种取值同样失败；移除该选项后 init 恢复成功（回到 50331649 的状态）。

尝试 4：换 Python 版本重跑（默认 miniconda Python 3.13.14 → `export PATH=/usr/bin:$PATH` 用系统 Python 3.12.3）——TBE 不再立即崩溃，但 180s 超时未完成；加大到 `timeout 600` 仍超时（10 分钟无结果）；清理时 `pkill -f "test_geir"` 本身也挂起（120s 超时），最终 `kill -9 $(pgrep -f "test_geir|te_cast|task_distribute")` 才清干净。

最终处置（可复用的收尾模式）：示例保留为"编译验证"交付；`run.sh` 加 Python 版本自动检测与切换；README 注明 GE IR 需要 TBE 在线编译、要求 Python ≥3.8 且 <3.13。`run.sh` 中新增的检测段（verbatim）：

```bash
# GE IR 示例需要 TBE 在线编译, 要求 Python >= 3.8 且 < 3.13
# 如果默认 Python >= 3.13, 尝试使用系统 Python 3.12
if [ "$RUN_GEIR" = true ]; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    if [ -n "$PY_VERSION" ] && python3 -c "import sys; exit(0 if sys.version_info < (3, 13) else 1)" 2>/dev/null; then
        : # Python version OK
    elif command -v /usr/bin/python3.12 &> /dev/null; then
        export PATH=/usr/bin:$PATH
        echo "[INFO] GE IR: 使用系统 Python 3.12 进行 TBE 编译 (PATH=/usr/bin:$PATH)"
    else
        echo "[WARN] GE IR: TBE 编译需要 Python < 3.13, 当前 Python ${PY_VERSION}, 可能编译失败"
    fi
fi
```

README 中新增的说明（verbatim）：

```markdown
> **注意**：GE IR 调用使用 `aclgrphBuildModel` + `aclmdlExecute` 方式。图构建过程中 GE 引擎会进行图优化（可能插入 Cast 等算子）并触发 TBE 在线编译，要求 Python 版本 ≥ 3.8 且 < 3.13。如默认 Python 版本不满足，`run.sh` 会自动尝试使用系统 Python 3.12。
```

## 设计决策

- 为何最终保留 GE IR 示例而不是删掉：构图、`aclgrphBuildModel`/`aclmdlExecute` 的 API 用法本身经编译与初始化阶段验证是正确的，失败点在环境（TBE 子进程），示例对满足 Python 约束的环境仍有复用价值。
- 为何用 `run.sh` 检测 PATH 而不是改 CANN 环境：PATH 切换是进程局部的、可自动回退（无 3.12 则只告警），不污染全局 Python。
- 当前路线属 banned：在 Python ≥3.13 的环境上依赖 GE IR 在线编译已被本轨迹证伪；ACLNN 两段式（预编译 kernel）是同一算子已验证的替代路线。

## 验证信号

段首来源说明：本验证来自该 LSTM 自定义算子 GE IR 示例的多轮运行尝试；所有信号均为同一天同一环境（Ascend950PR + cann-9.2.0）实测。

- `ge::Session` 路线：`CompileGraph` 成功后 `[FATAL] RunGraph failed: 1343225956`（0x50100064）+ `TBE Subprocess[task_distribute] raise error[], main process disappeared!` ×8 + Python 3.13 resource_tracker 9 个 semaphore 泄漏告警，日志显示正在在线编译 Cast（`te_cast_*`）（来源平台：950；证据等级：tool_observed）。
- `BuildGraph` 前置后：`[FATAL] BuildGraph failed: 1343225956`，同样 TBE 子进程崩溃（来源平台：950；证据等级：tool_observed）。
- `aclgrphBuildModel` 路线：`[FATAL] aclgrphBuildModel failed: 50331649`，同样 TBE 子进程崩溃 ×8（来源平台：950；证据等级：tool_observed）。
- `ge.socVersion=Ascend950PR` / `Ascend950`：`[FATAL] aclgrphBuildInitialize failed: 4294967295`；移除后 init 成功（来源平台：950；证据等级：tool_observed）。
- Python 3.12.3（PATH=/usr/bin）下：无崩溃输出但 180s 与 600s 两次超时未完成；`pkill` 挂起、`kill -9` 清理成功（来源平台：950；证据等级：tool_observed）。
- 对照：同一环境 ACLNN 示例运行 `PASS`（workspaceSize 查询与非全 0 输出正常）、PyTorch ST 44/44 通过——ACLNN 路径不触发 TBE 在线编译（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限于 CANN 9.2.0 + Python 3.13.14（立即崩溃）/ Python 3.12.3（挂起超过 10 分钟）实测；Python 3.12 下"挂起 vs 极慢"未区分，未验证，不推断。
- "TBE 在线编译要求 Python <3.13" 是从崩溃行为与切换实验归纳的边界（README 亦如此声明），CANN 官方支持矩阵未在轨迹内核对（未验证，推断可迁移需复现）。
- `ge.socVersion` 两取值导致 `aclgrphBuildInitialize` 返回 -1 的内部归因未查证；不代表该选项在别的水位/版本上不可用。
- GE IR 示例代码在满足 Python 约束的环境上端到端运行成功与否，未验证。

**未决项**：

- 【优先】在满足 Python ≥3.8 且 <3.13 的环境复现并追到机制层：确认 Python 3.13 下 TBE `task_distribute` 崩溃的确切依赖/语法不兼容点，以及 3.12 下长时间未完成是挂起还是极慢。
- 核对 CANN 官方 Python 支持矩阵与 `ge.socVersion` 合法取值清单，替代本卡的实测猜测边界。
## 坏实践

- 在默认 Python ≥3.13 的环境里反复换 GE 执行 API（RunGraph → BuildGraph → aclgrphBuildModel）试图绕过 TBE 子进程崩溃：三条路径都汇聚到同一在线编译点，换 API 不换环境无效。
- 给 options 加未核对的 `ge.socVersion` 字符串（`Ascend950PR`/`Ascend950` 猜测取值）：实测直接让 `aclgrphBuildInitialize` 返回 -1，比不加更糟。
- 挂起后用 `pkill` 清理卡死的 TBE 编译进程树：实测 `pkill` 自身挂起；用 `kill -9 $(pgrep -f "test_geir|te_cast|task_distribute")` 才有效。
- 把该环境失败当成示例代码 bug 去改构图代码：构图与 API 用法已经编译/初始化验证，问题在工具链子进程。

## 误诊

- 最初怀疑 `RunGraph` 失败是输入 tensor placement（HOST vs DEVICE）问题：`CompileGraph` 成功、`RunGraph` 失败的形态容易往输入摆放上想；实际观测显示失败点在 TBE 在线编译子进程（日志 `te_cast_*` + "main process disappeared"），与输入摆放无关（该 placement 假设未实施验证即被日志证据替代）。
