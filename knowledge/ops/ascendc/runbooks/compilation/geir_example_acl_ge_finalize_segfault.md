---
type: "Runbook"
title: GEIR 示例混用 ACL 清理致退出段错误
description: GE IR 图执行程序混用 aclInit/aclrtSetDevice/aclFinalize 与 GEFinalize，功能全部正确（RunGraph success、精度 PASS、输出写盘）但进程退出阶段段错误——ACL 清理先释放了 GE 会话仍引用的运行时资源；GE 经 ge.exec.deviceId 选项自管设备，纯图执行程序删除全部 ACL init/finalize 后段错误消失。
tags: [task.debugging, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-17T04:13:22Z'
updated_at: '2026-09-17T04:13:22Z'
platforms: ["950"]
---

# GEIR 示例混用 ACL 清理致退出段错误

## 现象

自定义算子（depthwise conv，fp16）的 GE IR 图模式调用示例（Ascend950PR，CANN 9.2.0）在修好图编译问题后功能完全正常：`RunGraph success`、输出写入 `geir_output_0.bin`、精度比对 `[PASS] MERE=7.33e-07, MARE=8.82e-04`。但进程在所有输出打印完之后、退出阶段崩溃：

```text
========================================
GE IR 示例执行结果: PASS
========================================
输出已写入: ./geir_output_0.bin
run.sh: ... Segmentation fault (core dumped) ./test_geir_<op> ${DEVICE_ID}
```

当时代码状态：示例在 `GEInitialize` 之后自行 `aclInit(nullptr)` + `aclrtSetDevice(deviceId)`，清理顺序为 `delete session; aclrtResetDevice(deviceId); aclFinalize(); GEFinalize();`。

## 适用场景

- 工程形态：GE IR 图模式调用示例/独立图执行程序（`ge::GEInitialize` + `ge::Session` + `AddGraph/RunGraph` + `GEFinalize`）。
- API 路径：`ge_api.h`（GEInitialize/GEFinalize/Session）、`acl/acl.h`（aclInit/aclrtSetDevice/aclFinalize/aclrtResetDevice）、GE 初始化选项 `ge.exec.deviceId`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——GE 与 ACL 运行时生命周期管理契约，与算子无关。
- 路线优先级：recommended_fastpath（纯图模式程序不引入 ACL 运行时生命周期）。
- 明确不适用 / 未验证场景：同一进程内 GE 图执行与 aclnn eager 直调混用、或需要显式管理 device/stream 的复杂宿主程序——本卡只验证了"纯 GE 图执行程序删除 ACL init/finalize"这一形态；混合场景未验证（未验证，推断须另行设计清理次序）。

## 根因

GE IR 图执行程序里混入了独立的 ACL 运行时生命周期（`aclInit/aclrtSetDevice` 与 `aclFinalize/aclrtResetDevice`），清理阶段 ACL 先于 GE 释放了 GE 会话仍引用的运行时资源，`GEFinalize` 再访问时崩溃；GE 通过初始化选项 `ge.exec.deviceId` 自管设备，纯图执行程序不需要外部 ACL 初始化。

机制层未追尽：精确崩溃点未做栈回溯确认，本卡结论是"删除 ACL init/finalize 后段错误消失"的反证式修复；`aclrtSetDevice` 与 GE `ge.exec.deviceId` 在同一进程并存时的资源归属官方语义未查证（见遗留问题）。

## 解决配方

修复前（示例 main 中 ACL 与 GE 混用）：

```cpp
    LOG_PRINT("%s - INFO: GEInitialize success\n", GetTime().c_str());

    // 设置 ACL 设备（GE 图执行需要 ACL 运行时）
    auto aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS && aclRet != ACL_ERROR_REPEAT_INITIALIZE) {
        LOG_PRINT("aclInit failed: %d\n", aclRet);
        GEFinalize();
        return FAILED;
    }
    aclRet = aclrtSetDevice(deviceId);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed: %d\n", aclRet);
        GEFinalize();
        aclFinalize();
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    // ... 构图 / AddGraph / RunGraph ...

    // 清理
    delete session;
    aclrtResetDevice(deviceId);
    aclFinalize();
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - WARN: GEFinalize failed\n", GetTime().c_str());
    }

    return SUCCESS;
}
```

修复后（删除全部 ACL init/finalize；设备由 GE 经 `ge.exec.deviceId` 选项管理；清理只保留 delete session → GEFinalize）：

```cpp
    LOG_PRINT("%s - INFO: GEInitialize success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT16;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR: CreateOppInGraph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    LOG_PRINT("%s - INFO: Create session\n", GetTime().c_str());
    std::map<AscendString, AscendString> build_options;
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        LOG_PRINT("%s - ERROR: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR: AddGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO: AddGraph success\n", GetTime().c_str());

    LOG_PRINT("%s - INFO: RunGraph...\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR: RunGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    // ... 结果比对 ...

    // 清理
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - WARN: GEFinalize failed\n", GetTime().c_str());
    }

    return SUCCESS;
}
```

GE 初始化侧保持（deviceId 经选项传入，不走 ACL）：

```cpp
    std::string deviceStr = std::to_string(deviceId);
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", AscendString(deviceStr.c_str())},
        {"ge.graphRunMode", "1"}
    };
```

## 设计决策

- 为什么直接删 ACL init/finalize 而不是调顺序（GEFinalize 提前）：对照同仓 GE IR 参考示例（add_example），其清理只有 `ge::GEFinalize()`，完全不使用 ACL——图执行程序的设备由 GE 内部管理；本示例当初加 ACL 是想"保险"，反而引入了生命周期冲突，回归参考实现即修复。
- 备选方案（只调换清理顺序，先 GEFinalize 再 aclFinalize）未实测——删除混用是更彻底且与参考一致的方案，优先采用。
- 当前方案属 recommended_fastpath。

## 预防

- 纯 GE 图执行程序（GEInitialize/Session/RunGraph/GEFinalize）不要"顺手"加 `aclInit/aclrtSetDevice` 当保险——设备经 `ge.exec.deviceId` 选项由 GE 自管，外部 ACL 生命周期会在退出清理阶段与 GE 冲突。
- 写 GE IR 示例时以官方参考示例（add_example）的初始化/清理骨架为基线，偏离基线的"增保险"代码须单独验证退出路径。

## 验证信号

来源说明：问题与验证来自该算子 GE IR 示例开发会话——RunGraph 成功后退出段错误，删除 ACL 生命周期代码后重跑验证。

- 修复前：`Segmentation fault (core dumped)`，崩溃发生在输出写入与结果打印之后（run.sh 执行日志）（来源平台：950；证据等级：tool_observed）。
- 修复后单跑：`RunGraph success` + `[PASS] MERE=7.33e-07, MARE=8.82e-04`，无段错误，进程正常退出（来源平台：950；证据等级：tool_observed）。
- 修复后从 clean 状态全量重建（`bash run.sh --clean` 后 eager + graph 两模式重编译重跑）：aclnn 与 GE IR 两示例均 `执行结果: PASS`，无崩溃（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 Ascend950PR + CANN 9.2.0 实测；其他版本未验证（推断可迁移）。
- 段错误只发生在退出清理阶段，不影响输出正确性——若只关心结果可暂时容忍，但交付示例必须修。
- 同进程混用 aclnn eager 与 GE 图执行的场景本卡未验证，不能据此推断"GE 程序一律禁止 aclInit"。
- 精确崩溃点未做 gdb/ASAN 栈回溯——机制层结论（哪个资源被提前释放）未追到源码层。

**未决项**：

- 【优先项】用 gdb/ASAN 对修复前版本做栈回溯，确认 `aclFinalize` 与 `GEFinalize` 之间具体哪个资源被提前释放——把机制层结论从反证式修复升级为直接证据。
- `aclrtSetDevice(deviceId)` 与 GE `ge.exec.deviceId` 在同一进程并存时资源归属的官方语义未查证（仅行为级结论）。
## 坏实践

- 给 GE 图执行程序"顺手"加 `aclInit/aclrtSetDevice` 当保险——与 GE 自管的设备生命周期冲突，退出时崩溃。
- 用 `return SUCCESS` 提前返回绕过清理代码来"消除"段错误——掩盖生命周期缺陷，资源全部泄漏。

## 误诊

- 最初怀疑是 `delete session` 后 GE 内部仍持有 session 资源导致 GEFinalize 崩溃，考虑过"GEFinalize 紧跟 delete session"的顺序调整；实际对照参考示例发现根因是 ACL 生命周期混入（推断与反证，修复以删除 ACL 代码验证）。
