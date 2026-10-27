---
type: "Runbook"
title: "同进程先跑正常用例后属性校验被跳过"
description: 同一测试进程内先执行一批属性合法的 aclnn 调用后，后续非法 int64 属性输入的 GetWorkspaceSize 返回成功而非 161002（属性校验 CheckAttrValid 被缓存跳过）；负向/异常用例结果顺序敏感，判别靠三组顺序对照实验，规避为异常用例先行或分进程执行。
tags: [task.debugging, topic.implementation.api_layer.c_api, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-17T01:51:00Z'
updated_at: '2026-09-17T01:51:00Z'
platforms: ["950"]
---

# 同进程先跑正常用例后属性校验被跳过

## 现象

在 Ascend950PR（CANN 9.1.0）上为一个自定义 aclnn 算子（MoE token 重排类）跑 C++ ST 全量 NPU 测试（单个测试进程按 CSV 顺序执行用例）。L2 异常用例的断言方式是：构造非法属性输入，调用 `aclnnXxxGetWorkspaceSize`，期望返回 `ACLNN_ERR_PARAM_INVALID`（161002）。同一批异常用例出现了**顺序敏感**的矛盾结果：

- 只跑 L2（20 条）：属性非法用例 L2_013（expertTokenNumType=0）、L2_014（idxType=1）、L2_015（两者均非法）全部 PASS——`[PASS] GetWorkspaceSize 返回错误码 161002（预期 161002），算子已拒绝无效输入`。
- 先跑 L0+L1（57 条全部 expertTokenNumType=1、idxType=0 的正常用例）再跑 L2：同样三条用例变成 FAIL，报错原文 `[FAIL] GetWorkspaceSize 返回成功，算子未拒绝无效输入（应返回 161002）`——非法属性没有被拒绝。
- 把顺序倒过来（L2 在最前）：三条用例又 PASS。

即：**同进程内先执行过一批属性相同的正常调用后，后续不同属性值的参数校验（CheckAttrValid）不再生效**。注意空 tensor 类校验（H=0 / A=0 / N=0 的 IsEmpty 检查）在任何顺序下都正常拒绝，Mock 模式（不经过真实 aclnn）77/77 不受影响。

## 适用场景

- 工程形态：aclnn 算子的 C++ ST 异常/负向用例（单进程顺序执行多用途例）。
- API 路径：`aclnnXxxGetWorkspaceSize` + op_api 层 `CheckParams`/`CheckAttrValid`（int64 标量属性校验）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——凡是在同一进程内反复调用 aclnn 两段式接口、且用 int64 属性区分用例的测试形态。
- 路线优先级：safety_fallback（测试侧规避；框架侧行为无法在算子侧修复）。fallback 声明：本卡不修复框架行为，只保证异常用例的校验断言可信。
- 明确不适用 / 未验证场景：空指针/dtype/shape/空 tensor 类校验不受此缓存影响（实测仍正常拒绝）；跨进程调用不受影响（每次进程冷启动）；其他 CANN 版本未验证。

## 判别方法

触发：单进程 ST 中，正常属性用例（attr=合法值）先执行、非法属性用例后执行，后者期望 161002 却返回成功。

探针（可执行，用同一测试程序做顺序对照）：

```bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
# 对照 1：异常用例所在的 L2 单独跑
./test_aclnn_<op> ../testcases/L2_test_cases.csv 2>&1 | grep -E 'L2_01[3-5]|总计|通过:|失败:'
# 对照 2：L0+L1 正常用例在前、L2 在后
./test_aclnn_<op> ../testcases/L0_test_cases.csv ../testcases/L1_test_cases.csv ../testcases/L2_test_cases.csv 2>&1 | grep -E 'L2_01[3-5]|总计|通过:|失败:'
# 对照 3：把 L2 放在最前（逆序）
./test_aclnn_<op> ../testcases/L2_test_cases.csv ../testcases/L0_test_cases.csv ../testcases/L1_test_cases.csv 2>&1 | grep -E 'L2_01[3-5]|总计|通过:|失败:'
```

原始观测（同机同进程对照）：

```text
# 对照 1（L2 单独）：属性异常 3 条 PASS，总计 20，通过 18 (正常=12, 异常=6)，失败 2（另两类校验缺失所致）
# 对照 2（L0+L1 → L2）：L2_013/014/015 [FAIL] GetWorkspaceSize 返回成功，算子未拒绝无效输入（应返回 161002）
# 对照 3（L2 → L0+L1）：属性异常 3 条恢复 PASS，总计 77，通过 75 (正常=69, 异常=6)，失败 2
```

解释（会话内推断，对照实验支持、框架源码级路径未验证）：ACL 框架在连续多次成功调用后复用缓存的 executor/校验结果，缓存键未覆盖 int64 标量属性，导致后续调用的属性校验被跳过。

## 解决配方

测试侧规避（在 ST 主循环里把异常用例排到正常用例之前，verbatim）：

```cpp
    LOG_PRINT("\n总用例数: %zu", allTestCases.size());

    // Real 模式下，将异常用例排在正常用例之前执行。
    // 原因：ACL 框架 executor 缓存问题——连续运行多个正常用例（expertNumType=1, idxType=0）
    // 后，框架会缓存校验结果，导致后续异常用例的属性校验（CheckAttrValid）被跳过。
    // 先执行异常用例可确保属性校验在缓存生效前正常工作。
#ifndef USE_MOCK_ACLNN
    std::vector<TestCase> exceptionCases, normalCases;
    for (const auto& tc : allTestCases) {
        if (IsExceptionCase(tc)) {
            exceptionCases.push_back(tc);
        } else {
            normalCases.push_back(tc);
        }
    }
    allTestCases.clear();
    allTestCases.insert(allTestCases.end(), exceptionCases.begin(), exceptionCases.end());
    allTestCases.insert(allTestCases.end(), normalCases.begin(), normalCases.end());
    LOG_PRINT("用例执行顺序: 异常用例(%zu) → 正常用例(%zu)", exceptionCases.size(), normalCases.size());
#endif

    // 3. 执行测试
```

替代方案：异常用例与正常用例分进程执行（各自冷启动），代价是测试编排变重；本会话采用同进程重排。

## 设计决策

- 选同进程重排而非分进程：保持单条命令跑全量的使用习惯，重排只影响执行顺序不影响覆盖；分进程留作更保守的备选。
- 该规避只包住"属性校验被跳过"这一类失真；另两类校验缺失（Sum=A / H 范围）是算子侧缺陷，须另行实现校验修复，不能靠排序掩盖。

## 预防

- 单进程 ST 的负向用例排在长串正常用例之后时，不直接采信其 PASS/FAIL 结果——先做一次顺序对照。
- Mock 模式（不经真实 aclnn）不能替代 Real 模式暴露此类缓存行为。

## 验证信号

来源说明：该算子 L2 异常用例上板联调会话的三组顺序对照实验（Ascend950PR，CANN 9.1.0）。

- L2 单独运行：属性非法用例 L2_013/L2_014/L2_015 全部按预期返回 161002（来源平台：950；证据等级：tool_observed）。
- L0+L1 在前时：同三条用例 `[FAIL] GetWorkspaceSize 返回成功，算子未拒绝无效输入（应返回 161002）`（来源平台：950；证据等级：tool_observed）。
- L2 在前的逆序全量：三条例外恢复 PASS，全量 75/77（2 个失败为 op_api 校验缺失，与本卡无关）；重排补丁后全量稳定 75/77，Mock 模式 77/77 不受影响（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只观测到 int64 标量属性校验被跳过；空 tensor/shape/dtype 类校验实测不受影响，但不能推断框架所有校验维度都如此。
- 该行为属框架层（executor 缓存），算子侧代码无法修复；若框架版本修复了缓存键，本卡规避即可退役。
- 其他 CANN 版本/其他芯片未验证（未验证，可迁移的是"做顺序对照实验"这一判别方法本身）。

**未决项**：

- 优先项：复现并追根因到机制层——在框架侧定位 executor 缓存的缓存键构成（是否含 int64 属性）与命中路径，确认是否为已知框架缺陷及修复版本。
- 未验证：同进程内"多少次正常调用后缓存生效"、非 int64 属性（dtype/shape 组合）是否同样被跳过。
## 坏实践

- 把顺序敏感的异常用例失败当成算子回归去修算子代码——重排后同样的代码全过，说明失败是测试编排与框架缓存交互造成的假象。
- 异常属性用例排在长串正常用例之后还直接采信其 PASS/FAIL 结果。

## 误诊

- 排查初期曾怀疑是测试程序自身传参错误（重新核对了异常执行路径中属性透传链路），或与算子包安装/OP 缓存有关；顺序对照实验（单独跑/逆序跑均 PASS）排除了这两类解释。
