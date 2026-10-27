---
type: "Runbook"
title: "L2 异常用例判据：两级 aclnn 报错即 PASS"
description: 黑盒 L2 异常用例的通过判据与正常用例相反：aclnnXxxGetWorkspaceSize / aclnnXxx 两级调用任一报错（实测均为 561002）即 PASS；dummy 零填充建张量省掉无意义的数据生成，本应报错却成功则 FAIL（反向哨兵）。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# L2 异常用例判据：两级 aclnn 报错即 PASS

## 现象

softplus_v2_grad（aclnnSoftplusV2Grad，ascend950）的黑盒 ST 用例集里有 9 条 L2 级"异常用例"：如 `float16_random_grad_output_NCHW_format_exception`（输出 NCHW 格式异常）、`float16_random_beta_attr_nan_exception`（beta=nan）、`float16_random_threshold_attr_nan_exception`（threshold=nan）。这类用例的语义是"非法输入应被 aclnn 层拒绝"——**报错才是正确行为**。如果黑盒框架沿用正常用例的"比对输出==期望"判据，这些用例要么无法定义期望，要么在算子错误处理正确时反而被记失败。

该框架的处理：manifest 生成时把 L2 级（或名字含 exception/error）的用例打上 `expect_error` 标记；执行时对这些用例不做数据生成与 golden 比对，而是检查两级 aclnn 调用的返回码——`GetWorkspaceSize` 报错或 `aclnn` 执行报错都判 PASS。全量实测中 9 条 L2 用例全部以如下输出通过（含错误码原文）：

```text
  [PASS] expect_error: GetWorkspaceSize errored (561002) — 报错即通过
```

（三条收尾用例原文：NCHW format exception、beta_attr_nan、threshold_attr_nan，均 `GetWorkspaceSize errored (561002)`。）

## 适用场景

- 工程形态：aclnn 黑盒 ST 的异常/负向用例（L2 级、非法输入、属性越界）执行与判定。
- API 路径：`aclnn<Op>GetWorkspaceSize`（参数校验发生处，非法输入通常在此报 561002）与 `aclnn<Op>` 两级调用的返回码；manifest 的 `expect_error` 字段。
- 验证过的平台与 CANN 版本：Ascend 950 系列（device 0，9 条 L2 全部实测通过）；CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`（任何 aclnn 算子黑盒框架的异常用例判定模式）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景："名义异常实可接受"的用例（框架预留了走 golden 比对的备选路径，本轨迹 9 条 L2 全部走报错判据，该备选路径未触发、未验证）；执行成功但输出部分非法的形态（两级调用都成功的用例只能靠比对判）；**运行时并不强制该约束的算子**——"期望报错但 kernel 成功"时应降级 SKIP 或转正常比对而非判 FAIL，见同域 [expectError 但 kernel 成功卡](expect_error_kernel_succeeded_not_enforced.md)；异常用例的非法性必须物化进真实输入，见同域 [L2 异常用例非法性物化卡](l2_exception_invalidity_not_materialized.md)。

## 判别方法

- 用例是否应走报错判据：看 manifest 标记（`level == "L2"` 或名字含 `exception`/`error`）。
- 报错发生在哪一级：`GetWorkspaceSize` 报错（参数/形状/dtype 校验，本轨迹 9 条全在此级，错误码 561002）或 `aclnn` 执行报错（执行期拒绝），两级任一报错即 PASS。
- 反向哨兵：expect_error 用例若两级调用都成功 → FAIL（"本应报错却成功"），见 runCase 的 dummyMode 分支。

## 解决配方

异常用例与正常用例共用一个 `runCase`，判据按 `expect_error` 分流（`test_aclnn_blackbox.cpp`，verbatim）：

```cpp
    // 调 aclnn
    double tKern0 = wallNow();
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus gs = aclnnSoftplusV2GradGetWorkspaceSize(tGo, tSelf, beta, thr, tOut, &wsSize, &exec);
    int result;
    if (gs != OK) {
        // GetWorkspaceSize 报错
        if (c.expectError) {
            fprintf(stdout, "  [PASS] expect_error: GetWorkspaceSize errored (%d) — 报错即通过\n", (int)gs);
            result = 0;
        } else {
            fprintf(stderr, "  [FAIL] unexpected GetWorkspaceSize error (%d)\n", (int)gs);
            result = 1;
        }
    } else {
        void* ws = nullptr;
        if (wsSize > 0) {
            if (aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                fprintf(stderr, "  [FAIL] workspace malloc failed\n");
                aclDestroyTensor(tGo); if (c.inputShapes.size()>1) aclDestroyTensor(tSelf);
                aclDestroyTensor(tOut); aclrtFree(devGo); if (c.inputShapes.size()>1) aclrtFree(devSelf); aclrtFree(devOut);
                return 1;  // binGuard 清理 /dev/shm
            }
        }
        aclnnStatus rs = aclnnSoftplusV2Grad(ws, wsSize, exec, stream);
        if (rs != OK) {
            if (c.expectError) {
                fprintf(stdout, "  [PASS] expect_error: aclnnSoftplusV2Grad errored (%d) — 报错即通过\n", (int)rs);
                result = 0;
            } else {
                fprintf(stderr, "  [FAIL] unexpected aclnn error (%d)\n", (int)rs);
                result = 1;
            }
        } else {
            // 成功：同步、读回、比对（正常用例路径）
            ...
            if (c.expectError && dummyMode) {
                // dummyMode 本应报错却成功 → FAIL（不该发生）
                fprintf(stderr, "  [FAIL] expect_error case succeeded unexpectedly\n");
                result = 1;
            } else if (nOut == 0) {
                ...
```

manifest 侧的标记生成（`prepare_blackbox_manifest.py` `_row_to_case`，verbatim）：

```python
    # expect_error：L2 异常用例（名字含 exception/error，或输出索引缺失）
    name = row.get("testcase_name", "")
    expect_error = (level == "L2") or ("exception" in name.lower()) or ("error" in name.lower())
```

配套的 dummy 模式（省掉无意义的全量数据生成，`runCase` 开头，verbatim）：

```cpp
    int64_t n0 = numel(c.inputShapes[0]);
    int64_t n1 = (c.inputShapes.size() > 1) ? numel(c.inputShapes[1]) : 0;
    bool dummyMode = (n0 != n1) || !supportedDtype(c.inputDtypes[0]) ||
                     (c.inputDtypes.size() > 1 && !supportedDtype(c.inputDtypes[1]));
    ...
    if (dummyMode) {
        // dummy：零填充（仅用于建张量，aclnn 应报错）
        goBytes.assign(bytes0 == 0 ? 1 : bytes0, 0);
```

## 设计决策

- **报错判据放在两级调用而非只看执行**：非法 dtype/shape/属性多数在 `GetWorkspaceSize` 的参数校验即被拒（实测 9 条 L2 全部 561002 于该级）；只盯执行级会把这些"更早被正确拒绝"的用例误判失败。
- **dummy 零填充建张量**：异常用例的数据内容无意义（反正期望报错），零填充避免为报错路径做昂贵的数据生成/落盘（对 1.4B 元素级 shape 的格式异常用例尤其重要——那条 NCHW 异常用例 shape 不一致自动落 dummy 模式）。
- **"本应报错却成功 → FAIL"**：防止异常用例因算子静默吞错而空转变绿；这是报错判据的自反哨兵。
- **备选路径（未触发）**：名义异常实可接受的用例走 golden 比对——为"算子对某些异常输入有定义良好的降级行为"预留，本轨迹 9 条 L2 均未走到。

## 预防

- 新算子黑盒框架从第一天就把异常用例判据设计成"报错即通过 + 反向哨兵"，不要复用正常比对判据再打补丁。
- 用例分级（L0/L1/L2）时把"异常/负向"独立成级并让 manifest 转换自动打 `expect_error`，避免手工逐条标注漏标。
- 错误码写进 PASS 输出（`errored (561002)`）：排障时能直接对上 aclnn 错误码表。

## 验证信号

以下信号来自 softplus_v2_grad 黑盒 ST 全量运行（Ascend 950，device 0）：

- 9 条 L2 异常用例全部通过，收尾三条原文均为 `[PASS] expect_error: GetWorkspaceSize errored (561002) — 报错即通过`（NCHW format exception（dummy 模式）、beta=nan、threshold=nan）（来源平台：950；证据等级：tool_observed）。
- 全量 summary 将 9 条 expect_error 计入 PASSED：`total=517  PASSED=517  FAILED=0`（8 L0 + 500 L1 + 9 L2）（来源平台：950；证据等级：tool_observed）。
- 反向哨兵路径（expect_error 却成功 → FAIL）在全量中未触发（所有 expect_error 用例均在两级调用处报错）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡判据适用于"算子侧确实会拒绝该非法输入"的场景；若实测两级调用都成功（运行时不强制约束），不能判 FAIL，应降级 SKIP 或转正常比对（见同域 expectError 但 kernel 成功卡）。
- dummy 模式以 shape 不一致/不支持 dtype 为触发条件；"形状与 dtype 都合法、仅属性值非法"的用例不走 dummy，需真实生成合法张量再注入非法属性（见同域非法性物化卡）。
- 其他平台未验证，推断可迁移（判据只依赖 aclnn 返回码语义）。

**未决项**：

- 无。
## 坏实践

- 异常用例沿用"比对输出==期望"判据，算子正确报错反而被记失败。
- 只检查执行级返回码，漏掉 GetWorkspaceSize 级的正确拒绝。
- expect_error 用例两级都成功时静默记 PASS（丢失"算子未拒绝非法输入"的信号）。

## 误诊

无。
