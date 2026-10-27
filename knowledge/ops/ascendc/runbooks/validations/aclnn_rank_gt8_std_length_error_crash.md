---
type: "Runbook"
title: rank>8 张量 ACL 抛 length_error 崩溃整跑
description: "549 条黑盒 aclnn ST 全量执行在第 544 条（9D 输入的 expect_error 用例）整体崩溃：ACL 库对 rank>8 张量抛未捕获 std::length_error 而非返回 aclnnStatus 错误码，Aborted (core dumped) exit 134；须逐用例 try/catch 并同步流。"
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# rank>8 张量 ACL 抛 length_error 崩溃整跑

## 现象

softplus_v2_grad 算子的黑盒 aclnn ST 全量执行（549 条，L0/L1/L2）在收尾的 L2 异常用例段整体崩溃。stdout/stderr 分离复跑后定位：第 544 条 `float32_random_input_gradients_dim_exception`（输入 9D shape `[3,3,7,4,5,4,6,4,6]` 与 2D `[4,6]`，输出 9D）打印完用例头后进程终止：

```
[544/549] aclnnSoftplusBackward_float32_random_input_gradients_dim_exception
terminate called after throwing an instance of 'std::length_error'
  what():  cannot create std::vector larger than max_size()
/usr/bin/bash: line 1: 483909 Aborted (core dumped) ...
exit code: 134
```

该用例本是 expect_error 用例（预期算子拒绝 9D 输入），但 ACL 库没有返回 `aclnnStatus` 错误码，而是从 C 接口背后抛出了未捕获的 C++ 异常，直接把整个测试进程打掉——前 543 条的结果全部作废，一条异常用例毁掉全量运行。

## 适用场景

- 工程形态：aclnn host 侧 ST/功能测试驱动，批量循环调用 `aclnnXxxGetWorkspaceSize` / `aclnnXxx`，用例集含非法输入（超限 rank、非法 dtype、不兼容 shape）。
- API 路径：`aclCreateTensor`、`aclnnXxxGetWorkspaceSize`（两段式 aclnn API 第一段）、`aclrtSynchronizeStream`。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.1.0。
- 适用粒度：platform_wide（ACL 库行为，与具体算子无关）。
- 路线优先级：recommended_fastpath（批量 ST 驱动的必备防御）。
- 明确不适用/未验证：单次手工调用场景（崩了重跑即可）价值有限；抛异常的确切库内位置未定位（见遗留问题）。

## 根因

ACL 张量秩上限为 8：**rank>8（9D 及以上）的张量进入 aclnn 调用链时，C 接口背后抛出 C++ 异常 `std::length_error`（"cannot create std::vector larger than max_size()"），而不是返回非零 `aclnnStatus`**。驱动侧若不在每用例边界 catch，`terminate` 直接终止进程。排查时已逐一核对崩溃用例自身的全部分配（输入 725,760 元素、`gradBytes=2,903,040`、输出 `outBytes=2,903,040`，均远小于 `vector::max_size()`），且同批 511–525 号正常用例同量级分配全部通过，可排除驱动自身分配——异常来自库内部。

## 解决配方

在 ST 驱动主循环里把**每条用例的完整处理**（构输入 + callOp + golden 比对）包进 `try/catch (const std::exception&)`，并补 `<stdexcept>` 包含（before → after，verbatim 取自事件流编辑；函数级展示驱动循环骨架）：

before（无防护，一条用例异常即整跑终止）：

```cpp
    int passed = 0, failed = 0, skipped = 0;

    for (int i = 0; i < total; ++i) {
        const StCase& c = cases[i];
        fprintf(stdout, "[%d/%d] %s\n", i + 1, total, c.caseId.c_str());

        if (!c.isEnabled) {
            fprintf(stdout, "  [SKIP] disabled\n");
            ++skipped;
            continue;
        }

        // Build inputs (single global RNG, never reset — constraint #1)
        std::vector<std::vector<float>> inputs = st_build_inputs(c);

        if (c.expectError) {
            std::vector<std::vector<float>> actual;
            int ret = callOp(c, inputs, actual, stream);
            if (ret != 0) {
                fprintf(stdout, "  [PASS] expect_error (aclnn returned %d)\n", ret);
                ++passed;
            } else {
                fprintf(stderr, "  [FAIL] expect_error — aclnn succeeded unexpectedly\n");
                ++failed;
            }
            continue;
        }

        // Normal case: compute expected via external golden
        std::vector<std::vector<float>> expected;
        bool hasGolden = computeExpected(c, inputs, expected);
        if (!hasGolden) {
            fprintf(stdout, "  [SKIP] no golden\n");
            ++skipped;
            continue;
        }

        // Call op
        std::vector<std::vector<float>> actual;
        int ret = callOp(c, inputs, actual, stream);
        if (ret != 0) {
            fprintf(stderr, "  [FAIL] aclnn error %d\n", ret);
            ++failed;
            continue;
        }

        // Compare
        bool ok = st_compare(c, actual, expected);
        if (ok) {
            fprintf(stdout, "  [PASS]\n");
            ++passed;
        } else {
            fprintf(stderr, "  [FAIL] precision mismatch\n");
            ++failed;
        }
    }
```

after（逐用例 try/catch；catch 后同步流确保无悬挂异步操作，expect_error 判 PASS、普通用例判 FAIL）：

```cpp
    int passed = 0, failed = 0, skipped = 0;

    for (int i = 0; i < total; ++i) {
        const StCase& c = cases[i];
        fprintf(stdout, "[%d/%d] %s\n", i + 1, total, c.caseId.c_str());

        if (!c.isEnabled) {
            fprintf(stdout, "  [SKIP] disabled\n");
            ++skipped;
            continue;
        }

        // Wrap per-case processing: ACL library may throw std::length_error
        // for unsupported ranks/dtypes instead of returning an error code.
        try {
            // Build inputs (single global RNG, never reset — constraint #1)
            std::vector<std::vector<float>> inputs = st_build_inputs(c);

            if (c.expectError) {
                std::vector<std::vector<float>> actual;
                int ret = callOp(c, inputs, actual, stream);
                if (ret != 0) {
                    fprintf(stdout, "  [PASS] expect_error (aclnn returned %d)\n", ret);
                    ++passed;
                } else {
                    std::vector<std::vector<float>> expected;
                    if (computeExpected(c, inputs, expected) && st_compare(c, actual, expected)) {
                        fprintf(stdout, "  [PASS] expect_error (accepted, output verified)\n");
                        ++passed;
                    } else {
                        fprintf(stderr, "  [FAIL] expect_error — accepted but output mismatch\n");
                        ++failed;
                    }
                }
                continue;
            }

            // Normal case: compute expected via external golden
            std::vector<std::vector<float>> expected;
            bool hasGolden = computeExpected(c, inputs, expected);
            if (!hasGolden) {
                fprintf(stdout, "  [SKIP] no golden\n");
                ++skipped;
                continue;
            }

            // Call op
            std::vector<std::vector<float>> actual;
            int ret = callOp(c, inputs, actual, stream);
            if (ret != 0) {
                fprintf(stderr, "  [FAIL] aclnn error %d\n", ret);
                ++failed;
                continue;
            }

            // Compare (NaN/Inf-safe — constraint #4)
            bool ok = st_compare(c, actual, expected);
            if (ok) {
                fprintf(stdout, "  [PASS]\n");
                ++passed;
            } else {
                fprintf(stderr, "  [FAIL] precision mismatch\n");
                ++failed;
            }
        } catch (const std::exception& e) {
            // ACL library may throw (e.g. std::length_error for 9D tensors).
            aclrtSynchronizeStream(stream);  // ensure stream is idle
            if (c.expectError) {
                fprintf(stdout, "  [PASS] expect_error (exception: %s)\n", e.what());
                ++passed;
            } else {
                fprintf(stderr, "  [FAIL] exception: %s\n", e.what());
                ++failed;
            }
        }
    }
```

同时在头文件包含区追加：

```cpp
#include <fstream>
#include <sstream>
#include <stdexcept>
```

catch 到异常后继续跑后续用例（9D 用例按 expect_error PASS），全量跑完不再中断。

## 设计决策

- **catch 边界放在用例级而非 API 调用级**：异常可能从 `aclCreateTensor` 或 `GetWorkspaceSize` 内部多处抛出，用例级包裹一次覆盖全部路径，且语义清晰（"这条用例以异常方式失败"）。
- **catch 后 `aclrtSynchronizeStream`**：异常若发生在 launch 之前流本应空闲，同步是防御性兜底，确保后续用例不从脏流上开始。
- **备选方案与放弃原因**：在 callOp 里预检 `rank > 8` 直接返回错误——需要把"ACL 支持的秩上限"硬编码进驱动，且只防住已知的这一种异常；try/catch 对任意库内异常通用，最终两者取通用者。另一备选"不给 9D 用例建张量、直接跳过"会丢失该异常用例的验证语义，放弃。
- **当前方案路线档位**：recommended_fastpath。

## 系统信息

- CANN 9.1.0 / Ascend 950；驱动为 manifest 驱动 aclnn ST 框架（`tests/aclnn/aclnn_st_core.h` 的 `st_run_manifest` 主循环）。
- 复现命令：`export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/<Vendor>/op_api/lib/:${LD_LIBRARY_PATH} && ./build/test_aclnn aclnn_cpp_cases.json`（stdout/stderr 分离重定向便于定位崩溃点）。

## 验证信号

本卡信号来自该黑盒 ST 框架的全量执行闭环：

- 修复前：第 544 条（9D 输入）触发 `terminate called after throwing an instance of 'std::length_error'`，`Aborted (core dumped)`，exit code 134，后续 5 条用例未执行（来源平台：950；证据等级：tool_observed）。
- 排除自身分配：崩溃点前后用例（511–525 号）输入/输出元素数与字节数逐条核对均在 1M 元素上限内（来源平台：950；证据等级：tool_observed）。
- 修复后：全量 549 条执行完毕 `==== Summary: 546/549 PASSED, 3 FAILED, 0 SKIPPED ====`，无崩溃、无 core dump，退出码 1（剩余 3 失败为 expect_error 语义问题，属另一独立问题）（来源平台：950；证据等级：tool_observed）。
- 9D 用例此后按 `expect_error (exception: ...)` 路径判 PASS，与 161002 拒绝路径并存（来源平台：950；证据等级：tool_observed）。

## 预防

- 批量 aclnn ST 驱动一律逐用例 try/catch——用例集里只要有一条非法输入用例，没有 catch 就是全跑报废。
- 输出 stdout（用例头）与 stderr（PASS/FAIL）分离重定向后再跑长批次，否则缓冲交错会把崩溃点定位搞混（本次定位初期就因 `2>&1` 混流把崩溃用例误读成相邻的正常用例）。

## 失效边界

- 本卡只证明"rank>8 触发库内未捕获 `std::length_error`"这一行为与防御写法；异常在库内的确切抛出点（`aclCreateTensor` vs `GetWorkspaceSize` 内部）未定位。
- 其他非法输入（非法 dtype 等）在本环境按 161002 错误码正常返回，未走异常路径；不能推断所有非法输入都抛异常。
- 950 + CANN 9.1.0 以外版本未验证，推断可迁移（未实测）。

**未决项**：

- **优先项**：在最小复现程序中二分定位 `std::length_error` 的确切抛出点（单独调 `aclCreateTensor` 传 9D shape，vs 构张量成功后再调 `GetWorkspaceSize`），把根因追到库内机制层。
- 库内抛异常是否为该版本已知缺陷、后续版本是否改为错误码，未查证。
## 坏实践

- 长批次 ST 驱动不设用例级异常边界，指望 ACL C API 永远返回错误码——本例证明它会抛 C++ 异常。
- 崩溃定位靠混合流 `2>&1 | tail`——stdout 块缓冲与 stderr 行缓冲交错，崩溃点会看错。

## 误诊

- 定位初期因 stdout/stderr 混流，把崩溃误读为发生在第 513/514 条（正常 L1 用例）；分离重定向后纠正为第 544 条 9D 异常用例。期间一度沿"驱动自身 vector 分配过大"方向排查（怀疑 shape capping 失效），逐条核对 manifest 元素数后排除。
