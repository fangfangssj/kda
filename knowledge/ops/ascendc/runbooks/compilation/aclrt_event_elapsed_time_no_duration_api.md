---
type: "Runbook"
title: ACL 事件计时 API 是 aclrtEventElapsedTime
description: CANN 9.1.0 无 aclrtEventDuration/aclrtGetDeviceSocName，ACLNN_SUCCESS 宏未定义；事件计时用 aclrtEventElapsedTime(float*, start, end)（出参在前），aclnn 返回值用 ret==0 判定。
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T04:39:31Z'
updated_at: '2026-09-17T04:39:31Z'
platforms: ["950"]
---

# ACL 事件计时 API 是 aclrtEventElapsedTime

## 现象

在 Ascend950PR（CANN 9.1.0）上为 BenchNoHarnessSwiGlu 编写 C++ 原生性能测试程序（tests/perf/perf_test.cpp，用 aclrtEvent 对 kernel 做 device 侧计时、5 次 warmup + 50 次测量）。首次 `make -j` 编译报两个 API 不存在（原文）：

```
perf_test.cpp:343:9: error: 'aclrtEventDuration' was not declared in this scope; did you mean 'aclrtEventStatus'?
  343 |         aclrtEventDuration(startEvent, endEvent, &timeMs[i]);
perf_test.cpp:506:5: error: 'aclrtGetDeviceSocName' was not declared in this scope; did you mean 'aclrtGetDeviceSatMode'?
  506 |     aclrtGetDeviceSocName(&socName);
```

后续重写版本又报第三个同类错误：

```
perf_test.cpp:251:25: error: 'ACLNN_SUCCESS' was not declared in this scope; did you mean 'ACL_SUCCESS'?
  251 |         return (ret2 == ACLNN_SUCCESS);
```

三处都是按"想当然的命名"写 host 侧 ACL/aclnn 调用，而 CANN 9.1.0 公共头文件里没有这些符号。

## 适用场景

- 工程形态：aclnn 算子的 C++ host 侧工具开发（性能/计时/诊断 harness）。
- API 路径：`acl/acl_rt.h`（事件计时）、`aclnn` 两段式接口返回值（aclnnStatus）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 ACL host API 的实际命名与返回值判定。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他 CANN 版本（命名可能随版本增加/变化，以头文件为准）。

## 根因

CANN 9.1.0 的 `include/acl/acl_rt.h` 中事件耗时接口只有 `aclrtEventElapsedTime`，不存在 `aclrtEventDuration`；设备 SoC 名查询接口在该版本头文件中不存在（编译器提示的 `aclrtGetDeviceSatMode` 是无关接口）；aclnn 状态码 `ACLNN_SUCCESS` 宏在公共头（`aclnn_base.h`/`acl_base_rt.h` 链）未导出（宏兜底方案见 [ACLNN_SUCCESS 公开头未导出须自定义宏](aclnn_success_not_in_public_headers.md)），但 aclnnStatus 就是 int，成功值为 0（与 ACL_SUCCESS 数值一致）。

## 解决配方

事件计时修复（tests/perf/perf_test.cpp，verbatim）：

```cpp
        aclrtSynchronizeStream(stream);
        aclrtEventElapsedTime(&timeMs[i], startEvent, endEvent);
```

（修复前为 `aclrtEventDuration(startEvent, endEvent, &timeMs[i]);`——参数顺序也错：正确签名出参在前，`aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent)`。）

设备名查询修复：直接删除该调用（verbatim）：

```cpp
    // 打印设备信息
    std::cout << "Device ID: " << deviceId << "\n\n";
```

（修复前为 `const char* socName = nullptr; aclrtGetDeviceSocName(&socName); ...`。）

aclnn 返回值判定修复（verbatim）：

```cpp
        return (ret2 == 0);  // ACLNN_SUCCESS == 0
```

（修复前为 `return (ret2 == ACLNN_SUCCESS);`。）

## 设计决策

- 计时后因事件方案与 executor 生命周期问题纠缠，最终 harness 改用 CPU 侧 chrono + `aclrtSynchronizeStream` 计时；但 `aclrtEventElapsedTime` 本身的签名结论不受影响（头文件 grep 确认）。事件法在可预构建 executor 的场景下仍是剔除 host 开销的推荐口径（见 [aclnn 带宽计时用事件法剔除 host 开销](../performance/aclnn_timing_aclrtevent_kernel_only.md)）。
- 设备 SoC 名改为从环境/平台先验信息获取（报告头部直接写明平台），不再调不存在的 API。

## 预防

写 ACL host 代码时对"想当然命名"先 grep 头文件再落笔：

```bash
grep -rn "aclrtEvent" ${ASCEND_HOME_PATH}/include/acl/acl_rt.h | grep -i "duration\|elapsed\|query"
# 命中: ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent);
```

aclnn 两段式返回值统一按 `ret != 0` 判失败，不依赖 ACLNN_SUCCESS 宏。

## 验证信号

来源说明：性能验收会话中 perf harness 的三轮编译修复。

- 报错原文 `'aclrtEventDuration' was not declared in this scope; did you mean 'aclrtEventStatus'?` 与 `'aclrtGetDeviceSocName' was not declared in this scope`（来源平台：950；证据等级：tool_observed）。
- `grep acl_rt.h` 实测仅 `aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent)` 存在（另有 aclrtQueryEvent/QueryEventStatus/QueryEventWaitStatus 状态查询族）（来源平台：950；证据等级：tool_observed）。
- 修复后 `make -j` `[100%] Built target perf_test`（首轮修复后编译通过）；ACLNN_SUCCESS 改 `ret2 == 0` 后再次编译通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡是 CANN 9.1.0 头文件事实；新版本 CANN 可能引入 ACLNN_SUCCESS 宏或 SoC 名查询接口，使用前以 `${ASCEND_HOME_PATH}/include` 实际声明为准（未验证，推断可迁移到"先查头文件"的方法论）。

**未决项**：

- 无。
## 坏实践

- 按编译器 "did you mean" 提示顺手换近名 API（如把 aclrtEventDuration 换成 aclrtEventStatus/GetDeviceSatMode）——提示的是词面相近而非语义相同的接口。
