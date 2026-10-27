---
type: "Runbook"
title: "aclnn内部sync吞错误致ST假通过"
description: "aclnn 第二段接口里额外的 aclrtSynchronizeStream 会取走 stream 上的异步错误码，调用方随后的 sync 返回成功、kernel 崩溃被掩盖成用例假通过；用去内部 sync 重跑对照即可判定。"
tags: [task.debugging, topic.implementation.synchronization, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# aclnn内部sync吞错误致ST假通过

## 现象

`bench_no_harness_unique` 算子（ascend950，CANN 9.1.0）迭代二给 aclnn 第二段接口加了 K 回读逻辑：`CommonOpExecutorRun` 之后追加 `aclrtSynchronizeStream(stream)`，成功则 `aclrtMemcpy` 读 workspace 里的 K 并 `SetViewShape`。此时 kernel 其实已经全量崩溃（向量核异常 507035），但 ST 测试的表现是 **74 用例通过 60**。把这十几行 host 侧回读（含内部 `aclrtSynchronizeStream`）删掉、kernel 一字不改再跑，通过数跌到 **31**，int32 回归范围 **0/16 全部 `[FAIL] sync stream: 507035`**。同一个崩溃的 kernel，只因为 aclnn 里有没有那一次额外 sync，测试结论完全相反——内部 sync 在"帮"测试假通过。

## 适用场景

- 工程形态：aclnn 两段式自定义算子接口开发 + C++ ST 上板测试；任何想在 aclnn 第二段（执行段）内部加流同步/回读的逻辑。
- API 路径：`CommonOpExecutorRun(...)` 之后的 `aclrtSynchronizeStream` / `aclrtMemcpy` / `SetViewShape` 组合；测试侧惯用的"调用 aclnn → 自己 `aclrtSynchronizeStream` → 校验返回值 → 读输出"模式。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0；仅 950 有实测证据。
- 适用粒度：aclnn 接口与测试脚手架的通用行为（workflow 层）到 operator_family。
- 路线优先级：`recommended_fastpath`——判定手法（去内部 sync 对照重跑）是排障正解；aclnn 内部额外 sync 本身应视为待审设计。
- 明确不适用/未验证场景：其他 CANN 版本的 stream 错误语义未验证；同步接口（非 stream 异步路径）不涉及。

## 判别方法

**触发信号**：aclnn 执行段内部含 `aclrtSynchronizeStream`，且测试侧自己也会 sync 并检查返回值；出现"删掉一段看似无关的 host 逻辑后通过数大幅下降"的异常敏感度。

**探针（可执行，verbatim）**——去掉 aclnn 内部 sync 前后各跑一轮同一测试并统计 507035 计数：

```bash
cd operators/bench_no_harness_unique/tests/st/build-real && \
  export ASCEND_TOOLKIT_HOME=<CANN 安装根> && \
  export ASCEND_HOME_PATH=<CANN 安装根> && \
  export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/opp/vendors/bench_no_harness_unique_custom/op_api/lib/:${LD_LIBRARY_PATH} && \
  ./test_aclnn_bench_no_harness_unique --scope <int32 回归范围> 2>&1 | grep -c "507035" && echo "--- iter1 total:" && \
  ./test_aclnn_bench_no_harness_unique --scope <int32 回归范围> 2>&1 | grep -E "总计|通过|失败"
```

**原始观测**（本轨迹 tool_observed）：

- 有内部 sync（K 回读版）：全量 60/74 通过；测试侧 sync 不报错。
- 无内部 sync（仅删 aclnn 内 `aclrtSynchronizeStream`/回读，kernel 相同）：全量 31/74，其中 39/43 个失败是 `sync stream: 507035`；int32 回归范围 0/16，`grep -c "507035"` = 16。
- 彻底修复 kernel 崩溃后：全量 67/74，`grep -c "507035"` = 0（此时有无内部 sync 不再影响结论）。

**解释（interpretation）**：`aclrtSynchronizeStream` 对 stream 上的异步错误是**消费型**的——第一次调用取走错误码（返回 507035），aclnn 内部逻辑只把它当"回读条件不满足"吞掉并照常返回 `ACLNN_SUCCESS`；测试侧随后的 sync 拿到的是成功，于是崩溃被翻译成"用例通过"。该消费语义是从上述成对计数对照推出的（未从 ACL 文档逐条核实），但三组观测自洽。

## 验证信号

来源说明：本卡验证来自同一次开发会话的三组成对运行（同一测试二进制、kernel 仅在最后一组修复）。

- 有内部 sync：60/74 通过（崩溃 kernel 假通过 29 例）（来源平台：950；证据等级：tool_observed）。
- 无内部 sync、kernel 未修：31/74，`grep -c "507035"`（int32 回归范围）= 16/16（来源平台：950；证据等级：tool_observed）。
- 无内部 sync、kernel 已修（移除 workspace 标量写，见同批 `kernel_workspace_param_direct_bind_507035`）：67/74，507035 计数 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "错误码被第一次 sync 消费"的机制解释未从 ACL 官方文档验证，属计数对照支持的推断；引用时按判别手法（对照重跑）使用，不要把机制表述当规范事实。
- 本卡只覆盖"aclnn 内部 sync + 测试侧 sync"的双 sync 形态；测试侧若不检查 sync 返回值，则表现为另一种假通过（读到的输出是旧数据），判别方式不同。
- ascend950 + CANN 9.1.0 实测；其他版本未验证（推断可迁移）。

**未决项**：

- `aclrtSynchronizeStream` 错误消费语义的规范级确认（ACL 文档/源码），以及多消费者场景（aclnn 内部、测试侧、框架）下错误码的所有权规则。
## 预防

- aclnn 执行段内部原则上不要追加 `aclrtSynchronizeStream`；确需回读中间结果时，把"是否吞错"设计进返回路径（内部 sync 非 ACL_SUCCESS 必须向调用方返回错误，而不是只跳过回读）。
- 测试侧判定通过的条件不要只依赖自己那次 sync 的返回码；对关键用例可加输出内容校验（本例 all_same 用例 y[0] 内容校验在假通过版本里其实是能暴露问题的）。
- 排查"通过率对无关 host 改动异常敏感"的测试时，第一反应做去内部 sync 对照重跑。

## 坏实践

- aclnn 内部 `syncRet != ACL_SUCCESS` 时仅跳过回读、仍返回 `ACLNN_SUCCESS`——把执行错误静默降级成"可选功能未生效"。
- 看到"删掉一段逻辑掉分"就得出"该逻辑有功能收益"的结论——本例它只是错误掩体。

## 误诊

- 初判"K 回读在帮部分用例通过（60 → 31 说明回读有益）"——方向完全相反；掉分的 29 例是原本被吞掉的崩溃用例。
