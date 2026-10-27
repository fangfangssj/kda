---
type: "Runbook"
title: "GEIR 算子缓存不感知 .h 修改，掩盖 kernel 修复"
description: "TTK-GEIR 算子编译缓存按其提取的 .cpp 判定「op source unchanged」，kernel 修复写在头文件时缓存复用修复前旧二进制，表现为修复完全无效；需对比缓存 mtime 并清 binary/lock/stamp。"
tags: [task.debugging, topic.compilation, topic.graph_execution]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# GEIR 算子缓存不感知 .h 修改，掩盖 kernel 修复

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义算子 BenchHarnessScatter 排查 GEIR 图模式执行失败（EZ9999 aivec error）：定位到 kernel 多核分片与实际启动网格失配的根因后，把修复写进 `op_kernel/arch35/bench_harness_scatter_kernel.h`（头文件，被算子入口 .cpp include），随后完整重建算子包（`bash build.sh BenchHarnessScatter_package` 输出 `SUCCESS`）并重跑 GEIR 图模式验证——结果 8 个用例仍然全部失败，`RESULT total=8 pass=0 fail=8`，错误形态与修复前几乎逐位相同，第一反应自然是「修复无效/根因判错」。

翻 GEIR 测试日志发现两行关键记录（原文）：

```text
2026-09-05 05:35:49,161 [INFO] [3768719 static_f16_i32 OnGeirCompile] [graph_builder.py:197]:
GEIR op source unchanged: /tmp/ops-test-kit/geir/ops/bench_harness...
2026-09-05 05:35:49,162 [INFO] [3768719 static_f16_i32 OnGeirCompile] [compiler.py:133]:
GEIR op binary cached: /tmp/ops-test-kit/geir/ops/bench_harness_scatter
```

再对比缓存目录时间戳：`/tmp/ops-test-kit/geir/ops/bench_harness_scatter`（二进制）、`.cpp`、`.stamp` 的 mtime 全部停在 `09-04 21:04`，而 kernel 修复发生在 `09-05 03:1x`——GEIR 复测跑的是修复前的旧二进制，修复根本没有被编译进去。

## 适用场景

- 工程形态：aclnn 自定义算子开发 + TTK（ops-test-kit）GEIR 图模式验证；修改过 kernel 后重跑 GEIR 用例验证修复效果。
- API 路径：TTK GEIR 编译缓存（`/tmp/ops-test-kit/geir/ops/<op_name>` 及 `.cpp` / `.lock` / `.stamp` 伴生文件）；算子入口 `op_kernel/<op>_apt.cpp` 与其 include 的 kernel 头文件。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（TTK GEIR 模式，与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有经 TTK-GEIR 流程验证的自定义算子，只要 kernel 修复落在被 include 的头文件（AscendC kernel 主体几乎都在 .h 中，命中概率高）。
- 路线优先级：recommended_fastpath（改 .cpp 或每次修复后清缓存都是低成本动作）。
- 明确不适用 / 未验证场景：修复只写在 .cpp 本体时的缓存行为未观测；TTK 以外的 GEIR/图编译缓存未验证；缓存判定内部实现（hash 还是 mtime）未验证，本卡只描述观测到的行为。

## 判别方法

触发条件：kernel 修复后重跑 GEIR 图模式验证，失败集合与错误形态与修复前相同（既没变好也没变形）——先别推翻根因结论，先查缓存。

可执行探针（原文命令，`<op_name>`/`<pkg>` 按实际替换）：

```bash
# 1) 日志签名：复测日志里出现这两行即命中缓存复用
grep -m2 "cached\|GEIR op source" <pkg>/docs/develop/geir_verify/impl-test.log

# 2) 时间戳对照：缓存二进制 mtime 早于修复时间即确认为旧二进制
ls -la --time-style=+%m-%d_%H:%M /tmp/ops-test-kit/geir/ops/<op_name>*

# 3) 失效缓存（保留会被重新生成的 .cpp 源，删除二进制/锁/戳），并 touch 入口 .cpp 兜底
rm -f /tmp/ops-test-kit/geir/ops/<op_name> \
      /tmp/ops-test-kit/geir/ops/<op_name>.lock \
      /tmp/ops-test-kit/geir/ops/<op_name>.stamp
touch <pkg>/op_kernel/<op>_apt.cpp

# 4) 重跑 GEIR 验证后，确认缓存二进制 mtime 已更新为本次编译时间
ls -la --time-style=+%m-%d_%H:%M /tmp/ops-test-kit/geir/ops/<op_name>
```

原始观测（本例实测）：

```text
$ ls -la --time-style=+%m-%d_%H:%M /tmp/ops-test-kit/geir/ops/bench_harness_scatter*
-rwxrwxr-x  1 <user> <group> 307912 09-04_21:04 /tmp/ops-test-kit/geir/ops/bench_harness_scatter
-rw-rw-r--  1 <user> <group>  24307 09-04_21:04 /tmp/ops-test-kit/geir/ops/bench_harness_scatter.cpp
-rw-rw-r--  1 <user> <group>      0 09-05_05:40 /tmp/ops-test-kit/geir/ops/bench_harness_scatter.lock
-rw-rw-r--  1 <user> <group>     31 09-04_21:04 /tmp/ops-test-kit/geir/ops/bench_harness_scatter.stamp
```

解释：GEIR 编译流程先把算子源提取为 `/tmp/ops-test-kit/geir/ops/<op_name>.cpp` 再编译缓存二进制；其「source unchanged」判定只覆盖该 .cpp，而本次修复写在被 include 的 `bench_harness_scatter_kernel.h`——.cpp 未变 → 判定 unchanged → 复用 09-04 21:04 的旧二进制。`touch` 入口 .cpp 是为了兜底任何基于 mtime 的比较。

## 设计决策

- 清理时保留 `.cpp` 只删 `binary/lock/stamp`：.cpp 是 GEIR 提取的源，会被流程重新生成/复用，删除二进制即可强制重编译；lock/stamp 是缓存有效性的伴生标记，残留会导致命中判断异常。
- 选择「kernel 修复后顺手清缓存」而不是「把修复写进 .cpp」：AscendC kernel 主体放头文件是该工程的既定结构，不应为绕开测试工具的缓存缺陷而改变代码组织。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0，TTK（ops-test-kit）GEIR 评测模式。
- 相关文件（与探针读写一一对应）：
  - `<算子包>/op_kernel/arch35/bench_harness_scatter_kernel.h`——修复落点（.h，缓存不感知）。
  - `<算子包>/op_kernel/bench_harness_scatter_apt.cpp`——算子入口（缓存检测对象，touch 兜底点）。
  - `<算子包>/docs/develop/geir_verify/impl-test.log`——缓存命中日志。
  - `/tmp/ops-test-kit/geir/ops/bench_harness_scatter{,.cpp,.lock,.stamp}`——GEIR 编译缓存四件套。

## 预防

- 每次修改 kernel（尤其改在 .h）并重建算子包后、重跑 GEIR 验证前，先 `ls` 缓存目录 mtime 或 grep 日志 `binary cached`，确认本次测试用的是新二进制。
- 把「清 GEIR 缓存」写进 kernel 修复的回归清单，与「重建算子包」并列。
- 判定修复是否生效时，以缓存二进制 mtime > 修复时间为前置条件，否则结论无效。

## 验证信号

以下信号来自该算子 GEIR 图模式失败排查会话：kernel 网格修复后复测仍全败，通过日志签名与缓存 mtime 对照定位缓存掩盖，清缓存后重编译确认。

- 复测日志命中缓存签名：`GEIR op source unchanged` + `GEIR op binary cached: /tmp/ops-test-kit/geir/ops/bench_harness_scatter`（时间 05:35:49），缓存二进制/.cpp/.stamp mtime 均为 09-04 21:04，早于修复时间 09-05 03:1x（来源平台：950；证据等级：tool_observed）。
- 清缓存（rm binary/lock/stamp + touch apt.cpp）后重跑：缓存条目 `/tmp/ops-test-kit/geir/ops/bench_harness_scatter` 重建于 09-05 06:59，确认新二进制包含修复（来源平台：950；证据等级：tool_observed）。
- 新二进制下 GEIR 图模式仍 8/8 失败（aivec error 95×5 / 263×1）——此时才构成对修复效果的干净复测（失败本身属另一独立问题，见 [GEIR aivec 95/263 网格缓存修复后仍全败卡](../runtime/geir_aivec_95_persists_after_grid_cache_fixes.md)）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只证明「.h 修改不被 GEIR 缓存的 source-unchanged 判定感知」这一行为；缓存判定的内部实现（内容 hash 还是 mtime、检测哪些文件）未验证，清缓存手法可能在 TTK 版本更新后变化。
- 若修复本身写在 GEIR 提取的 .cpp 里，缓存应能感知（未验证）。
- 清缓存后的问题域变化：旧二进制失败 ≠ 新二进制通过——本例清缓存后图模式仍失败，需按新失败形态继续排查，不能把「清缓存」当成修复手段。
- 相邻缓存层的区分：CANN 侧 kernel 二进制缓存（`kernel_meta`，host 侧改动不进 device hash，加 PRINTF 才能扰动）见 [kernel 源码改动未生效因 binary 缓存卡](kernel_binary_cache_printf_hash_toggle.md)；本卡是 TTK-GEIR 的算子编译缓存（按提取 .cpp 判定），两层缓存机制不同、清法不同。

**未决项**：

- GEIR 缓存「source unchanged」的判定实现（graph_builder.py:197 / compiler.py:133 附近）未读源确认——是 hash 还是 mtime、是否覆盖 include 闭包，值得下个 agent 核实后把本卡从行为级升级为机制级。
## 坏实践

- 修复后不复核缓存时效，直接把「复测仍失败」当作「根因判错」的证据推翻分析结论——本例差一点因此放弃正确的网格失配修复。
- 只 `touch` 头文件（缓存检测的 .cpp 不变，未必触发重编译）；正确做法是删缓存二进制/lock/stamp 并 touch 入口 .cpp。

## 误诊

- 复测 8/8 失败的最初解读是「网格对齐修复无效，需要另找根因」（agent_reported，会话内推理）；实际是缓存掩盖，修复已正确落地且动态路径受益。
