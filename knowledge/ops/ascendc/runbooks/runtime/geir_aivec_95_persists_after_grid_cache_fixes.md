---
type: "Runbook"
title: "GEIR aivec 95/263：网格与缓存修复后仍全败"
description: "网格失配修复+GEIR 缓存清理+重建确认后，图模式 8 用例仍 0 pass（aivec error 95×5/263×1），而同二进制 aclnn 动态路径全过；负结果：图模式失败另有机制层根因未解。"
tags: [task.debugging, topic.graph_execution, topic.memory]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# GEIR aivec 95/263：网格与缓存修复后仍全败

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义算子 BenchHarnessScatter 排查 GEIR 图模式执行失败：先修复了 kernel 多核分片与实际启动网格失配的越界 bug（三处 `AscendC::GetBlockNum()` 对齐，见 [507035 多核分发未拦截多余核越界卡](507035_unguarded_extra_cores_oob_dispatch.md)），又排除了 GEIR 编译缓存复用旧二进制的干扰（清缓存后确认测试二进制已含修复，见 [GEIR 算子缓存不感知 .h 修改卡](../compilation/geir_op_cache_ignores_header_edits.md)）。在此基础上重跑 GEIR 图模式验证——8 个用例仍然全部失败：

```text
2026-09-05 07:03:42,248 [INFO] [32667 static_f32_dim1 OnGeirCompile] [compiler.py:133]:
GEIR op binary cached: /tmp/ops-test-kit/geir/ops/bench_harness_scatter   ← 新二进制
...
==================== TTK Test Summary ===================
  Total    : 8
  PASS     : 0
  FAIL     : 2
  SKIP/NA  : 6
  PassRate : 0.00%  (0/8)
==========================================================
RESULT total=8 pass=0 fail=8 skip=0
```

错误统计（对 impl-test.log 的 EZ9999 行聚合）：`6 aivec error`、`error code = 95` ×5、`error code = 263` ×1、`mte error info: 0x1fb8adf800020264 / 0x1fb8adf800020278 / 0x1fb8adf800020293`——vector 核（aivec）异常集中在 core 0，带 MTE 错误信息。设备侧报错形态与修复前同族（修复前一次复测的原文样例：`there is an aivec error exception, core id is 0, error code = 95, ... mte error info: 0xe0000c8000020264, ... mte error: 0x80000000`）。

与此同时，**同一个算子二进制的 aclnn 动态路径完全正常**：aclnn 直连回归 `Test overwrite: PASSED / Test add: PASSED / summary: SUCCESS`，L2 反向回归符合预期，正向冒烟 30/30 PASS，黑盒/白盒链路此前的验收全部通过。

## 适用场景

- 工程形态：aclnn 自定义算子开发，同一算子需同时通过 aclnn 动态执行与 GEIR 图模式（GraphEngine IR）执行验证。
- API 路径：TTK（ops-test-kit）GEIR 评测模式（RunGraph）；AscendC kernel（TPipe/TBuf/事件同步/DataCopyPad）；算子含 fp32 workspace 累加路径（`GetUserWorkspace` 取用户工作区）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：same_operator（该算子图模式形态）；排查顺序部分（先缓存、后网格）可外推到 operator_family。
- 路线优先级：safety_fallback——图模式通路在本迭代以「已知限制+完整记录」披露放行，交付口径以 aclnn 动态通路为准（该通路全链路验证通过）。fallback 声明：这是止损决策而非修复，采用本路线意味着图模式执行能力暂不可用，功能覆盖有损；主接口不受影响。
- 明确不适用 / 未验证场景：本卡不宣称图模式失败的一般性原因（根因未定位）；其他算子/架构的图模式行为未验证。

## 根因

未追到机制层。已排除与已证实：

- 已排除①（编译缓存）：清缓存并确认新二进制后仍失败，缓存掩盖不再成立。
- 已排除②（网格失配的越界机制）：`GetBlockNum()` 对齐修复已落地且动态路径受益；该机制不能解释图模式残余失败。
- 未证实（假设）：图模式下 kernel 与图编译器在 workspace 传递 / TPipe 内存模型 / tiling 数据流上存在更深层不兼容——本会话的推理依据仅为「aclnn 动态路径同二进制全过、图模式全败」的对照，无直接机制证据。

## 尝试与失败观测

- 尝试（可重建）：三处 kernel 分片 `GetBlockNum()` 对齐修复 → 重建算子包 SUCCESS → 清 GEIR 缓存（rm 二进制/lock/stamp + touch 入口 .cpp）→ 重跑图模式验证。
- 客观失败（含修复的新二进制，mtime 09-05 06:59）：`RESULT total=8 pass=0 fail=8 skip=0`；EZ9999 行聚合 `6 aivec error / error code=95 ×5 / error code=263 ×1 / mte error info 0x1fb8adf800020264、0x1fb8adf800020278、0x1fb8adf800020293`。
- 边界对照（同二进制 aclnn 动态路径）：aclnn 回归 SUCCESS（overwrite/add 双用例）；L2 反向 24 FAIL（正确拒绝）+1 PASS；正向冒烟 30/30 PASS。
- 止损处置：三次图模式验证重试均失败后，记录已知限制并以披露方式放行，交付以 aclnn 通路为准。

## 设计决策

- 为什么止损而不是继续追：三次重试（含两次修复迭代）后仍 0/8，剩余假设（workspace/管道语义）需要重构 kernel 内存模型级别的调试，超出该迭代的合理投入；主接口（aclnn）已全链路验证，功能交付不受影响。
- 为什么把「披露放行」记为 safety_fallback 而非 pass：知识上必须保留「图模式未通过」的完整证据链（失败日志、错误码、已排除项），不允许以「通过」掩盖负结果。
- 当前方案路线档位：safety_fallback（图模式以已知限制披露；aclnn 动态路径为交付主口径）。

## 预防

- 排查 GEIR 图模式 EZ9999/aivec 类失败的推荐顺序：① 先核 GEIR 编译缓存时效（日志 `binary cached` + mtime 对照，见 [GEIR 算子缓存不感知 .h 修改卡](../compilation/geir_op_cache_ignores_header_edits.md)）；② 再核网格/分片失配（`GetBlockNum()` 对齐，见 [507035 多核分发未拦截多余核越界卡](507035_unguarded_extra_cores_oob_dispatch.md)）；③ 两者排除后按 workspace/管道语义差异方向追（本卡止步于此）。
- 交付双形态（图模式+动态）算子时，把图模式验证排进迭代早期，避免在收尾阶段才发现深层不兼容只能止损。
- 与负向用例的 EZ9999 区分：负向非法输入在图执行阶段以 rc=255/EZ9999 被拒是**预期形态**（见 [GEIR 负向拒绝签名卡](../validations/geir_negative_reject_rungraph_rc255_ez9999.md)）；本卡是正向用例的执行失败，两者错误码同形、语义相反。

## 验证信号

以下信号来自该算子 GEIR 图模式失败排查会话的第三轮复测（缓存已清、二进制确认含网格修复）与同期的动态路径对照。

- 图模式（新二进制 09-05 06:59）：8 用例 `RESULT total=8 pass=0 fail=8 skip=0`；TTK Summary `Total 8 / PASS 0 / FAIL 2 / SKIP-NA 6 / PassRate 0.00%`；EZ9999 聚合 `aivec error ×6、error code=95 ×5、error code=263 ×1`、`mte error info 0x1fb8adf800020264/78/93`（来源平台：950；证据等级：tool_observed）。
- 动态路径对照（同算子包）：aclnn 直连回归 `Test overwrite: PASSED / Test add: PASSED / summary: SUCCESS`（来源平台：950；证据等级：tool_observed）。
- 动态路径黑盒对照：L2 反向 25 例 24 FAIL（正确拒绝）+1 PASS；正向冒烟 30/30 PASS（来源平台：950；证据等级：tool_observed）。
- 修复前同族错误形态（旧二进制复测样例）：`aivec error exception, core id is 0, error code = 95, ... mte error info: 0xe0000c8000020264, ... mte error: 0x80000000`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限于该算子在 Ascend950PR + CANN 9.2.0 的图模式形态；不外推「GEIR 图模式普遍不可用」。
- 「workspace/管道语义不兼容」是假设，未验证；据此重构 kernel 前必须先复现并定位。
- 披露放行的处置依赖该算子 aclnn 通路为交付主口径的前提；若图模式是硬性交付项，本止损路线不适用。

**未决项**：

- 【优先项】复现并追图模式 aivec error 95/263 到机制层根因：建议接续点——对比图模式与动态路径下 kernel 实际收到的 workspace 指针/大小与 tiling 字段（kernel 内调试输出或 plog args 对照），核查 `GetUserWorkspace` 在图编译分配规则下的取址与 TPipe UB 预算假设是否成立。
- 图模式 staging 实际启动网格数（6）的决策链（为何 TilingParse 的 coreNum=1 未生效）未追查。
- 8 用例中 2 FAIL + 6 SKIP/NA 的 skip 语义未核对（是否为前置失败导致的跳过）。
## 坏实践

- 在未核缓存时效、未核网格失配的情况下，把图模式 EZ9999 直接归因为「图模式不支持自定义算子」而放弃。
- 反过来，在缓存与网格两因都排除后仍反复盲试相同修复路径期望图模式转绿——本卡即为该路径的负结果证据。
- 以状态机无 N/A 终态为由，把图模式未通过静默记为通过、不留失败证据。

## 误诊

- 首轮修复后的复测失败曾被归因为「网格修复无效」（agent_reported，会话内推理）——实际该轮跑的是缓存里的旧二进制；清缓存后才能得到干净复测。
- 「图模式失败 = 网格失配的另一种表现」的推断在干净复测后证伪：网格修复落地且缓存排除后仍全败。
