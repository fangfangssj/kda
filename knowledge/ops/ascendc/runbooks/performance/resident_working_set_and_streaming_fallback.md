---
type: Runbook
title: "完整工作集驻留 UB 前核对分配账与流式回退"
description: "从固定 GroupNorm 源码整理驻留工作集、dtype 条件暂存和流式分支参数区的完整静态账目，明确驻留条件与超预算停止边界。"
tags: [task.optimization, topic.memory, topic.implementation.tiling]
status: draft
sources: []
created_at: "2026-10-19T05:14:19Z"
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# 完整工作集驻留 UB 前核对分配账与流式回退

## 适用范围与证据

这是固定 Ascend C 源码的静态采用检查，平台和实际 CANN runtime 未在本轮复核。只分析 `group_norm_kernel` 入口内的驻留与两遍流式分支，不认证外层其他路由、编译产物或端到端性能。

来源身份为原源包（SHA256 `712076cb729cb501dc02c8070942053706c60a97f6653c456aae8b2ccf73ef3b`）的 `csrc/ops/group_norm/op_kernel/group_norm_kernel.cpp`（SHA256 `00866372c5c1335b3a1e5f9181cad3dd20364f2a017a0bbcf39545984c1056ab`）。行号均指这个完整文件。静态材料见[原贡献审阅包](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)（ZIP SHA-256 `15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`），精确目录为 `cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/`，其中 `groupnorm_allocation_excerpt.txt` 与 `source_structure.json` 的 `/allocation_statements/GroupNorm` 提供分配语句和原行号，`verify_budget_models.py`、`inputs.json`、`expected_output.json` 提供原创 CPU 整数账目模型及实际输出，许可声明见 `NOTICE.txt`、`LICENSE.CANN-2.0`。公开附件提供证据入口，技术证据为上述具体文件。

## 驻留条件不等于只计算输入大小

源文件 L26–28 的固定容量为 `SP_MAX_ELEMS=16384`、`SP_MAX_CPG=4096`、`SP_STATS=2048`；L50–51 只有同时满足 `Cpg*S <= 16384`、`Cpg <= 4096`、`S % 8 == 0` 才进入驻留分支。这里 `Cpg=C/num_groups`，调用前还应确认正维度及分组整除。本卡不证明源码已检查全部输入合同。

L75–84 的实际 `InitBuffer` 是按容量分配，并非按本次 `Cpg*S` 缩减。记 `d=sizeof(T)`：

| 分配项 | 字节 | 条件 |
|---|---:|---|
| xFBuf | 16384×4 | 所有 dtype |
| tmpBuf、accBuf、accSqBuf | 3×2048×4 | 所有 dtype |
| gBuf | 2×4096×4 | 所有 dtype |
| rBuf | 512 | 所有 dtype |
| xRawBuf | 16384×d | 仅非 float |
| gbRawBuf | 2×4096×d | 仅非 float |

因此 float 分支为 **123392B**；2B 输入分支为 **172544B**。这些容量均是 32B 倍数。数值只代表该源的显式 UB 分配，不含尚未核验的编译器／框架额外占用，不是芯片容量或通用安全余量。L96 的低精度 beta 起点还要按 16 个元素向上对齐；不能删掉参数暂存或只以 `x` 大小估预算。

## 流式分支也必须重算

L242、L255–270 选择 float 的 `T=5120`，或 2B 输入的 `T=7168`。完整显式账为：

`B_stream = 4*T*d + 3*T*4 + Align32(max(T*4, (Align8(Cpg)+Cpg)*4)) + 512`。

第一项是输入和输出各 depth=2；第二项包括 fBuf、accBuf、accSqBuf；第三项是 fBuf2 的整块分配，先作平方暂存、后容纳 gamma/beta。**不同阶段复用不意味着两份独立分配，取二者最大容量；也不能漏掉较大的参数区。**

当参数区不超过 tile 工作区时，两 dtype 分别为 164352B 与 172544B。参数区随 `Cpg` 增长；例如合成 `Cpg=20000,d=2` 的显式账为 303872B。原入口没有在这组分配前读取实际 UB 容量并作总量拒绝，因此本卡不把“进入流式 fallback”当成所有形状的安全证明。迁移时应按目标资源加最终适配检查，超额则缩小／拆分参数工作集或停止该路径。

## 驻留与复用还要满足完成关系

完整工作集可容纳只是必要条件。向量读者、MTE2 写者及 MTE3 输出读者可能跨阶段存活；只有最终读者完成后才可覆写共享存储。这里不输出源中的调度经验，也不由源码顺序推断异步完成。复核方向见[按 dtype 分配 UB 必须匹配 tiling 预算](../compilation/dtype_conditional_ub_allocation_matches_tiling_budget.md)；具体同步迁移须另外核对编译与执行证据。

## 已执行验证与停止边界

本轮只核对精确内包／源码哈希、上述全部分配语句和分支条件，并执行原创 CPU 整数账目模型。合成用例覆盖驻留容量等号两侧、对齐拒绝、参数区扩大与流式超额；它们不是历史评测输入，也未运行原 kernel。

该旧公开目录 `source_receipts.json` 的 `/sources/0` 记录原源码、源包和 report 哈希及同提交的历史关联摘要；它是派生 receipt，不是公开的原单条评测记录，不据此新增平台认证或本轮设备通过结论。这个关联不构成驻留机制的单因素收益、精度优势或本轮设备验证。只有取得同版本／平台、同调用路径的独立对照后，才可增加性能或同步充分性结论。