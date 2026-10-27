---
type: "Runbook"
title: "arch35 后端 64 位 div/mod 报 bitwidth>32"
description: "ascend950/dav-3510 后端对 uint64_t 的 / 与 % 报 Div of bitwidth greater than 32 not supported；kernel 索引运算须降至 uint32_t。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.language_extension"]
status: stable
sources: []
created_at: "2026-08-18T07:20:36Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["Div_of_bitwidth"]
---

# arch35 后端 64 位 div/mod 报 bitwidth>32

## 现象

在 ascend950 (dav-3510) 上编译含 `uint64_t` 除法/取模的 `__simd_vf__` kernel 时，前端编译通过，但 backend (LLVM/dav-3510 codegen) 阶段对全部 14 个 dtype 实例化报同一硬约束错误：

```
fatal error: error in backend: Div of bitwidth greater than 32 not supported
```

判别信号：**前端无语法错误 + backend 阶段全 dtype 一致失败 + 错误文本含 "Div of bitwidth greater than 32"** → 指向 64 位整数 div/mod 未被后端降级，而非算子逻辑或前端类型错误。

## 适用场景

- 工程形态：kernel 直调（`__simd_vf__` + `asc_vf_call` MicroAPI regbase 范式）
- API 路径：`uint64_t` 的 `/` 与 `%` 运算 → dav-3510 backend codegen
- 验证过的平台与 CANN 版本：ascend950 (dav-3510)，CANN 9.1.0（与 frontmatter `platforms` 严格一致）
- 适用粒度：`platform_wide`（dav-3510 平台所有 kernel 中的 64 位 div/mod 均受此 backend 约束，与算子语义无关）
- 路线优先级：`recommended_fastpath`（降至 uint32_t 是验证过的、可消除 backend 错误的直接路线）
- 明确不适用或未验证场景：a2/a3 平台未验证；shape 超过 uint32 范围（>42 亿元素）时降至 uint32 不安全，需软件 64 位除法或拆维

## 判别方法

**trigger（触发）**：kernel 中存在 `uint64_t` 类型的 `/` 或 `%` 运算。典型场景是用 `uint64_t` 存全局线性偏移 `gmOffset`、矩阵元素数 `mn`、列数 `N`，并做 `gmOffset % mn`（求当前 tile 起始的行内偏移）、`ij0 / N`（求行号）、`ij0 % N`（求列号）、`newJ / N` 与 `newJ % N`（推进行列索引）等索引推算；前端编译通过，但编译进入 backend 阶段后失败。

**probe（探针）**：清构建缓存后重编，grep 抓 backend div 错误与退出码：

```bash
rm -rf operators/<pkg>/build $HOME/atc_data/kernel_cache/ 2>/dev/null
cd operators && ./build.sh <pkg> 2>&1 | grep -E "Div of|error:|SUCCESS" | sort -u | head -10
echo "EXIT=${PIPESTATUS[0]}"
```

**observation（观测）**：probe 输出 `fatal error: error in backend: Div of bitwidth greater than 32 not supported`，且对算子全部 dtype 实例化（本例 14 个）一致出现；前端无语法错误，错误定位在 backend codegen 阶段而非 source 行号。这是 dav-3510 backend 的硬约束，非随机或偶发。

**interpretation（判读）**：dav-3510 后端（LLVM codegen）不支持将 64 位整数除法/取模指令降级为硬件指令，因此任何 `uint64_t` 的 `/`、`%` 在 backend 阶段触发该硬约束。这是平台指令集约束，与算子语义无关——改 compute 逻辑无效，必须在 div/mod 之前把操作数降至 ≤32 位整数（如 `static_cast<uint32_t>` 后再运算），或对超 uint32 范围的 shape 采用软件 64 位除法（移位减法循环）/拆维避免单次 div 超限。

## 预防

- 在 dav-3510 kernel 的 `__simd_vf__` 函数内，**所有 div/mod 运算的整数类型必须 ≤ 32 位**。
- 若 `gmOffset`/`mn`/`N` 从 host 侧以 `uint64_t`/`int64_t` 传入，函数签名可保持 `uint64_t` 以兼容 host 侧 TilingData 字段，但在函数入口 `static_cast<uint32_t>` 后再做 div/mod。
- 对超过 uint32 范围的 shape（如 `(16,8,8192,8192)` bf16 ≈ 86 亿元素），`static_cast<uint32_t>` 会截断，须改用软件 64 位除法或拆维避免单次 div 超限。

## 验证信号

来源说明：本判别规则来自 ascend950 (dav-3510) 上一次 Triu 算子 kernel 编译会话；清缓存重编后 backend 阶段对全部 dtype 实例化一致报 `Div of bitwidth greater than 32 not supported`，据此定位为后端 64 位 div/mod 不支持。

- `fatal error: error in backend: Div of bitwidth greater than 32 not supported` 在全 14 dtype 实例化复现（来源平台：950；证据等级：tool_observed）
- 前端编译通过、错误仅出现在 backend codegen 阶段（来源平台：950；证据等级：tool_observed）

## 失效边界

- a2/a3 平台：未验证，推断可迁移（未写入 frontmatter `platforms`）。
- shape 超过 uint32 范围（>42 亿元素）：降至 uint32 会截断，本判别的 uint32 解法不安全，需软件 64 位除法或拆维。
- 测试用例最大 shape 落在 uint32 范围内时，降至 uint32 安全。
